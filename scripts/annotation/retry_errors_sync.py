"""
Retry annotation errors using synchronous LiteLLM requests.

This helper mirrors the retry-errors path for batch jobs but executes the
requests synchronously via classify_chats-style calls. It prefers non-error
records when deduplicating and writes outputs in the same JSONL format as
classify_chats.py.
"""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS,
    add_llm_common_arguments,
    disable_litellm_logging,
)
from llm_delusions_annotations.classify_messages import classify_tasks_batch
from llm_delusions_annotations.configs import AnnotationConfig
from llm_delusions_annotations.llm_utils.client import DEFAULT_CHAT_MODEL

from annotation.io import (
    infer_job_stem_from_filename,
    is_quote_mismatch_error,
    iter_jsonl_records,
)
from annotation.pipeline import ensure_output_and_write_outcomes_for_context
from annotation.retry_utils import build_retry_tasks, load_retry_meta
from utils.cli import (
    add_model_argument,
    add_output_path_argument,
    extract_non_default_arguments_with_model,
)
from utils.io import collect_family_files
from utils.participants import participant_from_record
from utils.utils import normalize_arg_value

DEFAULT_MAX_WORKERS = 16
DEFAULT_SYNC_BATCH_SIZE = 128

disable_litellm_logging()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for sync retry errors."""

    parser = argparse.ArgumentParser(
        description=(
            "Retry errored annotation outputs using synchronous LiteLLM calls."
        )
    )
    parser.add_argument(
        "--retry-errors-from",
        type=Path,
        required=True,
        help=(
            "Reference JSONL output file to scan for retryable errors. "
            "Only records with non-quote-mismatch errors are retried."
        ),
    )
    parser.add_argument(
        "--retry-errors-all-ppts",
        action="store_true",
        help=(
            "When set, scan all participant JSONL files sharing the same job "
            "stem under --output instead of limiting to the reference file."
        ),
    )
    add_output_path_argument(
        parser,
        default_path="annotation_outputs",
        help_text=(
            "Root directory for annotation outputs (default: annotation_outputs)."
        ),
    )
    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)
    add_llm_common_arguments(parser)
    parser.add_argument(
        "--job",
        type=str,
        help=(
            "Optional job name to use when writing outputs. Defaults to the "
            "stem of --retry-errors-from so outputs append to the same family."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Maximum concurrent requests for synchronous batch completion "
            f"(default: {DEFAULT_MAX_WORKERS})."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_SYNC_BATCH_SIZE,
        help=(
            "Number of retry tasks to send per synchronous batch "
            f"(default: {DEFAULT_SYNC_BATCH_SIZE})."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help=(
            "Optional cap on the number of retry tasks to run. " "Set to 0 to disable."
        ),
    )

    args = parser.parse_args(argv)
    defaults = {key: parser.get_default(key) for key in vars(args)}
    setattr(args, "_defaults", defaults)
    return args


def _collect_retry_family_files(
    retry_path: Path,
    outputs_root: Path,
    *,
    include_all_participants: bool,
) -> List[Path]:
    """Return retry JSONL files under ``outputs_root`` matching the job stem."""

    family_files = collect_family_files(retry_path, outputs_root)
    if include_all_participants:
        return family_files or [retry_path]
    same_parent = [path for path in family_files if path.parent == retry_path.parent]
    return same_parent or [retry_path]


def _chunk_tasks(
    tasks: Sequence,
    batch_size: int,
) -> Iterable[List]:
    """Yield tasks in fixed-size batches."""

    if batch_size <= 0:
        batch_size = DEFAULT_SYNC_BATCH_SIZE
    for start in range(0, len(tasks), batch_size):
        yield list(tasks[start : start + batch_size])


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Script entry point for synchronous retry errors."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    retry_path = args.retry_errors_from.expanduser().resolve()
    if not retry_path.exists():
        logging.error("Retry JSONL not found: %s", retry_path)
        return 2

    output_root = Path(args.output).expanduser().resolve()
    if not output_root.exists() or not output_root.is_dir():
        logging.error("Outputs root not found: %s", output_root)
        return 2

    retry_files = _collect_retry_family_files(
        retry_path,
        output_root,
        include_all_participants=bool(getattr(args, "retry_errors_all_ppts", False)),
    )
    if not retry_files:
        logging.error("No retry outputs found under %s", output_root)
        return 2

    meta = load_retry_meta(retry_path)
    defaults = getattr(args, "_defaults", {})
    if isinstance(meta.get("model"), str) and normalize_arg_value(
        getattr(args, "model", None)
    ) == normalize_arg_value(defaults.get("model")):
        args.model = str(meta.get("model"))

    annotation_specs: dict[str, Mapping[str, object]] = {
        str(item.get("id")): item for item in ANNOTATIONS
    }

    latest_error_records: dict[Tuple[str, str, int, int, str], dict[str, object]] = {}
    success_keys: Set[Tuple[str, str, int, int, str]] = set()
    for file_path in retry_files:
        try:
            for record in iter_jsonl_records(file_path):
                participant = participant_from_record(record)
                source_path = str(record.get("source_path") or "").strip()
                annotation_id = str(record.get("annotation_id") or "").strip()
                if not participant or not source_path or not annotation_id:
                    continue
                try:
                    chat_index = int(record.get("chat_index"))
                    message_index = int(record.get("message_index"))
                except (ValueError, TypeError):
                    continue
                key = (
                    participant,
                    source_path,
                    chat_index,
                    message_index,
                    annotation_id,
                )
                error_value = record.get("error")
                if not error_value:
                    success_keys.add(key)
                    if key in latest_error_records:
                        latest_error_records.pop(key, None)
                    continue
                error_text = str(error_value)
                if is_quote_mismatch_error(error_text):
                    continue
                if key in success_keys:
                    continue
                latest_error_records[key] = record
        except OSError:
            continue

    tasks = build_retry_tasks(
        latest_error_records,
        success_keys,
        annotation_specs=annotation_specs,
    )
    if not tasks:
        logging.info("No retryable error records found.")
        return 0
    if args.max_messages and args.max_messages > 0:
        tasks = tasks[: int(args.max_messages)]
    logging.info("Sync retry will process %s tasks.", len(tasks))

    job_name = str(args.job or infer_job_stem_from_filename(retry_path.name)).strip()
    if not job_name:
        logging.error("Retry job name could not be derived from %s", retry_path)
        return 2

    configs = [
        AnnotationConfig(spec=spec, allowed_roles=None)
        for spec in annotation_specs.values()
    ]
    non_default_arguments = extract_non_default_arguments_with_model(args)
    non_default_arguments = dict(non_default_arguments)
    non_default_arguments["retry_errors_from"] = str(retry_path)

    args_for_output = argparse.Namespace(
        model=args.model,
        preceding_context=0,
        follow_links=False,
    )

    total_written = 0
    output_handles = {}
    with ExitStack() as stack:
        for batch_index, task_batch in enumerate(
            _chunk_tasks(tasks, int(args.batch_size)),
            start=1,
        ):
            logging.info(
                "Running sync batch %s with %s tasks.",
                batch_index,
                len(task_batch),
            )
            outcomes = classify_tasks_batch(
                task_batch,
                model=args.model,
                timeout=int(args.timeout),
                max_workers=int(args.max_workers),
            )
            for outcome in outcomes:
                wrote = ensure_output_and_write_outcomes_for_context(
                    context=outcome.task.context,
                    outcomes=[outcome],
                    args=args_for_output,
                    configs=configs,
                    output_dir=output_root,
                    single_output_file=None,
                    resolved_output_name=f"{job_name}.jsonl",
                    non_default_arguments=non_default_arguments,
                    output_handles=output_handles,
                    stack=stack,
                )
                if not wrote:
                    return 2
                total_written += 1
            if args.sleep and args.sleep > 0:
                time.sleep(float(args.sleep))

    logging.info("Sync retry complete. Wrote %s records.", total_written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
