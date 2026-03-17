"""
Batch-oriented entry point for chat annotation classification.

This script mirrors the ``classify_chats.py`` interface but is organized
around provider batch workflows. The two primary subcommands are:

- ``submit``: select and package messages into provider batches and write
  manifest JSONL files describing each batch.
- ``harvest``: read existing manifests, attach to completed batches, and
  write classification outputs under the annotation outputs directory.

Behavior is not yet implemented; this module currently provides a typed
argument parser and subcommand skeletons ready for wiring.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, TextIO, Tuple

import litellm
import openai
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATIONS,
    add_llm_common_arguments,
    disable_litellm_logging,
)
from llm_delusions_annotations.classify_messages import (
    MAX_CLASSIFICATION_TOKENS,
    ClassificationError,
    ClassificationOutcome,
    ClassificationTask,
    MessageContext,
    build_completion_messages,
    extract_matches_from_response_text,
)
from llm_delusions_annotations.configs import AnnotationConfig, load_annotation_configs
from llm_delusions_annotations.llm_utils.client import (
    DEFAULT_CHAT_MODEL,
    apply_reasoning_defaults,
)
from llm_delusions_annotations.utils import to_litellm_messages
from tqdm import tqdm

from annotation.batch_manifest import (
    ManifestConfig,
    create_manifest,
    decode_custom_id,
    encode_custom_id,
    iter_manifests,
    load_manifest_tasks,
    update_manifest_status,
)
from annotation.io import (
    ReplayKey,
    SeenKey,
    infer_job_stem_from_filename,
    is_quote_mismatch_error,
    iter_jsonl_meta,
    iter_jsonl_records,
    load_latest_retryable_error_keys,
    load_resume_keys,
)
from annotation.manifest_summary import (
    print_annotation_stats,
    print_duplicate_warnings,
    print_participant_stats,
    print_token_cost_summary,
)
from annotation.pipeline import (
    build_base_record,
    build_classification_tasks_for_context,
    ensure_output_and_write_outcomes_for_context,
    prepare_message_iterator,
)
from annotation.retry_utils import build_retry_tasks, load_retry_meta
from llm_utils.litellm_batch import (
    BatchFailedError,
    BatchTimeoutError,
    create_litellm_batch,
    resume_litellm_batch,
    to_batch_json,
)
from utils.cli import (
    Spinner,
    add_chat_io_arguments,
    add_chat_sampling_arguments,
    add_follow_links_argument,
    add_model_argument,
    add_participants_argument,
    add_score_cutoff_argument,
    extract_non_default_arguments_with_model,
)
from utils.io import collect_family_files
from utils.participants import (
    normalize_participant_filter,
    normalize_participant_value,
    participant_from_record,
)
from utils.utils import normalize_arg_value

# When not debugging, silence LiteLLM's own logger for batch workflows.
disable_litellm_logging()

# Soft cap for the LiteLLM batch input file size. Provider limits are around
# 100 MB; this margin aims to stay comfortably under that.
MAX_BATCH_INPUT_FILE_BYTES = 90 * 1024 * 1024

# Default daily token budget for gpt-5.1 batch classification jobs.
DEFAULT_MAX_TOKENS = 15_000_000_000

MAX_MIN_WORKERS = 32


def _add_common_submit_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments shared with classify_chats to a submit subparser."""

    add_chat_io_arguments(
        parser,
        default_output_dir="annotation_outputs",
        output_help=(
            "Root directory for annotation outputs (default: annotation_outputs)."
        ),
    )

    parser.add_argument(
        "--annotation",
        "-a",
        action="append",
        help=(
            "Annotation ID to use for classification (repeatable). "
            "When omitted, defaults to all non-test annotations."
        ),
    )

    add_participants_argument(
        parser,
        help_text=(
            "Restrict processing to chats under these participant IDs "
            "(repeatable). Defaults to all participants."
        ),
    )

    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)

    add_chat_sampling_arguments(
        parser,
        max_messages_help=(
            "Optional cap on the number of messages to include across all batches. "
            "Set to 0 to process all messages."
        ),
    )

    add_follow_links_argument(parser)
    parser.add_argument(
        "--prefilter-conversations",
        action="store_true",
        help=(
            "Skip the remainder of a chat when no annotations match within the "
            "first few turns."
        ),
    )

    # Shared scoring-related arguments used during later analysis.
    add_score_cutoff_argument(parser)

    parser.add_argument(
        "--job",
        type=str,
        help=(
            "Optional short job name used in output basenames and manifests. "
            "When omitted, a name will be derived from non-default arguments."
        ),
    )

    # Batch-specific knobs that will be consumed by the provider workflow.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help=("Maximum number of classification tasks to include per provider batch "),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help=(
            "Directory where batch manifest JSONL files will be written. "
            "Defaults to a job-specific subdirectory under the output root."
        ),
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help=(
            "Provider-specific completion window for batches (for example '24h'). "
            "Exact semantics depend on the LiteLLM provider."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prepare batch manifests without creating provider batches. "
            "All manifests will be written with pending status for later enqueue."
        ),
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum estimated tokens to submit in this run. Set to 0 to disable "
            "token budgeting. Estimates rely on litellm.token_counter."
        ),
    )

    add_llm_common_arguments(parser)


def _add_harvest_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments for the harvest subcommand."""

    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        default="annotation_outputs",
        help="Root directory where annotation JSONL outputs will be written.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help=(
            "Directory containing batch manifest JSONL files to harvest. "
            "When omitted, a job-specific subdirectory under the output root "
            "will be used when --job is provided."
        ),
    )


def _add_enqueue_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments for the enqueue subcommand."""

    parser.add_argument(
        "--manifest-dir",
        type=Path,
        required=True,
        help="Directory containing batch manifest JSONL files to enqueue.",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help=(
            "Provider-specific completion window for batches (for example '24h'). "
            "Exact semantics depend on the LiteLLM provider."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum estimated tokens to submit in this run. Set to 0 to disable "
            "token budgeting. Estimates rely on litellm.token_counter."
        ),
    )
    parser.add_argument(
        "--job",
        type=str,
        help=(
            "Optional job name used to select which manifests to harvest when "
            "multiple jobs share the same manifest root."
        ),
    )


def _add_retry_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments for retrying errored records from prior outputs."""

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
    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        default="annotation_outputs",
        help=(
            "Root directory containing annotation JSONL outputs "
            "(default: annotation_outputs)."
        ),
    )
    parser.add_argument(
        "--job",
        type=str,
        help=(
            "Optional job name to use when writing retry manifests. Defaults "
            "to the stem of --retry-errors-from so harvest appends to the "
            "same output family."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help=("Maximum number of retry tasks to include per batch manifest."),
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help=(
            "Directory where retry manifest JSONL files will be written. "
            "Defaults to a job-specific subdirectory under the output root."
        ),
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help=(
            "Provider-specific completion window for batches (for example '24h'). "
            "Exact semantics depend on the LiteLLM provider."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Prepare retry manifests without creating provider batches. "
            "All manifests will be written with pending status for later enqueue."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=(
            "Maximum estimated tokens to submit in this retry run. Set to 0 to "
            "disable token budgeting. Estimates rely on litellm.token_counter."
        ),
    )
    add_model_argument(parser, default_model=DEFAULT_CHAT_MODEL)
    add_llm_common_arguments(parser)


def _add_retry_missing_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach arguments for retrying missing records from prior outputs."""

    _add_common_submit_arguments(parser)
    parser.add_argument(
        "--missing-from",
        type=Path,
        required=True,
        help=(
            "Reference JSONL output file to scan for coverage. Missing records "
            "and non-quote errors are computed relative to this output family."
        ),
    )
    parser.add_argument(
        "--missing-single-ppt",
        action="store_true",
        help=(
            "When set, only scan the reference JSONL file instead of including "
            "all participant files with the same job stem."
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for classify_chats_batch."""

    parser = argparse.ArgumentParser(
        description=(
            "Batch-oriented annotation classifier using LiteLLM provider batches. "
            "Use the 'submit' subcommand to enqueue work and 'harvest' to "
            "collect completed results."
        )
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Subcommand to run (submit, enqueue, or harvest).",
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Prepare and submit provider batches for classification.",
    )
    _add_common_submit_arguments(submit_parser)

    harvest_parser = subparsers.add_parser(
        "harvest",
        help="Harvest completed batches described by manifest files.",
    )
    _add_harvest_arguments(harvest_parser)

    enqueue_parser = subparsers.add_parser(
        "enqueue",
        help="Submit pending batch manifests without re-reading transcripts.",
    )
    _add_enqueue_arguments(enqueue_parser)

    retry_parser = subparsers.add_parser(
        "retry-errors",
        help="Retry non-quote-mismatch errors from prior annotation outputs.",
    )
    _add_retry_arguments(retry_parser)
    retry_missing_parser = subparsers.add_parser(
        "retry-missing",
        help="Retry missing coverage from prior annotation outputs.",
    )
    _add_retry_missing_arguments(retry_missing_parser)

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO).",
    )

    args = parser.parse_args(argv)
    defaults = {key: parser.get_default(key) for key in vars(args)}
    setattr(args, "_defaults", defaults)
    return args


def _estimate_tokens_for_messages(
    model: str,
    messages: List[dict[str, object]],
) -> int:
    """Return estimated prompt and completion tokens for a single request.

    This helper uses :func:`litellm.token_counter`, which can be relatively
    expensive. Callers should avoid using it in dry runs or when token
    budgeting is disabled.
    """

    return int(
        litellm.token_counter(
            model=model,
            messages=messages,
            count_response_tokens=True,
            default_token_count=MAX_CLASSIFICATION_TOKENS,
        )
    )


def _build_batch_payload_for_tasks(
    *,
    tasks: List[ClassificationTask],
    args: argparse.Namespace,
) -> tuple[dict[str, List[dict[str, object]]], List[dict[str, object]], int]:
    """Return LiteLLM payload, manifest records, and estimated tokens."""

    keys_to_messages: dict[str, List[dict[str, object]]] = {}
    manifest_tasks: List[dict[str, object]] = []

    for task in tasks:
        context = task.context
        annotation = task.annotation
        custom_id = encode_custom_id(
            context.participant,
            str(context.source_path),
            context.chat_index,
            context.message_index,
            str(annotation.get("id", "")),
        )
        messages = to_litellm_messages(
            build_completion_messages(
                task.prompt,
                system_prompt=ANNOTATION_SYSTEM_PROMPT,
            )
        )
        keys_to_messages[custom_id] = messages

        base_record = build_base_record(context)
        manifest_record: dict[str, object] = {
            "custom_id": custom_id,
            **base_record,
            "annotation_id": annotation.get("id", ""),
            "annotation_name": annotation.get("name", ""),
            "prompt": task.prompt,
            # Persist the message content so harvest can perform quote
            # validation against the correct text rather than an empty
            # placeholder.
            "content": context.content,
            "preceding": context.preceding or [],
        }
        manifest_tasks.append(manifest_record)

    estimated_tokens = 0
    track_tokens = bool(
        getattr(args, "max_tokens", 0) or getattr(args, "dry_run", False)
    )
    if track_tokens and keys_to_messages:
        max_workers = min(MAX_MIN_WORKERS, len(keys_to_messages))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _estimate_tokens_for_messages,
                    args.model,
                    messages,
                )
                for messages in keys_to_messages.values()
            ]
            for future in as_completed(futures):
                estimated_tokens += int(future.result())

    return keys_to_messages, manifest_tasks, estimated_tokens


def _flush_batch(
    *,
    job_name: str,
    batch_index: int,
    tasks: List[ClassificationTask],
    args: argparse.Namespace,
    manifest_dir: Path,
    non_default_arguments: Mapping[str, object],
) -> None:
    """Create a provider batch and corresponding manifest for a group of tasks."""

    if not tasks:
        return

    keys_to_messages, manifest_tasks, estimated_tokens = _build_batch_payload_for_tasks(
        tasks=tasks,
        args=args,
    )

    batch_model, provider, _api_key, _api_base = litellm.get_llm_provider(args.model)
    request_parameters: dict[str, object] = {"model": batch_model}
    apply_reasoning_defaults(
        args.model,
        request_parameters,
        max_completion_tokens=MAX_CLASSIFICATION_TOKENS,
    )

    batch_bytes = len(
        to_batch_json(
            keys_to_messages,
            endpoint="/v1/chat/completions",
            request_parameters=request_parameters,
        )
    )
    if batch_bytes > MAX_BATCH_INPUT_FILE_BYTES:
        logging.error(
            "Batch %s for job %s exceeds max input file size: %s bytes "
            "(limit %s bytes). Reduce --batch-size and retry.",
            batch_index,
            job_name,
            batch_bytes,
            MAX_BATCH_INPUT_FILE_BYTES,
        )
        raise ValueError("Batch input file too large for provider limits.")

    if getattr(args, "dry_run", False):
        manifest_arguments = dict(non_default_arguments)
        if estimated_tokens:
            manifest_arguments["estimated_tokens"] = int(estimated_tokens)
        manifest_arguments["batch_bytes"] = int(batch_bytes)

        manifest_config = ManifestConfig(
            job_name=job_name,
            batch_id="",
            input_file_id="",
            model=args.model,
            provider=provider,
            endpoint="/v1/chat/completions",
            arguments=manifest_arguments,
        )
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{job_name}_batch_{batch_index:05d}.jsonl"
        create_manifest(
            manifest_config,
            tasks=manifest_tasks,
            manifest_path=manifest_path,
        )
        update_manifest_status(manifest_path, "pending")
        if estimated_tokens:
            current_tokens = getattr(args, "_tokens_used", 0)
            setattr(args, "_tokens_used", current_tokens + estimated_tokens)
        logging.info(
            "Dry run: prepared manifest %s with %s tasks (no provider batch created).",
            manifest_path,
            len(tasks),
        )
        return

    max_tokens = getattr(args, "max_tokens", DEFAULT_MAX_TOKENS)
    if max_tokens is not None and max_tokens > 0:
        current_tokens = getattr(args, "_tokens_used", 0)
        projected_total = current_tokens + estimated_tokens
        if projected_total > max_tokens:
            logging.info(
                "Token budget reached (current=%s, batch_estimate=%s, limit=%s); "
                "writing pending manifest without submitting provider batch.",
                current_tokens,
                estimated_tokens,
                max_tokens,
            )
            manifest_arguments = dict(non_default_arguments)
            if estimated_tokens:
                manifest_arguments["estimated_tokens"] = int(estimated_tokens)
            manifest_arguments["batch_bytes"] = int(batch_bytes)

            manifest_config = ManifestConfig(
                job_name=job_name,
                batch_id="",
                input_file_id="",
                model=args.model,
                provider=provider,
                endpoint="/v1/chat/completions",
                arguments=manifest_arguments,
            )
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{job_name}_batch_{batch_index:05d}.jsonl"
            create_manifest(
                manifest_config,
                tasks=manifest_tasks,
                manifest_path=manifest_path,
            )
            update_manifest_status(manifest_path, "pending")
            return

    batch_id, input_file_id = create_litellm_batch(
        keys_to_messages,
        litellm_client=litellm,
        custom_llm_provider=provider,
        endpoint="/v1/chat/completions",
        completion_window=args.completion_window,
        request_parameters=request_parameters,
    )

    manifest_arguments = dict(non_default_arguments)
    if estimated_tokens:
        manifest_arguments["estimated_tokens"] = int(estimated_tokens)
    manifest_arguments["batch_bytes"] = int(batch_bytes)

    manifest_config = ManifestConfig(
        job_name=job_name,
        batch_id=batch_id,
        input_file_id=input_file_id,
        model=args.model,
        provider=provider,
        endpoint="/v1/chat/completions",
        arguments=manifest_arguments,
    )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{job_name}_batch_{batch_index:05d}.jsonl"
    create_manifest(
        manifest_config,
        tasks=manifest_tasks,
        manifest_path=manifest_path,
    )

    logging.info(
        "Created batch %s (job=%s, tasks=%s) with manifest %s",
        batch_id,
        job_name,
        len(tasks),
        manifest_path,
    )
    current_tokens = getattr(args, "_tokens_used", 0)
    setattr(args, "_tokens_used", current_tokens + estimated_tokens)


def _flush_batch_with_splitting(
    *,
    job_name: str,
    batch_index: int,
    tasks: List[ClassificationTask],
    args: argparse.Namespace,
    manifest_dir: Path,
    non_default_arguments: Mapping[str, object],
) -> int:
    """Flush tasks into one or more batches, splitting when too large.

    Returns the next available batch index after all sub-batches have been
    written. When a single-task batch exceeds provider limits, a ValueError
    is raised.
    """

    if not tasks:
        return batch_index

    try:
        _flush_batch(
            job_name=job_name,
            batch_index=batch_index,
            tasks=tasks,
            args=args,
            manifest_dir=manifest_dir,
            non_default_arguments=non_default_arguments,
        )
        return batch_index + 1
    except ValueError:
        if len(tasks) <= 1:
            # Individual task cannot be represented within provider limits.
            raise
        mid = len(tasks) // 2
        next_index = _flush_batch_with_splitting(
            job_name=job_name,
            batch_index=batch_index,
            tasks=tasks[:mid],
            args=args,
            manifest_dir=manifest_dir,
            non_default_arguments=non_default_arguments,
        )
    return _flush_batch_with_splitting(
        job_name=job_name,
        batch_index=next_index,
        tasks=tasks[mid:],
        args=args,
        manifest_dir=manifest_dir,
        non_default_arguments=non_default_arguments,
    )


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


def _update_task_stats(
    tasks: Sequence[ClassificationTask],
    *,
    annotation_counts: Counter[str],
    participant_request_counts: Counter[str],
    participant_annotation_counts: Dict[str, Counter[str]],
    participant_message_keys: Dict[str, Set[Tuple[str, int, int]]],
    duplicate_key_counts: Counter[Tuple[str, str, int, int, str]],
) -> None:
    """Update summary counters based on classification tasks."""

    for task in tasks:
        context = task.context
        annotation = task.annotation
        ann_id_raw = annotation.get("id")
        ann_id = str(ann_id_raw or "").strip()
        if ann_id:
            annotation_counts[ann_id] += 1

        participant_raw = context.participant
        participant = normalize_participant_value(participant_raw)
        if participant:
            participant_request_counts[participant] += 1
            if ann_id:
                participant_annotation_counts[participant][ann_id] += 1

            source_path = str(context.source_path)
            chat_index = context.chat_index
            message_index = context.message_index
            key = (source_path, chat_index, message_index)
            participant_message_keys[participant].add(key)
            if ann_id:
                dup_key = (
                    participant,
                    source_path,
                    chat_index,
                    message_index,
                    ann_id,
                )
                duplicate_key_counts[dup_key] += 1


def run_submit(args: argparse.Namespace) -> int:
    """Entry point for the 'submit' subcommand."""

    if not args.job or not args.job.strip():
        logging.error(
            "The submit subcommand requires a non-empty --job name to "
            "organize batch manifests."
        )
        return 2

    input_root = Path(args.input).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        logging.error("Input directory not found: %s", input_root)
        return 2

    configs: List[AnnotationConfig] = load_annotation_configs(args.annotation)
    if not configs:
        logging.error("No annotation configurations resolved for submit.")
        return 2

    normalized_filter = normalize_participant_filter(args.participants)
    participants_filter: Optional[Sequence[str]] = (
        sorted(normalized_filter) if normalized_filter else None
    )
    replay_keys: Optional[Set[ReplayKey]] = None

    message_iter, max_messages, progress_total, _ = prepare_message_iterator(
        args,
        input_root,
        configs,
        participants_filter,
        replay_keys,
    )

    non_default_arguments = extract_non_default_arguments_with_model(args)
    manifest_dir = (
        args.manifest_dir
        if args.manifest_dir is not None
        else Path(args.output_dir).expanduser().resolve() / "batch_manifests" / args.job
    )

    resume_seen_keys: Set[SeenKey] = set()
    min_positive = 0
    positive_counts: Mapping[str, int] = {}

    batch_index = 0
    tasks_for_batch: List[ClassificationTask] = []
    processed_messages = 0
    total_tasks = 0
    annotation_counts: Counter[str] = Counter()
    participant_request_counts: Counter[str] = Counter()
    participant_annotation_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    participant_message_keys: Dict[str, Set[Tuple[str, int, int]]] = defaultdict(set)
    duplicate_key_counts: Counter[Tuple[str, str, int, int, str]] = Counter()
    setattr(args, "_tokens_used", 0)

    with tqdm(
        total=progress_total,
        desc="Preparing batches",
        disable=not sys.stderr.isatty(),
    ) as progress:
        for context in message_iter:
            if max_messages is not None and processed_messages >= max_messages:
                break

            tasks_for_context = build_classification_tasks_for_context(
                context,
                configs,
                resume_seen_keys,
                min_positive,
                positive_counts,
                args=args,
                target_keys=None,
            )
            if not tasks_for_context:
                continue

            # Update aggregate statistics for pretty-printing at the end
            # of the submit run.
            for task in tasks_for_context:
                context = task.context
                annotation = task.annotation
                ann_id_raw = annotation.get("id")
                ann_id = str(ann_id_raw or "").strip()
                if ann_id:
                    annotation_counts[ann_id] += 1

                participant_raw = context.participant
                participant = normalize_participant_value(participant_raw)
                if participant:
                    participant_request_counts[participant] += 1
                    if ann_id:
                        participant_annotation_counts[participant][ann_id] += 1

                    source_path = str(context.source_path)
                    chat_index = context.chat_index
                    message_index = context.message_index
                    key = (source_path, chat_index, message_index)
                    participant_message_keys[participant].add(key)
                    if ann_id:
                        dup_key = (
                            participant,
                            source_path,
                            chat_index,
                            message_index,
                            ann_id,
                        )
                        duplicate_key_counts[dup_key] += 1

            tasks_for_batch.extend(tasks_for_context)
            total_tasks += len(tasks_for_context)

            processed_messages += 1
            progress.update(1)

            if len(tasks_for_batch) >= max(1, int(args.batch_size)):
                batch_index = _flush_batch_with_splitting(
                    job_name=args.job,
                    batch_index=batch_index,
                    tasks=tasks_for_batch,
                    args=args,
                    manifest_dir=manifest_dir,
                    non_default_arguments=non_default_arguments,
                )
                tasks_for_batch = []

    if tasks_for_batch:
        batch_index = _flush_batch_with_splitting(
            job_name=args.job,
            batch_index=batch_index,
            tasks=tasks_for_batch,
            args=args,
            manifest_dir=manifest_dir,
            non_default_arguments=non_default_arguments,
        )

    total_tokens = int(getattr(args, "_tokens_used", 0))
    if getattr(args, "dry_run", False):
        logging.info(
            "Submit dry run complete. Prepared %s batches for job %s. "
            "Estimated tokens: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            args.job,
            total_tokens,
        )
    else:
        logging.info(
            "Submit complete. Prepared %s batches for job %s. "
            "Estimated tokens submitted: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            args.job,
            total_tokens,
        )

    # Pretty-print a token and request summary similar to
    # tmp_sum_manifest_tokens.py so that large runs do not require a
    # separate analysis step.
    print_token_cost_summary(
        model=args.model,
        total_tokens=total_tokens,
        total_tasks=total_tasks,
    )
    if annotation_counts:
        print_annotation_stats(annotation_counts)
    if participant_request_counts:
        print_participant_stats(
            participant_request_counts,
            participant_annotation_counts,
            participant_message_keys,
        )
    print_duplicate_warnings(duplicate_key_counts)

    return 0


def run_enqueue(args: argparse.Namespace) -> int:
    """Entry point for the 'enqueue' subcommand."""

    manifest_root = args.manifest_dir.expanduser().resolve()
    if not manifest_root.exists() or not manifest_root.is_dir():
        logging.error("Manifest directory not found: %s", manifest_root)
        return 2

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else 0
    tokens_used = 0

    # Collect all pending manifests first so we can validate budgeting
    # requirements before enqueuing any batches.
    pending_manifests = []
    for summary in iter_manifests(manifest_root):
        status = str(summary.meta.get("status") or "submitted").lower()
        if status == "pending":
            pending_manifests.append(summary)

    if max_tokens:
        for summary in pending_manifests:
            args_meta = dict(summary.meta.get("arguments") or {})
            estimated_tokens = int(args_meta.get("estimated_tokens") or 0)
            if not estimated_tokens:
                logging.error(
                    "Manifest %s is missing estimated_tokens required for "
                    "token budgeting. Ensure submit was run with token "
                    "estimation enabled.",
                    summary.path,
                )
                return 2

    skipped_manifests: List[Path] = []

    for summary in pending_manifests:
        meta = summary.meta
        model = str(meta.get("model") or DEFAULT_CHAT_MODEL)
        manifest_meta, task_records = load_manifest_tasks(summary.path)
        if not task_records:
            logging.info("Skipping empty manifest %s", summary.path)
            continue

        args_meta = dict(manifest_meta.get("arguments") or {})
        estimated_tokens = int(args_meta.get("estimated_tokens") or 0)

        if max_tokens:
            projected_total = tokens_used + estimated_tokens
            if projected_total > max_tokens:
                logging.info(
                    "Token budget reached while enqueueing (current=%s, "
                    "batch_estimate=%s, limit=%s); leaving manifest pending: %s",
                    tokens_used,
                    estimated_tokens,
                    max_tokens,
                    summary.path,
                )
                continue

        keys_to_messages: dict[str, List[dict[str, object]]] = {}
        for task in task_records:
            custom_id = str(task.get("custom_id") or "").strip()
            prompt = str(task.get("prompt") or "")
            if not custom_id or not prompt:
                continue
            messages = to_litellm_messages(
                build_completion_messages(
                    prompt,
                    system_prompt=ANNOTATION_SYSTEM_PROMPT,
                )
            )
            keys_to_messages[custom_id] = messages

        if not keys_to_messages:
            logging.info("No usable tasks found in manifest %s", summary.path)
            update_manifest_status(summary.path, "failed")
            continue

        batch_model, inferred_provider, _api_key, _api_base = litellm.get_llm_provider(
            model,
            custom_llm_provider=str(manifest_meta.get("provider") or ""),
        )
        request_parameters: dict[str, object] = {"model": batch_model}
        apply_reasoning_defaults(
            model,
            request_parameters,
            max_completion_tokens=MAX_CLASSIFICATION_TOKENS,
        )

        batch_bytes = len(
            to_batch_json(
                keys_to_messages,
                endpoint="/v1/chat/completions",
                request_parameters=request_parameters,
            )
        )
        if batch_bytes > MAX_BATCH_INPUT_FILE_BYTES:
            logging.error(
                "Manifest %s would create a batch exceeding max input file size: "
                "%s bytes (limit %s bytes).",
                summary.path,
                batch_bytes,
                MAX_BATCH_INPUT_FILE_BYTES,
            )
            update_manifest_status(summary.path, "failed")
            continue

        try:
            provider = str(
                manifest_meta.get("provider") or inferred_provider or "openai"
            )
            batch_id, input_file_id = create_litellm_batch(
                keys_to_messages,
                litellm_client=litellm,
                custom_llm_provider=provider,
                endpoint=str(manifest_meta.get("endpoint") or "/v1/chat/completions"),
                completion_window=args.completion_window,
                request_parameters=request_parameters,
            )
        except openai.APIConnectionError as err:
            logging.warning(
                "Connection error while creating batch for manifest %s: %s. "
                "Leaving manifest pending for later retry.",
                summary.path,
                err,
            )
            skipped_manifests.append(summary.path)
            continue

        manifest_config = ManifestConfig(
            job_name=str(manifest_meta.get("job_name") or ""),
            batch_id=batch_id,
            input_file_id=input_file_id,
            model=model,
            provider=provider,
            endpoint=str(manifest_meta.get("endpoint") or "/v1/chat/completions"),
            arguments=dict(manifest_meta.get("arguments") or {}),
        )
        manifest_config.arguments["batch_bytes"] = int(batch_bytes)

        create_manifest(
            manifest_config,
            tasks=task_records,
            manifest_path=summary.path,
        )
        logging.info(
            "Enqueued batch %s for manifest %s with %s tasks.",
            batch_id,
            summary.path,
            len(task_records),
        )
        if estimated_tokens:
            tokens_used += estimated_tokens

    logging.info("Enqueue complete. Estimated tokens submitted: %s.", tokens_used)
    if skipped_manifests:
        logging.warning(
            "Skipped %s manifests due to connection errors. "
            "These manifests remain in 'pending' status and can be retried by "
            "re-running the enqueue subcommand. Example skipped manifest: %s",
            len(skipped_manifests),
            skipped_manifests[0],
        )
    return 0


def run_retry_errors(args: argparse.Namespace) -> int:
    """Entry point for the 'retry-errors' subcommand."""

    retry_path = args.retry_errors_from.expanduser().resolve()
    if not retry_path.exists():
        logging.error("Retry JSONL not found: %s", retry_path)
        return 2

    outputs_root = Path(args.output_dir).expanduser().resolve()
    if not outputs_root.exists() or not outputs_root.is_dir():
        logging.error("Outputs root not found: %s", outputs_root)
        return 2

    retry_files = _collect_retry_family_files(
        retry_path,
        outputs_root,
        include_all_participants=bool(getattr(args, "retry_errors_all_ppts", False)),
    )
    if not retry_files:
        logging.error("No retry outputs found under %s", outputs_root)
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

    latest_error_records: dict[SeenKey, dict[str, object]] = {}
    success_keys: Set[SeenKey] = set()
    for file_path in tqdm(
        sorted(retry_files),
        desc="Scanning retry files",
        unit="file",
    ):
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
                key: SeenKey = (
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

    job_name = str(args.job or infer_job_stem_from_filename(retry_path.name)).strip()
    if not job_name:
        logging.error("Retry job name could not be derived from %s", retry_path)
        return 2

    manifest_dir = (
        args.manifest_dir
        if args.manifest_dir is not None
        else outputs_root / "batch_manifests" / job_name
    )

    non_default_arguments = extract_non_default_arguments_with_model(args)
    non_default_arguments = dict(non_default_arguments)
    non_default_arguments["retry_errors_from"] = str(retry_path)

    setattr(args, "_tokens_used", 0)
    batch_index = 0
    tasks_for_batch: List[ClassificationTask] = []

    for task in tasks:
        tasks_for_batch.append(task)
        if len(tasks_for_batch) >= max(1, int(args.batch_size)):
            batch_index = _flush_batch_with_splitting(
                job_name=job_name,
                batch_index=batch_index,
                tasks=tasks_for_batch,
                args=args,
                manifest_dir=manifest_dir,
                non_default_arguments=non_default_arguments,
            )
            tasks_for_batch = []

    if tasks_for_batch:
        batch_index = _flush_batch_with_splitting(
            job_name=job_name,
            batch_index=batch_index,
            tasks=tasks_for_batch,
            args=args,
            manifest_dir=manifest_dir,
            non_default_arguments=non_default_arguments,
        )

    total_tokens = int(getattr(args, "_tokens_used", 0))
    if getattr(args, "dry_run", False):
        logging.info(
            "Retry dry run complete. Prepared %s batches for job %s. "
            "Estimated tokens: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            job_name,
            total_tokens,
        )
    else:
        logging.info(
            "Retry submit complete. Prepared %s batches for job %s. "
            "Estimated tokens submitted: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            job_name,
            total_tokens,
        )
    return 0


def run_retry_missing(args: argparse.Namespace) -> int:
    """Entry point for the 'retry-missing' subcommand."""

    missing_path = args.missing_from.expanduser().resolve()
    if not missing_path.exists():
        logging.error("Reference JSONL not found: %s", missing_path)
        return 2

    outputs_root = Path(args.output_dir).expanduser().resolve()
    if not outputs_root.exists() or not outputs_root.is_dir():
        logging.error("Outputs root not found: %s", outputs_root)
        return 2

    include_all_participants = not bool(getattr(args, "missing_single_ppt", False))
    family_files = _collect_retry_family_files(
        missing_path,
        outputs_root,
        include_all_participants=include_all_participants,
    )
    if not family_files:
        logging.error("No output files found under %s", outputs_root)
        return 2

    meta = load_retry_meta(missing_path)
    defaults = getattr(args, "_defaults", {})
    if isinstance(meta.get("model"), str) and normalize_arg_value(
        getattr(args, "model", None)
    ) == normalize_arg_value(defaults.get("model")):
        args.model = str(meta.get("model"))

    annotation_ids = args.annotation
    configs: List[AnnotationConfig] = load_annotation_configs(annotation_ids)
    if not configs:
        logging.error("No annotation configurations resolved for retry-missing.")
        return 2

    input_root = Path(args.input).expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        logging.error("Input directory not found: %s", input_root)
        return 2

    normalized_filter = normalize_participant_filter(args.participants)
    participants_filter: Optional[Sequence[str]] = (
        sorted(normalized_filter) if normalized_filter else None
    )

    resume_seen_keys: Set[SeenKey] = set()
    with tqdm(
        total=len(family_files),
        desc="Indexing outputs",
        disable=not sys.stderr.isatty(),
    ) as progress:
        for path in family_files:
            if "batch_manifests" in path.parts:
                progress.update(1)
                continue
            keys, _ = load_resume_keys(path, None)
            resume_seen_keys.update(keys)
            progress.update(1)

    message_iter, max_messages, progress_total, _ = prepare_message_iterator(
        args,
        input_root,
        configs,
        participants_filter,
        replay_keys=None,
    )

    job_name = str(args.job or infer_job_stem_from_filename(missing_path.name)).strip()
    if not job_name:
        logging.error("Retry job name could not be derived from %s", missing_path)
        return 2

    manifest_dir = (
        args.manifest_dir
        if args.manifest_dir is not None
        else outputs_root / "batch_manifests" / job_name
    )

    non_default_arguments = extract_non_default_arguments_with_model(args)
    non_default_arguments = dict(non_default_arguments)
    non_default_arguments["retry_missing_from"] = str(missing_path)

    message_entries: List[Tuple[str, str, int, int, str]] = []
    processed_messages = 0
    with tqdm(
        total=progress_total,
        desc="Indexing transcripts",
        disable=not sys.stderr.isatty(),
    ) as progress:
        for context in message_iter:
            if max_messages is not None and processed_messages >= max_messages:
                break
            message_entries.append(
                (
                    context.participant,
                    str(context.source_path),
                    context.chat_index,
                    context.message_index,
                    context.role,
                )
            )
            processed_messages += 1
            progress.update(1)

    missing_keys: Set[SeenKey] = set()
    unrestricted_configs = [cfg for cfg in configs if cfg.allowed_roles is None]
    configs_by_role: Dict[str, List[AnnotationConfig]] = defaultdict(list)
    for cfg in configs:
        if cfg.allowed_roles is None:
            continue
        for role in cfg.allowed_roles:
            configs_by_role[role].append(cfg)

    with tqdm(
        total=len(message_entries),
        desc="Diffing missing coverage",
        disable=not sys.stderr.isatty(),
    ) as progress:
        for (
            participant,
            source_path,
            chat_index,
            message_index,
            role,
        ) in message_entries:
            applicable = unrestricted_configs + configs_by_role.get(role, [])
            for cfg in applicable:
                ann_id = str(cfg.spec.get("id"))
                key = (participant, source_path, chat_index, message_index, ann_id)
                if key not in resume_seen_keys:
                    missing_keys.add(key)
            progress.update(1)

    if not missing_keys:
        logging.info("No missing coverage found for retry-missing.")
        return 0

    logging.info(
        "Identified %s missing (message, annotation) keys for retry-missing.",
        len(missing_keys),
    )

    replay_keys: Set[ReplayKey] = {
        (key[0], key[1], key[2], key[3]) for key in missing_keys
    }
    message_iter, max_messages, progress_total, _ = prepare_message_iterator(
        args,
        input_root,
        configs,
        participants_filter,
        replay_keys=replay_keys,
    )

    batch_index = 0
    tasks_for_batch: List[ClassificationTask] = []
    processed_messages = 0
    total_tasks = 0
    annotation_counts: Counter[str] = Counter()
    participant_request_counts: Counter[str] = Counter()
    participant_annotation_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    participant_message_keys: Dict[str, Set[Tuple[str, int, int]]] = defaultdict(set)
    duplicate_key_counts: Counter[Tuple[str, str, int, int, str]] = Counter()
    setattr(args, "_tokens_used", 0)

    with tqdm(
        total=progress_total,
        desc="Preparing missing batches",
        disable=not sys.stderr.isatty(),
    ) as progress:
        for context in message_iter:
            if max_messages is not None and processed_messages >= max_messages:
                break

            tasks_for_context = build_classification_tasks_for_context(
                context,
                configs,
                resume_seen_keys,
                0,
                {},
                args=args,
                target_keys=missing_keys,
            )
            if not tasks_for_context:
                processed_messages += 1
                progress.update(1)
                continue

            _update_task_stats(
                tasks_for_context,
                annotation_counts=annotation_counts,
                participant_request_counts=participant_request_counts,
                participant_annotation_counts=participant_annotation_counts,
                participant_message_keys=participant_message_keys,
                duplicate_key_counts=duplicate_key_counts,
            )

            tasks_for_batch.extend(tasks_for_context)
            total_tasks += len(tasks_for_context)
            processed_messages += 1
            progress.update(1)

            if len(tasks_for_batch) >= max(1, int(args.batch_size)):
                batch_index = _flush_batch_with_splitting(
                    job_name=job_name,
                    batch_index=batch_index,
                    tasks=tasks_for_batch,
                    args=args,
                    manifest_dir=manifest_dir,
                    non_default_arguments=non_default_arguments,
                )
                tasks_for_batch = []

    if tasks_for_batch:
        batch_index = _flush_batch_with_splitting(
            job_name=job_name,
            batch_index=batch_index,
            tasks=tasks_for_batch,
            args=args,
            manifest_dir=manifest_dir,
            non_default_arguments=non_default_arguments,
        )

    total_tokens = int(getattr(args, "_tokens_used", 0))
    if getattr(args, "dry_run", False):
        logging.info(
            "Retry-missing dry run complete. Prepared %s batches for job %s. "
            "Estimated tokens: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            job_name,
            total_tokens,
        )
    else:
        logging.info(
            "Retry-missing submit complete. Prepared %s batches for job %s. "
            "Estimated tokens submitted: %s.",
            batch_index + (1 if tasks_for_batch else 0),
            job_name,
            total_tokens,
        )

    print_token_cost_summary(
        model=args.model,
        total_tokens=total_tokens,
        total_tasks=total_tasks,
    )
    if annotation_counts:
        print_annotation_stats(annotation_counts)
    if participant_request_counts:
        print_participant_stats(
            participant_request_counts,
            participant_annotation_counts,
            participant_message_keys,
        )
    print_duplicate_warnings(duplicate_key_counts)
    return 0


def run_harvest(args: argparse.Namespace) -> int:
    """Entry point for the 'harvest' subcommand."""

    output_root = Path(args.output_dir).expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logging.error("Failed to create output directory: %s", err)
        return 2

    if args.manifest_dir is not None:
        manifest_root = args.manifest_dir.expanduser().resolve()
    elif args.job:
        manifest_root = (
            output_root / "batch_manifests" / str(args.job).strip()
        ).resolve()
    else:
        logging.error("The harvest subcommand requires either --manifest-dir or --job.")
        return 2

    if not manifest_root.exists() or not manifest_root.is_dir():
        logging.error("Manifest directory not found: %s", manifest_root)
        return 2

    resume_seen_keys: Set[SeenKey] = set()
    for path, _meta in iter_jsonl_meta(output_root):
        # Ignore manifest JSONL files when rebuilding resume keys; they do not
        # contain classification records and would otherwise cause tasks to be
        # treated as already written.
        if "batch_manifests" in path.parts:
            continue
        keys, _ = load_resume_keys(path, None)
        resume_seen_keys.update(keys)
    retryable_error_keys: Optional[Set[SeenKey]] = None

    participant_counts: dict[str, int] = {}
    total_written = 0

    try:
        for summary in iter_manifests(manifest_root):
            meta = summary.meta
            status = str(meta.get("status") or "submitted").lower()
            if status in {"written", "failed"}:
                continue

            batch_id = meta.get("batch_id")
            if not isinstance(batch_id, str) or not batch_id.strip():
                logging.warning(
                    "Skipping manifest without batch_id: %s",
                    summary.path,
                )
                continue

            provider = str(meta.get("provider") or "openai")
            manifest_meta, task_records = load_manifest_tasks(summary.path)

            spinner_message = (
                f"Harvesting batch {batch_id} for manifest {summary.path.name}"
            )
            try:
                with Spinner(spinner_message):
                    custom_id_order = [
                        str(task.get("custom_id") or "").strip()
                        for task in task_records
                        if str(task.get("custom_id") or "").strip()
                    ]
                    results_by_custom_id = resume_litellm_batch(
                        batch_id,
                        litellm_client=litellm,
                        custom_llm_provider=provider,
                        custom_id_order=custom_id_order,
                    )
            except openai.InternalServerError as err:
                logging.warning(
                    "Transient OpenAI internal error while resuming batch %s for "
                    "manifest %s: %s",
                    batch_id,
                    summary.path,
                    err,
                )
                update_manifest_status(summary.path, "timeout")
                continue
            except openai.PermissionDeniedError as err:
                logging.error(
                    "Permission denied while fetching batch output for batch %s "
                    "and manifest %s: %s",
                    batch_id,
                    summary.path,
                    err,
                )
                update_manifest_status(summary.path, "failed")
                continue
            except (
                BatchTimeoutError,
                BatchFailedError,
                litellm.OpenAIError,
                OSError,
                ValueError,
            ) as err:
                if isinstance(err, BatchTimeoutError):
                    logging.warning(
                        "Batch %s timed out during harvest for manifest %s: %s",
                        batch_id,
                        summary.path,
                        err,
                    )
                    update_manifest_status(summary.path, "timeout")
                else:
                    logging.error(
                        "Error while resuming batch %s for manifest %s: %s",
                        batch_id,
                        summary.path,
                        err,
                    )
                    update_manifest_status(summary.path, "failed")
                continue

            manifest_arguments = dict(manifest_meta.get("arguments") or {})
            retry_errors_from = manifest_arguments.get("retry_errors_from")
            retry_mode = bool(retry_errors_from)
            if retry_mode and retryable_error_keys is None:
                retryable_error_keys = set()
                output_paths: List[Path] = []
                for path, _meta in iter_jsonl_meta(output_root):
                    if "batch_manifests" in path.parts:
                        continue
                    output_paths.append(path)
                if output_paths:
                    retryable_error_keys = load_latest_retryable_error_keys(
                        output_paths
                    )

            # Derive minimal annotation configs from manifest tasks.
            specs_by_id: dict[str, dict[str, object]] = {}
            for task in task_records:
                ann_id = str(task.get("annotation_id") or "").strip()
                if not ann_id or ann_id in specs_by_id:
                    continue
                specs_by_id[ann_id] = {
                    "id": ann_id,
                    "name": str(task.get("annotation_name") or ""),
                    "description": "",
                }
            configs: List[AnnotationConfig] = [
                AnnotationConfig(spec=spec, allowed_roles=None)
                for spec in specs_by_id.values()
            ]

            non_default_arguments = manifest_arguments

            args_for_output = argparse.Namespace(
                model=manifest_meta.get("model") or DEFAULT_CHAT_MODEL,
                preceding_context=int(manifest_meta.get("preceding_context", 0) or 0),
                follow_links=False,
            )

            output_handles: dict[Path, TextIO] = {}

            with ExitStack() as stack:
                for task in task_records:
                    custom_id = str(task.get("custom_id") or "").strip()
                    if not custom_id:
                        continue
                    if custom_id not in results_by_custom_id:
                        logging.error(
                            "No result found for custom_id %s in batch %s (manifest %s).",
                            custom_id,
                            batch_id,
                            summary.path,
                        )
                        continue

                    try:
                        (
                            participant,
                            source_path,
                            chat_index,
                            message_index,
                            ann_id,
                        ) = decode_custom_id(custom_id)
                    except ValueError as err:
                        logging.error(
                            "Unable to decode custom_id %s in manifest %s: %s",
                            custom_id,
                            summary.path,
                            err,
                        )
                        continue

                    key: SeenKey = (
                        participant,
                        source_path,
                        chat_index,
                        message_index,
                        ann_id,
                    )
                    if key in resume_seen_keys and not (
                        retry_mode
                        and retryable_error_keys is not None
                        and key in retryable_error_keys
                    ):
                        continue

                    message_content_raw = task.get("content")
                    message_content = str(message_content_raw or "")
                    if not message_content:
                        logging.error(
                            "Missing message content for custom_id %s in manifest %s; "
                            "marking manifest as failed. Regenerate manifests with an "
                            "updated classify_chats_batch submit run.",
                            custom_id,
                            summary.path,
                        )
                        update_manifest_status(summary.path, "failed")
                        continue

                    content = str(results_by_custom_id.get(custom_id) or "")

                    context = MessageContext(
                        participant=participant,
                        source_path=Path(source_path),
                        chat_index=chat_index,
                        chat_key=task.get("chat_key"),
                        chat_date=task.get("chat_date"),
                        message_index=message_index,
                        role=str(task.get("role") or "user"),
                        timestamp=task.get("timestamp"),
                        content=message_content,
                        preceding=task.get("preceding"),
                    )

                    annotation_spec = specs_by_id.get(ann_id) or {
                        "id": ann_id,
                        "name": str(task.get("annotation_name") or ""),
                        "description": "",
                    }

                    classification_task = ClassificationTask(
                        context=context,
                        annotation=annotation_spec,
                        prompt=str(task.get("prompt") or ""),
                    )

                    try:
                        rationale, matches, score = extract_matches_from_response_text(
                            content
                        )
                        error = None
                    except ClassificationError as err:
                        matches = []
                        rationale = None
                        score = None
                        error = str(err)

                    outcome = ClassificationOutcome(
                        task=classification_task,
                        matches=matches,
                        error=error,
                        rationale=rationale,
                        score=score if error is None else None,
                    )

                    resolved_output_name = (
                        f"{manifest_meta.get('job_name')}.jsonl"
                        if manifest_meta.get("job_name")
                        else f"{batch_id}.jsonl"
                    )

                    wrote = ensure_output_and_write_outcomes_for_context(
                        context=context,
                        outcomes=[outcome],
                        args=args_for_output,
                        configs=configs,
                        output_dir=output_root,
                        single_output_file=None,
                        resolved_output_name=resolved_output_name,
                        non_default_arguments=non_default_arguments,
                        output_handles=output_handles,
                        stack=stack,
                    )
                    if not wrote:
                        return 2

                    resume_seen_keys.add(key)
                    participant_counts[participant] = (
                        participant_counts.get(participant, 0) + 1
                    )
                    total_written += 1

            update_manifest_status(summary.path, "written")
    except KeyboardInterrupt:
        logging.warning("Harvest interrupted by user.")
        return 130

    logging.info(
        "Harvest complete. Wrote %s records across %s participants.",
        total_written,
        len(participant_counts),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry point for classify_chats_batch."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "submit":
        return run_submit(args)
    if args.command == "harvest":
        return run_harvest(args)
    if args.command == "enqueue":
        return run_enqueue(args)
    if args.command == "retry-errors":
        return run_retry_errors(args)
    if args.command == "retry-missing":
        return run_retry_missing(args)

    logging.error("Unknown command: %s", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
