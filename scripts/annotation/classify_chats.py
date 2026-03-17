"""
Classify individual chat messages with configurable annotation prompts.

The script scans chat JSON exports, optionally filters by participant, and
submits each message to an annotation classifier powered by LiteLLM.
Results are written to JSON Lines for downstream analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Set, TextIO

from llm_delusions_annotations.annotation_ids import normalize_annotation_ids
from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATIONS,
    add_llm_common_arguments,
    build_prompt,
    disable_litellm_logging,
)
from llm_delusions_annotations.chat import (
    iter_loaded_chats,
    resolve_bucket_and_rel_path,
)
from llm_delusions_annotations.chat.chat_utils import (
    build_preceding_entry,
    normalize_optional_string,
)
from llm_delusions_annotations.classify_messages import (
    MAX_CLASSIFICATION_TOKENS,
    ClassificationError,
    ClassificationOutcome,
    MessageContext,
    build_completion_messages,
    classify_tasks_batch,
    parse_optional_int,
)
from llm_delusions_annotations.configs import (
    AnnotationConfig,
    derive_allowed_roles,
    load_annotation_configs,
    parse_annotation_scope,
    resolve_annotation,
)
from llm_delusions_annotations.llm_utils.client import DEFAULT_CHAT_MODEL
from llm_delusions_annotations.utils import AnnotationRequest, to_litellm_messages
from tqdm import tqdm

from annotation.io import (
    IndexPair,
    ReplayKey,
    SeenKey,
    collect_replay_files_for_job,
    infer_job_stem_from_filename,
    iter_jsonl_meta,
    iter_jsonl_records,
    load_latest_retryable_error_keys,
    load_replay_message_annotation_ids,
    load_replay_message_keys,
    load_resume_keys,
    parse_message_indices,
)
from annotation.outputs_summary import compute_positive_counts
from annotation.pipeline import (
    MAX_OUTPUT_NAME_LENGTH,
    PrefilterState,
    build_classification_tasks_for_context,
    build_meta_record,
    ensure_output_and_write_outcomes_for_context,
    ensure_participant_directory,
    prepare_message_iterator,
)
from llm_utils import (
    print_cost_summary,
    safe_estimate_max_request_cost,
    summarize_token_totals,
)
from utils.cli import (
    add_chat_io_arguments,
    add_chat_sampling_arguments,
    add_follow_links_argument,
    add_model_argument,
    add_participants_argument,
    add_score_cutoff_argument,
)
from utils.io import collect_family_files
from utils.param_strings import dict_to_string, string_to_dict
from utils.participants import (
    normalize_participant_filter,
    normalize_participant_id,
    normalize_participant_value,
    participant_from_record,
)
from utils.utils import (
    extract_non_default_arguments,
    normalize_arg_value,
    pick_latest_per_parent,
)

# Uncomment the following line when debugging LiteLLM calls (requires importing litellm).
# litellm._turn_on_debug()
# When not debugging, silence LiteLLM's own logger.
disable_litellm_logging()

DEFAULT_OUTPUT_DIR = "annotation_outputs"
# My max RPM for gpt-4.1-nano is 30k
# needed_concurrency ~= target_RPS × avg_latency_seconds. (Little's law)
# With 30k RPM = 500 RPS and expected latency of 1s = 500 calls
# Single-process ThreadPoolExecutor: 128–512 threads is a reasonable ceiling
# on typical 8–32 GB RAM machines.
MAX_WORKERS = 256
PREFILTER_TURN_LIMIT = 10


def _build_parser(annotation_ids: Sequence[str]) -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser.

    Parameters
    ----------
    annotation_ids: Sequence[str]
        Available annotation identifiers used to constrain arguments.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description="Classify chat messages using LiteLLM annotation prompts."
    )

    add_chat_io_arguments(
        parser,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        output_help=(
            f"Directory for annotation outputs (default: {DEFAULT_OUTPUT_DIR})."
        ),
    )

    parser.add_argument(
        "--job",
        type=str,
        help=(
            "Optional short job name used as part of output filenames. "
            "When the auto-generated filename would exceed the maximum "
            "length, the run will fail and ask you to re-run with a "
            "shorter --job value."
        ),
    )

    parser.add_argument(
        "--annotation",
        "-a",
        action="append",
        choices=annotation_ids,
        help=(
            "Annotation ID to use for classification (repeatable). "
            "Defaults to all annotations except those with category 'test' when "
            "omitted."
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
            "Optional cap on the number of messages to classify. "
            "Set to 0 to process all messages."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS,
        help=(
            "Maximum worker threads used by the LiteLLM batch_completion "
            f"API (default: {MAX_WORKERS}). Lower this to reduce concurrency "
            "when hitting provider rate limits."
        ),
    )

    add_follow_links_argument(parser)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the prepared prompt for the first matching message and exit.",
    )
    add_score_cutoff_argument(
        parser,
        help_text=(
            "Optional minimum score (0-10) required for a classification to count as "
            "positive when enforcing --min-positive-per-annotation."
        ),
    )
    parser.add_argument(
        "--min-positive-per-annotation",
        type=int,
        default=0,
        help=(
            "When positive, continue classifying until each selected annotation has "
            "at least this many messages with one or more matches, subject to the "
            "overall --max-messages cap when provided."
        ),
    )
    parser.add_argument(
        "--resume-auto",
        action="store_true",
        help=(
            "Automatically locate the most recent JSONL per participant under the "
            "output directory that matches current annotation criteria and resume "
            "from them (skipping already-seen messages). Prompts for confirmation."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Resume classification by appending to an existing JSONL output file. "
            "Participants and annotations are inferred from the file metadata, and "
            "messages already present in the file are skipped."
        ),
    )
    parser.add_argument(
        "--prefilter-conversations",
        action="store_true",
        help=(
            "Skip the remainder of a chat when no annotations match within the first "
            f"{PREFILTER_TURN_LIMIT} turns."
        ),
    )

    parser.add_argument(
        "--replay-from",
        type=Path,
        help=(
            "Replay the exact message sample from a prior JSONL output file. "
            "Sampling and conversation limiting flags are ignored when this is set."
        ),
    )
    parser.add_argument(
        "--replay-from-manual",
        type=Path,
        action="append",
        help=(
            "Replay messages from manual annotation JSONL files. "
            "Sampling and conversation limiting flags are ignored when this is set, "
            "and only annotation ids present in the manual records are re-run."
        ),
    )
    parser.add_argument(
        "--replay-all-ppts",
        action="store_true",
        help=(
            "When used with --replay-from, replay the exact message sample across all "
            "participant JSONL files in the output directory that share the same "
            "output filename and annotation parameters as the provided replay file."
        ),
    )
    parser.add_argument(
        "--replay-from-source",
        action="store_true",
        help=(
            "Replay using the original source transcripts from --input while "
            "reusing the same message keys as the prior JSONL. When omitted, "
            "replay operates directly on the JSONL contents without loading "
            "transcripts or reconstructing conversation context."
        ),
    )
    parser.add_argument(
        "--retry-errors-from",
        type=Path,
        help=(
            "Retry only messages with non-quote-mismatch errors from a prior JSONL "
            "output file. This re-runs only the failing (message, annotation_id) "
            "pairs and appends results to the same output family."
        ),
    )
    parser.add_argument(
        "--retry-errors-all-ppts",
        action="store_true",
        help=(
            "When used with --retry-errors-from, include all participant JSONL files "
            "under the outputs root that share the same job stem."
        ),
    )
    add_llm_common_arguments(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv: Sequence[str] | None
        Optional command-line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with defaults.
    """

    annotation_ids = [spec["id"] for spec in ANNOTATIONS]
    parser = _build_parser(annotation_ids)
    args = parser.parse_args(argv)
    defaults = {key: parser.get_default(key) for key in vars(args)}
    setattr(args, "_defaults", defaults)
    if args.min_positive_per_annotation < 0:
        parser.error("--min-positive-per-annotation must be non-negative.")
    if args.score_cutoff is not None and (
        args.score_cutoff < 0 or args.score_cutoff > 10
    ):
        parser.error("--score-cutoff must be between 0 and 10.")
    if args.replay_from is not None and args.replay_from_manual:
        parser.error("--replay-from-manual cannot be combined with --replay-from.")
    if args.replay_from_manual and getattr(args, "replay_from_source", False):
        parser.error(
            "--replay-from-manual cannot be combined with --replay-from-source."
        )
    if args.replay_from_manual and getattr(args, "replay_all_ppts", False):
        parser.error("--replay-from-manual cannot be combined with --replay-all-ppts.")
    return args


def _load_positive_counts_from_files(
    jsonl_paths: Sequence[Path],
    *,
    score_cutoff: Optional[int],
) -> dict[str, int]:
    """Return positive counts per annotation id from existing JSONL outputs.

    This is a thin shim over the logic used by ``summarize_annotation_positives``
    so the definition of a "positive" record remains consistent across tools.

    Parameters
    ----------
    jsonl_paths:
        One or more JSONL files produced by ``classify_chats.py``.
    score_cutoff:
        Optional minimum score required for a record to count as positive.

    Returns
    -------
    dict[str, int]
        Mapping from annotation id to the number of positive records found
        across all provided files.
    """

    positive_counts, _ = compute_positive_counts(
        jsonl_paths,
        score_cutoff=score_cutoff,
        annotation_filter_set=None,
    )
    return {
        annotation_id: int(count) for annotation_id, count in positive_counts.items()
    }


def _read_resume_annotation_ids(resume_path: Path) -> Set[str]:
    """Return annotation ids from the JSONL meta line only."""
    try:
        first = resume_path.open("r", encoding="utf-8", errors="ignore").readline()
    except OSError:
        return set()
    try:
        meta = json.loads(first)
    except (json.JSONDecodeError, ValueError, TypeError):
        return set()
    if not isinstance(meta, dict) or meta.get("type") != "meta":
        return set()
    ids: Set[str] = set()
    ann_ids = meta.get("annotation_ids")
    if isinstance(ann_ids, list):
        ids.update(normalize_annotation_ids(ann_ids))
    snap = meta.get("annotation_snapshots")
    if isinstance(snap, dict) and not ids:
        ids.update(normalize_annotation_ids(snap.keys()))
    return ids


def _load_configs_from_annotation_ids(
    annotation_ids: Iterable[str],
    *,
    label: str,
    source_path: Optional[Path] = None,
) -> List[AnnotationConfig]:
    """Return annotation configs resolved from annotation ids.

    Parameters
    ----------
    annotation_ids: Iterable[str]
        Annotation identifiers to resolve.
    label: str
        Human-readable label used in log messages (for example, "Replay JSONL").
    source_path: Optional[Path]
        Optional source path used in error messages.

    Returns
    -------
    list[AnnotationConfig]
        Resolved annotation configs (empty when none can be resolved).
    """

    specs: List[dict[str, object]] = []
    unknown_ids: List[str] = []
    for ann_id in sorted({str(item) for item in annotation_ids if item}):
        try:
            specs.append(resolve_annotation(ann_id))
        except ValueError as err:
            unknown_ids.append(ann_id)
            logging.warning(
                "%s references unknown annotation %s: %s",
                label,
                ann_id,
                err,
            )

    if not specs:
        if source_path is not None:
            logging.error(
                "%s did not resolve any known annotations from %s; "
                "cannot prepare replay/resume state.",
                label,
                source_path,
            )
        else:
            logging.error(
                "%s did not resolve any known annotations; "
                "cannot prepare replay/resume state.",
                label,
            )
        return []

    return [
        AnnotationConfig(spec=spec, allowed_roles=parse_annotation_scope(spec))
        for spec in specs
    ]


def _load_configs_and_participants_from_jsonl(
    jsonl_path: Path,
    *,
    label: str,
) -> tuple[List[AnnotationConfig], List[str], Optional[str], dict[str, object]]:
    """Return annotation configs, participants, model, and settings from a JSONL meta.

    Parameters
    ----------
    jsonl_path: Path
        Path to a JSONL file whose first line is a meta record.
    label: str
        Human-readable label used in log messages (for example, "Replay JSONL").

    Returns
    -------
    tuple[List[AnnotationConfig], List[str], Optional[str], dict[str, object]]
        Loaded annotation configs, any participant identifiers extracted from
        the meta record, the model name when available, and a dictionary of
        additional run settings such as context depth.
        On error, returns empty configs and participants with details logged.
    """

    configs: List[AnnotationConfig] = []
    participants: List[str] = []
    model: Optional[str] = None
    settings: dict[str, object] = {}

    if not jsonl_path.exists() or not jsonl_path.is_file():
        logging.error("%s not found: %s", label, jsonl_path)
        return configs, participants, model, settings

    annotation_ids = _read_resume_annotation_ids(jsonl_path)
    if not annotation_ids:
        logging.error(
            "%s does not contain usable annotation metadata: %s",
            label,
            jsonl_path,
        )
        return configs, participants, model, settings
    configs = _load_configs_from_annotation_ids(
        annotation_ids,
        label=label,
        source_path=jsonl_path,
    )
    if not configs:
        return configs, participants, model, settings

    try:
        first_line = jsonl_path.open("r", encoding="utf-8", errors="ignore").readline()
    except OSError as err:
        logging.error("Failed to read %s metadata: %s", label.lower(), err)
        return configs, participants, model, settings
    try:
        meta = json.loads(first_line)
    except (json.JSONDecodeError, ValueError, TypeError):
        logging.error(
            "%s does not start with a valid meta record: %s",
            label,
            jsonl_path,
        )
        return configs, participants, model, settings
    if not isinstance(meta, dict) or meta.get("type") != "meta":
        logging.error(
            "%s does not start with a meta record: %s",
            label,
            jsonl_path,
        )
        return configs, participants, model, settings

    model_value = meta.get("model")
    if isinstance(model_value, str) and model_value.strip():
        model = model_value.strip()
    ppt_id = meta.get("ppt_id")
    if isinstance(ppt_id, str) and ppt_id.strip():
        participants.append(normalize_participant_value(ppt_id))
    participants_meta = meta.get("participants")
    if isinstance(participants_meta, list):
        for item in participants_meta:
            normalized = normalize_participant_value(item)
            if normalized and normalized not in participants:
                participants.append(normalized)

    # Optional run settings that should be restored on resume when available.
    preceding_value = meta.get("preceding_context")
    if isinstance(preceding_value, int):
        settings["preceding_context"] = preceding_value

    return configs, participants, model, settings


def _apply_prior_run_arguments(
    args: argparse.Namespace,
    *,
    jsonl_path: Path,
    settings: Mapping[str, object],
    run_model: Optional[str],
    label: str,
) -> None:
    """Apply arguments and settings from a prior run JSONL onto ``args``.

    Only fields whose current values still match their parser defaults are
    overridden, so explicit command-line flags always win. Core location-style
    fields (for example, ``input``, ``output_dir``) are never modified.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments for the current invocation.
    jsonl_path: Path
        Path to the JSONL file whose filename encodes original parameters.
    settings: Mapping[str, object]
        Optional run settings loaded from the JSONL meta, such as
        ``preceding_context``.
    run_model: Optional[str]
        Model name recorded in the prior run meta, when available.
    label: str
        Human-readable label for logging context (for example, ``\"Replay\"``
        or ``\"Resume\"``).
    """

    defaults = getattr(args, "_defaults", {})

    original_params = _load_original_params_from_filename(jsonl_path)
    settings_overrides = _build_settings_overrides(settings)

    _apply_overrides_to_args(args, defaults, original_params)
    _apply_overrides_to_args(args, defaults, settings_overrides)

    _maybe_override_model(args, defaults, run_model, label)


def _load_original_params_from_filename(jsonl_path: Path) -> dict[str, object]:
    """Return original non-default parameters encoded in a JSONL filename.

    The filename is expected to contain a ``__``-separated parameter fragment
    ending in ``.jsonl``. When the fragment is missing, equal to ``\"defaults\"``,
    or cannot be parsed into a dictionary, an empty mapping is returned.

    Parameters
    ----------
    jsonl_path: Path
        Path to a JSONL file whose filename may encode original parameters.

    Returns
    -------
    dict[str, object]
        Mapping of parameter names to their original values, or an empty
        dictionary when no usable fragment is present.
    """

    try:
        name = jsonl_path.name
        param_fragment = name.split("__", 1)[1].rsplit(".jsonl", 1)[0]
    except (IndexError, ValueError):
        param_fragment = ""

    original_params: dict[str, object] = {}
    if param_fragment and param_fragment != "defaults":
        try:
            original_params = string_to_dict(param_fragment)
        except (ValueError, TypeError):
            original_params = {}

    return original_params


def _build_settings_overrides(settings: Mapping[str, object]) -> dict[str, object]:
    """Return argument overrides derived from JSONL meta-only settings.

    Only settings that are intended to be restored from meta when the user
    has not explicitly overridden them are included.

    Parameters
    ----------
    settings: Mapping[str, object]
        Mapping of settings loaded from the JSONL meta record.

    Returns
    -------
    dict[str, object]
        Mapping of argument names to values that should be applied as
        overrides when the corresponding arguments still match their parser
        defaults.
    """

    overrides: dict[str, object] = {}
    preceding_value = settings.get("preceding_context")
    if isinstance(preceding_value, int):
        overrides["preceding_context"] = preceding_value
    return overrides


def _validate_replay_sampling_flags(args: argparse.Namespace) -> bool:
    """Return True when replay flags are compatible with sampling options."""

    if (
        getattr(args, "randomize", False)
        or getattr(args, "randomize_conversations", False)
        or getattr(args, "max_conversations", 0) > 0
    ):
        logging.error(
            "--replay-from/--replay-from-manual cannot be combined with --randomize, "
            "--randomize-conversations, or --max-conversations.",
        )
        return False
    return True


def _validate_retry_sampling_flags(args: argparse.Namespace) -> bool:
    """Return True when retry-error flags are compatible with sampling options."""

    if (
        getattr(args, "randomize", False)
        or getattr(args, "randomize_conversations", False)
        or getattr(args, "max_conversations", 0) > 0
        or getattr(args, "max_messages", 0) > 0
    ):
        logging.error(
            "--retry-errors-from cannot be combined with --randomize, "
            "--randomize-conversations, --max-conversations, or --max-messages.",
        )
        return False
    return True


def _collect_retry_family_files(
    retry_path: Path,
    outputs_root: Path,
    *,
    include_all_participants: bool,
) -> List[Path]:
    """Return JSONL files that belong to the retry error family."""

    family_files = collect_family_files(retry_path, outputs_root)
    if include_all_participants:
        return family_files or [retry_path]
    same_parent = [path for path in family_files if path.parent == retry_path.parent]
    return same_parent or [retry_path]


def _collect_retry_error_keys(
    jsonl_paths: Sequence[Path],
) -> tuple[Set[SeenKey], Set[ReplayKey]]:
    """Return retryable error keys and message keys for filtering."""

    target_keys = load_latest_retryable_error_keys(jsonl_paths)
    message_keys: Set[ReplayKey] = {
        (participant, source_path, chat_index, message_index)
        for participant, source_path, chat_index, message_index, _ann_id in target_keys
    }
    return target_keys, message_keys


def _prepare_replay_configs_and_files(
    args: argparse.Namespace,
    *,
    label: str,
) -> tuple[int, Optional[List[AnnotationConfig]], Optional[List[Path]]]:
    """Return configs and JSONL files for a replay run.

    Parameters
    ----------
    args:
        Parsed command-line arguments for the current invocation.
    label:
        Human-readable label used in log messages when loading the replay
        JSONL (for example, ``\"Replay JSONL\"``).

    Returns
    -------
    tuple
        A tuple containing the status code (zero on success), the loaded
        annotation configs, and the list of JSONL files that belong to the
        same replay job. On error, returns a non-zero status and ``None``
        placeholders.
    """

    replay_from = getattr(args, "replay_from", None)
    if replay_from is None:
        logging.error("--replay-from must be provided for replay runs.")
        return 2, None, None

    replay_path = Path(replay_from).expanduser().resolve()
    (
        configs,
        _participants_from_meta,
        replay_model,
        replay_settings,
    ) = _load_configs_and_participants_from_jsonl(
        replay_path,
        label=label,
    )
    if not configs:
        return 2, None, None
    if args.annotation:
        configs = load_annotation_configs(args.annotation)

    _apply_prior_run_arguments(
        args,
        jsonl_path=replay_path,
        settings=replay_settings,
        run_model=replay_model,
        label=label.lower(),
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    replay_files = collect_replay_files_for_job(
        replay_path,
        output_root=output_root,
        include_all_participants=bool(getattr(args, "replay_all_ppts", False)),
        read_annotation_ids=_read_resume_annotation_ids,
    )

    return 0, configs, replay_files


def _collect_manual_replay_files(args: argparse.Namespace) -> List[Path]:
    """Return replay JSONL paths provided via --replay-from-manual."""

    replay_from_manual = getattr(args, "replay_from_manual", None) or []
    replay_files: List[Path] = []
    for item in replay_from_manual:
        if item is None:
            continue
        replay_files.append(Path(item).expanduser().resolve())
    return replay_files


def _collect_annotation_ids_for_manual_replay(
    replay_annotation_ids: Mapping[ReplayKey, Set[str]],
) -> Set[str]:
    """Return annotation ids present across replay records."""

    annotation_ids: Set[str] = set()
    for ids in replay_annotation_ids.values():
        annotation_ids.update(ids)
    return annotation_ids


def _collect_replay_annotation_ids(
    replay_files: Sequence[Path],
) -> dict[ReplayKey, Set[str]]:
    """Return replay message keys mapped to annotation ids across files.

    Parameters
    ----------
    replay_files:
        Sequence of JSONL files produced by ``classify_chats.py``.

    Returns
    -------
    dict[ReplayKey, Set[str]]
        Mapping from message keys to annotation ids observed in the replay
        JSONL records.
    """

    combined: dict[ReplayKey, Set[str]] = {}
    for path in replay_files:
        per_file = load_replay_message_annotation_ids(path)
        for key, annotation_ids in per_file.items():
            combined.setdefault(key, set()).update(annotation_ids)
    return combined


def _run_jsonl_replay_from_files(
    args: argparse.Namespace,
    *,
    replay_files: Sequence[Path],
    configs: Sequence[AnnotationConfig],
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]],
    label: str,
    restrict_roles: bool = True,
) -> int:
    """Run a JSONL-only replay using precomputed inputs."""

    contexts = _load_message_contexts_from_jsonls(
        replay_files,
        configs,
        participants_filter=args.participants,
        restrict_roles=restrict_roles,
    )
    if not contexts:
        logging.error(
            "%s did not yield any messages for classification.",
            label,
        )
        return 2

    max_messages: Optional[int]
    if args.max_messages and args.max_messages > 0:
        max_messages = min(args.max_messages, len(contexts))
        contexts = contexts[:max_messages]
    else:
        max_messages = len(contexts)
    progress_total = max_messages

    if args.dry_run:
        return run_dry_run(
            iter(contexts),
            configs,
            args.model,
            max_messages,
            replay_annotation_ids=replay_annotation_ids,
        )

    dummy_root = Path(args.input).expanduser().resolve()
    (
        status,
        output_dir,
        resolved_output_name,
        non_default_arguments,
    ) = prepare_output_files(
        args,
        dummy_root,
        configs,
        participants_filter=args.participants,
        single_output_file=None,
        resume_job_basename=None,
        forced_output_name=None,
    )
    if status != 0 or output_dir is None:
        return status

    return classify_messages(
        args,
        iter(contexts),
        configs,
        output_dir,
        single_output_file=None,
        non_default_arguments=non_default_arguments,
        resolved_output_name=resolved_output_name,
        existing_positive_counts={},
        resume_seen_keys=set(),
        target_keys=None,
        replay_annotation_ids=replay_annotation_ids,
        progress_total=progress_total,
        max_messages=max_messages,
    )


def _load_message_contexts_from_jsonls(
    jsonl_paths: Sequence[Path],
    configs: Sequence[AnnotationConfig],
    participants_filter: Optional[Sequence[str]],
    *,
    restrict_roles: bool = True,
) -> List[MessageContext]:
    """Return unique message contexts reconstructed from one or more JSONL files.

    Parameters
    ----------
    jsonl_paths:
        One or more JSONL files produced by ``classify_chats.py``.
    configs:
        Annotation configurations used to derive allowed roles.
    participants_filter:
        Optional participant identifiers used to limit which messages are
        included.
    restrict_roles:
        When True, drop messages whose roles fall outside of the annotation
        scopes implied by ``configs``. When False, include all roles and rely
        on per-annotation filtering later.

    Returns
    -------
    list[MessageContext]
        Unique message contexts keyed by participant, source path, chat index,
        and message index. When available, per-record ``preceding`` context
        from JSONL outputs is restored onto each context.
    """
    participants_set = normalize_participant_filter(participants_filter)
    base_allowed_roles = derive_allowed_roles(configs)
    restrict_to_scope = restrict_roles and base_allowed_roles is not None
    allowed_roles: Set[str] = set(base_allowed_roles or [])

    contexts: List[MessageContext] = []
    seen_keys: Set[ReplayKey] = set()

    for jsonl_path in jsonl_paths:
        for obj in iter_jsonl_records(jsonl_path):
            participant = participant_from_record(obj)
            source_path = str(obj.get("source_path") or "").strip()
            if not participant or not source_path:
                continue

            pair: Optional[IndexPair] = parse_message_indices(obj)
            if pair is None:
                continue
            chat_index, message_index = pair

            key: ReplayKey = (
                participant,
                source_path,
                chat_index,
                message_index,
            )
            if key in seen_keys:
                continue

            if participants_set is not None and (
                participant.lower() not in participants_set
            ):
                continue

            role = str(obj.get("role") or "").strip() or "user"
            if restrict_to_scope and role not in allowed_roles:
                continue

            content = str(obj.get("content") or "")
            chat_key = str(obj.get("chat_key") or "").strip() or None
            timestamp = obj.get("timestamp")
            chat_date = obj.get("chat_date")

            preceding_messages: Optional[List[dict[str, str]]] = None
            raw_preceding = obj.get("preceding")
            if isinstance(raw_preceding, list):
                normalized_preceding: List[dict[str, str]] = []
                for item in raw_preceding:
                    if not isinstance(item, dict):
                        continue
                    prev_role = str(item.get("role") or "").strip() or "unknown"
                    prev_content = str(item.get("content") or "").strip()
                    if not prev_content:
                        continue
                    index_parsed = parse_optional_int(item.get("index"))
                    timestamp_value = normalize_optional_string(item.get("timestamp"))
                    entry = build_preceding_entry(
                        prev_role,
                        prev_content,
                        index=index_parsed,
                        timestamp=timestamp_value,
                    )
                    normalized_preceding.append(entry)
                if normalized_preceding:
                    preceding_messages = normalized_preceding

            context = MessageContext(
                participant=participant,
                source_path=Path(source_path),
                chat_index=chat_index,
                chat_key=chat_key,
                chat_date=chat_date,
                message_index=message_index,
                role=role,
                timestamp=timestamp,
                content=content,
                preceding=preceding_messages,
            )
            contexts.append(context)
            seen_keys.add(key)

    return contexts


def _apply_overrides_to_args(
    args: argparse.Namespace,
    defaults: Mapping[str, object],
    overrides: Mapping[str, object],
) -> None:
    """Apply parameter overrides onto ``args`` when still at default values.

    For each key in ``overrides``, the corresponding attribute on ``args`` is
    updated only when its normalized value still matches the parser default.
    Core location-style arguments such as ``input`` and ``output_dir`` are
    never modified.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments for the current invocation.
    defaults: Mapping[str, object]
        Mapping of argument names to their parser default values.
    overrides: Mapping[str, object]
        Mapping of argument names to original values that should be restored
        when the user did not explicitly override them.
    """

    protected_keys = {
        "input",
        "output_dir",
        "output_name",
        "replay_from",
        "replay_from_manual",
        "resume_from",
    }

    for key, original_value in overrides.items():
        if key in protected_keys:
            continue
        if not hasattr(args, key):
            continue
        default_value = defaults.get(key, None)
        current_value = getattr(args, key, None)
        if normalize_arg_value(current_value) != normalize_arg_value(default_value):
            continue
        setattr(args, key, original_value)


def _maybe_override_model(
    args: argparse.Namespace,
    defaults: Mapping[str, object],
    run_model: Optional[str],
    label: str,
) -> None:
    """Update ``args.model`` from a prior run when appropriate.

    The model recorded in the prior run meta is applied only when the current
    model value still matches the parser default. When the user has provided
    an explicit model on the command line, that choice is preserved and a
    short diagnostic is logged.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments for the current invocation.
    defaults: Mapping[str, object]
        Mapping of argument names to their parser default values.
    run_model: Optional[str]
        Model name recorded in the prior run meta, when available.
    label: str
        Human-readable label for logging context (for example, ``\"replay\"``
        or ``\"resume\"``).
    """

    if not run_model:
        return

    default_model = defaults.get("model")
    if normalize_arg_value(args.model) == normalize_arg_value(default_model):
        if args.model != run_model:
            logging.info(
                "Overriding default model %s with model from %s JSONL: %s",
                args.model,
                label,
                run_model,
            )
        args.model = run_model
    elif args.model != run_model:
        logging.info(
            "Using model %s from command line instead of %s JSONL model %s",
            args.model,
            label,
            run_model,
        )


def _gather_participant_targets(
    root: Path, participants: Optional[Sequence[str]], *, followlinks: bool
) -> List[tuple[Path, str]]:
    """Return a list of (relative_directory, participant_name) pairs.

    relative_directory matches where files will be written (mirrors the
    later output path logic), and participant_name is the ppt id used in meta.
    Only chats whose paths resolve to a bucket label (for example, ``105`` or
    ``207``) are considered; other directories (such as ``public``) are
    ignored. Respects the provided participants filter when given.
    """
    participant_filter = normalize_participant_filter(participants)
    seen: dict[Path, str] = {}
    for file_path, _chats in iter_loaded_chats(root, followlinks=followlinks):
        bucket, rel_path = resolve_bucket_and_rel_path(file_path, root)
        if not bucket:
            # Skip files that do not belong to a labeled bucket
            continue
        participant = normalize_participant_id(bucket)
        if participant_filter and participant.lower() not in participant_filter:
            continue
        rel_dir = rel_path.parent
        if rel_dir == Path("."):
            rel_dir = Path(participant)
        # Record the first participant observed for this output directory
        if rel_dir not in seen:
            seen[rel_dir] = participant
    return [(rel_dir, seen[rel_dir]) for rel_dir in sorted(seen.keys(), key=str)]


def run_dry_run(
    message_iter: Iterator[MessageContext],
    configs: Sequence[AnnotationConfig],
    model: str,
    max_messages: Optional[int],
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]] = None,
) -> int:
    """Print a prompt preview and cost estimate for the first message.

    Parameters
    ----------
    message_iter: Iterator[MessageContext]
        Iterator over candidate message contexts.
    configs: Sequence[AnnotationConfig]
        Annotation configurations selected for the run.
    model: str
        LiteLLM model identifier to use for estimation.
    max_messages: Optional[int]
        Optional cap on the number of messages considered during estimation.
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]]
        Optional mapping that limits each replayed message to specific
        annotation ids.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """
    (
        status,
        first,
        applicable,
        first_requests,
        participant_counts,
    ) = _select_first_message_and_configs(
        message_iter,
        configs,
        replay_annotation_ids=replay_annotation_ids,
    )
    if status != 0 or first is None:
        return status

    if max_messages is None:
        print(
            "Dry run will scan all available messages to estimate cost. "
            "Use --max-messages to limit the scope.",
            file=sys.stderr,
        )

    progress_bar = None
    if sys.stderr.isatty():
        progress_bar = tqdm(
            desc="Estimating cost",
            unit="request",
            leave=False,
        )
    total_cost, cost_breakdown, max_tokens, total_requests = (
        safe_estimate_max_request_cost(
            model,
            _iter_all_requests_for_dry_run(
                first_requests,
                message_iter,
                configs,
                max_messages=max_messages,
                replay_annotation_ids=replay_annotation_ids,
                participant_counts=participant_counts,
            ),
            max_completion_tokens=MAX_CLASSIFICATION_TOKENS,
            progress_callback=progress_bar.update if progress_bar is not None else None,
        )
    )
    if progress_bar is not None:
        progress_bar.close()

    _print_dry_run_output(
        first,
        applicable,
        configs,
        model,
        first_requests,
        total_cost,
        cost_breakdown,
        max_tokens,
        total_requests,
        participant_counts,
    )
    return 0


def _select_first_message_and_configs(
    message_iter: Iterator[MessageContext],
    configs: Sequence[AnnotationConfig],
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]] = None,
) -> tuple[
    int,
    Optional[MessageContext],
    List[AnnotationConfig],
    List[AnnotationRequest],
    dict[str, int],
]:
    """Return the first message, applicable configs, initial requests, and counts.

    Handles early exit conditions when no messages match or when none of the
    provided configurations apply to the first message.

    Parameters
    ----------
    message_iter: Iterator[MessageContext]
        Iterator over candidate message contexts.
    configs: Sequence[AnnotationConfig]
        Annotation configurations selected for the run.
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]]
        Optional mapping that limits each replayed message to specific
        annotation ids.

    Returns
    -------
    tuple[
        int,
        Optional[MessageContext],
        List[AnnotationConfig],
        List[AnnotationRequest],
        dict[str, int],
    ]
        A status code (zero on success, non-zero on failure), the first
        message context, the list of applicable configs, the initial
        annotation requests for the first message, and a participant
        message-count mapping seeded with the first message when available.
    """

    first: Optional[MessageContext] = None
    applicable: List[AnnotationConfig] = []
    for candidate in message_iter:
        filtered_configs = _filter_configs_for_replay(
            configs,
            replay_annotation_ids,
            candidate,
        )
        applicable = [
            config
            for config in filtered_configs
            if config.allowed_roles is None or candidate.role in config.allowed_roles
        ]
        if applicable:
            first = candidate
            break

    if first is None:
        print("No messages matched the provided filters.", file=sys.stderr)
        return 1, None, [], [], {}

    participant_counts: dict[str, int] = {first.participant: 1}

    first_requests: List[AnnotationRequest] = []
    for config in applicable:
        prompt = build_prompt(
            config.spec,
            first.content,
            role=first.role,
            context_messages=first.preceding,
        )
        first_requests.append(
            (
                config.spec["id"],
                config.spec["name"],
                build_completion_messages(prompt),
            )
        )

    return 0, first, applicable, first_requests, participant_counts


def _iter_all_requests_for_dry_run(
    first_requests: Sequence[AnnotationRequest],
    message_iter: Iterator[MessageContext],
    configs: Sequence[AnnotationConfig],
    max_messages: Optional[int],
    *,
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]] = None,
    participant_counts: dict[str, int],
) -> Iterator[AnnotationRequest]:
    """Yield annotation requests for cost estimation in dry-run mode.

    The iterator first yields requests for the initial message and then
    continues through the remaining messages, respecting the optional
    ``max_messages`` cap and updating participant message counts.

    Parameters
    ----------
    first_requests: Sequence[AnnotationRequest]
        Requests prepared for the first matching message.
    message_iter: Iterator[MessageContext]
        Iterator over remaining candidate message contexts.
    configs: Sequence[AnnotationConfig]
        Annotation configurations selected for the run.
    max_messages: Optional[int]
        Optional cap on the number of messages considered during estimation.
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]]
        Optional mapping that limits each replayed message to specific
        annotation ids.
    participant_counts: dict[str, int]
        Mutable mapping of participant identifiers to message counts that
        will be updated as additional messages are processed.
    """

    messages_counted = 1
    yield from first_requests

    for context in message_iter:
        filtered_configs = _filter_configs_for_replay(
            configs,
            replay_annotation_ids,
            context,
        )
        applicable_configs = [
            config
            for config in filtered_configs
            if config.allowed_roles is None or context.role in config.allowed_roles
        ]
        if not applicable_configs:
            continue
        if max_messages is not None and messages_counted >= max_messages:
            break
        participant_counts[context.participant] = (
            participant_counts.get(context.participant, 0) + 1
        )
        messages_counted += 1
        for config in applicable_configs:
            prompt = build_prompt(
                config.spec,
                context.content,
                role=context.role,
                context_messages=context.preceding,
            )
            yield (
                config.spec["id"],
                config.spec["name"],
                build_completion_messages(prompt),
            )


def _print_dry_run_output(
    first: MessageContext,
    applicable: Sequence[AnnotationConfig],
    configs: Sequence[AnnotationConfig],
    model: str,
    first_requests: Sequence[AnnotationRequest],
    total_cost: float,
    cost_breakdown: List[dict[str, object]],
    max_tokens: Optional[int],
    total_requests: int,
    participant_counts: Mapping[str, int],
) -> None:
    """Print prompt preview, payload, cost summary, and metadata for a dry run."""

    total_messages_tracked = sum(participant_counts.values())

    total_prompt_tokens: Optional[int]
    total_assumed_completion_tokens: Optional[int]
    if cost_breakdown:
        (
            total_prompt_tokens,
            total_assumed_completion_tokens,
        ) = summarize_token_totals(cost_breakdown)
    else:
        total_prompt_tokens = None
        total_assumed_completion_tokens = None

    print("Dry run prompt preview:")
    separator = "-" * 80
    for config in applicable:
        annotation = config.spec
        print(separator)
        print(f"[{annotation['id']}] {annotation['name']}")
        print(separator)
        print(
            build_prompt(
                annotation,
                first.content,
                role=first.role,
                context_messages=first.preceding,
            )
        )
        print()
    print(separator)

    first_payload = to_litellm_messages(first_requests[0][2])
    print("First message payload:")
    print(json.dumps(first_payload, ensure_ascii=False, indent=2))

    print_cost_summary(
        model,
        MAX_CLASSIFICATION_TOKENS,
        total_cost,
        cost_breakdown,
        max_tokens,
        total_requests,
    )

    if participant_counts:
        summary = ", ".join(
            f"{name} ({count})" for name, count in participant_counts.items()
        )
        print(
            f"\nParticipants encountered ({len(participant_counts)} participants, "
            f"{total_messages_tracked} messages): {summary}"
        )

    print("Metadata:")
    print(
        json.dumps(
            {
                "participant": first.participant,
                "source_path": str(first.source_path),
                "chat_index": first.chat_index,
                "chat_key": first.chat_key,
                "message_index": first.message_index,
                "role": first.role,
                "annotation_ids": [config.spec["id"] for config in configs],
                "applicable_annotation_ids": [
                    config.spec["id"] for config in applicable
                ],
                "max_potential_cost_usd": total_cost if cost_breakdown else None,
                "estimated_request_count": total_requests if cost_breakdown else None,
                "total_prompt_tokens": total_prompt_tokens,
                "total_assumed_completion_tokens": total_assumed_completion_tokens,
                "max_completion_tokens": MAX_CLASSIFICATION_TOKENS,
                "model_max_tokens": max_tokens,
                "participant_message_counts": dict(participant_counts),
                "total_messages_considered": total_messages_tracked,
            },
            indent=2,
        )
    )


def _filter_configs_for_replay(
    configs: Sequence[AnnotationConfig],
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]],
    context: MessageContext,
) -> Sequence[AnnotationConfig]:
    """Return configs restricted to replay annotation ids for ``context``."""

    if replay_annotation_ids is None:
        return configs
    replay_key: ReplayKey = (
        context.participant,
        str(context.source_path),
        context.chat_index,
        context.message_index,
    )
    allowed = replay_annotation_ids.get(replay_key)
    if not allowed:
        return []
    return [config for config in configs if str(config.spec.get("id")) in allowed]


def classify_messages(
    args: argparse.Namespace,
    message_iter: Iterator[MessageContext],
    configs: Sequence[AnnotationConfig],
    output_dir: Path,
    single_output_file: Optional[Path],
    non_default_arguments: Mapping[str, object],
    resolved_output_name: str,
    existing_positive_counts: Mapping[str, int],
    resume_seen_keys: Set[SeenKey],
    target_keys: Optional[Set[SeenKey]],
    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]],
    progress_total: Optional[int],
    max_messages: Optional[int],
) -> int:
    """Run the main classification loop and write JSONL outputs.

    Parameters
    ----------
    args:
        Parsed command-line arguments for the current invocation.
    message_iter:
        Iterator over message contexts to classify.
    configs:
        Annotation configurations selected for the run.
    output_dir:
        Output directory for JSONL results.
    single_output_file:
        Optional explicit JSONL path when resuming or forcing output.
    non_default_arguments:
        Argument overrides to persist in output metadata.
    resolved_output_name:
        Output filename resolved from job parameters.
    existing_positive_counts:
        Prior positive counts used for min-positive skipping.
    resume_seen_keys:
        Already-classified keys to skip.
    target_keys:
        Optional set of keys to target for retry runs.
    replay_annotation_ids:
        Optional mapping that limits each replayed message to specific
        annotation ids.
    progress_total:
        Optional total for the progress bar.
    max_messages:
        Optional cap on the number of messages to process.

    Returns
    -------
    int
        Zero on success, non-zero on failure.
    """
    processed = 0
    participant_counts: dict[str, int] = {}
    prefilter_states: dict[tuple[str, str, int, str], PrefilterState] = {}

    # NB: This currently loops through each of the different messages
    # and then does all of the annotations at once for that message.
    # We later may want to do the opposite depending on the max
    # batch size.

    min_positive = max(0, int(getattr(args, "min_positive_per_annotation", 0) or 0))
    positive_counts: dict[str, int] = {}
    if min_positive > 0:
        for config in configs:
            annotation_id = str(config.spec.get("id"))
            base_count = existing_positive_counts.get(annotation_id, 0)
            positive_counts[annotation_id] = base_count
            if base_count >= min_positive:
                logging.info(
                    "Annotation %s (%s) already has %s positives from prior runs; "
                    "skipping further classification for this annotation.",
                    annotation_id,
                    config.spec.get("name", ""),
                    base_count,
                )

    try:
        output_handles: dict[Path, TextIO] = {}
        with (
            ExitStack() as stack,
            tqdm(
                total=progress_total,
                desc="Classifying messages",
                disable=not sys.stderr.isatty(),
            ) as progress,
        ):
            for context in message_iter:
                if max_messages is not None and processed >= max_messages:
                    break

                chat_identifier: Optional[tuple[str, str, int, str]] = None
                prefilter_state: Optional[PrefilterState] = None
                if args.prefilter_conversations:
                    chat_identifier = (
                        context.participant,
                        str(context.source_path),
                        context.chat_index,
                        context.chat_key or "",
                    )
                    prefilter_state = prefilter_states.get(chat_identifier)
                    if prefilter_state is None:
                        prefilter_state = PrefilterState()
                        prefilter_states[chat_identifier] = prefilter_state
                    if prefilter_state.skip_remaining:
                        continue

                tasks = build_classification_tasks_for_context(
                    context,
                    _filter_configs_for_replay(
                        configs,
                        replay_annotation_ids,
                        context,
                    ),
                    resume_seen_keys,
                    min_positive,
                    positive_counts,
                    args=args,
                    target_keys=target_keys,
                )
                if not tasks:
                    continue

                try:
                    outcomes = classify_tasks_batch(
                        tasks,
                        model=args.model,
                        timeout=args.timeout,
                        max_workers=max(
                            1, int(getattr(args, "max_workers", MAX_WORKERS))
                        ),
                        system_prompt=ANNOTATION_SYSTEM_PROMPT,
                    )
                except ClassificationError as err:
                    outcomes = [
                        ClassificationOutcome(
                            task=task,
                            matches=[],
                            error=str(err),
                            rationale=None,
                            score=None,
                        )
                        for task in tasks
                    ]

                if not outcomes:
                    continue

                participant_counts[context.participant] = (
                    participant_counts.get(context.participant, 0) + 1
                )

                wrote = ensure_output_and_write_outcomes_for_context(
                    context=context,
                    outcomes=outcomes,
                    args=args,
                    configs=configs,
                    output_dir=output_dir,
                    single_output_file=single_output_file,
                    resolved_output_name=resolved_output_name,
                    non_default_arguments=non_default_arguments,
                    output_handles=output_handles,
                    stack=stack,
                )
                if not wrote:
                    return 2

                if args.sleep:
                    time.sleep(args.sleep)

                if args.prefilter_conversations and prefilter_state:
                    prefilter_state.turns_evaluated += 1
                    if any(outcome.matches for outcome in outcomes):
                        prefilter_state.has_match = True
                    if (
                        prefilter_state.turns_evaluated >= PREFILTER_TURN_LIMIT
                        and not prefilter_state.has_match
                    ):
                        prefilter_state.skip_remaining = True
                        logging.debug(
                            "Prefilter skipping remaining messages for chat %s "
                            "(participant=%s, index=%s).",
                            context.chat_key,
                            context.participant,
                            context.chat_index,
                        )

                processed += 1
                progress.update(1)

                if min_positive > 0 and positive_counts:
                    score_cutoff = getattr(args, "score_cutoff", None)
                    for outcome in outcomes:
                        if not outcome.matches:
                            continue
                        if score_cutoff is not None:
                            if outcome.score is None or outcome.score < score_cutoff:
                                continue
                        annotation_id = str(outcome.task.annotation.get("id"))
                        if annotation_id not in positive_counts:
                            continue
                        current = positive_counts.get(annotation_id, 0)
                        if current < min_positive:
                            new_count = current + 1
                            positive_counts[annotation_id] = new_count
                            if new_count == min_positive:
                                logging.info(
                                    "Reached minimum positive quota (%s) for "
                                    "annotation %s (%s).",
                                    min_positive,
                                    annotation_id,
                                    outcome.task.annotation.get("name", ""),
                                )
                    if all(count >= min_positive for count in positive_counts.values()):
                        logging.info(
                            "Reached minimum positive quota (%s) for all %s annotations; "
                            "stopping early after %s messages.",
                            min_positive,
                            len(positive_counts),
                            processed,
                        )
                        break
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130

    if (
        min_positive > 0
        and positive_counts
        and not all(count >= min_positive for count in positive_counts.values())
        and max_messages is not None
        and processed >= max_messages
    ):
        logging.warning(
            "Reached --max-messages=%s before satisfying "
            "--min-positive-per-annotation=%s for all annotations. "
            "Final positive counts: %s",
            max_messages,
            min_positive,
            positive_counts,
        )

    logging.info("Finished classifying %s messages.", processed)
    logging.info("Results written under %s", output_dir)
    if participant_counts:
        formatted_participants = ", ".join(
            f"{name} ({count})" for name, count in participant_counts.items()
        )
        logging.info(
            "Participants processed (%s): %s",
            len(participant_counts),
            formatted_participants,
        )
    return 0


def prepare_replay_and_resume_state(
    args: argparse.Namespace,
) -> tuple[
    int,
    Optional[List[AnnotationConfig]],
    Optional[Sequence[str]],
    Optional[Set[ReplayKey]],
    Optional[str],
    dict[str, int],
    Optional[str],
    Optional[Path],
    Set[SeenKey],
]:
    """Prepare configs, participant filters, and replay/resume state for a run.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments for the current invocation.
        Unused here; kept for future compatibility.

    Returns
    -------
    tuple
        A tuple containing the status code (zero on success), the loaded
        annotation configs, participant filter, replay keys, resume path,
        existing positive counts, resume job basename, single output file
        path when resuming from a specific JSONL, and the set of seen keys
        used for skipping already processed messages.
    """

    participants_filter: Optional[Sequence[str]] = args.participants

    resume_from = getattr(args, "resume_from", None)
    replay_keys: Optional[Set[ReplayKey]] = None
    resume_model: Optional[str] = None
    resume_settings: dict[str, object] = {}
    existing_positive_counts: dict[str, int] = {}

    replay_from = getattr(args, "replay_from", None)
    if resume_from is not None and replay_from is not None:
        logging.error("--resume-from cannot be combined with --replay-from.")
        return 2, None, None, None, None, existing_positive_counts, None, None, set()

    if replay_from is not None:
        if not _validate_replay_sampling_flags(args):
            return _replay_resume_error(2, existing_positive_counts)

        status, configs, replay_files = _prepare_replay_configs_and_files(
            args,
            label="Replay JSONL",
        )
        if status != 0 or configs is None or replay_files is None:
            return _replay_resume_error(status or 2, existing_positive_counts)

        combined_replay_keys: Set[ReplayKey] = set()
        for path in replay_files:
            combined_replay_keys.update(load_replay_message_keys(path))
        replay_keys = combined_replay_keys

        if not replay_keys:
            logging.error(
                "Replay JSONL%s contained no usable message records.",
                " files" if len(replay_files) > 1 else "",
            )
            return _replay_resume_error(2, existing_positive_counts)
        logging.info(
            "Replaying %s unique messages from %s file(s)",
            len(replay_keys),
            len(replay_files),
        )
        if participants_filter is None:
            participants_filter = sorted({key[0] for key in replay_keys})
    elif resume_from is not None:
        if args.annotation:
            logging.error(
                "--annotation cannot be used with --resume-from; "
                "annotations are fixed by the resume JSONL.",
            )
            return _replay_resume_error(2, existing_positive_counts)
        if args.participants:
            logging.error(
                "--participant/--participants cannot be used with --resume-from; "
                "participants are fixed by the resume JSONL.",
            )
            return _replay_resume_error(2, existing_positive_counts)

        resume_path = Path(resume_from).expanduser().resolve()
        (
            configs,
            inferred_participants,
            resume_model,
            resume_settings,
        ) = _load_configs_and_participants_from_jsonl(
            resume_path,
            label="Resume JSONL",
        )
        if not configs:
            return _replay_resume_error(2, existing_positive_counts)
        if not inferred_participants:
            logging.error(
                "Resume JSONL meta does not contain a usable participant identifier: %s",
                resume_path,
            )
            return _replay_resume_error(2, existing_positive_counts)
        participants_filter = inferred_participants

        _apply_prior_run_arguments(
            args,
            jsonl_path=resume_path,
            settings=resume_settings,
            run_model=resume_model,
            label="resume",
        )
    else:
        configs = load_annotation_configs(args.annotation)

    # Resume support: collect previously seen keys for per-annotation skipping.
    single_output_file: Optional[Path] = None
    resume_seen_keys: Set[SeenKey] = set()
    resume_positive_sources: List[Path] = []
    resume_job_basename: Optional[str] = None

    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        single_output_file = resume_path
        keys, _ = load_resume_keys(resume_path, None)
        resume_seen_keys.update(keys)

    if (
        getattr(args, "resume_auto", False)
        and single_output_file is None
        and replay_keys is None
    ):
        output_root = Path(args.output_dir).expanduser().resolve()
        if args.annotation:
            wanted_ann_ids: Set[str] = {str(item) for item in args.annotation if item}
        else:
            wanted_ann_ids = {str(cfg.spec.get("id")) for cfg in configs}

        candidates: List[Path] = []
        for path, meta in iter_jsonl_meta(output_root):
            ids_in_meta: Set[str] = set()
            snap = meta.get("annotation_snapshots")
            if isinstance(snap, dict):
                ids_in_meta.update(str(key) for key in snap.keys())
            ann_ids = meta.get("annotation_ids")
            if isinstance(ann_ids, list):
                ids_in_meta.update(
                    str(item) for item in ann_ids if isinstance(item, str)
                )
            if (
                wanted_ann_ids
                and ids_in_meta
                and wanted_ann_ids.isdisjoint(ids_in_meta)
            ):
                continue
            candidates.append(path)

        selected = pick_latest_per_parent(candidates)
        if selected:
            print("Found prior outputs to resume from:")
            for selected_path in sorted(selected):
                print(f" - {selected_path}")
            response = input(
                "Resume by skipping seen messages from these files? [y/N] "
            )
            if response.strip().lower() not in {"y", "yes"}:
                print("Resume cancelled by user; proceeding without resume.")
            else:
                for selected_path in selected:
                    keys, _ = load_resume_keys(selected_path, None)
                    resume_seen_keys.update(keys)
                    resume_positive_sources.append(selected_path)
                    if resume_job_basename is None:
                        resume_job_basename = selected_path.name

    if resume_from is not None:
        existing_positive_counts = _load_positive_counts_from_files(
            [Path(resume_from).expanduser().resolve()],
            score_cutoff=getattr(args, "score_cutoff", None),
        )
    elif resume_positive_sources:
        existing_positive_counts = _load_positive_counts_from_files(
            resume_positive_sources,
            score_cutoff=getattr(args, "score_cutoff", None),
        )

    return (
        0,
        configs,
        participants_filter,
        replay_keys,
        resume_from,
        existing_positive_counts,
        resume_job_basename,
        single_output_file,
        resume_seen_keys,
    )


def _replay_resume_error(
    status: int,
    existing_positive_counts: dict[str, int],
) -> tuple[
    int,
    None,
    None,
    None,
    None,
    dict[str, int],
    None,
    None,
    Set[SeenKey],
]:
    """Return a standardized error tuple for replay/resume preparation."""

    return status, None, None, None, None, existing_positive_counts, None, None, set()


def run_jsonl_only_replay(args: argparse.Namespace) -> int:
    """Execute a replay run that reads messages directly from JSONL outputs.

    This mode reuses the original message sample encoded in existing JSONL
    files without reading source transcripts from disk. Target messages and
    metadata are reconstructed from the JSONL records themselves, and
    conversation context is not restored.
    """

    if getattr(args, "resume_from", None) is not None or getattr(
        args, "resume_auto", False
    ):
        logging.error(
            "--replay-from without --replay-from-source cannot be combined with "
            "--resume-from or --resume-auto."
        )
        return 2

    if not _validate_replay_sampling_flags(args):
        return 2

    status, configs, replay_files = _prepare_replay_configs_and_files(
        args,
        label="Replay JSONL",
    )
    if status != 0 or configs is None or replay_files is None:
        return status or 2

    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]] = None

    label = "Replay JSONL"
    if len(replay_files) > 1:
        label = f"{label} files"
    return _run_jsonl_replay_from_files(
        args,
        replay_files=replay_files,
        configs=configs,
        replay_annotation_ids=replay_annotation_ids,
        label=label,
    )


def run_jsonl_only_manual_replay(args: argparse.Namespace) -> int:
    """Execute a replay run using manual annotation JSONL files."""

    if getattr(args, "resume_from", None) is not None or getattr(
        args, "resume_auto", False
    ):
        logging.error(
            "--replay-from-manual cannot be combined with --resume-from or --resume-auto."
        )
        return 2

    if not _validate_replay_sampling_flags(args):
        return 2

    replay_files = _collect_manual_replay_files(args)
    if not replay_files:
        logging.error("--replay-from-manual must be provided for replay runs.")
        return 2
    missing = [path for path in replay_files if not path.exists()]
    if missing:
        logging.error(
            "Manual replay JSONL file%s not found: %s",
            "s" if len(missing) > 1 else "",
            ", ".join(str(path) for path in missing),
        )
        return 2

    replay_annotation_ids = _collect_replay_annotation_ids(replay_files)
    if not replay_annotation_ids:
        logging.error("Manual replay JSONL did not yield annotation ids for filtering.")
        return 2

    if args.annotation:
        configs = load_annotation_configs(args.annotation)
    else:
        annotation_ids = _collect_annotation_ids_for_manual_replay(
            replay_annotation_ids
        )
        configs = _load_configs_from_annotation_ids(
            annotation_ids,
            label="Manual replay JSONL",
        )
    if not configs:
        return 2

    label = "Manual replay JSONL"
    if len(replay_files) > 1:
        label = f"{label} files"
    return _run_jsonl_replay_from_files(
        args,
        replay_files=replay_files,
        configs=configs,
        replay_annotation_ids=replay_annotation_ids,
        label=label,
        restrict_roles=False,
    )


def run_retry_errors(args: argparse.Namespace) -> int:
    """Retry only records with non-quote-mismatch errors from prior outputs."""

    retry_from = getattr(args, "retry_errors_from", None)
    if retry_from is None:
        logging.error("--retry-errors-from must be provided for retry runs.")
        return 2
    if getattr(args, "resume_from", None) is not None or getattr(
        args, "resume_auto", False
    ):
        logging.error("--retry-errors-from cannot be combined with resume options.")
        return 2
    if getattr(args, "replay_from", None) is not None or getattr(
        args, "replay_from_manual", None
    ):
        logging.error(
            "--retry-errors-from cannot be combined with --replay-from or "
            "--replay-from-manual."
        )
        return 2
    if not _validate_retry_sampling_flags(args):
        return 2

    retry_path = Path(retry_from).expanduser().resolve()
    if not retry_path.exists():
        logging.error("Retry JSONL not found: %s", retry_path)
        return 2

    (
        configs,
        _participants_from_meta,
        retry_model,
        retry_settings,
    ) = _load_configs_and_participants_from_jsonl(
        retry_path,
        label="Retry JSONL",
    )
    if not configs:
        return 2
    if args.annotation:
        configs = load_annotation_configs(args.annotation)

    _apply_prior_run_arguments(
        args,
        jsonl_path=retry_path,
        settings=retry_settings,
        run_model=retry_model,
        label="retry",
    )

    output_root = Path(args.output_dir).expanduser().resolve()
    retry_files = _collect_retry_family_files(
        retry_path,
        output_root,
        include_all_participants=bool(getattr(args, "retry_errors_all_ppts", False)),
    )
    if not retry_files:
        logging.error("No retry outputs found under %s", output_root)
        return 2

    target_keys, message_keys = _collect_retry_error_keys(retry_files)
    if not target_keys:
        logging.info("No retryable errors found in %s file(s).", len(retry_files))
        return 0

    participants_filter = sorted({key[0] for key in message_keys})
    args.min_positive_per_annotation = 0
    args.prefilter_conversations = False

    contexts = _load_message_contexts_from_jsonls(
        retry_files,
        configs,
        participants_filter,
    )
    contexts = [
        context
        for context in contexts
        if (
            context.participant,
            str(context.source_path),
            context.chat_index,
            context.message_index,
        )
        in message_keys
    ]
    if not contexts:
        logging.info("No retryable contexts found after filtering.")
        return 0
    max_messages = len(contexts)
    progress_total = max_messages
    if args.dry_run:
        return run_dry_run(
            iter(contexts),
            configs,
            args.model,
            max_messages,
            replay_annotation_ids=None,
        )
    message_iter = iter(contexts)
    root = Path(args.input).expanduser().resolve()

    (
        status,
        output_dir,
        resolved_output_name,
        non_default_arguments,
    ) = prepare_output_files(
        args,
        root,
        configs,
        participants_filter,
        single_output_file=None,
        resume_job_basename=None,
        forced_output_name=f"{infer_job_stem_from_filename(retry_path.name)}.jsonl",
    )
    if status != 0 or output_dir is None:
        return status

    return classify_messages(
        args,
        message_iter,
        configs,
        output_dir,
        single_output_file=None,
        non_default_arguments=non_default_arguments,
        resolved_output_name=resolved_output_name,
        existing_positive_counts={},
        resume_seen_keys=set(),
        target_keys=target_keys,
        replay_annotation_ids=None,
        progress_total=progress_total,
        max_messages=max_messages,
    )


def prepare_output_files(
    args: argparse.Namespace,
    root: Path,
    configs: Sequence[AnnotationConfig],
    participants_filter: Optional[Sequence[str]],
    single_output_file: Optional[Path],
    resume_job_basename: Optional[str],
    forced_output_name: Optional[str] = None,
) -> tuple[int, Optional[Path], str, Mapping[str, object]]:
    """Prepare output directory, filename, and per-participant meta files."""

    defaults = getattr(args, "_defaults", {})
    non_default_arguments = extract_non_default_arguments(args, defaults)
    non_default_arguments = dict(non_default_arguments)
    non_default_arguments["model"] = args.model

    explicit_job = getattr(args, "job", None)
    if explicit_job is not None:
        # Use a normalized variant of the user-provided job name as the
        # parameter fragment so related tooling can still parse filenames
        # consistently when needed.
        parameter_fragment = normalize_arg_value(explicit_job)
    else:
        parameter_fragment = dict_to_string(non_default_arguments)

    timestamp_fragment = datetime.now().strftime("%Y%m%d-%H%M%S")

    if forced_output_name is not None:
        resolved_output_name = forced_output_name
    elif resume_job_basename is not None and getattr(args, "resume_auto", False):
        resolved_output_name = resume_job_basename
    else:
        resolved_output_name = f"{timestamp_fragment}__{parameter_fragment}.jsonl"

    if len(resolved_output_name) > MAX_OUTPUT_NAME_LENGTH:
        if explicit_job is not None:
            logging.error(
                "Output filename %s is longer than %s characters even with the "
                "provided --job value. Please re-run with a shorter job name.",
                resolved_output_name,
                MAX_OUTPUT_NAME_LENGTH,
            )
        else:
            logging.error(
                "Output filename %s is longer than %s characters. Please re-run "
                "classify_chats with a shorter custom job name using --job.",
                resolved_output_name,
                MAX_OUTPUT_NAME_LENGTH,
            )
        return 2, None, resolved_output_name, non_default_arguments

    if single_output_file is not None:
        output_dir = single_output_file.parent
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logging.error("Failed to create output directory: %s", err)
            return 2, None, resolved_output_name, non_default_arguments

    if single_output_file is None:
        targets = _gather_participant_targets(
            root,
            participants_filter,
            followlinks=args.follow_links,
        )
        for rel_dir, ppt in targets:
            participant_dir = ensure_participant_directory(output_dir, rel_dir)
            if participant_dir is None:
                return 2, None, resolved_output_name, non_default_arguments
            output_file_path = participant_dir / resolved_output_name
            if not output_file_path.exists():
                try:
                    with output_file_path.open("w", encoding="utf-8") as handle:
                        meta_record = build_meta_record(
                            args=args,
                            configs=configs,
                            participant=ppt,
                            non_default_arguments=non_default_arguments,
                        )
                        handle.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
                except OSError as err:
                    logging.error(
                        "Failed to precreate output file %s: %s", output_file_path, err
                    )
                    return 2, None, resolved_output_name, non_default_arguments

    return 0, output_dir, resolved_output_name, non_default_arguments


def main(argv: Sequence[str] | None = None) -> int:
    """Script entry point."""

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if getattr(args, "retry_errors_from", None) is not None:
        return run_retry_errors(args)
    if getattr(args, "replay_from_manual", None):
        return run_jsonl_only_manual_replay(args)
    if getattr(args, "replay_from", None) is not None and not getattr(
        args, "replay_from_source", False
    ):
        return run_jsonl_only_replay(args)

    root = Path(args.input).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        logging.error("Input directory not found: %s", root)
        return 2

    (
        status,
        configs,
        participants_filter,
        replay_keys,
        resume_from,
        existing_positive_counts,
        resume_job_basename,
        single_output_file,
        resume_seen_keys,
    ) = prepare_replay_and_resume_state(args)
    if status != 0 or configs is None:
        return status

    # Resume support: collect previously seen keys for per-annotation skipping.
    single_output_file: Optional[Path] = None
    resume_seen_keys: Set[SeenKey] = set()
    resume_positive_sources: List[Path] = []
    resume_job_basename: Optional[str] = None

    if resume_from is not None:
        resume_path = Path(resume_from).expanduser().resolve()
        single_output_file = resume_path
        keys, _ = load_resume_keys(resume_path, None)
        resume_seen_keys.update(keys)

    if (
        getattr(args, "resume_auto", False)
        and single_output_file is None
        and replay_keys is None
    ):
        # Infer candidate files under the output directory and confirm.
        output_root = Path(args.output_dir).expanduser().resolve()
        wanted_ann_ids: Set[str]
        if args.annotation:
            wanted_ann_ids = {str(item) for item in args.annotation if item}
        else:
            # If no explicit annotation, prefer all currently selected configs.
            wanted_ann_ids = {str(cfg.spec.get("id")) for cfg in configs}

        # Collect files with overlapping annotation_ids.
        candidates: List[Path] = []
        for path, meta in iter_jsonl_meta(output_root):
            ids_in_meta: Set[str] = set()
            snap = meta.get("annotation_snapshots")
            if isinstance(snap, dict):
                ids_in_meta.update(str(key) for key in snap.keys())
            ann_ids = meta.get("annotation_ids")
            if isinstance(ann_ids, list):
                ids_in_meta.update(
                    str(item) for item in ann_ids if isinstance(item, str)
                )
            # Require at least one overlap with desired annotations.
            if (
                wanted_ann_ids
                and ids_in_meta
                and wanted_ann_ids.isdisjoint(ids_in_meta)
            ):
                continue
            candidates.append(path)

        selected = pick_latest_per_parent(candidates)
        if selected:
            print("Found prior outputs to resume from:")
            for selected_path in sorted(selected):
                print(f" - {selected_path}")
            response = input(
                "Resume by skipping seen messages from these files? [y/N] "
            )
            if response.strip().lower() not in {"y", "yes"}:
                print("Resume cancelled by user; proceeding without resume.")
            else:
                for selected_path in selected:
                    keys, _ = load_resume_keys(selected_path, None)
                    resume_seen_keys.update(keys)
                    resume_positive_sources.append(selected_path)
                    if resume_job_basename is None:
                        resume_job_basename = selected_path.name

    if resume_from is not None:
        existing_positive_counts = _load_positive_counts_from_files(
            [Path(resume_from).expanduser().resolve()],
            score_cutoff=getattr(args, "score_cutoff", None),
        )
    elif resume_positive_sources:
        existing_positive_counts = _load_positive_counts_from_files(
            resume_positive_sources,
            score_cutoff=getattr(args, "score_cutoff", None),
        )

    replay_annotation_ids: Optional[Mapping[ReplayKey, Set[str]]] = None

    message_iter, max_messages, progress_total, sampled_contexts = (
        prepare_message_iterator(
            args,
            root,
            configs,
            participants_filter,
            replay_keys,
        )
    )

    if args.dry_run:
        if sampled_contexts is not None:
            dry_run_iter: Iterator[MessageContext] = iter(sampled_contexts)
        else:
            dry_run_iter = message_iter
        return run_dry_run(
            dry_run_iter,
            configs,
            args.model,
            max_messages,
            replay_annotation_ids=replay_annotation_ids,
        )

    (
        status,
        output_dir,
        resolved_output_name,
        non_default_arguments,
    ) = prepare_output_files(
        args,
        root,
        configs,
        participants_filter,
        single_output_file,
        resume_job_basename,
        forced_output_name=None,
    )
    if status != 0 or output_dir is None:
        return status

    return classify_messages(
        args,
        message_iter,
        configs,
        output_dir,
        single_output_file,
        non_default_arguments,
        resolved_output_name,
        existing_positive_counts,
        resume_seen_keys,
        target_keys=None,
        replay_annotation_ids=replay_annotation_ids,
        progress_total=progress_total,
        max_messages=max_messages,
    )


if __name__ == "__main__":
    sys.exit(main())
