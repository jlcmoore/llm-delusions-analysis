"""
Shared helpers for annotation classification flows.

This module centralizes common logic used by ``classify_chats.py`` and
related tools, including message iteration, configuration selection, and
JSONL output handling. Keeping this code under ``src/`` allows multiple
entry points to share consistent behavior.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence, Set, TextIO, Tuple

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATION_SYSTEM_PROMPT,
    ANNOTATION_TEMPLATE,
    build_prompt,
)
from llm_delusions_annotations.chat.chat_utils import (
    iter_chat_messages,
    iter_loaded_chats,
    normalize_optional_string,
    resolve_bucket_and_rel_path,
)
from llm_delusions_annotations.classify_messages import (
    ClassificationOutcome,
    ClassificationTask,
    MessageContext,
    filter_quotes_to_content,
    find_unmatched_quotes,
)
from llm_delusions_annotations.configs import AnnotationConfig, derive_allowed_roles

from annotation.io import (
    PART_SUFFIX_MARKER,
    ReplayKey,
    SeenKey,
    infer_job_stem_from_filename,
)
from annotation.sampling import (
    limit_conversations_by_participant,
    sample_conversations_within_participant,
    sample_messages_by_participant,
)
from utils.participants import normalize_participant_filter, normalize_participant_id

MAX_OUTPUT_NAME_LENGTH = 200
SHARD_MAX_BYTES = 50_000_000
"""Target maximum size in bytes for an output JSONL shard."""

_SHARD_PATHS: dict[tuple[Path, str], Path] = {}
"""Mapping from (participant_dir, logical output name) to current shard path."""


def ensure_participant_directory(
    output_dir: Path,
    relative_directory: Path,
) -> Optional[Path]:
    """Ensure a participant directory exists under ``output_dir``.

    Parameters
    ----------
    output_dir:
        Root output directory for annotation files.
    relative_directory:
        Participant-specific subdirectory relative to ``output_dir``.

    Returns
    -------
    Optional[Path]
        The created or existing participant directory, or ``None`` on error.
    """

    participant_dir = output_dir / relative_directory
    try:
        participant_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logging.error(
            "Failed to create participant output directory %s: %s",
            participant_dir,
            err,
        )
        return None
    return participant_dir


def build_meta_record(
    *,
    args: argparse.Namespace,
    configs: Sequence[AnnotationConfig],
    participant: str,
    non_default_arguments: Mapping[str, object],
) -> dict[str, object]:
    """Return a meta record capturing shared annotation run settings.

    Parameters
    ----------
    args:
        Parsed command-line arguments providing model and contextual flags.
    configs:
        Annotation configurations used for the current run.
    participant:
        Participant identifier associated with the output file.
    non_default_arguments:
        Mapping of non-default CLI arguments recorded for traceability.

    Returns
    -------
    dict[str, object]
        Metadata dictionary suitable for writing as the first JSONL line.
    """

    annotation_snapshots = {
        cfg.spec["id"]: {
            "name": cfg.spec.get("name", ""),
            "description": cfg.spec.get("description", ""),
        }
        for cfg in configs
    }
    return {
        "type": "meta",
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "system_prompt": ANNOTATION_SYSTEM_PROMPT,
        "template": ANNOTATION_TEMPLATE,
        "arguments": non_default_arguments,
        "preceding_context": int(getattr(args, "preceding_context", 0) or 0),
        "annotation_snapshots": annotation_snapshots,
        "ppt_id": participant,
        "annotation_ids": [cfg.spec["id"] for cfg in configs],
        "participants": [participant],
    }


def build_base_record(context: MessageContext) -> dict[str, object]:
    """Return shared metadata fields for a message record.

    Parameters
    ----------
    context:
        Message context describing the participant, location, and message
        metadata.

    Returns
    -------
    dict[str, object]
        Dictionary containing participant, location, and role metadata shared
        by classifier outputs and manual annotation datasets.
    """

    return {
        "participant": context.participant,
        "ppt_id": context.participant,
        "source_path": str(context.source_path),
        "chat_index": context.chat_index,
        "chat_key": context.chat_key,
        "chat_date": context.chat_date,
        "message_index": context.message_index,
        "role": context.role,
        "timestamp": context.timestamp,
    }


@dataclass
class PrefilterState:
    """Track prefilter progress for a single conversation."""

    turns_evaluated: int = 0
    has_match: bool = False
    skip_remaining: bool = False


def select_applicable_configs_for_context(
    context: MessageContext,
    configs: Sequence[AnnotationConfig],
    resume_seen_keys: Mapping[SeenKey, object] | Sequence[SeenKey] | set[SeenKey],
    min_positive: int,
    positive_counts: Mapping[str, int],
    target_keys: Optional[Set[SeenKey]] = None,
) -> List[AnnotationConfig]:
    """Return configs that should be applied to a single message context.

    Parameters
    ----------
    context:
        Target message context being classified.
    configs:
        Annotation configurations available for the run.
    resume_seen_keys:
        Set-like container containing seen (participant, path, chat, message,
        annotation) keys that should be skipped.
    min_positive:
        Minimum required positives per annotation when quota tracking is
        enabled.
    positive_counts:
        Mapping from annotation id to the number of existing positive records.
    target_keys:
        Optional set of specific (participant, path, chat, message, annotation)
        keys to include. When provided, only annotations whose keys appear in
        ``target_keys`` are eligible.

    Returns
    -------
    List[AnnotationConfig]
        Configurations that are still eligible for classification for this
        message.
    """

    applicable = [
        config
        for config in configs
        if config.allowed_roles is None or context.role in config.allowed_roles
    ]
    if target_keys is not None:
        target_set = set(target_keys)
        applicable = [
            cfg
            for cfg in applicable
            if (
                context.participant,
                str(context.source_path),
                context.chat_index,
                context.message_index,
                str(cfg.spec.get("id")),
            )
            in target_set
        ]
    if resume_seen_keys:
        seen_set = set(resume_seen_keys)
        applicable = [
            cfg
            for cfg in applicable
            if (
                context.participant,
                str(context.source_path),
                context.chat_index,
                context.message_index,
                str(cfg.spec.get("id")),
            )
            not in seen_set
        ]
    if min_positive > 0 and positive_counts:
        applicable = [
            cfg
            for cfg in applicable
            if positive_counts.get(str(cfg.spec.get("id")), 0) < min_positive
        ]
    return applicable


def ensure_output_handle_for_context(
    context: MessageContext,
    args: argparse.Namespace,
    configs: Sequence[AnnotationConfig],
    output_dir: Path,
    single_output_file: Optional[Path],
    resolved_output_name: str,
    non_default_arguments: Mapping[str, object],
    output_handles: dict[Path, TextIO],
    stack: ExitStack,
) -> Tuple[Optional[Path], Optional[TextIO]]:
    """Return an open output file handle for the given message context.

    Creates participant-specific directories and writes a meta record when
    needed. On error, logs a message and returns (None, None).
    """

    relative_directory = context.source_path.parent
    if relative_directory == Path("."):
        relative_directory = Path(context.participant)

    if single_output_file is not None:
        output_file_path = single_output_file
    else:
        participant_dir = ensure_participant_directory(output_dir, relative_directory)
        if participant_dir is None:
            return None, None
        key = (participant_dir, resolved_output_name)
        mapped_path = _SHARD_PATHS.get(key)
        if mapped_path is not None:
            output_file_path = mapped_path
        else:
            output_file_path = participant_dir / resolved_output_name
            _SHARD_PATHS[key] = output_file_path

    if output_file_path not in output_handles:
        if single_output_file is not None:
            out_file = stack.enter_context(output_file_path.open("a", encoding="utf-8"))
        else:
            if output_file_path.exists():
                out_file = stack.enter_context(
                    output_file_path.open("a", encoding="utf-8")
                )
            else:
                out_file = stack.enter_context(
                    output_file_path.open("w", encoding="utf-8")
                )
                meta_record = build_meta_record(
                    args=args,
                    configs=configs,
                    participant=context.participant,
                    non_default_arguments=non_default_arguments,
                )
                out_file.write(json.dumps(meta_record, ensure_ascii=False) + "\n")
        output_handles[output_file_path] = out_file

    return output_file_path, output_handles[output_file_path]


def _rotate_shard_if_needed(
    *,
    context: MessageContext,
    args: argparse.Namespace,
    configs: Sequence[AnnotationConfig],
    output_dir: Path,
    resolved_output_name: str,
    non_default_arguments: Mapping[str, object],
    output_handles: dict[Path, TextIO],
    stack: ExitStack,
    output_file_path: Path,
    out_file: TextIO,
) -> tuple[Optional[Path], Optional[TextIO]]:
    """Rotate to a new shard file when ``output_file_path`` exceeds the size cap.

    Returns the potentially updated ``(output_file_path, out_file)`` pair,
    or ``(None, None)`` when the rotation fails.
    """

    try:
        current_size = output_file_path.stat().st_size
    except OSError:
        current_size = 0
    if current_size < SHARD_MAX_BYTES:
        return output_file_path, out_file

    participant_dir = output_file_path.parent
    stem = infer_job_stem_from_filename(output_file_path.name)

    max_index = 0
    pattern = f"{stem}{PART_SUFFIX_MARKER}*.jsonl"
    for candidate in participant_dir.glob(pattern):
        cand_name = candidate.name
        cand_base = (
            cand_name[:-6] if cand_name.lower().endswith(".jsonl") else cand_name
        )
        idx_pos = cand_base.rfind(PART_SUFFIX_MARKER)
        if idx_pos == -1:
            continue
        suffix = cand_base[idx_pos + len(PART_SUFFIX_MARKER) :]
        if not suffix.isdigit():
            continue
        try:
            value = int(suffix)
        except ValueError:
            continue
        max_index = max(max_index, value)

    next_index = max_index + 1
    new_name = f"{stem}{PART_SUFFIX_MARKER}{next_index:04d}.jsonl"
    new_path = participant_dir / new_name

    try:
        out_file.close()
    except OSError:
        pass
    output_handles.pop(output_file_path, None)

    shard_key = (participant_dir, resolved_output_name)
    _SHARD_PATHS[shard_key] = new_path

    new_path, new_out_file = ensure_output_handle_for_context(
        context=context,
        args=args,
        configs=configs,
        output_dir=output_dir,
        single_output_file=None,
        resolved_output_name=resolved_output_name,
        non_default_arguments=non_default_arguments,
        output_handles=output_handles,
        stack=stack,
    )
    return new_path, new_out_file


def write_outcomes_for_context(
    context: MessageContext,
    outcomes: Sequence[ClassificationOutcome],
    *,
    args: argparse.Namespace,
    out_file: TextIO,
) -> None:
    """Write classification outcomes for a single context to a JSONL file.

    Handles quote validation, logging, and record construction for each
    annotation outcome targeting the provided message context.

    Parameters
    ----------
    context:
        Message context that all ``outcomes`` share.
    outcomes:
        Completed classification outcomes for the message.
    args:
        Parsed command-line arguments providing model and optional COT flags.
    out_file:
        Open text file handle to which JSONL records will be written.
    """

    for outcome in outcomes:
        annotation = outcome.task.annotation
        record_error = outcome.error
        unmatched_quotes = find_unmatched_quotes(outcome.matches, context.content)
        matches_for_record = filter_quotes_to_content(outcome.matches, context.content)
        unmatched_fragment = None
        if unmatched_quotes:
            unmatched_fragment = "Quoted text not found in transcript: " + ", ".join(
                f'"{quote}"' for quote in unmatched_quotes
            )
            if record_error:
                record_error = f"{record_error} | {unmatched_fragment}"
            else:
                record_error = unmatched_fragment
            logging.warning(
                ("Quoted text mismatch for %s #%s (message %s) [%s]: %s"),
                context.chat_key,
                context.chat_index,
                context.message_index,
                annotation["id"],
                unmatched_fragment,
            )
        if outcome.error:
            logging.error(
                "Failed to classify %s #%s (message %s) [%s]: %s",
                context.chat_key,
                context.chat_index,
                context.message_index,
                annotation["id"],
                outcome.error,
            )

        base_record = build_base_record(context)
        record = {
            **base_record,
            "annotation_id": annotation["id"],
            "annotation": annotation["name"],
            "preceding": context.preceding or [],
            "rationale": outcome.rationale,
            "reasoning_content": outcome.reasoning_content,
            "thinking_blocks": outcome.thinking_blocks,
            # Numeric strength/confidence score from 0 to 10 when available.
            "score": outcome.score,
            "matches": matches_for_record,
            "model": args.model,
            "error": record_error,
            "content": context.content,
        }
        try:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        except UnicodeEncodeError as err:
            logging.error(
                "Failed to write record for %s #%s (message %s) [%s]: %s",
                context.chat_key,
                context.chat_index,
                context.message_index,
                annotation["id"],
                err,
            )


def ensure_output_and_write_outcomes_for_context(
    *,
    context: MessageContext,
    outcomes: Sequence[ClassificationOutcome],
    args: argparse.Namespace,
    configs: Sequence[AnnotationConfig],
    output_dir: Path,
    single_output_file: Optional[Path],
    resolved_output_name: str,
    non_default_arguments: Mapping[str, object],
    output_handles: dict[Path, TextIO],
    stack: ExitStack,
) -> bool:
    """Open an output handle for ``context`` and write ``outcomes``.

    This is a small convenience wrapper that combines
    :func:`ensure_output_handle_for_context` and
    :func:`write_outcomes_for_context`. It returns ``True`` on success and
    ``False`` when the output handle could not be created.
    """

    output_file_path, out_file = ensure_output_handle_for_context(
        context=context,
        args=args,
        configs=configs,
        output_dir=output_dir,
        single_output_file=single_output_file,
        resolved_output_name=resolved_output_name,
        non_default_arguments=non_default_arguments,
        output_handles=output_handles,
        stack=stack,
    )
    if output_file_path is None or out_file is None:
        return False

    if single_output_file is None and SHARD_MAX_BYTES > 0:
        output_file_path, out_file = _rotate_shard_if_needed(
            context=context,
            args=args,
            configs=configs,
            output_dir=output_dir,
            resolved_output_name=resolved_output_name,
            non_default_arguments=non_default_arguments,
            output_handles=output_handles,
            stack=stack,
            output_file_path=output_file_path,
            out_file=out_file,
        )
        if output_file_path is None or out_file is None:
            return False

    write_outcomes_for_context(
        context,
        outcomes,
        args=args,
        out_file=out_file,
    )
    return True


def build_classification_tasks_for_context(
    context: MessageContext,
    configs: Sequence[AnnotationConfig],
    resume_seen_keys: Mapping[SeenKey, object] | Sequence[SeenKey] | set[SeenKey],
    min_positive: int,
    positive_counts: Mapping[str, int],
    *,
    args: argparse.Namespace,
    target_keys: Optional[Set[SeenKey]] = None,
) -> List[ClassificationTask]:
    """Return classification tasks for all applicable configs for a context.

    This helper applies :func:`select_applicable_configs_for_context` and
    constructs :class:`ClassificationTask` instances using the shared
    :func:`build_prompt` helper.
    """

    applicable_configs = select_applicable_configs_for_context(
        context,
        configs,
        resume_seen_keys,
        min_positive,
        positive_counts,
        target_keys,
    )
    if not applicable_configs:
        return []

    return [
        ClassificationTask(
            context=context,
            annotation=config.spec,
            prompt=build_prompt(
                config.spec,
                context.content,
                role=context.role,
                context_messages=context.preceding,
            ),
        )
        for config in applicable_configs
    ]


def iter_message_contexts(
    root: Path,
    participants: Optional[Sequence[str]],
    *,
    followlinks: bool,
    allowed_roles: Optional[set[str]],
    reverse_conversations: bool,
    preceding_count: int = 0,
) -> Iterator[MessageContext]:
    """Yield message contexts from chat exports beneath ``root``.

    Parameters
    ----------
    root:
        Root directory containing chat JSON exports.
    participants:
        Optional collection of participant identifiers to include.
    followlinks:
        When ``True``, symbolic links are followed while scanning for files.
    allowed_roles:
        Optional set of message roles to include; when ``None``, all roles are
        accepted.
    reverse_conversations:
        When ``True``, emit messages for each conversation in reverse order so
        the most recent turns are processed first.
    preceding_count:
        Maximum number of preceding messages to include as context.
    """

    participant_filter = normalize_participant_filter(participants)

    for file_path, chats in iter_loaded_chats(root, followlinks=followlinks):
        bucket, rel_path = resolve_bucket_and_rel_path(file_path, root)
        if not bucket:
            continue
        participant = normalize_participant_id(bucket.strip())
        if participant_filter and participant.lower() not in participant_filter:
            continue

        for chat_index, chat in enumerate(chats):
            for message_context_data in iter_chat_messages(
                chat.messages,
                allowed_roles=allowed_roles,
                reverse_conversations=reverse_conversations,
                preceding_count=preceding_count,
            ):
                yield MessageContext(
                    participant=participant,
                    source_path=rel_path,
                    chat_index=chat_index,
                    chat_key=chat.key,
                    chat_date=chat.date_label,
                    message_index=message_context_data.message_index,
                    role=message_context_data.role,
                    content=message_context_data.content,
                    timestamp=message_context_data.timestamp,
                    preceding=message_context_data.preceding,
                )


def prepare_message_iterator(
    args: argparse.Namespace,
    root: Path,
    configs: Sequence[AnnotationConfig],
    participants_filter: Optional[Sequence[str]],
    replay_keys: Optional[Sequence[ReplayKey]],
) -> tuple[
    Iterator[MessageContext],
    Optional[int],
    Optional[int],
    Optional[List[MessageContext]],
]:
    """Return a configured message iterator and sampling metadata.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed command-line arguments controlling filtering and sampling.
    root: Path
        Root directory containing chat exports.
    configs: Sequence[AnnotationConfig]
        Annotation configurations selected for the run.
    participants_filter: Optional[Sequence[str]]
        Optional participant identifiers used to limit iteration.
    replay_keys: Optional[Sequence[ReplayKey]]
        When provided, restricts iteration to messages present in the replay
        set.

    Returns
    -------
    tuple[Iterator[MessageContext], Optional[int], Optional[int], Optional[List[MessageContext]]]
        The message iterator, the maximum number of messages to process,
        the expected progress bar total, and an optional list of sampled
        contexts when randomization is enabled.
    """

    allowed_roles = derive_allowed_roles(configs)
    message_iter: Iterator[MessageContext] = iter_message_contexts(
        root,
        participants_filter,
        followlinks=args.follow_links,
        allowed_roles=allowed_roles,
        reverse_conversations=args.reverse_conversations,
        preceding_count=max(0, int(args.preceding_context or 0)),
    )

    if replay_keys is not None:
        replay_set = {
            (
                key[0],
                normalize_optional_string(key[1]),
                key[2],
                key[3],
            )
            for key in replay_keys
        }

        def _filter_replay(
            iterable: Iterator[MessageContext],
        ) -> Iterator[MessageContext]:
            for context in iterable:
                key: ReplayKey = (
                    context.participant,
                    str(context.source_path),
                    context.chat_index,
                    context.message_index,
                )
                normalized_key = (
                    key[0],
                    normalize_optional_string(key[1]),
                    key[2],
                    key[3],
                )
                if normalized_key in replay_set:
                    yield context

        message_iter = _filter_replay(message_iter)

    if replay_keys is not None:
        max_messages: Optional[int] = len(replay_keys)
        max_conversations: Optional[int] = None
    else:
        max_messages = args.max_messages if args.max_messages > 0 else None
        max_conversations = (
            args.max_conversations if args.max_conversations > 0 else None
        )
    if max_conversations is not None:
        message_iter = limit_conversations_by_participant(
            message_iter,
            max_conversations,
        )

    sampled_contexts: Optional[List[MessageContext]] = None
    progress_total = max_messages
    if replay_keys is not None:
        progress_total = max_messages
    elif args.randomize_conversations:
        rng = random.Random()
        sample_limit = max_messages if max_messages is not None else 0
        sampled_contexts = sample_conversations_within_participant(
            message_iter,
            sample_limit,
            rng,
        )
        rng.shuffle(sampled_contexts)
        message_iter = iter(sampled_contexts)
        if max_messages is None or len(sampled_contexts) < max_messages:
            max_messages = len(sampled_contexts)
        progress_total = len(sampled_contexts)
    elif args.randomize:
        if max_messages is None:
            logging.info(
                "Ignoring --randomize because --max-messages was not provided."
            )
        else:
            rng = random.Random()
            per_ppt_mode = getattr(args, "randomize_per_ppt", "proportional")
            sampled_contexts = sample_messages_by_participant(
                message_iter,
                max_messages,
                rng,
                equal=(per_ppt_mode == "equal"),
            )
            rng.shuffle(sampled_contexts)
            message_iter = iter(sampled_contexts)
            max_messages = len(sampled_contexts)
            progress_total = len(sampled_contexts)
    else:
        progress_total = max_messages

    return message_iter, max_messages, progress_total, sampled_contexts
