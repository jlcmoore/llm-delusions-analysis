"""Helpers for working with annotation job families.

This module centralises logic for iterating over classify_chats JSONL outputs
and constructing shared state used by multiple analysis scripts:

* ConversationKey: stable identifiers for conversation loci.
* collect_family_state: message-level and annotation-level aggregates shared
  by global frequency and participant-profile analyses.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import pandas as pd
from llm_delusions_annotations.annotation_ids import normalize_annotation_id
from llm_delusions_annotations.annotation_metadata import (
    load_annotation_metadata_or_exit_code,
)
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping
from llm_delusions_annotations.utils import is_positive_score

from annotation.io import (
    ParticipantMessageKey,
    build_participant_message_key,
    extract_conversation_key,
    get_annotation_id,
    iter_family_records_with_error_filter,
)
from llm_delusions_data.loaders import load_annotations_preprocessed_parquet
from utils.io import resolve_family_files
from utils.participants import is_excluded_participant

FAILED_QUOTE_PREFIX = "Quoted text not found in transcript"


@dataclass(frozen=True)
class ConversationKey:
    """Stable identifier for a conversation locus."""

    participant: str
    transcript_rel_path: str
    conversation_index: int


def role_from_record(record: Mapping[str, object]) -> Optional[str]:
    """Return the message role from a record when available."""

    role_raw = record.get("role") or record.get("message_role")
    if not role_raw:
        return None
    role_text = str(role_raw).strip()
    return role_text or None


def collect_family_state(
    family_files: Sequence[Path],
    *,
    metadata_by_id: Mapping[str, object],
    cutoffs_by_id: Mapping[str, int],
    failed_quote_prefix: str = FAILED_QUOTE_PREFIX,
) -> Tuple[
    Dict[ParticipantMessageKey, Tuple[str, ConversationKey]],
    Dict[ConversationKey, Counter[str]],
    Dict[Tuple[str, ParticipantMessageKey], bool],
]:
    """Return message, conversation, and positivity state for a job family.

    Parameters
    ----------
    family_files:
        JSONL files that belong to a single classification job.
    metadata_by_id:
        Mapping from annotation id to metadata objects; only ids present in
        this mapping are considered.
    cutoffs_by_id:
        Mapping from annotation id to per-annotation LLM score cutoff. Every
        annotation id present in ``metadata_by_id`` and encountered in the
        job family must have an entry in this mapping; missing cutoffs are
        treated as hard errors.
    failed_quote_prefix:
        Error-message prefix used to retain quote-mismatch errors while
        dropping other error records.

    Returns
    -------
    message_info:
        Mapping from participant message key to (role, conversation key).
    conversation_messages:
        Mapping from conversation key to a counter of message keys.
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to a boolean flag
        marking whether the pair was ever positive at the configured cutoff.
    """

    message_info: Dict[ParticipantMessageKey, Tuple[str, ConversationKey]] = {}
    conversation_messages: Dict[ConversationKey, Counter[str]] = defaultdict(Counter)
    annotation_message_positive: Dict[Tuple[str, ParticipantMessageKey], bool] = {}
    seen_pairs: set[Tuple[str, ParticipantMessageKey]] = set()

    records = iter_family_records_with_error_filter(
        family_files,
        allowed_error_prefixes=[failed_quote_prefix],
        drop_other_errors=True,
    )
    for record in records:
        annotation_id_raw = get_annotation_id(record)
        if annotation_id_raw is None:
            continue
        role = role_from_record(record)
        annotation_id = normalize_annotation_id(
            annotation_id_raw,
            role=role,
            strict_role=True,
        )
        if annotation_id is None:
            continue
        if annotation_id not in metadata_by_id:
            continue

        message_key = build_participant_message_key(record)
        if message_key is None:
            continue
        pair = (annotation_id, message_key)
        if pair in seen_pairs:
            raise RuntimeError(
                "Duplicate annotation/message pair encountered for "
                f"{annotation_id} at {message_key}",
            )
        seen_pairs.add(pair)

        conv_fields = extract_conversation_key(record)
        if conv_fields is None:
            continue
        participant, transcript_rel_path, conversation_index, _chat_key, _chat_date = (
            conv_fields
        )
        if is_excluded_participant(str(participant)):
            continue
        conv_key = ConversationKey(
            participant=participant,
            transcript_rel_path=transcript_rel_path,
            conversation_index=conversation_index,
        )

        if role is None:
            continue

        if message_key in message_info:
            existing_role, existing_conv = message_info[message_key]
            if existing_role != role or existing_conv != conv_key:
                raise RuntimeError(
                    "Inconsistent metadata for message key "
                    f"{message_key}: {existing_role!r}, {existing_conv!r} "
                    f"vs {role!r}, {conv_key!r}",
                )
        else:
            message_info[message_key] = (role, conv_key)

        conversation_messages[conv_key][str(message_key)] += 1

        if annotation_id not in cutoffs_by_id:
            raise RuntimeError(
                "No LLM score cutoff provided for annotation "
                f"{annotation_id!r}; please ensure the cutoffs CSV includes "
                "this id.",
            )
        cutoff = cutoffs_by_id[annotation_id]
        is_positive = is_positive_score(record, cutoff=cutoff)
        ap_key = (annotation_id, message_key)
        if ap_key in annotation_message_positive:
            annotation_message_positive[ap_key] = (
                annotation_message_positive[ap_key] or is_positive
            )
        else:
            annotation_message_positive[ap_key] = is_positive

    return message_info, conversation_messages, annotation_message_positive


FamilyState = Tuple[
    Dict[ParticipantMessageKey, Tuple[str, ConversationKey]],
    Dict[ConversationKey, Counter[str]],
    Dict[Tuple[str, ParticipantMessageKey], bool],
]


def compute_family_state_or_warn(
    family_files: Sequence[Path],
    *,
    metadata_by_id: Mapping[str, object],
    cutoffs_by_id: Mapping[str, int],
    failed_quote_prefix: str = FAILED_QUOTE_PREFIX,
    empty_message: str = "No usable messages discovered in the selected job family.",
) -> Optional[FamilyState]:
    """Return shared family state or warn and ``None`` if empty.

    This helper wraps :func:`collect_family_state` so that multiple analysis
    scripts can share the same empty-family handling without duplicating the
    pattern that checks for an empty ``message_info`` mapping and prints a
    warning.
    """

    (
        message_info,
        conversation_messages,
        annotation_message_positive,
    ) = collect_family_state(
        family_files,
        metadata_by_id=metadata_by_id,
        cutoffs_by_id=cutoffs_by_id,
        failed_quote_prefix=failed_quote_prefix,
    )
    if not message_info:
        print(empty_message)
        return None

    return message_info, conversation_messages, annotation_message_positive


def run_annotation_job(
    args: object,
    analysis_fn: Callable[
        [Sequence[Path], FamilyState, Mapping[str, object], Mapping[str, int], object],
        int,
    ],
) -> int:
    """Resolve a job family and metadata, then invoke ``analysis_fn``.

    This helper centralises the boilerplate used by multiple analysis scripts:

    * Resolve ``file`` and ``--outputs-root``.
    * Discover the job family via :func:`collect_family_files`.
    * Load annotation metadata and per-annotation cutoffs.

    The ``analysis_fn`` callback receives ``(family_files, metadata_by_id,
    cutoffs_by_id, args)`` and is responsible for computing the final result.
    """

    # The args namespace is expected to expose these attributes; callers should
    # ensure they are added via shared CLI helpers.
    reference_file = Path(getattr(args, "file"))
    outputs_root = Path(getattr(args, "outputs_root"))
    annotations_csv = Path(getattr(args, "annotations_csv"))
    llm_cutoffs_json = getattr(args, "llm_cutoffs_json")
    global_cutoff = getattr(args, "llm_score_cutoff", None)

    family_files, status = resolve_family_files(reference_file, outputs_root)
    if status != 0:
        return status
    if not family_files:
        return 0

    metadata_by_id, status = load_annotation_metadata_or_exit_code(annotations_csv)
    if status != 0:
        return status

    if llm_cutoffs_json is not None:
        cutoffs_by_id = load_cutoffs_mapping(llm_cutoffs_json)
        if not cutoffs_by_id and global_cutoff is None:
            print(
                "No usable LLM score cutoffs were loaded; please provide a "
                "cutoffs CSV via --llm-cutoffs-json.",
            )
            return 2
        if global_cutoff is not None:
            fallback_value = int(global_cutoff)
            for annotation_id in metadata_by_id:
                cutoffs_by_id.setdefault(annotation_id, fallback_value)
    else:
        if global_cutoff is None:
            print(
                "No LLM score cutoffs were provided; please set "
                "--llm-cutoffs-json or --llm-score-cutoff.",
            )
            return 2
        cutoff_value = int(global_cutoff)
        cutoffs_by_id = {aid: cutoff_value for aid in metadata_by_id}

    family_state = compute_family_state_or_warn(
        family_files,
        metadata_by_id=metadata_by_id,
        cutoffs_by_id=cutoffs_by_id,
    )
    if family_state is None:
        return 0

    return analysis_fn(
        family_files,
        family_state,
        metadata_by_id,
        cutoffs_by_id,
        args,
    )


def load_family_state_from_preprocessed_table(
    csv_path: Path,
    *,
    metadata_by_id: Mapping[str, object],
    cutoffs_by_id: Mapping[str, int],
    use_human_validation: bool = False,
) -> Optional[FamilyState]:
    """Return family state loaded from a preprocessed annotation table.

    This helper expects a wide per-message table produced by the
    ``analysis/preprocess_annotation_family.py`` script. The table must
    contain the columns:

    * ``participant``
    * ``source_path``
    * ``chat_index``
    * ``message_index``
    * ``role``
    * ``score__<annotation_id>`` for each annotation identifier

    Parameters
    ----------
    csv_path:
        Path to the preprocessed table file (CSV or Parquet).
    metadata_by_id:
        Mapping from annotation id to metadata objects; only ids present in
        this mapping are considered.
    cutoffs_by_id:
        Mapping from annotation id to per-annotation LLM score cutoff. Every
        annotation id present in ``metadata_by_id`` and encountered in the
        CSV must have an entry in this mapping; missing cutoffs are treated
        as hard errors.

    Returns
    -------
    Optional[FamilyState]
        Family state tuple when at least one usable message is discovered;
        otherwise ``None`` after printing a warning.
    """

    resolved = csv_path.expanduser().resolve()
    if not resolved.exists():
        print(f"Preprocessed table not found at {resolved}")
        return None

    # Load the Parquet preprocessed table into a DataFrame. Legacy CSV inputs
    # are no longer produced by the preprocessing pipeline.
    frame = load_annotations_preprocessed_parquet(resolved)

    if frame.empty:
        print(
            "No usable messages discovered in the selected preprocessed "
            "annotation table.",
        )
        return None

    message_info: Dict[ParticipantMessageKey, Tuple[str, ConversationKey]] = {}
    conversation_messages: Dict[ConversationKey, Counter[str]] = defaultdict(Counter)
    annotation_message_positive: Dict[Tuple[str, ParticipantMessageKey], bool] = {}

    metadata_ids = set(metadata_by_id.keys())
    score_columns = [name for name in frame.columns if name.startswith("score__")]

    for row in frame.to_dict(orient="records"):
        participant = str((row.get("participant") or "")).strip()
        source_path = str((row.get("source_path") or "")).strip()
        chat_index_raw = row.get("chat_index")
        message_index_raw = row.get("message_index")
        role = str(row.get("role") or "").strip()
        if not participant or not source_path or role == "":
            continue
        if is_excluded_participant(participant):
            continue
        try:
            chat_index = int(chat_index_raw)  # type: ignore[arg-type]
            message_index = int(message_index_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        conv_key = ConversationKey(
            participant=participant,
            transcript_rel_path=source_path,
            conversation_index=chat_index,
        )
        message_key: ParticipantMessageKey = (
            participant,
            source_path,
            chat_index,
            message_index,
        )

        if message_key in message_info:
            existing_role, existing_conv = message_info[message_key]
            if existing_role != role or existing_conv != conv_key:
                raise RuntimeError(
                    "Inconsistent metadata for message key "
                    f"{message_key}: {existing_role!r}, {existing_conv!r} "
                    f"vs {role!r}, {conv_key!r}",
                )
        else:
            message_info[message_key] = (role, conv_key)

        conversation_messages[conv_key][str(message_key)] += 1

        for column in score_columns:
            annotation_id = column[len("score__") :]
            if annotation_id not in metadata_ids:
                continue
            score_raw = row.get(column)
            if annotation_id not in cutoffs_by_id:
                # Mirror JSONL semantics: only raise when a record for an
                # annotation id without a cutoff is actually present.
                if score_raw not in (None, "", " ") and not pd.isna(score_raw):
                    raise RuntimeError(
                        "No LLM score cutoff provided for annotation "
                        f"{annotation_id!r}; please ensure the cutoffs "
                        "JSON includes this id.",
                    )
                continue
            if score_raw is None or score_raw == "" or pd.isna(score_raw):
                continue
            try:
                score_value = float(score_raw)
            except (TypeError, ValueError):
                continue
            cutoff = cutoffs_by_id[annotation_id]
            is_positive = int(score_value) >= int(cutoff)

            if use_human_validation:
                human_col = f"human_validated__{annotation_id}"
                if human_col in row:
                    val = row.get(human_col)
                    if not pd.isna(val) and val is not None:
                        is_positive = bool(val)

            key = (annotation_id, message_key)
            if key in annotation_message_positive:
                annotation_message_positive[key] = (
                    annotation_message_positive[key] or is_positive
                )
            else:
                annotation_message_positive[key] = is_positive

    return message_info, conversation_messages, annotation_message_positive


def run_preprocessed_annotation_job(
    args: object,
    analysis_fn: Callable[
        [Sequence[Path], FamilyState, Mapping[str, object], Mapping[str, int], object],
        int,
    ],
) -> int:
    """Load preprocessed family state and invoke ``analysis_fn``.

    This helper mirrors :func:`run_annotation_job` but operates on a
    preprocessed per-message CSV instead of raw JSONL files. It:

    * Loads annotation metadata and per-annotation LLM score cutoffs.
    * Loads family state from ``args.input_csv`` via
      :func:`load_family_state_from_preprocessed_table`.
    * Invokes ``analysis_fn`` with an empty ``family_files`` sequence and
      the constructed state.

    The ``args`` namespace is expected to expose ``input_csv``,
    ``annotations_csv``, and ``llm_cutoffs_json`` attributes.
    """

    input_csv = Path(getattr(args, "input_csv"))
    annotations_csv = Path(getattr(args, "annotations_csv"))
    llm_cutoffs_json = getattr(args, "llm_cutoffs_json", None)
    global_cutoff = getattr(args, "llm_score_cutoff", None)

    metadata_by_id, status = load_annotation_metadata_or_exit_code(annotations_csv)
    if status != 0:
        return status

    if llm_cutoffs_json is not None:
        cutoffs_by_id = load_cutoffs_mapping(llm_cutoffs_json)
        if global_cutoff is not None:
            fallback_value = int(global_cutoff)
            for annotation_id in metadata_by_id:
                cutoffs_by_id.setdefault(annotation_id, fallback_value)
    else:
        cutoff_value = int(global_cutoff)
        cutoffs_by_id = {aid: cutoff_value for aid in metadata_by_id}

    family_state = load_family_state_from_preprocessed_table(
        input_csv,
        metadata_by_id=metadata_by_id,
        cutoffs_by_id=cutoffs_by_id,
        use_human_validation=getattr(args, "use_human_validation", False),
    )
    if family_state is None:
        return 0

    return analysis_fn(
        [],
        family_state,
        metadata_by_id,
        cutoffs_by_id,
        args,
    )


__all__ = [
    "ConversationKey",
    "FAILED_QUOTE_PREFIX",
    "role_from_record",
    "collect_family_state",
    "FamilyState",
    "compute_family_state_or_warn",
    "run_annotation_job",
    "load_family_state_from_preprocessed_table",
    "run_preprocessed_annotation_job",
]
