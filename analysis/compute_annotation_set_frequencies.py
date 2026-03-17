"""Compute frequencies for named sets of annotations.

This script aggregates message-level and participant-level frequencies for
user-defined sets of annotation identifiers from a preprocessed per-message
annotation table. It is designed for questions of the form:

* "What percentage of unique messages contain any of these codes?"
* "What is the corresponding rate averaged over participants, with 95% CIs?"

The script expects a Parquet table produced by
``analysis/preprocess_annotation_family.py`` together with annotation
metadata and LLM score cutoffs. For each named set of annotation ids, it
computes:

* The proportion of scoped messages that are positive for at least one
  annotation in the set, with a Beta(1, 1) posterior mean and 95 percent
  normal-approximate interval.
* The mean per-participant positive rate over scoped messages together with
  a normal-approximate 95 percent confidence interval across participants.

Default usage groups annotations by their ``category`` metadata field so
that each output row summarises a single category across all of its member
annotations. This avoids the need to define custom sets on the command
line and is suitable for most analyses.

When you need per-model summaries, pass ``--model MODEL_ID`` one or more
times together with ``--transcripts-parquet``. The script will join the
annotation rows to the transcripts table via the shared message keys and
restrict the analysis to messages whose ``model_id`` matches each provided
value, emitting one row per (set, model) combination.

For advanced use, annotation sets can still be provided via repeated
``--set`` arguments:

.. code-block:: bash

    python analysis/compute_annotation_set_frequencies.py \\
        annotations/all_annotations__preprocessed.parquet \\
        --llm-cutoffs-json analysis/agreement/validation/cutoffs.csv \\
        --set violent_user=user-violent-intent,user-suicidal-intent \\
        --set assistant_validate=assistant-validates-violent-feelings,\\
assistant-validates-self-harm-feelings

Each ``--set`` takes the form ``NAME:id1,id2,...`` where ``NAME`` is an
arbitrary identifier used in the output table and ``id*`` are annotation
identifiers from ``annotations.csv``. When no ``--set`` arguments are
given, the script falls back to category-based sets.

By default, message scopes follow the per-annotation ``scope`` metadata
field and include both user and assistant messages where the annotations
apply. To restrict a custom set to a single role, add matching
``--set-role`` arguments. Valid role filters are ``user`` and
``assistant``; an omitted role filter uses the union of per-annotation
scopes.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
)

from analysis.lib.stats.beta_utils import beta_normal_ci
from analysis.lib.utils.formatting import round3
from annotation.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from annotation.io import ParticipantMessageKey
from llm_delusions_data.loaders import load_transcripts_parquet
from utils.cli import (
    add_annotation_metadata_arguments,
    add_output_path_argument,
    add_preprocessed_input_csv_argument,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the annotation-set frequency script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute message-level and participant-level frequencies for "
            "named sets of annotations from a preprocessed per-message "
            "annotation table."
        )
    )
    add_preprocessed_input_csv_argument(parser)
    add_annotation_metadata_arguments(parser)
    parser.add_argument(
        "--transcripts-parquet",
        type=Path,
        default=None,
        help=(
            "Optional transcripts parquet path (for example, "
            "transcripts_data/transcripts.parquet). Required when filtering "
            "by model ids."
        ),
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        metavar="MODEL_ID",
        help=(
            "Restrict analysis to messages associated with this model id "
            "(as recorded in transcripts.parquet). May be provided multiple "
            "times to compute separate rows for each model."
        ),
    )
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        metavar="NAME:id1,id2,...",
        help=(
            "Named annotation set definition. May be provided multiple times. "
            "Each value must take the form NAME:id1,id2,... where NAME is an "
            "identifier for the set and id* are annotation ids."
        ),
    )
    parser.add_argument(
        "--set-role",
        dest="set_roles",
        action="append",
        metavar="NAME:role",
        help=(
            "Optional role filter applied to a named set. The role must be "
            "one of 'user' or 'assistant'. When omitted for a set, both "
            "roles are included wherever the annotations are in scope."
        ),
    )
    add_output_path_argument(
        parser,
        default_path=Path("analysis/data/annotation_set_frequencies.csv"),
        help_text="Output CSV path for the annotation-set frequency table.",
    )
    return parser


def _parse_set_definitions(
    raw_sets: Optional[Sequence[str]],
) -> Dict[str, Sequence[str]]:
    """Return a mapping from set name to annotation ids.

    Parameters
    ----------
    raw_sets:
        Raw ``--set`` argument values of the form ``NAME:id1,id2,...``.

    Returns
    -------
    Dict[str, Sequence[str]]
        Mapping from set identifier to a non-empty sequence of annotation ids.

    Notes
    -----
    When ``raw_sets`` is empty or ``None``, an empty mapping is returned so
    that callers can fall back to alternative grouping strategies (for
    example, grouping by annotation category).
    """

    sets_by_name: Dict[str, Sequence[str]] = {}
    if not raw_sets:
        return sets_by_name
    for value in raw_sets:
        parts = value.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --set value {value!r}; expected NAME:id1,id2,...",
            )
        name_raw, ids_raw = parts
        name = name_raw.strip()
        if not name:
            raise ValueError(
                f"Invalid --set value {value!r}; set name must be non-empty"
            )
        if name in sets_by_name:
            raise ValueError(f"Duplicate set name {name!r} in --set arguments")
        id_parts = [part.strip() for part in ids_raw.split(",") if part.strip()]
        if not id_parts:
            raise ValueError(
                f"Invalid --set value {value!r}; at least one annotation id is required"
            )
        sets_by_name[name] = tuple(id_parts)
    return sets_by_name


def _parse_set_roles(
    raw_roles: Optional[Sequence[str]],
    sets_by_name: Mapping[str, Sequence[str]],
) -> Dict[str, Optional[str]]:
    """Return a mapping from set name to an optional role filter.

    Parameters
    ----------
    raw_roles:
        Raw ``--set-role`` argument values of the form ``NAME:role``.
    sets_by_name:
        Parsed set definitions keyed by set name.

    Returns
    -------
    Dict[str, Optional[str]]
        Mapping from set name to role filter string (``\"user\"`` or
        ``\"assistant\"``) or ``None`` when no filter is applied.

    Raises
    ------
    ValueError
        If a role mapping is malformed, references an unknown set name,
        or uses an unsupported role token.
    """

    roles_by_name: Dict[str, Optional[str]] = {}
    if not raw_roles:
        return roles_by_name

    for value in raw_roles:
        parts = value.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --set-role value {value!r}; expected NAME:role where "
                "role is 'user' or 'assistant'",
            )
        name_raw, role_raw = parts
        name = name_raw.strip()
        if not name:
            raise ValueError(
                f"Invalid --set-role value {value!r}; set name must be non-empty"
            )
        if name not in sets_by_name:
            raise ValueError(
                f"--set-role refers to unknown set {name!r}; define the set "
                "first with --set NAME:id1,id2,...",
            )
        role_normalized = role_raw.strip().lower()
        if role_normalized in {"", "all", "any", "both", "auto"}:
            roles_by_name[name] = None
            continue
        if role_normalized not in {"user", "assistant"}:
            raise ValueError(
                f"Invalid role {role_raw!r} for set {name!r}; expected one of "
                "'user', 'assistant', 'both', or 'all'.",
            )
        roles_by_name[name] = role_normalized
    return roles_by_name


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the annotation-set frequency script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, ``sys.argv``
        semantics are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with additional ``set_definitions`` and
        ``set_roles_by_name`` attributes attached.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        set_definitions = _parse_set_definitions(args.sets)
        set_roles_by_name = _parse_set_roles(args.set_roles, set_definitions)
    except ValueError as exc:
        parser.error(str(exc))

    args.set_definitions = set_definitions
    args.set_roles_by_name = set_roles_by_name
    if args.models:
        models = [model.strip() for model in args.models if model and model.strip()]
        if not models:
            parser.error("--model values must be non-empty strings")
        if args.transcripts_parquet is None:
            parser.error("--transcripts-parquet is required when using --model")
        args.models = models
    else:
        args.models = []
    return args


def _compute_message_and_participant_counts_for_set(
    set_name: str,
    annotation_ids: Sequence[str],
    role_filter: Optional[str],
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[Tuple[str, ParticipantMessageKey], bool],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
) -> Optional[Dict[str, object]]:
    """Return count aggregates for a single annotation set.

    This helper computes:

    * The number of scoped messages and positive messages where at least one
      annotation in the set is positive.
    * Per-participant scoped and positive message counts.

    Parameters
    ----------
    set_name:
        Identifier of the annotation set.
    annotation_ids:
        Annotation identifiers included in the set.
    role_filter:
        Optional role restriction (``\"user\"`` or ``\"assistant\"``) applied
        on top of per-annotation scopes. When ``None``, both roles are
        included wherever at least one annotation is in scope.
    message_info:
        Mapping from participant message key to (role, conversation key).
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to a boolean flag
        indicating whether the annotation is positive on that message.
    metadata_by_id:
        Annotation metadata keyed by annotation id.

    Returns
    -------
    Optional[Dict[str, object]]
        Dictionary of count aggregates keyed by field name, or ``None`` when
        no messages are in scope for the set.
    """

    # Restrict to annotations that are present in the metadata table.
    active_ids = [aid for aid in annotation_ids if aid in metadata_by_id]
    if not active_ids:
        return None

    scoped_message_keys: set[ParticipantMessageKey] = set()
    positive_message_keys: set[ParticipantMessageKey] = set()

    per_participant_scoped: Dict[str, int] = {}
    per_participant_positive: Dict[str, int] = {}

    for message_key, (role, _conv_key) in message_info.items():
        if role_filter is not None and role != role_filter:
            continue

        participant = message_key[0]

        # Determine whether this message is within the union of annotation
        # scopes for the set and whether any annotation is positive.
        in_scope = False
        is_positive_any = False
        for annotation_id in active_ids:
            meta = metadata_by_id.get(annotation_id)
            if meta is None:
                continue
            if not is_role_in_scope(role, meta.scope):
                continue
            in_scope = True
            if annotation_message_positive.get((annotation_id, message_key), False):
                is_positive_any = True
        if not in_scope:
            continue

        scoped_message_keys.add(message_key)
        per_participant_scoped[participant] = (
            per_participant_scoped.get(participant, 0) + 1
        )
        if is_positive_any:
            positive_message_keys.add(message_key)
            per_participant_positive[participant] = (
                per_participant_positive.get(participant, 0) + 1
            )

    n_messages_scoped = len(scoped_message_keys)
    n_messages_positive = len(positive_message_keys)
    if n_messages_scoped <= 0:
        return None

    return {
        "set_id": set_name,
        "annotation_ids": ",".join(active_ids),
        "role_filter": role_filter or "",
        "n_messages_scoped": n_messages_scoped,
        "n_messages_positive": n_messages_positive,
        "per_participant_scoped": per_participant_scoped,
        "per_participant_positive": per_participant_positive,
    }


def _compute_participant_rate_ci(
    per_participant_scoped: Mapping[str, int],
    per_participant_positive: Mapping[str, int],
    *,
    z_value: float = 1.96,
) -> Tuple[float, float, float, int]:
    """Return mean per-participant rate and an approximate CI.

    Parameters
    ----------
    per_participant_scoped:
        Mapping from participant id to the number of scoped messages for the
        set.
    per_participant_positive:
        Mapping from participant id to the number of positive messages for
        the set.
    z_value:
        Normal multiplier for the interval; the default 1.96 corresponds to
        an approximate 95 percent interval.

    Returns
    -------
    Tuple[float, float, float, int]
        (mean_rate, low, high, n_participants_scoped) where the bounds are
        clipped to [0, 1]. When no participants have scoped messages, all
        returned values are zero.
    """

    rates: list[float] = []
    for participant, denom in per_participant_scoped.items():
        numer = per_participant_positive.get(participant, 0)
        if denom <= 0:
            continue
        rates.append(float(numer) / float(denom))

    n_participants = len(rates)
    if n_participants == 0:
        return 0.0, 0.0, 0.0, 0

    mean_rate = sum(rates) / float(n_participants)
    if n_participants == 1 or z_value <= 0.0:
        return mean_rate, mean_rate, mean_rate, n_participants

    variance = 0.0
    for rate in rates:
        diff = rate - mean_rate
        variance += diff * diff
    variance /= float(max(1, n_participants - 1))
    if variance <= 0.0:
        return mean_rate, mean_rate, mean_rate, n_participants

    sd = math.sqrt(variance)
    se = sd / math.sqrt(float(n_participants))
    delta = z_value * se
    low = max(0.0, mean_rate - delta)
    high = min(1.0, mean_rate + delta)
    return mean_rate, low, high, n_participants


def _summarise_set_counts(
    counts: Mapping[str, object],
    *,
    model_id: str = "",
) -> Dict[str, object]:
    """Return a flat summary row for an annotation set.

    Parameters
    ----------
    counts:
        Dictionary returned by
        :func:`_compute_message_and_participant_counts_for_set`.

    Returns
    -------
    Dict[str, object]
        Flat dictionary suitable for CSV output with rate and interval
        fields rounded to three decimal places.
    """

    set_id = str(counts["set_id"])
    annotation_ids = str(counts["annotation_ids"])
    role_filter = str(counts.get("role_filter") or "")
    n_messages_scoped = int(counts["n_messages_scoped"])
    n_messages_positive = int(counts["n_messages_positive"])
    per_participant_scoped = counts["per_participant_scoped"]
    per_participant_positive = counts["per_participant_positive"]

    assert isinstance(per_participant_scoped, dict)
    assert isinstance(per_participant_positive, dict)

    # Message-level rate and Beta posterior interval with a uniform prior.
    alpha = float(n_messages_positive) + 1.0
    beta_count = float(n_messages_scoped - n_messages_positive) + 1.0
    mean_messages, low_messages, high_messages = beta_normal_ci(alpha, beta_count)

    ppt_mean, ppt_low, ppt_high, n_participants_scoped = _compute_participant_rate_ci(
        per_participant_scoped,
        per_participant_positive,
    )

    return {
        "set_id": set_id,
        "model_id": model_id,
        "annotation_ids": annotation_ids,
        "role_filter": role_filter,
        "n_messages_scoped": n_messages_scoped,
        "n_messages_positive": n_messages_positive,
        "rate_messages": round3(mean_messages),
        "rate_messages_ci_low": round3(low_messages),
        "rate_messages_ci_high": round3(high_messages),
        "n_participants_scoped": n_participants_scoped,
        "ppt_rate_mean": round3(ppt_mean),
        "ppt_rate_ci_low": round3(ppt_low),
        "ppt_rate_ci_high": round3(ppt_high),
    }


def _write_rows(output_csv: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write summary rows to ``output_csv``."""

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "set_id",
        "model_id",
        "annotation_ids",
        "role_filter",
        "n_messages_scoped",
        "n_messages_positive",
        "rate_messages",
        "rate_messages_ci_low",
        "rate_messages_ci_high",
        "n_participants_scoped",
        "ppt_rate_mean",
        "ppt_rate_ci_low",
        "ppt_rate_ci_high",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_message_model_ids(
    transcripts_parquet: Path,
) -> Dict[ParticipantMessageKey, str]:
    """Return message-key -> model_id mapping from transcripts parquet."""

    columns = [
        "participant",
        "source_path",
        "chat_index",
        "message_index",
        "model_id",
    ]
    frame = load_transcripts_parquet(transcripts_parquet, columns=columns)
    if frame.empty or "model_id" not in frame.columns:
        return {}

    mapping: Dict[ParticipantMessageKey, str] = {}
    for row in frame.itertuples(index=False):
        model_id = getattr(row, "model_id", None)
        if model_id is None:
            continue
        model_text = str(model_id).strip()
        if not model_text:
            continue
        try:
            participant = str(getattr(row, "participant"))
            source_path = str(getattr(row, "source_path"))
            chat_index = int(getattr(row, "chat_index"))
            message_index = int(getattr(row, "message_index"))
        except (TypeError, ValueError):
            continue
        key: ParticipantMessageKey = (
            participant,
            source_path,
            chat_index,
            message_index,
        )
        mapping[key] = model_text
    return mapping


def _filter_family_state_for_model(
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[Tuple[str, ParticipantMessageKey], bool],
    *,
    model_id: str,
    message_model_ids: Mapping[ParticipantMessageKey, str],
) -> Tuple[
    Dict[ParticipantMessageKey, Tuple[str, ConversationKey]],
    Dict[Tuple[str, ParticipantMessageKey], bool],
]:
    """Return family state restricted to a single model id."""

    allowed_keys = {
        key for key, value in message_model_ids.items() if value == model_id
    }
    filtered_message_info = {
        key: value for key, value in message_info.items() if key in allowed_keys
    }
    filtered_annotation_positive = {
        (annotation_id, key): value
        for (annotation_id, key), value in annotation_message_positive.items()
        if key in allowed_keys
    }
    return filtered_message_info, filtered_annotation_positive


def _run_annotation_set_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    _cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with :func:`run_preprocessed_annotation_job`."""

    message_info, _conversation_messages, annotation_message_positive = family_state

    # Apply shared analysis filters so that excluded annotations (for example,
    # test-category or synthetic ids) do not participate in the summaries.
    metadata_by_id_filtered = filter_analysis_metadata(metadata_by_id)

    rows: list[Dict[str, object]] = []
    message_model_ids: Dict[ParticipantMessageKey, str] = {}
    models: Sequence[str] = args.models or []
    if models:
        message_model_ids = _load_message_model_ids(args.transcripts_parquet)

    def _append_rows(
        *,
        message_info_subset: Mapping[
            ParticipantMessageKey, Tuple[str, ConversationKey]
        ],
        annotation_positive_subset: Mapping[Tuple[str, ParticipantMessageKey], bool],
        model_id: str,
    ) -> None:
        if args.set_definitions:
            # Custom sets supplied on the CLI: respect the provided definitions
            # and any matching role filters.
            for set_name, annotation_ids in args.set_definitions.items():
                role_filter = args.set_roles_by_name.get(set_name)
                counts = _compute_message_and_participant_counts_for_set(
                    set_name,
                    annotation_ids,
                    role_filter,
                    message_info_subset,
                    annotation_positive_subset,
                    metadata_by_id=metadata_by_id_filtered,
                )
                if counts is None:
                    continue
                rows.append(_summarise_set_counts(counts, model_id=model_id))
        else:
            # Default mode: group annotations by category and compute one row per
            # category. Empty or missing category values are ignored.
            sets_by_category: Dict[str, list[str]] = {}
            for annotation_id, meta in metadata_by_id_filtered.items():
                category = meta.category.strip()
                if not category:
                    continue
                sets_by_category.setdefault(category, []).append(annotation_id)

            for set_name, annotation_ids in sorted(sets_by_category.items()):
                counts = _compute_message_and_participant_counts_for_set(
                    set_name,
                    annotation_ids,
                    None,
                    message_info_subset,
                    annotation_positive_subset,
                    metadata_by_id=metadata_by_id_filtered,
                )
                if counts is None:
                    continue
                rows.append(_summarise_set_counts(counts, model_id=model_id))

    if models:
        _append_rows(
            message_info_subset=message_info,
            annotation_positive_subset=annotation_message_positive,
            model_id="overall",
        )
        for model_id in models:
            filtered_message_info, filtered_annotation_positive = (
                _filter_family_state_for_model(
                    message_info,
                    annotation_message_positive,
                    model_id=model_id,
                    message_model_ids=message_model_ids,
                )
            )
            _append_rows(
                message_info_subset=filtered_message_info,
                annotation_positive_subset=filtered_annotation_positive,
                model_id=model_id,
            )
    else:
        _append_rows(
            message_info_subset=message_info,
            annotation_positive_subset=annotation_message_positive,
            model_id="",
        )

    if not rows:
        print("No annotation sets produced usable counts; nothing to write.")
        return 0

    _write_rows(args.output, rows)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for computing annotation-set frequencies."""

    args = parse_args(argv)
    return run_preprocessed_annotation_job(args, _run_annotation_set_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
