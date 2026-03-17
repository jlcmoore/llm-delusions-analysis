"""Compute global and marginal annotation frequencies from LLM outputs.

This module aggregates message-level frequencies for all annotations in a
single classification job family. It reads JSONL outputs produced by
``scripts/annotation/classify_chats.py``, applies per-annotation LLM score
cutoffs, and summarises:

* Message-pooled positive rates by role.
* Participant-averaged positive rates.
* An aggregate \"any harmful\" annotation built from metadata.

The script expects:

* A reference JSONL file within the target job family (``--file``).
* An outputs root containing sibling JSONL files (``--outputs-root``).
* A cutoffs CSV file mapping annotation ids to score cutoffs
  (``--llm-cutoffs-json``).
* The annotation metadata table (``--annotations-csv``).

Results are written as a CSV table designed to map cleanly into LaTeX.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
)

from analysis.lib.plotting.labels import filter_annotation_ids_for_display
from analysis.lib.utils.formatting import round3
from annotation.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from annotation.io import ParticipantMessageKey
from utils.cli import (
    add_annotation_metadata_arguments,
    add_output_path_argument,
    add_preprocessed_input_csv_argument,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the frequency script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute global and marginal annotation frequencies from a "
            "preprocessed per-message annotation table."
        )
    )
    add_preprocessed_input_csv_argument(parser)
    add_annotation_metadata_arguments(parser)
    add_output_path_argument(
        parser,
        default_path=Path("analysis/data/annotation_frequencies.csv"),
        help_text="Output CSV path for the annotation frequency table.",
    )
    parser.add_argument(
        "--k-threshold",
        type=int,
        default=5,
        help=(
            "Minimum number of positive messages per participant required to "
            "count as positive when computing participant-level positive rates."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the frequency script.

    Parameters
    ----------
    argv:
        Optional sequence of command-line arguments. When omitted, ``sys.argv``
        semantics are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments populated with defaults.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    return args


def _safe_fraction(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` with a zero-safe denominator."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _aggregate_frequencies(
    message_info: Mapping[ParticipantMessageKey, tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    k_threshold: int,
) -> list[dict]:
    """Return a list of frequency summary rows."""

    # Preserve the original order from the annotations CSV by iterating over
    # ids in insertion order rather than sorting alphabetically.
    annotation_ids = list(metadata_by_id.keys())

    per_annotation_message_keys: Dict[str, set[ParticipantMessageKey]] = {
        aid: set() for aid in annotation_ids
    }
    for (annotation_id, message_key), _flag in annotation_message_positive.items():
        if annotation_id in per_annotation_message_keys:
            per_annotation_message_keys[annotation_id].add(message_key)

    participant_roles: Dict[str, Counter[str]] = defaultdict(Counter)
    for message_key, (role, _conv_key) in message_info.items():
        participant = message_key[0]
        participant_roles[participant][role] += 1

    rows: list[dict] = []

    for annotation_id in annotation_ids:
        meta = metadata_by_id[annotation_id]
        scope = meta.scope
        message_keys = per_annotation_message_keys.get(annotation_id, set())

        if not message_keys:
            # Skip annotations that have no annotated messages in this job
            # family so that synthetic or unused ids (such as test labels)
            # do not appear in the summary table.
            continue

        n_messages_all = 0
        n_messages_user = 0
        n_messages_assistant = 0
        n_positive_all = 0
        n_positive_user = 0
        n_positive_assistant = 0

        per_participant_numerators: Dict[str, int] = defaultdict(int)
        per_participant_denominators: Dict[str, int] = defaultdict(int)
        per_participant_numerators_user: Dict[str, int] = defaultdict(int)
        per_participant_denominators_user: Dict[str, int] = defaultdict(int)
        per_participant_numerators_assistant: Dict[str, int] = defaultdict(int)
        per_participant_denominators_assistant: Dict[str, int] = defaultdict(int)

        for message_key in message_keys:
            role, _conv_key = message_info[message_key]
            participant = message_key[0]
            is_positive = annotation_message_positive.get(
                (annotation_id, message_key), False
            )

            n_messages_all += 1
            if role == "user":
                n_messages_user += 1
            elif role == "assistant":
                n_messages_assistant += 1

            if is_positive:
                n_positive_all += 1
                if role == "user":
                    n_positive_user += 1
                elif role == "assistant":
                    n_positive_assistant += 1

            if is_role_in_scope(role, scope):
                per_participant_denominators[participant] += 1
                if is_positive:
                    per_participant_numerators[participant] += 1
                if role == "user":
                    per_participant_denominators_user[participant] += 1
                    if is_positive:
                        per_participant_numerators_user[participant] += 1
                elif role == "assistant":
                    per_participant_denominators_assistant[participant] += 1
                    if is_positive:
                        per_participant_numerators_assistant[participant] += 1

        rate_all: float | str
        rate_user: float | str
        rate_assistant: float | str

        if n_messages_all > 0:
            rate_all = round3(_safe_fraction(n_positive_all, n_messages_all))
        else:
            rate_all = ""

        if n_messages_user > 0:
            rate_user = round3(_safe_fraction(n_positive_user, n_messages_user))
        else:
            rate_user = ""

        if n_messages_assistant > 0:
            rate_assistant = round3(
                _safe_fraction(n_positive_assistant, n_messages_assistant)
            )
        else:
            rate_assistant = ""

        participant_rates: list[float] = []
        n_participants_with_positive = 0
        for participant, denom in per_participant_denominators.items():
            numer = per_participant_numerators.get(participant, 0)
            if denom <= 0:
                continue
            rate_p = _safe_fraction(numer, denom)
            participant_rates.append(rate_p)
            if numer >= k_threshold:
                n_participants_with_positive += 1

        rate_participants_positive: float | str
        rate_participants_mean: float | str
        rate_participants_std: float | str
        n_participants_scoped: int
        rate_participants_mean_user: float | str
        rate_participants_std_user: float | str
        n_participants_scoped_user: int
        rate_participants_mean_assistant: float | str
        rate_participants_std_assistant: float | str
        n_participants_scoped_assistant: int

        if per_participant_denominators:
            n_participants_scoped = len(participant_rates)
            denom_participants = float(len(per_participant_denominators))
            rate_participants_positive = round3(
                _safe_fraction(n_participants_with_positive, int(denom_participants))
            )
            if participant_rates:
                mean_rate = sum(participant_rates) / float(len(participant_rates))
                mean_rate = float(round3(mean_rate))
                variance = 0.0
                for rate_p in participant_rates:
                    diff = rate_p - mean_rate
                    variance += diff * diff
                variance /= float(len(participant_rates))
                std_dev = variance**0.5
                rate_participants_mean = mean_rate
                rate_participants_std = round3(std_dev)
            else:
                rate_participants_mean = ""
                rate_participants_std = ""
        else:
            n_participants_scoped = 0
            rate_participants_positive = ""
            rate_participants_mean = ""
            rate_participants_std = ""

        # Compute role-specific participant-normalised rates so that
        # user-scoped and assistant-scoped analyses can treat dual-scoped
        # annotations using only messages from the corresponding role.
        participant_rates_user: list[float] = []
        if per_participant_denominators_user:
            for participant, denom in per_participant_denominators_user.items():
                numer = per_participant_numerators_user.get(participant, 0)
                if denom <= 0:
                    continue
                participant_rates_user.append(_safe_fraction(numer, denom))
            if participant_rates_user:
                n_participants_scoped_user = len(participant_rates_user)
                mean_rate_user = sum(participant_rates_user) / float(
                    len(participant_rates_user)
                )
                mean_rate_user = float(round3(mean_rate_user))
                variance_user = 0.0
                for rate_p in participant_rates_user:
                    diff_user = rate_p - mean_rate_user
                    variance_user += diff_user * diff_user
                variance_user /= float(len(participant_rates_user))
                std_dev_user = variance_user**0.5
                rate_participants_mean_user = mean_rate_user
                rate_participants_std_user = round3(std_dev_user)
            else:
                n_participants_scoped_user = 0
                rate_participants_mean_user = ""
                rate_participants_std_user = ""
        else:
            n_participants_scoped_user = 0
            rate_participants_mean_user = ""
            rate_participants_std_user = ""

        participant_rates_assistant: list[float] = []
        if per_participant_denominators_assistant:
            for participant, denom in per_participant_denominators_assistant.items():
                numer = per_participant_numerators_assistant.get(participant, 0)
                if denom <= 0:
                    continue
                participant_rates_assistant.append(_safe_fraction(numer, denom))
            if participant_rates_assistant:
                n_participants_scoped_assistant = len(participant_rates_assistant)
                mean_rate_assistant = sum(participant_rates_assistant) / float(
                    len(participant_rates_assistant)
                )
                mean_rate_assistant = float(round3(mean_rate_assistant))
                variance_assistant = 0.0
                for rate_p in participant_rates_assistant:
                    diff_assistant = rate_p - mean_rate_assistant
                    variance_assistant += diff_assistant * diff_assistant
                variance_assistant /= float(len(participant_rates_assistant))
                std_dev_assistant = variance_assistant**0.5
                rate_participants_mean_assistant = mean_rate_assistant
                rate_participants_std_assistant = round3(std_dev_assistant)
            else:
                n_participants_scoped_assistant = 0
                rate_participants_mean_assistant = ""
                rate_participants_std_assistant = ""
        else:
            n_participants_scoped_assistant = 0
            rate_participants_mean_assistant = ""
            rate_participants_std_assistant = ""
        # Apply role scope to per-role counts and rates. When a role is not
        # part of the annotation scope (for example, a user-only label for an
        # assistant role), the corresponding counts and rates are left empty in
        # the output table rather than shown as zero.
        if "user" not in scope:
            n_messages_user = ""
            n_positive_user = ""
            rate_user = ""
            n_participants_scoped_user = 0
            rate_participants_mean_user = ""
            rate_participants_std_user = ""
        if "assistant" not in scope:
            n_messages_assistant = ""
            n_positive_assistant = ""
            rate_assistant = ""
            n_participants_scoped_assistant = 0
            rate_participants_mean_assistant = ""
            rate_participants_std_assistant = ""

        rows.append(
            {
                "annotation_id": annotation_id,
                "category": meta.category,
                "scope": ",".join(scope),
                "is_harmful": int(meta.is_harmful),
                "score_cutoff": cutoffs_by_id.get(annotation_id, ""),
                "n_messages_all": n_messages_all,
                "n_messages_user": n_messages_user,
                "n_messages_assistant": n_messages_assistant,
                "n_positive_all": n_positive_all,
                "n_positive_user": n_positive_user,
                "n_positive_assistant": n_positive_assistant,
                "rate_all": rate_all,
                "rate_user": rate_user,
                "rate_assistant": rate_assistant,
                "rate_participants_positive": rate_participants_positive,
                "rate_participants_mean": rate_participants_mean,
                "rate_participants_std": rate_participants_std,
                "n_participants_scoped": n_participants_scoped,
                "rate_participants_mean_user": rate_participants_mean_user,
                "rate_participants_std_user": rate_participants_std_user,
                "n_participants_scoped_user": n_participants_scoped_user,
                "rate_participants_mean_assistant": rate_participants_mean_assistant,
                "rate_participants_std_assistant": rate_participants_std_assistant,
                "n_participants_scoped_assistant": n_participants_scoped_assistant,
            }
        )

    return rows


def _write_rows(output_csv: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Write summary rows to ``output_csv``."""

    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    def _row_sort_key(row: Mapping[str, object]) -> tuple[object, object, object]:
        """Return a sort key grouping by category then participant mean rate."""

        category = str(row.get("category", ""))
        rate_participants_mean = row.get("rate_participants_mean")
        if isinstance(rate_participants_mean, (int, float)):
            # Sort by descending participant mean within each category.
            rate_key: float | int | float = -float(rate_participants_mean)
        else:
            # Place rows without a participant mean at the end of each category.
            rate_key = float("inf")
        annotation_id = str(row.get("annotation_id", ""))
        return (category, rate_key, annotation_id)

    fieldnames = [
        "annotation_id",
        "category",
        "scope",
        "is_harmful",
        "score_cutoff",
        "n_messages_all",
        "n_messages_user",
        "n_messages_assistant",
        "n_positive_all",
        "n_positive_user",
        "n_positive_assistant",
        "rate_all",
        "rate_user",
        "rate_assistant",
        "rate_participants_positive",
        "rate_participants_mean",
        "rate_participants_std",
        "n_participants_scoped",
        "rate_participants_mean_user",
        "rate_participants_std_user",
        "n_participants_scoped_user",
        "rate_participants_mean_assistant",
        "rate_participants_std_assistant",
        "n_participants_scoped_assistant",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=_row_sort_key):
            writer.writerow(row)


def _run_frequency_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with run_annotation_job to compute frequencies."""

    (message_info, _conversation_messages, annotation_message_positive) = family_state

    # Apply shared analysis filters so that excluded annotations (for example,
    # test-category or synthetic ids) are omitted from the frequency table.
    metadata_by_id_filtered = filter_analysis_metadata(metadata_by_id)

    # When both canonical and role-split variants are present for selected
    # annotations (for example, ``platonic-affinity`` alongside
    # ``user-platonic-affinity`` and ``assistant-platonic-affinity``),
    # restrict the per-annotation rows to the role-split identifiers so that
    # each behaviour appears exactly once in the summary table.
    annotation_ids_for_rows = filter_annotation_ids_for_display(
        list(metadata_by_id_filtered.keys()),
    )
    metadata_for_rows = {
        annotation_id: metadata_by_id_filtered[annotation_id]
        for annotation_id in annotation_ids_for_rows
        if annotation_id in metadata_by_id_filtered
    }

    rows = _aggregate_frequencies(
        message_info=message_info,
        annotation_message_positive=annotation_message_positive,
        metadata_by_id=metadata_for_rows,
        cutoffs_by_id=cutoffs_by_id,
        k_threshold=args.k_threshold,
    )
    _write_rows(args.output, rows)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for computing annotation frequencies."""

    args = parse_args(argv)
    return run_preprocessed_annotation_job(args, _run_frequency_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
