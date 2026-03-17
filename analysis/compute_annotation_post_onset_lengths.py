"""Estimate how annotation presence relates to remaining conversation length.

This script fits per-annotation regression models in which the outcome is the
remaining conversation length after each message and the key predictor is
whether the message is positively annotated. It operates on the same
preprocessed per-message annotation tables used by other analysis scripts
and respects annotation role scopes.

For each annotation id, the script:

* Orders messages within conversations.
* For each scoped message, computes the number of messages remaining after
  that point.
* Builds a message-level dataset with a transformed remaining-length outcome,
  a binary indicator for annotation presence, and a time-within-conversation
  covariate.
* Fits a linear regression of the form

    transform(remaining_t) = beta_0 + beta_1 * I(annotated_t) + beta_2 * time_frac_t

  where ``time_frac_t`` is the fraction of the conversation completed by
  message ``t``.
* Optionally re-computes standard errors with participant-level clustering.

Results are written, for each role scope (all in-scope roles, user-only,
assistant-only), as a CSV table with one row per annotation id containing
support sizes and an estimated remaining-length effect for annotated versus
unannotated messages at the same position in the conversation. Scope-specific
tables are suffixed ``__scope-user`` or ``__scope-assistant`` in the output
filename.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
)

from analysis.lib.cli.length_cli import add_length_model_arguments, parse_length_args
from analysis.lib.io.csv_utils import write_rows_with_fieldnames
from analysis.lib.stats.effect_summaries import summarise_linear_effect
from analysis.lib.stats.regression_utils import fit_ols_with_time_fraction
from analysis.lib.utils.formatting import round3
from annotation.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from annotation.io import ParticipantMessageKey
from utils.cli import add_output_path_argument


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the remaining-length script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Estimate how annotation presence relates to the remaining "
            "conversation length after each message using per-annotation "
            "regression models."
        )
    )
    add_length_model_arguments(parser)
    add_output_path_argument(
        parser,
        default_path=Path("analysis/data/annotation_remaining_length_effects.csv"),
        help_text=(
            "Output CSV path for per-annotation remaining-length estimates. "
            "Each row corresponds to a single annotation id."
        ),
    )
    return parser


def _build_remaining_sample_for_annotation(
    annotation_id: str,
    *,
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
    messages_by_conversation: Mapping[ConversationKey, Sequence[ParticipantMessageKey]],
    annotation_message_positive: Mapping[Tuple[str, ParticipantMessageKey], bool],
    metadata_by_id: Mapping[str, AnnotationMetadata],
    length_transform: str,
    role_filter: Optional[str] | None = None,
) -> Optional[
    Tuple[
        list[float],
        list[float],
        list[float],
        list[str],
        int,
        int,
        int,
        int,
    ]
]:
    """Return remaining-length sample vectors and support sizes for an annotation.

    Parameters
    ----------
    annotation_id:
        Identifier of the annotation to analyse.
    message_info:
        Mapping from participant message key to (role, conversation key).
    messages_by_conversation:
        Mapping from conversation key to ordered message keys.
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to positivity flags.
    metadata_by_id:
        Annotation metadata keyed by annotation id.
    length_transform:
        Name of the transform applied to remaining length as the outcome;
        ``\"log\"`` or ``\"raw\"``.
    role_filter:
        Optional role restriction for the messages contributing to the
        sample. When ``\"user\"`` or ``\"assistant\"``, only messages with
        that role and within the annotation scope are included. When
        ``None``, all in-scope roles are included.

    Returns
    -------
    Optional[tuple]
        ``None`` when no scoped messages contribute to the sample;
        otherwise a tuple containing:

        * y: transformed remaining conversation lengths per scoped message.
        * annot_values: binary predictors indicating annotation presence per
          scoped message.
        * time_fractions: fraction of the conversation completed at each
          message index.
        * clusters: participant ids per message for potential clustering.
        * n_conversations_scoped: number of conversations contributing at
          least one scoped message.
        * n_conversations_with_positive: number of conversations with at
          least one positive scoped occurrence for the annotation.
        * n_messages_scoped: total number of scoped messages contributing
          to the sample.
        * n_positive_messages: total number of messages in the sample with
          positive annotation.
    """

    meta = metadata_by_id.get(annotation_id)
    if meta is None:
        return None
    scope = meta.scope
    role_token: Optional[str] = role_filter.lower() if role_filter else None

    y: list[float] = []
    annot_values: list[float] = []
    time_fractions: list[float] = []
    clusters: list[str] = []

    conversations_scoped: set[ConversationKey] = set()
    conversations_with_positive: set[ConversationKey] = set()

    for conv_key, message_keys in messages_by_conversation.items():
        if not message_keys:
            continue
        ordered_keys = sorted(
            message_keys,
            key=lambda key: (key[1], key[2], key[3]),
        )
        length = len(ordered_keys)
        if length <= 1:
            continue

        conv_has_scoped = False
        conv_has_positive = False

        for index, message_key in enumerate(ordered_keys, start=1):
            role, _conv_for_message = message_info[message_key]
            role_lower = role.lower()

            if role_token is not None and role_lower != role_token:
                continue

            if not is_role_in_scope(role_lower, scope):
                continue

            remaining = length - index
            if remaining <= 0:
                continue

            is_positive = annotation_message_positive.get(
                (annotation_id, message_key),
                False,
            )
            conv_has_scoped = True
            if is_positive:
                conv_has_positive = True

            if length_transform == "log":
                outcome_value = math.log(float(remaining))
            else:
                outcome_value = float(remaining)

            time_fraction = float(index) / float(length)
            participant_id = str(message_key[0])

            y.append(outcome_value)
            annot_values.append(1.0 if is_positive else 0.0)
            time_fractions.append(time_fraction)
            clusters.append(participant_id)

        if conv_has_scoped:
            conversations_scoped.add(conv_key)
        if conv_has_positive:
            conversations_with_positive.add(conv_key)

    n_messages_scoped = len(y)
    n_positive_messages = int(sum(annot_values))
    n_conversations_scoped = len(conversations_scoped)
    n_conversations_with_positive = len(conversations_with_positive)

    if n_messages_scoped <= 0 or n_conversations_scoped <= 0:
        return None

    return (
        y,
        annot_values,
        time_fractions,
        clusters,
        n_conversations_scoped,
        n_conversations_with_positive,
        n_messages_scoped,
        n_positive_messages,
    )


def _fit_remaining_effect(
    y: Sequence[float],
    annot_values: Sequence[float],
    time_fractions: Sequence[float],
    clusters: Sequence[str],
    *,
    cluster_by_participant: bool,
) -> Optional[tuple[float, float, float, float]]:
    """Return (beta, se, z, p) for the annotation effect on remaining length.

    The function fits a simple linear regression model:

        outcome = beta_0 + beta_1 * I(annotated) + beta_2 * time_frac + error,

    where ``outcome`` is the transformed remaining conversation length, and
    the annotation coefficient ``beta_1`` captures the average difference in
    transformed remaining length between annotated and unannotated messages
    at the same conversation-relative position.
    """

    return fit_ols_with_time_fraction(
        y,
        annot_values,
        time_fractions,
        clusters,
        cluster_by_participant=cluster_by_participant,
    )


def _write_results(
    output_path: Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Write per-annotation remaining-length estimates to ``output_path``."""

    fieldnames = [
        "annotation_id",
        "category",
        "scope_role",
        "n_conversations_scoped",
        "n_conversations_with_positive",
        "share_conversations_with_positive",
        "n_messages_scoped",
        "n_positive_messages",
        "rate_positive",
        "model_type",
        "length_transform",
        "cluster_by_participant",
        "beta_effect",
        "se_effect",
        "z_value",
        "p_value",
        "remaining_ratio_annotated_vs_not",
        "ci_lower",
        "ci_upper",
    ]

    write_rows_with_fieldnames(
        output_path,
        fieldnames,
        rows,
        description="annotation remaining-length effects",
    )


def append_remaining_length_row(
    rows: list[Dict[str, object]],
    estimate: Optional[tuple[float, float, float, float]],
    *,
    annotation_id: str,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    length_transform: str,
    cluster_by_participant: bool,
    scope_label: str,
    n_conversations_scoped: int,
    n_conversations_with_positive: int,
    share_conversations_with_positive: float | str,
    n_messages_scoped: int,
    n_positive_messages: int,
    rate_positive: float | str,
) -> None:
    """Append a remaining-length output row for ``annotation_id``."""

    meta = metadata_by_id[annotation_id]
    summary = summarise_linear_effect(
        estimate,
        length_transform=length_transform,
    )
    row: Dict[str, object] = {
        "annotation_id": annotation_id,
        "category": meta.category,
        "scope_role": scope_label,
        "n_conversations_scoped": n_conversations_scoped,
        "n_conversations_with_positive": n_conversations_with_positive,
        "share_conversations_with_positive": share_conversations_with_positive,
        "n_messages_scoped": n_messages_scoped,
        "n_positive_messages": n_positive_messages,
        "rate_positive": rate_positive,
        "model_type": "linear",
        "length_transform": length_transform,
        "cluster_by_participant": int(cluster_by_participant),
        "beta_effect": summary["beta"],
        "se_effect": summary["se"],
        "z_value": summary["z"],
        "p_value": summary["p"],
        "remaining_ratio_annotated_vs_not": summary["ratio"],
        "ci_lower": summary["ci_lower"],
        "ci_upper": summary["ci_upper"],
    }
    rows.append(row)


def _run_remaining_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with run_preprocessed_annotation_job for remaining lengths."""

    (
        message_info,
        _conversation_messages,
        annotation_message_positive,
    ) = family_state

    metadata_by_id = filter_analysis_metadata(metadata_by_id)

    if not message_info:
        print("No usable messages discovered in the preprocessed table.")
        return 0

    base_output_path = Path(args.output)

    messages_by_conversation: Dict[ConversationKey, list[ParticipantMessageKey]] = (
        defaultdict(list)
    )
    for message_key, (_role, conv_key) in message_info.items():
        messages_by_conversation[conv_key].append(message_key)

    present_ids = {aid for (aid, _mkey) in annotation_message_positive.keys()}
    annotation_ids = [
        aid for aid in metadata_by_id if aid in present_ids and aid in cutoffs_by_id
    ]

    min_messages: int = int(args.min_messages)
    length_transform: str = str(args.length_transform)
    cluster_by_participant: bool = bool(args.cluster_by_participant)
    # Compute remaining-length effects separately for all in-scope roles,
    # user-only messages, and assistant-only messages so that annotations
    # such as romantic-interest can be inspected independently for user
    # and assistant turns.
    roles: list[Optional[str]] = [None, "user", "assistant"]
    any_rows_written = False

    for role_filter in roles:
        scope_rows: list[dict] = []
        scope_label = role_filter if role_filter is not None else "any"

        for annotation_id in annotation_ids:
            sample = _build_remaining_sample_for_annotation(
                annotation_id,
                message_info=message_info,
                messages_by_conversation=messages_by_conversation,
                annotation_message_positive=annotation_message_positive,
                metadata_by_id=metadata_by_id,
                length_transform=length_transform,
                role_filter=role_filter,
            )
            if sample is None:
                continue

            (
                y,
                annot_values,
                time_fractions,
                clusters,
                n_conversations_scoped,
                n_conversations_with_positive,
                n_messages_scoped,
                n_positive_messages,
            ) = sample

            if n_messages_scoped < min_messages:
                continue

            share_conversations_with_positive: float | str
            if n_conversations_scoped > 0:
                share_conversations_with_positive = round3(
                    float(n_conversations_with_positive)
                    / float(n_conversations_scoped),
                )
            else:
                share_conversations_with_positive = ""

            rate_positive: float | str
            if n_messages_scoped > 0:
                rate_positive = round3(
                    float(n_positive_messages) / float(n_messages_scoped),
                )
            else:
                rate_positive = ""

            estimate = _fit_remaining_effect(
                y,
                annot_values,
                time_fractions,
                clusters,
                cluster_by_participant=cluster_by_participant,
            )

            append_remaining_length_row(
                scope_rows,
                estimate,
                annotation_id=annotation_id,
                metadata_by_id=metadata_by_id,
                length_transform=length_transform,
                cluster_by_participant=cluster_by_participant,
                scope_label=scope_label,
                n_conversations_scoped=n_conversations_scoped,
                n_conversations_with_positive=n_conversations_with_positive,
                share_conversations_with_positive=share_conversations_with_positive,
                n_messages_scoped=n_messages_scoped,
                n_positive_messages=n_positive_messages,
                rate_positive=rate_positive,
            )

        if not scope_rows:
            print(
                "No annotations met the minimum message threshold for "
                "remaining-length modelling at scope "
                f"{scope_label}.",
            )
            continue

        any_rows_written = True

        if role_filter in {"user", "assistant"}:
            output_path = base_output_path.with_name(
                f"{base_output_path.stem}__scope-{scope_label}{base_output_path.suffix}"
            )
        else:
            output_path = base_output_path

        _write_results(output_path, scope_rows)

    if not any_rows_written:
        return 0

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for estimating remaining-length effects."""

    args = parse_length_args(_build_parser, argv)
    return run_preprocessed_annotation_job(args, _run_remaining_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
