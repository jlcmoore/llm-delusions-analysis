"""Compute sequential annotation dynamics within conversations.

This script analyses within-conversation co-occurrence dynamics for pairs of
annotations at multiple horizons K, using JSONL outputs produced by
``scripts/annotation/classify_chats.py``. For each source annotation X, target
annotation Y, and window size K, it summarises:

* The number of messages containing X (N_X).
* The number of times Y appears within K messages after X in the same
  conversation (C_K[X, Y]).
* The expected count of Y within the K-window given X.
* The global base rate of Y across scoped messages.
* An enrichment ratio comparing windowed co-occurrence to the base rate.

In addition to pairwise X->Y summaries, the script can emit sparse co-window
triple tables that record how often a third annotation Z appears in the same
K-message window where both X occurs and Y is present somewhere in that
window. These X,Y,Z tables are designed to support downstream queries of
quantities such as P(Z | X, Y-in-window) / P(Z | X) without requiring the
script to be rerun for specific annotation ids.

The statistics are inherently message-weighted: longer conversations that
contain more occurrences of X contribute more windows and therefore have
greater influence on the results. The current implementation does not
stratify by conversation length, so users should keep this bias in mind when
interpreting enrichment patterns.

The analysis operates at the message level and respects annotation scopes, so
role-incompatible labels are excluded from both base-rate and co-occurrence
calculations.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
    normalize_role_filter,
)

from analysis.lib.cli.sequential_dynamics_cli import (
    add_window_k_argument,
    parse_window_k_arguments,
)
from analysis.lib.stats.beta_utils import beta_posterior_sd as _beta_posterior_sd
from analysis.lib.utils.formatting import round3
from annotation.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from annotation.io import ParticipantMessageKey
from utils.cli import (
    add_annotation_metadata_arguments,
    add_preprocessed_input_csv_argument,
)

# Hyperparameters for the Beta window model. The prior strength controls how
# strongly the global K-window rate influences the posterior when only a few
# X windows are observed. Z_BETA_CI encodes the normal-approximation multiplier
# for an approximate 95% interval.
BETA_PRIOR_STRENGTH = 2.0
BETA_CI_Z = 1.96


def _parse_override_scopes(
    raw_values: Sequence[str],
) -> Dict[str, Sequence[str]]:
    """Return per-annotation role overrides parsed from CLI values.

    Parameters
    ----------
    raw_values:
        Raw ``--override-scope`` strings of the form
        ``annotation_id:role1,role2,...``.

    Returns
    -------
    Dict[str, Sequence[str]]
        Mapping from annotation id to a non-empty sequence of roles drawn
        from ``{\"user\", \"assistant\"}``.

    Raises
    ------
    ValueError
        If any value is malformed or uses an unsupported role token.
    """

    overrides: Dict[str, Sequence[str]] = {}
    for value in raw_values:
        parts = value.split(":", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Invalid --override-scope value {value!r}; expected "
                "annotation_id:role1,role2,...",
            )
        annotation_id_raw, roles_raw = parts
        annotation_id = annotation_id_raw.strip()
        if not annotation_id:
            raise ValueError(
                f"Invalid --override-scope value {value!r}; annotation id "
                "must be non-empty",
            )
        role_tokens = [
            token.strip().lower() for token in roles_raw.split(",") if token.strip()
        ]
        if not role_tokens:
            raise ValueError(
                f"Invalid --override-scope value {value!r}; at least one "
                "role is required",
            )
        for token in role_tokens:
            if token not in {"user", "assistant"}:
                raise ValueError(
                    f"Invalid role {token!r} for annotation {annotation_id!r}; "
                    "expected 'user' or 'assistant'.",
                )
        overrides[annotation_id] = tuple(role_tokens)
    return overrides


def _build_scope_prefix_name(
    base_name: str,
    source_role: Optional[str],
    target_role: Optional[str],
) -> str:
    """Return an output-prefix name augmented with role scope suffixes.

    Parameters
    ----------
    base_name:
        Basename of the output prefix (for example, ``\"sequential_dynamics\"``).
    source_role:
        Optional role restriction for source annotations X (``\"user\"``,
        ``\"assistant\"``, or ``None`` for any in-scope role).
    target_role:
        Optional role restriction for target annotations Y (``\"user\"``,
        ``\"assistant\"``, or ``None`` for any in-scope role).

    Returns
    -------
    str
        New basename with ``\"__source-<role>\"`` and/or ``\"__scope-<role>\"``
        suffixes appended where applicable.
    """

    name = base_name
    if source_role in {"user", "assistant"}:
        name = f"{name}__source-{source_role}"
    if target_role in {"user", "assistant"}:
        name = f"{name}__scope-{target_role}"
    return name


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the sequential-dynamics script."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute within-conversation sequential annotation dynamics for "
            "multiple window sizes K using a preprocessed per-message "
            "annotation table."
        )
    )
    add_preprocessed_input_csv_argument(parser)
    add_annotation_metadata_arguments(parser)
    add_window_k_argument(parser)
    parser.add_argument(
        "--min-nx",
        type=int,
        default=50,
        help=(
            "Minimum number of messages containing X required for a source "
            "annotation to be included in the top-pairs summary (default: 50)."
        ),
    )
    parser.add_argument(
        "--min-cooccurrences",
        type=int,
        default=10,
        help=(
            "Minimum number of windowed co-occurrences C_K[X, Y] required "
            "for an X->Y pair to be included in the top-pairs summary "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--source-role",
        type=str,
        default="auto",
        help=(
            "Optional role restriction for source annotations X. When set to "
            "'user' or 'assistant', only messages with that role contribute "
            "to N_X and windowed co-occurrence counts. When omitted or set "
            "to 'auto'/'both', both roles are included wherever annotations "
            "are in scope."
        ),
    )
    parser.add_argument(
        "--target-role",
        type=str,
        default="auto",
        help=(
            "Optional role restriction for target annotations Y. When set to "
            "'user' or 'assistant', only Y occurrences on messages with that "
            "role contribute to base rates and windowed co-occurrences. When "
            "omitted or set to 'auto'/'both', both roles are included "
            "wherever annotations are in scope."
        ),
    )
    parser.add_argument(
        "--override-scope",
        dest="override_scopes",
        action="append",
        metavar="annotation_id:role1,role2,...",
        help=(
            "Optional per-annotation role overrides applied on top of the "
            "metadata scopes for this analysis only. Each value must take "
            "the form annotation_id:role1,role2,... where roles are drawn "
            "from 'user' and 'assistant'."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis") / "data" / "sequential_dynamics" / "base",
        help=(
            "Output prefix used for CSV tables. Full paths include the K "
            "value and optional role suffixes, for example "
            "'analysis/data/sequential_dynamics/base_K10_matrix.csv' or "
            "'analysis/data/sequential_dynamics/base_K10_triples_cowindow.csv'. "
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the sequential-dynamics script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    args.window_k = parse_window_k_arguments(parser, args.window_k)

    if args.min_nx < 0:
        parser.error("--min-nx must be non-negative")
    if args.min_cooccurrences < 0:
        parser.error("--min-cooccurrences must be non-negative")

    try:
        args.source_role = normalize_role_filter(getattr(args, "source_role", None))
        args.target_role = normalize_role_filter(getattr(args, "target_role", None))
    except ValueError as exc:
        parser.error(str(exc))

    raw_overrides = getattr(args, "override_scopes", None)
    try:
        args.override_scopes_by_id = (
            _parse_override_scopes(raw_overrides) if raw_overrides else {}
        )
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _build_conversation_messages(
    message_info: Mapping[ParticipantMessageKey, Tuple[str, ConversationKey]],
) -> Dict[ConversationKey, list[tuple[ParticipantMessageKey, str]]]:
    """Return per-conversation ordered message keys and roles.

    Parameters
    ----------
    message_info:
        Mapping from participant message key to (role, conversation key).

    Returns
    -------
    Dict[ConversationKey, list[tuple[ParticipantMessageKey, str]]]
        Mapping from conversation key to a list of (message key, role) pairs
        ordered by chat and message indices.
    """

    grouped: Dict[ConversationKey, list[tuple[ParticipantMessageKey, str]]] = (
        defaultdict(list)
    )
    for message_key, (role, conv_key) in message_info.items():
        grouped[conv_key].append((message_key, role))

    for conv_key, items in grouped.items():
        items.sort(key=lambda pair: (pair[0][2], pair[0][3]))
    return grouped


def _build_message_annotations(
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
) -> Dict[ParticipantMessageKey, set[str]]:
    """Return positive annotations per message key."""

    per_message: Dict[ParticipantMessageKey, set[str]] = defaultdict(set)
    for (
        annotation_id,
        message_key,
    ), is_positive in annotation_message_positive.items():
        if is_positive:
            per_message[message_key].add(annotation_id)
    return per_message


def _compute_base_rates(
    message_info: Mapping[ParticipantMessageKey, tuple[str, ConversationKey]],
    message_annotations: Mapping[ParticipantMessageKey, set[str]],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    target_role_filter: Optional[str] = None,
) -> tuple[Dict[str, int], Dict[str, int]]:
    """Return per-annotation (messages_with_Y, total_scoped_messages_for_Y).

    The scoped denominator counts messages for which the annotation could
    apply given its role scope, regardless of whether the label is positive
    on that message.
    """

    messages_with_y: Dict[str, int] = Counter()
    total_scoped_messages: Dict[str, int] = Counter()

    # Precompute annotation ids for each role to avoid repeated scope checks.
    role_to_annotations: Dict[str, list[str]] = defaultdict(list)
    for annotation_id, meta in metadata_by_id.items():
        for role in ("user", "assistant"):
            if is_role_in_scope(role, meta.scope):
                role_to_annotations[role].append(annotation_id)

    for message_key, (role, _conv_key) in message_info.items():
        role_lower = role.lower()
        if target_role_filter is not None and role_lower != target_role_filter:
            continue
        scoped_annotations = role_to_annotations.get(role_lower, [])
        if not scoped_annotations:
            continue
        present = message_annotations.get(message_key, set())

        for annotation_id in scoped_annotations:
            total_scoped_messages[annotation_id] += 1
            if annotation_id in present:
                messages_with_y[annotation_id] += 1

    return messages_with_y, total_scoped_messages


def _accumulate_sequential_counts(
    conversation_messages: Mapping[
        ConversationKey, list[tuple[ParticipantMessageKey, str]]
    ],
    message_annotations: Mapping[ParticipantMessageKey, set[str]],
    *,
    ks: Sequence[int],
    metadata_by_id: Mapping[str, AnnotationMetadata],
    annotation_ids: Sequence[str],
    source_role_filter: Optional[str] = None,
    target_role_filter: Optional[str] = None,
) -> tuple[
    Dict[str, int],
    Dict[int, Dict[tuple[str, str], int]],
    Dict[int, Dict[str, int]],
    Dict[int, Dict[tuple[str, str], int]],
    Dict[int, Dict[tuple[str, str, str], int]],
]:
    """Return N_X and per-K windowed co-occurrence statistics.

    In addition to the original per-K co-occurrence counts C_K[X, Y], this
    function also tracks per-window binary events suitable for a simple Beta
    model. For each source annotation X, target annotation Y, and window size
    K it accumulates:

    * C_K[X, Y]: total count of Y messages within the K-message window after X
      (unchanged from the original implementation).
    * trials_K[X]: number of X messages that have at least one message in the
      forward K-window (used as the binomial denominator).
    * successes_K[X, Y]: number of X messages for which Y appears at least
      once anywhere in the K-window (used as the binomial numerator).

    The binary (trials, successes) statistics ensure that the per-window
    probability of observing Y given X is bounded in [0, 1] and therefore
    compatible with a Beta posterior, while leaving the original count-based
    C_K statistics intact for existing analyses.
    """

    n_x: Dict[str, int] = Counter()
    counts_by_k: Dict[int, Dict[tuple[str, str], int]] = {k: Counter() for k in ks}
    trials_by_k: Dict[int, Dict[str, int]] = {k: Counter() for k in ks}
    successes_by_k: Dict[int, Dict[tuple[str, str], int]] = {k: Counter() for k in ks}
    triple_successes_by_k: Dict[int, Dict[tuple[str, str, str], int]] = {
        k: Counter() for k in ks
    }

    # Precompute source-eligible annotations per role so that we can
    # efficiently derive both positive and absence-based sources. Only
    # annotations in ``annotation_ids`` are considered; this keeps the
    # synthetic ``not:<id>`` sources aligned with the rest of the pipeline.
    role_to_source_annotations: Dict[str, list[str]] = defaultdict(list)
    for annotation_id in annotation_ids:
        meta = metadata_by_id.get(annotation_id)
        if meta is None:
            continue
        for role in ("user", "assistant"):
            if is_role_in_scope(role, meta.scope):
                role_to_source_annotations[role].append(annotation_id)

    for _conv_key, items in conversation_messages.items():
        if not items:
            continue
        source_at: list[set[str]] = []
        target_at: list[set[str]] = []
        for message_key, role in items:
            role_lower = role.lower()
            present = message_annotations.get(message_key, set())

            scoped_source: set[str] = set()
            scoped_target: set[str] = set()

            # Positive targets are based on actual annotations present on
            # this message that are in scope for the role and target filter.
            for annotation_id in present:
                meta = metadata_by_id.get(annotation_id)
                if meta is None:
                    continue
                if is_role_in_scope(role_lower, meta.scope):
                    if target_role_filter is None or role_lower == target_role_filter:
                        scoped_target.add(annotation_id)

            # Sources include both positive occurrences of an annotation and
            # synthetic absence-based sources of the form ``not:<id>`` for
            # annotations that are in scope for this role but not present on
            # this message. This allows downstream plots to treat
            # ``not:user-suicidal-intent`` as a valid source id whose
            # dynamics are computed directly from the underlying data rather
            # than by complementing existing probabilities.
            source_candidates = role_to_source_annotations.get(role_lower, [])
            for annotation_id in source_candidates:
                if source_role_filter is not None and role_lower != source_role_filter:
                    continue
                if annotation_id in present:
                    scoped_source.add(annotation_id)
                else:
                    scoped_source.add(f"not:{annotation_id}")

            source_at.append(scoped_source)
            target_at.append(scoped_target)

        length = len(source_at)
        if length == 0:
            continue

        for index in range(length):
            source_annotations = source_at[index]
            if not source_annotations:
                continue

            for window in ks:
                if window == 0:
                    window_indices = [index]
                else:
                    end = min(index + window, length - 1)
                    window_indices = list(range(index + 1, end + 1))

                if not window_indices:
                    continue

                # Build the set of unique target annotations that appear at
                # least once within the K-message window, which is used for
                # binary success counts in the Beta model and co-window triple
                # statistics.
                window_targets: set[str] = set()
                for target_index in window_indices:
                    window_targets.update(target_at[target_index])

                for source in source_annotations:
                    # Count each occurrence of X that has a non-empty forward
                    # window so that trials_K[X] reflects the number of
                    # binomial trials for the X->Y event.
                    trials_by_k[window][source] += 1

                    # Preserve the original occurrence-weighted co-occurrence
                    # counts C_K[X, Y].
                    for target_index in window_indices:
                        for target in target_at[target_index]:
                            counts_by_k[window][(source, target)] += 1

                    # Record binary successes for the Beta model: a single
                    # success whenever Y appears at least once anywhere in the
                    # window after X.
                    for target in window_targets:
                        successes_by_k[window][(source, target)] += 1

                    # Record co-window triple statistics: for each target Y
                    # that appears at least once within the X-window, record a
                    # success for all targets Z that co-occur anywhere in that
                    # same window. Per-(X, Y) trial counts can be recovered
                    # from the pairwise successes, so only successes for
                    # (X, Y, Z) are stored explicitly.
                    for y in window_targets:
                        for z in window_targets:
                            triple_successes_by_k[window][(source, y, z)] += 1

                # N_X counts every message that contains X, regardless of
                # whether a particular K-window is available. This preserves
                # the historical interpretation of N_X while the Beta model
                # uses trials_K[X] as its own denominator.
                for source in source_annotations:
                    n_x[source] += 1

    return n_x, counts_by_k, trials_by_k, successes_by_k, triple_successes_by_k


def _safe_fraction(numerator: int, denominator: int) -> float:
    """Return numerator / denominator guarded against zero denominators."""

    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _compute_beta_window_stats(
    *,
    k: int,
    x: str,
    y: str,
    p_y_anywhere: float,
    trials_k: Mapping[str, int],
    successes_k: Mapping[tuple[str, str], int],
    prior_strength: float = BETA_PRIOR_STRENGTH,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Return Beta posterior statistics for the X->Y K-window event.

    The model treats, for a fixed K, each occurrence of X with a non-empty
    K-message forward window as a Bernoulli trial whose outcome is whether Y
    appears at least once in that window. A weak Beta prior is centred at an
    approximate global K-window rate derived from the single-message base
    rate p(Y).
    """

    if p_y_anywhere <= 0.0:
        return 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Approximate the global K-window probability for Y as the independent
    # message expectation, clamped into (0, 1) so that it can safely serve as
    # the mean of a Beta prior.
    if k > 0:
        p_window_global = 1.0 - (1.0 - p_y_anywhere) ** float(k)
    else:
        p_window_global = p_y_anywhere

    epsilon = 1e-6
    p_window_global_clamped = max(
        epsilon,
        min(1.0 - epsilon, p_window_global),
    )

    trial_count = int(trials_k.get(x, 0))
    success_count = int(successes_k.get((x, y), 0))
    if trial_count <= 0:
        return trial_count, success_count, p_window_global, 0.0, 0.0, 0.0, 0.0, 0.0

    alpha0 = prior_strength * p_window_global_clamped
    beta0 = prior_strength * (1.0 - p_window_global_clamped)

    alpha_post = alpha0 + float(success_count)
    beta_post = beta0 + float(trial_count - success_count)
    denom = alpha_post + beta_post
    if denom <= 0.0:
        return (
            trial_count,
            success_count,
            p_window_global,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    beta_p_window_given_x = alpha_post / denom
    beta_log_lift = math.log(beta_p_window_given_x / p_window_global_clamped)
    beta_posterior_sd = _beta_posterior_sd(alpha_post, beta_post)
    if beta_posterior_sd > 0.0:
        ci_delta = BETA_CI_Z * beta_posterior_sd
        ci_low = max(0.0, beta_p_window_given_x - ci_delta)
        ci_high = min(1.0, beta_p_window_given_x + ci_delta)
    else:
        ci_low = 0.0
        ci_high = 0.0
    return (
        trial_count,
        success_count,
        p_window_global,
        beta_p_window_given_x,
        beta_log_lift,
        beta_posterior_sd,
        ci_low,
        ci_high,
    )


def _compute_beta_triple_window_stats(
    *,
    p_z_given_x: float,
    trials_xy: int,
    successes_xyz: int,
    prior_strength: float = BETA_PRIOR_STRENGTH,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Return Beta posterior statistics for the X,Y->Z K-window event.

    The prior mean for the triple event P(Z | X, Y-in-window) is set to
    the pairwise K-window probability P(Z | X), so that the Beta model
    measures the incremental influence of Y relative to the existing
    X->Z dependence.
    """

    if p_z_given_x <= 0.0 or trials_xy <= 0:
        return trials_xy, successes_xyz, p_z_given_x, 0.0, 0.0, 0.0, 0.0, 0.0

    epsilon = 1e-6
    p_prior = max(epsilon, min(1.0 - epsilon, p_z_given_x))

    alpha0 = prior_strength * p_prior
    beta0 = prior_strength * (1.0 - p_prior)

    alpha_post = alpha0 + float(successes_xyz)
    beta_post = beta0 + float(trials_xy - successes_xyz)
    denom = alpha_post + beta_post
    if denom <= 0.0:
        return (
            trials_xy,
            successes_xyz,
            p_prior,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    beta_p_triple = alpha_post / denom
    beta_log_lift = math.log(beta_p_triple / p_prior)
    beta_posterior_sd = _beta_posterior_sd(alpha_post, beta_post)
    if beta_posterior_sd > 0.0:
        ci_delta = BETA_CI_Z * beta_posterior_sd
        ci_low = max(0.0, beta_p_triple - ci_delta)
        ci_high = min(1.0, beta_p_triple + ci_delta)
    else:
        ci_low = 0.0
        ci_high = 0.0

    return (
        trials_xy,
        successes_xyz,
        p_prior,
        beta_p_triple,
        beta_log_lift,
        beta_posterior_sd,
        ci_low,
        ci_high,
    )


def _write_full_matrix_csv(
    output_path: Path,
    k: int,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    n_x: Mapping[str, int],
    counts_k: Mapping[tuple[str, str], int],
    base_messages_with_y: Mapping[str, int],
    base_total_scoped_y: Mapping[str, int],
    trials_k: Mapping[str, int],
    successes_k: Mapping[tuple[str, str], int],
) -> None:
    """Write full X,Y matrix statistics for a single K to CSV."""

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "K",
        "X",
        "Y",
        "N_X",
        "C_K",
        "p_y_within_K_given_X",
        "p_y_anywhere",
        "messages_with_y",
        "total_scoped_y",
        "enrichment_K",
        "enrichment_K_per_step",
        "beta_trials",
        "beta_successes",
        "beta_p_window_given_X",
        "beta_p_window_global",
        "beta_log_lift",
        "beta_p_window_sd",
        "beta_p_window_ci_low",
        "beta_p_window_ci_high",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for x in source_ids:
            n_x_val = int(n_x.get(x, 0))
            for y in target_ids:
                pair = (x, y)
                c_k = int(counts_k.get(pair, 0))
                p_y_given_x = _safe_fraction(c_k, n_x_val)

                messages_with_y = int(base_messages_with_y.get(y, 0))
                scoped_y = int(base_total_scoped_y.get(y, 0))
                p_y_anywhere = _safe_fraction(messages_with_y, scoped_y)

                enrichment: float
                if p_y_anywhere > 0.0:
                    enrichment = _safe_fraction(p_y_given_x, p_y_anywhere)
                else:
                    enrichment = 0.0

                if k > 0:
                    enrichment_per_step = enrichment / float(k)
                else:
                    enrichment_per_step = enrichment

                (
                    beta_trials,
                    beta_successes,
                    beta_p_window_global,
                    beta_p_window_given_x,
                    beta_log_lift,
                    beta_p_window_sd,
                    beta_p_window_ci_low,
                    beta_p_window_ci_high,
                ) = _compute_beta_window_stats(
                    k=k,
                    x=x,
                    y=y,
                    p_y_anywhere=p_y_anywhere,
                    trials_k=trials_k,
                    successes_k=successes_k,
                )

                row = {
                    "K": k,
                    "X": x,
                    "Y": y,
                    "N_X": n_x_val,
                    "C_K": c_k,
                    "p_y_within_K_given_X": round3(p_y_given_x),
                    "p_y_anywhere": round3(p_y_anywhere),
                    "messages_with_y": messages_with_y,
                    "total_scoped_y": scoped_y,
                    "enrichment_K": round3(enrichment),
                    "enrichment_K_per_step": round3(enrichment_per_step),
                    "beta_trials": beta_trials,
                    "beta_successes": beta_successes,
                    "beta_p_window_given_X": round3(beta_p_window_given_x),
                    "beta_p_window_global": round3(beta_p_window_global),
                    "beta_log_lift": round3(beta_log_lift),
                    "beta_p_window_sd": round3(beta_p_window_sd),
                    "beta_p_window_ci_low": round3(beta_p_window_ci_low),
                    "beta_p_window_ci_high": round3(beta_p_window_ci_high),
                }
                writer.writerow(row)

    print(f"Wrote full sequential dynamics matrix for K={k} to {output_path}")


def _write_triple_cowindow_csv(
    output_path: Path,
    k: int,
    *,
    base_messages_with_y: Mapping[str, int],
    base_total_scoped_y: Mapping[str, int],
    trials_k: Mapping[str, int],
    successes_k: Mapping[tuple[str, str], int],
    triple_successes_k: Mapping[tuple[str, str, str], int],
) -> None:
    """Write X,Y,Z co-window statistics for a single K to CSV.

    The table is sparse: only triples (X, Y, Z) that co-occur within at least
    one K-message window are included. For each such triple the CSV records:

    * trials_X: number of X occurrences with a non-empty K-window.
    * trials_XY: number of X occurrences whose K-window contains Y
      (recovered from the pairwise successes).
    * successes_XZ: number of X occurrences whose K-window contains Z.
    * successes_XYZ: number of X occurrences whose K-window contains both Y
      and Z.

    These counts are sufficient to reconstruct, at plotting time, both the
    baseline probability P(Z | X) and the co-window conditional probability
    P(Z | X, Y-in-window) together with their Beta posterior summaries using
    the same prior as the pairwise model.
    """

    if not triple_successes_k:
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "K",
        "X",
        "Y",
        "Z",
        "trials_X",
        "trials_XY",
        "successes_XZ",
        "successes_XYZ",
        "p_z_within_K_given_X",
        "p_z_within_K_given_XY",
        "risk_ratio",
        "beta_trials_XY",
        "beta_successes_XYZ",
        "beta_p_window_prior",
        "beta_p_window_given_XY",
        "beta_log_lift_XY_vs_X",
        "beta_p_window_sd",
        "beta_p_window_ci_low",
        "beta_p_window_ci_high",
        "messages_with_z",
        "total_scoped_z",
        "p_z_anywhere",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate deterministically over observed triples.
        for (x, y, z), successes_xyz in sorted(triple_successes_k.items()):
            trials_x = int(trials_k.get(x, 0))
            trials_xy = int(successes_k.get((x, y), 0))
            successes_xz = int(successes_k.get((x, z), 0))

            p_z_given_x = _safe_fraction(successes_xz, trials_x)
            p_z_given_xy = _safe_fraction(successes_xyz, trials_xy)

            if p_z_given_x > 0.0:
                risk_ratio = _safe_fraction(p_z_given_xy, p_z_given_x)
            else:
                risk_ratio = 0.0

            (
                beta_trials_xy,
                beta_successes_xyz,
                beta_p_window_prior,
                beta_p_window_given_xy,
                beta_log_lift,
                beta_posterior_sd,
                beta_ci_low,
                beta_ci_high,
            ) = _compute_beta_triple_window_stats(
                p_z_given_x=p_z_given_x,
                trials_xy=trials_xy,
                successes_xyz=int(successes_xyz),
            )

            messages_with_z = int(base_messages_with_y.get(z, 0))
            scoped_z = int(base_total_scoped_y.get(z, 0))
            p_z_anywhere = _safe_fraction(messages_with_z, scoped_z)

            row = {
                "K": k,
                "X": x,
                "Y": y,
                "Z": z,
                "trials_X": trials_x,
                "trials_XY": trials_xy,
                "successes_XZ": successes_xz,
                "successes_XYZ": int(successes_xyz),
                "p_z_within_K_given_X": round3(p_z_given_x),
                "p_z_within_K_given_XY": round3(p_z_given_xy),
                "risk_ratio": round3(risk_ratio),
                "beta_trials_XY": beta_trials_xy,
                "beta_successes_XYZ": beta_successes_xyz,
                "beta_p_window_prior": round3(beta_p_window_prior),
                "beta_p_window_given_XY": round3(beta_p_window_given_xy),
                "beta_log_lift_XY_vs_X": round3(beta_log_lift),
                "beta_p_window_sd": round3(beta_posterior_sd),
                "beta_p_window_ci_low": round3(beta_ci_low),
                "beta_p_window_ci_high": round3(beta_ci_high),
                "messages_with_z": messages_with_z,
                "total_scoped_z": scoped_z,
                "p_z_anywhere": round3(p_z_anywhere),
            }
            writer.writerow(row)

    print(
        "Wrote co-window triple sequential dynamics table for "
        f"K={k} to {output_path}",
    )


def _write_top_pairs_csv(
    output_path: Path,
    k: int,
    source_ids: Sequence[str],
    target_ids: Sequence[str],
    n_x: Mapping[str, int],
    counts_k: Mapping[tuple[str, str], int],
    base_messages_with_y: Mapping[str, int],
    base_total_scoped_y: Mapping[str, int],
    *,
    trials_k: Mapping[str, int],
    successes_k: Mapping[tuple[str, str], int],
    min_nx: int,
    min_cooccurrences: int,
    max_rows: int = 200,
) -> None:
    """Write a compact top-enriched X->Y pairs table for a single K.

    Note
    ----
    A possible extension (not implemented here) is to compute simple lag
    summaries, such as the mean or median number of messages from X to the
    first Y within the K-message window, for a small subset of strongly
    enriched pairs. This script currently focuses on windowed counts and
    enrichment ratios only.
    """

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for x in source_ids:
        n_x_val = int(n_x.get(x, 0))
        if n_x_val < min_nx:
            continue
        for y in target_ids:
            if k == 0 and x == y:
                continue
            pair = (x, y)
            c_k = int(counts_k.get(pair, 0))
            if c_k < min_cooccurrences:
                continue

            p_y_given_x = _safe_fraction(c_k, n_x_val)

            messages_with_y = int(base_messages_with_y.get(y, 0))
            scoped_y = int(base_total_scoped_y.get(y, 0))
            p_y_anywhere = _safe_fraction(messages_with_y, scoped_y)
            if p_y_anywhere <= 0.0:
                continue

            (
                beta_trials,
                beta_successes,
                beta_p_window_global,
                beta_p_window_given_x,
                beta_log_lift,
                beta_p_window_sd,
                beta_p_window_ci_low,
                beta_p_window_ci_high,
            ) = _compute_beta_window_stats(
                k=k,
                x=x,
                y=y,
                p_y_anywhere=p_y_anywhere,
                trials_k=trials_k,
                successes_k=successes_k,
            )

            enrichment = _safe_fraction(p_y_given_x, p_y_anywhere)
            if k > 0:
                enrichment_per_step = enrichment / float(k)
            else:
                enrichment_per_step = enrichment
            records.append(
                {
                    "K": k,
                    "X": x,
                    "Y": y,
                    "N_X": n_x_val,
                    "C_K": c_k,
                    "p_y_within_K_given_X": round3(p_y_given_x),
                    "p_y_anywhere": round3(p_y_anywhere),
                    "enrichment_K": round3(enrichment),
                    "enrichment_K_per_step": round3(enrichment_per_step),
                    "beta_trials": beta_trials,
                    "beta_successes": beta_successes,
                    "beta_p_window_given_X": round3(beta_p_window_given_x),
                    "beta_p_window_global": round3(beta_p_window_global),
                    "beta_log_lift": round3(beta_log_lift),
                    "beta_p_window_sd": round3(beta_p_window_sd),
                    "beta_p_window_ci_low": round3(beta_p_window_ci_low),
                    "beta_p_window_ci_high": round3(beta_p_window_ci_high),
                }
            )

    # Sort by enrichment descending, then by C_K and N_X as tie-breakers.
    records.sort(
        key=lambda row: (
            float(row["enrichment_K"]),
            int(row["C_K"]),
            int(row["N_X"]),
        ),
        reverse=True,
    )
    if max_rows > 0:
        records = records[:max_rows]

    fieldnames = [
        "K",
        "X",
        "Y",
        "N_X",
        "C_K",
        "p_y_within_K_given_X",
        "p_y_anywhere",
        "enrichment_K",
        "enrichment_K_per_step",
        "beta_trials",
        "beta_successes",
        "beta_p_window_given_X",
        "beta_p_window_global",
        "beta_log_lift",
        "beta_p_window_sd",
        "beta_p_window_ci_low",
        "beta_p_window_ci_high",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"Wrote top enriched pairs for K={k} to {output_path}")


def _run_sequential_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with run_annotation_job to compute sequential dynamics."""

    message_info, _conversation_messages, annotation_message_positive = family_state

    # Apply shared analysis filters so that excluded annotations (for example,
    # test-category or synthetic ids) do not participate in sequential
    # dynamics summaries.
    metadata_by_id = filter_analysis_metadata(metadata_by_id)

    # Apply any per-annotation scope overrides requested on the CLI so that
    # dual-scoped annotations can be treated as user-only or assistant-only
    # for this analysis without modifying the underlying metadata table.
    override_scopes_by_id: Dict[str, Sequence[str]] = getattr(
        args,
        "override_scopes_by_id",
        {},
    )
    if override_scopes_by_id:
        adjusted: Dict[str, AnnotationMetadata] = {}
        for annotation_id, meta in metadata_by_id.items():
            override_scope = override_scopes_by_id.get(annotation_id)
            if override_scope is None:
                adjusted[annotation_id] = meta
                continue
            adjusted[annotation_id] = AnnotationMetadata(
                annotation_id=meta.annotation_id,
                category=meta.category,
                scope=tuple(override_scope),
                is_harmful=meta.is_harmful,
            )
        metadata_by_id = adjusted

    prefix: Path = args.output_prefix
    cutoffs_json = getattr(args, "llm_cutoffs_json", None)
    global_cutoff = getattr(args, "llm_score_cutoff", None)
    if cutoffs_json is None and global_cutoff is not None:
        cutoff_value = int(global_cutoff)
        prefix = prefix.with_name(f"{prefix.name}__scorecutoff{cutoff_value}")

    if not message_info:
        print("No usable messages for sequential dynamics analysis.")
        return 0

    # Restrict to annotations that both have LLM score cutoffs and appear in
    # the current job family. This excludes synthetic or test labels that are
    # present in metadata but never instantiated in the selected outputs.
    present_ids = {aid for (aid, _mkey) in annotation_message_positive.keys()}
    annotation_ids = [
        aid for aid in metadata_by_id if aid in present_ids and aid in cutoffs_by_id
    ]
    if not annotation_ids:
        print(
            "No annotations with both metrics and outputs were found for the "
            "selected job family."
        )
        return 0

    ks: list[int] = list(args.window_k)

    conversation_messages = _build_conversation_messages(message_info)
    message_annotations = _build_message_annotations(annotation_message_positive)

    # Include synthetic absence-based sources ``not:<id>`` for all
    # annotation ids that participate in the dynamics. These are used
    # only as sources; targets and base-rate summaries remain defined
    # over positive annotations.
    source_ids: list[str] = list(annotation_ids)
    for aid in annotation_ids:
        source_ids.append(f"not:{aid}")

    # Compute dynamics for multiple (source-role, target-role) scope views in
    # a single pass. For each combination, separate matrix and top-pairs CSV
    # tables are written using a suffix convention that encodes the roles.
    roles: list[Optional[str]] = [None, "user", "assistant"]

    for source_role in roles:
        for target_role in roles:
            base_messages_with_y, base_total_scoped_y = _compute_base_rates(
                message_info,
                message_annotations,
                metadata_by_id=metadata_by_id,
                target_role_filter=target_role,
            )

            (
                n_x,
                counts_by_k,
                trials_by_k,
                successes_by_k,
                triple_successes_by_k,
            ) = _accumulate_sequential_counts(
                conversation_messages,
                message_annotations,
                ks=ks,
                metadata_by_id=metadata_by_id,
                annotation_ids=annotation_ids,
                source_role_filter=source_role,
                target_role_filter=target_role,
            )

            if not n_x:
                label_source = source_role or "any"
                label_target = target_role or "any"
                print(
                    "No source annotations with positive occurrences were "
                    f"found for source_role={label_source}, "
                    f"target_role={label_target}.",
                )
                continue

            scope_prefix_name = _build_scope_prefix_name(
                prefix.name,
                source_role,
                target_role,
            )
            scope_prefix = prefix.with_name(scope_prefix_name)

            for k in ks:
                counts_k = counts_by_k.get(k, {})
                trials_k = trials_by_k.get(k, {})
                successes_k = successes_by_k.get(k, {})
                triple_successes_k = triple_successes_by_k.get(k, {})
                matrix_csv = scope_prefix.with_name(
                    f"{scope_prefix.name}_K{k}_matrix.csv"
                )
                top_pairs_csv = scope_prefix.with_name(
                    f"{scope_prefix.name}_K{k}_top_pairs.csv"
                )
                triple_csv = scope_prefix.with_name(
                    f"{scope_prefix.name}_K{k}_triples_cowindow.csv"
                )

                _write_full_matrix_csv(
                    matrix_csv,
                    k,
                    source_ids,
                    annotation_ids,
                    n_x,
                    counts_k,
                    base_messages_with_y,
                    base_total_scoped_y,
                    trials_k,
                    successes_k,
                )
                _write_top_pairs_csv(
                    top_pairs_csv,
                    k,
                    source_ids,
                    annotation_ids,
                    n_x,
                    counts_k,
                    base_messages_with_y,
                    base_total_scoped_y,
                    trials_k=trials_k,
                    successes_k=successes_k,
                    min_nx=args.min_nx,
                    min_cooccurrences=args.min_cooccurrences,
                )
                _write_triple_cowindow_csv(
                    triple_csv,
                    k,
                    base_messages_with_y=base_messages_with_y,
                    base_total_scoped_y=base_total_scoped_y,
                    trials_k=trials_k,
                    successes_k=successes_k,
                    triple_successes_k=triple_successes_k,
                )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for computing sequential annotation dynamics."""

    args = parse_args(argv)
    return run_preprocessed_annotation_job(args, _run_sequential_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
