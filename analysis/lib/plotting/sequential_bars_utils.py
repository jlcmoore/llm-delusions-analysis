"""Shared helpers for per-target sequential dynamics bar plots.

This module centralises utilities that are used by both the single-panel
and paired-panel sequential annotation bar plot scripts. The helpers
cover parsing of target specifications, construction of scope-specific
matrix prefixes, and loading per-target baseline and conditional rates
from precomputed X->Y matrices.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.cli.sequential_dynamics_cli import read_matrix_header
from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.plotting.style import annotation_color_for_label
from analysis.lib.stats.beta_utils import beta_normal_ci


def parse_target_specs(
    raw_targets: Optional[Sequence[str]],
) -> Tuple[list[Tuple[str, Optional[str]]], Dict[str, Optional[str]]]:
    """Return parsed target specifications and per-id role filters.

    Parameters
    ----------
    raw_targets:
        Raw target specification values. Each value may be either a bare
        annotation id (for example, ``theme-awakening-consciousness``) or
        an id plus a role suffix of the form ``id:role`` where ``role`` is
        ``\"user\"`` or ``\"assistant\"``. Role tokens such as ``\"any\"``
        or ``\"both\"`` are treated as unscoped.

    Returns
    -------
    specs:
        Ordered list of (annotation_id, role) pairs where ``role`` is
        ``None`` for unscoped targets.
    roles_by_id:
        Mapping from annotation id to its resolved role filter. Each
        annotation id may appear at most once with a consistent role;
        conflicting role suffixes raise ``ValueError``.
    """

    specs: list[Tuple[str, Optional[str]]] = []
    roles_by_id: Dict[str, Optional[str]] = {}

    if not raw_targets:
        return specs, roles_by_id

    for value in raw_targets:
        token = value.strip()
        if not token:
            raise ValueError("Empty target specification value is not allowed")

        # Support absence-based targets specified as ``not:<id>`` without a
        # role suffix. These are interpreted as a synthetic target id whose
        # metrics are derived downstream from the corresponding positive
        # annotation where applicable.
        if token.startswith("not:") and token.count(":") == 1:
            annotation_id = token
            role = None
        elif ":" in token:
            id_raw, role_raw = token.split(":", 1)
            annotation_id = id_raw.strip()
            role_token = role_raw.strip().lower()
            if not annotation_id:
                raise ValueError(
                    f"Invalid target specification value {value!r}; "
                    "annotation id must be non-empty",
                )
            if role_token in {"", "all", "any", "both", "auto"}:
                role = None
            elif role_token in {"user", "assistant"}:
                role = role_token
            else:
                raise ValueError(
                    f"Invalid role {role_raw!r} for target-id {annotation_id!r}; "
                    "expected 'user', 'assistant', or a synonym for both.",
                )
        else:
            annotation_id = token
            role = None

        previous = roles_by_id.get(annotation_id)
        if previous is not None and previous != role:
            raise ValueError(
                f"Conflicting roles for target-id {annotation_id!r}; "
                "received both scoped and unscoped values.",
            )
        roles_by_id[annotation_id] = role
        specs.append((annotation_id, role))

    return specs, roles_by_id


def format_target_tick_labels(targets: Sequence[str]) -> list[str]:
    """Return short tick labels for per-target axes.

    Parameters
    ----------
    targets:
        Sequence of target identifiers, including optional ``not:`` prefixes
        or role suffixes.

    Returns
    -------
    list[str]
        Short labels suitable for x-axis tick labels.
    """

    labels: list[str] = []
    for target in targets:
        label = format_annotation_display_label(target)
        labels.append(label.replace("-", "-\n"))
    return labels


def set_effect_source_ylabel(
    axis: plt.Axes,
    effect_source: str,
    k: int,
    *,
    fontsize: int,
) -> None:
    """Set the y-axis label for a sequential plot based on effect source."""

    if effect_source == "beta":
        axis.set_ylabel(
            f"P(occurs ≥ 1x in {k} msgs.)",
            fontsize=fontsize,
        )
    else:
        axis.set_ylabel(
            "Per-message rate of Y (per-step within K)",
            fontsize=fontsize,
        )


def compute_pair_figure_size(
    left_count: int,
    right_count: int,
    *,
    base_width: float = 0.5,
    per_target_width: float = 0.9,
    figure_height: float = 2.0,
) -> tuple[float, float, float, float]:
    """Return figure sizing values for paired per-target plots."""

    left_width = base_width + per_target_width * float(max(left_count, 1))
    right_width = base_width + per_target_width * float(max(right_count, 1))
    figure_width = left_width + right_width
    return left_width, right_width, figure_width, figure_height


def create_pair_axes(
    left_count: int,
    right_count: int,
    *,
    sharey: bool,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Return a paired-axes figure sized for two per-target panels."""

    left_width, right_width, figure_width, figure_height = compute_pair_figure_size(
        left_count,
        right_count,
    )
    figure, (axis_left, axis_right) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        sharey=sharey,
        gridspec_kw={"width_ratios": [left_width, right_width]},
    )
    return figure, axis_left, axis_right


def set_pair_titles(
    axis_left: plt.Axes,
    axis_right: plt.Axes,
    *,
    left_source_id: str,
    right_source_id: str,
    fontsize: int = 9,
) -> None:
    """Set per-panel titles for a paired per-target plot."""

    axis_left.set_title(
        f"Source: {format_annotation_display_label(left_source_id)}",
        fontsize=fontsize,
    )
    axis_right.set_title(
        f"Source: {format_annotation_display_label(right_source_id)}",
        fontsize=fontsize,
    )


def init_pair_legend_state(
    axis_left: plt.Axes,
    axis_right: plt.Axes,
    *,
    left_source_id: str,
    right_source_id: str,
    fontsize: int = 9,
) -> tuple[list[plt.Artist], list[str], Set[str]]:
    """Set pair titles and return empty legend containers."""

    set_pair_titles(
        axis_left,
        axis_right,
        left_source_id=left_source_id,
        right_source_id=right_source_id,
        fontsize=fontsize,
    )
    return [], [], set()


def add_shared_legend(
    figure: plt.Figure,
    *,
    legend_handles: Sequence[plt.Artist],
    legend_labels: Sequence[str],
    ncol: int,
    legend_kwargs: Dict[str, object],
    fontsize: int = 8,
) -> None:
    """Add a figure-level legend with shared formatting."""

    figure.legend(
        handles=legend_handles,
        labels=legend_labels,
        ncol=ncol,
        fontsize=fontsize,
        **legend_kwargs,
    )


def render_pair_legend(
    figure: plt.Figure,
    *,
    legend_handles: Sequence[plt.Artist],
    legend_labels: Sequence[str],
    compact_threshold: int,
    lower_loc: str,
    lower_anchor: tuple[float, float],
    upper_loc: str,
    upper_anchor: tuple[float, float],
    upper_ncol: int,
    fontsize: int = 8,
) -> int:
    """Render a figure legend with configurable layout thresholds."""

    if not legend_handles:
        return 0

    label_count = len(legend_labels)
    if label_count <= compact_threshold:
        ncol = label_count
        legend_kwargs = {"loc": lower_loc, "bbox_to_anchor": lower_anchor}
    else:
        ncol = upper_ncol
        legend_kwargs = {"loc": upper_loc, "bbox_to_anchor": upper_anchor}

    add_shared_legend(
        figure,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        ncol=ncol,
        legend_kwargs=legend_kwargs,
        fontsize=fontsize,
    )
    return label_count


def load_pairwise_window_counts(
    matrix_path: Path,
    k: int,
    source_id: str,
    target_ids: Sequence[str],
) -> Dict[str, Tuple[int, int]]:
    """Return per-target (trials, successes) for X->target K-windows."""

    resolved = matrix_path.expanduser().resolve()
    if not resolved.is_file():
        return {}

    wanted = set(target_ids)
    counts: Dict[str, Tuple[int, int]] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id:
                continue
            target = row.get("Y")
            if target not in wanted:
                continue
            try:
                trials = int(row.get("beta_trials", "0"))
            except ValueError:
                trials = 0
            try:
                successes = int(row.get("beta_successes", "0"))
            except ValueError:
                successes = 0
            counts[target] = (trials, successes)

    return counts


def load_triple_window_counts(
    triples_path: Path,
    k: int,
    source_id: str,
    cond_id: str,
    z_ids: Set[str],
) -> Dict[str, Tuple[int, int]]:
    """Return per-Z (trials_XY, successes_XYZ) for X,Y->Z K-windows."""

    counts: Dict[str, Tuple[int, int]] = {}
    for row in iter_triples_rows(triples_path, k, source_id, cond_id):
        z_value = row.get("Z")
        if z_value not in z_ids:
            continue
        try:
            trials_xy = int(row.get("trials_XY", "0"))
        except ValueError:
            trials_xy = 0
        try:
            successes_xyz = int(row.get("successes_XYZ", "0"))
        except ValueError:
            successes_xyz = 0
        counts[z_value] = (trials_xy, successes_xyz)

    return counts


def load_triple_beta_stats(
    triples_path: Path,
    k: int,
    source_id: str,
    cond_id: str,
    z_ids: Set[str],
) -> Dict[str, Tuple[float, float, float]]:
    """Return per-Z (mean, ci_low, ci_high) for X,Y->Z Beta posteriors."""

    stats: Dict[str, Tuple[float, float, float]] = {}
    for row in iter_triples_rows(triples_path, k, source_id, cond_id):
        z_value = row.get("Z")
        if z_value not in z_ids:
            continue
        try:
            mean = float(row.get("beta_p_window_given_XY", "0"))
        except (TypeError, ValueError):
            mean = 0.0
        try:
            ci_low = float(row.get("beta_p_window_ci_low", "0"))
        except (TypeError, ValueError):
            ci_low = 0.0
        try:
            ci_high = float(row.get("beta_p_window_ci_high", "0"))
        except (TypeError, ValueError):
            ci_high = 0.0
        stats[z_value] = (mean, ci_low, ci_high)

    return stats


def iter_triples_rows(
    triples_path: Path,
    k: int,
    source_id: str,
    cond_id: str,
) -> List[Dict[str, str]]:
    """Return matching rows from a co-window triples CSV."""

    resolved = triples_path.expanduser().resolve()
    if not resolved.is_file():
        return []

    rows: List[Dict[str, str]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                row_k = int(row.get("K", "0"))
            except ValueError:
                continue
            if row_k != k:
                continue
            if row.get("X") != source_id or row.get("Y") != cond_id:
                continue
            rows.append(row)
    return rows


def _annotate_or_step(
    axis: plt.Axes,
    *,
    x_position: float,
    y_start: float,
    y_end: float,
    magnitude: float,
    horizontal_alignment: str,
    text_color: str,
    show_text: bool = True,
) -> None:
    """Draw an odds- or risk-ratio arrow and label between two points."""

    midpoint_y = (y_start + y_end) / 2.0
    axis.annotate(
        "",
        xy=(x_position, y_end),
        xytext=(x_position, y_start),
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
            "lw": 0.8,
        },
        zorder=2,
    )
    if show_text:
        axis.text(
            x_position + (0.05 if horizontal_alignment == "left" else -0.05),
            midpoint_y,
            f"{magnitude:.1f}x",
            fontsize=7,
            ha=horizontal_alignment,
            va="center",
            color=text_color,
        )


def plot_triple_panel_on_axis(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    magnitude_metric: str,
    cond_id: str,
    triple_targets: Set[str],
    pairwise_counts: Dict[str, Tuple[int, int]],
    triple_counts: Dict[str, Tuple[int, int]],
    triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None,
    show_effect_annotations: bool = True,
) -> Tuple[plt.Container, plt.Container, plt.Container]:
    """Render a triple-conditional profile onto an existing axis."""

    baseline_artist, conditional_artist_pairwise = plot_per_target_profile_on_axis(
        axis,
        panel=panel,
        k=k,
        effect_source=effect_source,
        magnitude_metric=magnitude_metric,
        add_arrows=False,
        add_ylabel=False,
    )

    x_positions = np.arange(len(panel.targets), dtype=float)

    conditional_y = np.full(len(panel.targets), np.nan, dtype=float)
    conditional_yerr = np.zeros((2, len(panel.targets)), dtype=float)
    use_beta = effect_source == "beta" and triple_beta is not None
    for idx, target in enumerate(panel.targets):
        if target not in triple_targets:
            continue
        if use_beta and target in triple_beta:
            mean, ci_low, ci_high = triple_beta[target]
            conditional_y[idx] = mean
            conditional_yerr[0, idx] = mean - ci_low
            conditional_yerr[1, idx] = ci_high - mean
        else:
            triple_trials, triple_successes = triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            conditional_y[idx] = float(triple_successes) / float(triple_trials)
    cond_color = annotation_color_for_label(cond_id)
    triple_indices = [
        idx
        for idx, target in enumerate(panel.targets)
        if target in triple_targets and np.isfinite(conditional_y[idx])
    ]
    conditional_artist_triple = axis.errorbar(
        x_positions[triple_indices],
        conditional_y[triple_indices],
        yerr=conditional_yerr[:, triple_indices],
        fmt="o",
        color=cond_color,
        ecolor=cond_color,
        elinewidth=1.0,
        capsize=3.0,
        zorder=5,
    )

    if show_effect_annotations:
        for idx, target in enumerate(panel.targets):
            base_p = float(panel.baseline_means[target])
            pair_trials, pair_successes = pairwise_counts.get(target, (0, 0))
            if pair_trials <= 0:
                continue
            p_x = float(pair_successes) / float(pair_trials)

            if 0.0 < base_p < 1.0 and 0.0 < p_x < 1.0 and base_p != 1.0:
                base_odds = base_p / (1.0 - base_p)
                cond_odds = p_x / (1.0 - p_x)
                if base_odds > 0.0:
                    or_global = cond_odds / base_odds
                    _annotate_or_step(
                        axis,
                        x_position=float(x_positions[idx]),
                        y_start=base_p,
                        y_end=p_x,
                        magnitude=or_global,
                        horizontal_alignment="left",
                        text_color=annotation_color_for_label(panel.source_id),
                        show_text=show_effect_annotations,
                    )

            if target not in triple_targets:
                continue
            triple_trials, triple_successes = triple_counts.get(target, (0, 0))
            if triple_trials <= 0:
                continue
            p_xy = float(triple_successes) / float(triple_trials)
            if 0.0 < p_x < 1.0 and 0.0 < p_xy < 1.0 and p_x != 1.0:
                base_odds = p_x / (1.0 - p_x)
                cond_odds = p_xy / (1.0 - p_xy)
                if base_odds > 0.0:
                    or_cond = cond_odds / base_odds
                    _annotate_or_step(
                        axis,
                        x_position=float(x_positions[idx]),
                        y_start=p_x,
                        y_end=p_xy,
                        magnitude=or_cond,
                        horizontal_alignment="right",
                        text_color=cond_color,
                        show_text=show_effect_annotations,
                    )

    return baseline_artist, conditional_artist_pairwise, conditional_artist_triple


def build_scope_prefix(
    base_prefix: Path,
    source_role: Optional[str],
    target_role: Optional[str],
) -> Path:
    """Return an output-prefix path augmented with role scope suffixes.

    Parameters
    ----------
    base_prefix:
        Base output prefix path whose ``name`` attribute is used as the
        starting basename.
    source_role:
        Optional role restriction for source annotations X (``\"user\"``,
        ``\"assistant\"``, or ``None`` for any in-scope role).
    target_role:
        Optional role restriction for target annotations Y (``\"user\"``,
        ``\"assistant\"``, or ``None`` for any in-scope role).

    Returns
    -------
    pathlib.Path
        New prefix path with ``\"__source-<role>\"`` and/or
        ``\"__scope-<role>\"`` suffixes appended where applicable.
    """

    name = base_prefix.name
    if source_role in {"user", "assistant"}:
        name = f"{name}__source-{source_role}"
    if target_role in {"user", "assistant"}:
        name = f"{name}__scope-{target_role}"
    return base_prefix.with_name(name)


def load_rows_for_source_and_targets(
    matrix_path: Path,
    k: int,
    source_id: str,
    target_ids: Optional[Sequence[str]],
    effect_source: str,
) -> Tuple[
    List[str],
    Dict[str, float],
    Dict[str, Tuple[float, float]],
    Dict[str, float],
    Dict[str, Tuple[float, float]],
]:
    """Return per-target baseline and conditional rates and intervals.

    Parameters
    ----------
    matrix_path:
        Path to a single-K X->Y matrix CSV written by the sequential
        dynamics compute script.
    k:
        Window size K in messages used when computing sequential
        dynamics.
    source_id:
        Source annotation id X whose conditional effects are loaded.
    target_ids:
        Optional collection of target annotation ids Y to filter to.
        When ``None``, all targets for the selected source are loaded.
    effect_source:
        Effect-size source to use, matching the plotting scripts
        (``\"beta\"`` or ``\"enrichment\"``).

    Returns
    -------
    targets:
        Alphabetically ordered list of target annotation ids present in
        the matrix after filtering.
    baseline_means:
        Mapping from target id to global baseline mean.
    baseline_cis:
        Mapping from target id to (low, high) baseline confidence
        interval bounds.
    conditional_means:
        Mapping from target id to conditional mean following X.
    conditional_cis:
        Mapping from target id to (low, high) conditional confidence
        interval bounds.
    """

    resolved = matrix_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Matrix CSV for K={k} not found at {resolved}")

    # Determine which underlying positive annotation ids need to be loaded
    # from the matrix so that absence-based targets of the form ``not:<id>``
    # can be derived afterwards.
    positive_targets_filter: Optional[set[str]]
    absence_targets: Dict[str, str] = {}
    if target_ids is not None:
        positive_targets_filter = set()
        for tid in target_ids:
            if tid.startswith("not:"):
                base_id = tid[len("not:") :].strip()
                if base_id:
                    absence_targets[tid] = base_id
                    positive_targets_filter.add(base_id)
            else:
                positive_targets_filter.add(tid)
    else:
        positive_targets_filter = None

    baseline_means: Dict[str, float] = {}
    baseline_cis: Dict[str, Tuple[float, float]] = {}
    conditional_means: Dict[str, float] = {}
    conditional_cis: Dict[str, Tuple[float, float]] = {}

    with resolved.open("r", encoding="utf-8") as handle:
        _header, indices = read_matrix_header(handle)
        x_index = indices.get("X")
        y_index = indices.get("Y")
        p_any_index = indices.get("p_y_anywhere")
        messages_index = indices.get("messages_with_y")
        scoped_index = indices.get("total_scoped_y")
        p_cond_index = indices.get("p_y_within_K_given_X")
        c_k_index = indices.get("C_K")
        n_x_index = indices.get("N_X")
        beta_mean_index = indices.get("beta_p_window_given_X")
        beta_low_index = indices.get("beta_p_window_ci_low")
        beta_high_index = indices.get("beta_p_window_ci_high")
        beta_global_index = indices.get("beta_p_window_global")
        if x_index is None or y_index is None:
            raise ValueError(
                f"Matrix CSV for K={k} is missing required columns at {resolved}"
            )
        if p_any_index is None or messages_index is None or scoped_index is None:
            raise ValueError(
                f"Matrix CSV for K={k} is missing base-rate columns at {resolved}"
            )
        if effect_source == "beta":
            required = [
                beta_mean_index,
                beta_low_index,
                beta_high_index,
                beta_global_index,
            ]
            if any(index is None for index in required):
                raise ValueError(
                    "Requested effect-source 'beta' but required Beta columns "
                    f"were not found in {resolved}"
                )
        else:
            if p_cond_index is None or c_k_index is None or n_x_index is None:
                raise ValueError(
                    "Requested effect-source 'enrichment' but count-based "
                    f"columns were not found in {resolved}"
                )

        for line in handle:
            parts = line.rstrip("\n").split(",")
            x_value = parts[x_index]
            if x_value != source_id:
                continue
            y_value = parts[y_index]
            if (
                positive_targets_filter is not None
                and y_value not in positive_targets_filter
            ):
                continue

            messages_with_y = float(parts[messages_index])
            scoped_y = float(parts[scoped_index])
            if scoped_y <= 0.0:
                mean_base_p = 0.0
                base_low_p = 0.0
                base_high_p = 0.0
            else:
                alpha_base = messages_with_y + 1.0
                beta_base = scoped_y - messages_with_y + 1.0
                mean_base_p, base_low_p, base_high_p = beta_normal_ci(
                    alpha_base,
                    beta_base,
                )

            if effect_source == "beta":
                mean_cond = float(parts[beta_mean_index])
                low_cond = float(parts[beta_low_index])
                high_cond = float(parts[beta_high_index])
                mean_base = 1.0 - (1.0 - mean_base_p) ** float(k)
                low_base = 1.0 - (1.0 - base_low_p) ** float(k)
                high_base = 1.0 - (1.0 - base_high_p) ** float(k)
            else:
                mean_base = mean_base_p
                low_base = base_low_p
                high_base = base_high_p

                p_cond = float(parts[p_cond_index])
                c_k = float(parts[c_k_index])
                n_x = float(parts[n_x_index])
                if k > 0:
                    mean_cond = p_cond / float(k)
                    n_denom = max(1.0, n_x * float(k))
                else:
                    mean_cond = p_cond
                    n_denom = max(1.0, n_x)
                alpha_cond = c_k + 1.0
                beta_cond = max(1.0, n_denom - c_k) + 1.0
                _, low_cond, high_cond = beta_normal_ci(alpha_cond, beta_cond)

            baseline_means[y_value] = mean_base
            baseline_cis[y_value] = (low_base, high_base)
            conditional_means[y_value] = mean_cond
            conditional_cis[y_value] = (low_cond, high_cond)

    # Derive absence-based target metrics where requested. For a target id
    # ``not:<id>`` we treat the baseline and conditional probabilities as
    # complements of the corresponding positive annotation's probabilities.
    # Confidence intervals are mirrored accordingly.
    for absence_id, base_id in absence_targets.items():
        if base_id not in baseline_means or base_id not in conditional_means:
            continue
        base_mean = float(baseline_means[base_id])
        base_low, base_high = baseline_cis[base_id]
        cond_mean = float(conditional_means[base_id])
        cond_low, cond_high = conditional_cis[base_id]

        # Complement means.
        absence_base_mean = 1.0 - base_mean
        absence_cond_mean = 1.0 - cond_mean

        # Mirror intervals so that low/high remain ordered.
        absence_base_low = max(0.0, 1.0 - base_high)
        absence_base_high = min(1.0, 1.0 - base_low)
        absence_cond_low = max(0.0, 1.0 - cond_high)
        absence_cond_high = min(1.0, 1.0 - cond_low)

        baseline_means[absence_id] = absence_base_mean
        baseline_cis[absence_id] = (absence_base_low, absence_base_high)
        conditional_means[absence_id] = absence_cond_mean
        conditional_cis[absence_id] = (absence_cond_low, absence_cond_high)

    ordered_targets = sorted(
        baseline_means.keys(),
        key=lambda name: name.lower(),
    )
    return (
        ordered_targets,
        baseline_means,
        baseline_cis,
        conditional_means,
        conditional_cis,
    )


class PanelMetrics(NamedTuple):
    """Container for per-source per-target sequential dynamics metrics."""

    source_id: str
    targets: Sequence[str]
    baseline_means: Dict[str, float]
    baseline_cis: Dict[str, Tuple[float, float]]
    conditional_means: Dict[str, float]
    conditional_cis: Dict[str, Tuple[float, float]]


def compute_effect_metrics(
    panel: PanelMetrics,
) -> List[Tuple[str, float, float, float, float, float]]:
    """Return per-target baseline/conditional metrics, RR, and OR.

    Each returned tuple has the form:

    (target_label, baseline, conditional, delta, risk_ratio, odds_ratio)

    where ``delta = conditional - baseline``. When the risk or odds
    ratio is not well defined (for example, due to zero or extreme base
    probabilities), the corresponding value is ``float('nan')``.
    """

    results: List[Tuple[str, float, float, float, float, float]] = []
    for target in panel.targets:
        base = float(panel.baseline_means[target])
        cond = float(panel.conditional_means[target])
        delta = cond - base

        if base > 0.0:
            risk_ratio = cond / base
        else:
            risk_ratio = float("nan")

        if 0.0 < base < 1.0 and 0.0 < cond < 1.0 and base != 1.0:
            base_odds = base / (1.0 - base)
            cond_odds = cond / (1.0 - cond)
            if base_odds > 0.0:
                odds_ratio = cond_odds / base_odds
            else:
                odds_ratio = float("nan")
        else:
            odds_ratio = float("nan")

        results.append((target, base, cond, delta, risk_ratio, odds_ratio))

    return results


def compute_triple_effect_rows(
    panel: PanelMetrics,
    triple_targets: Set[str],
    pairwise_counts: Dict[str, Tuple[int, int]],
    triple_counts: Dict[str, Tuple[int, int]],
) -> List[Tuple[str, str, float, float, float, float, float, int, int]]:
    """Return per-target rows for pairwise and triple-conditioned summaries."""

    rows: List[Tuple[str, str, float, float, float, float, float, int, int]] = []

    for target in panel.targets:
        base_p = float(panel.baseline_means[target])
        pair_trials, pair_successes = pairwise_counts.get(target, (0, 0))
        p_x = float(pair_successes) / float(pair_trials) if pair_trials > 0 else 0.0

        delta_pair = p_x - base_p
        if base_p > 0.0:
            rr_pair = p_x / base_p
        else:
            rr_pair = float("nan")
        if 0.0 < base_p < 1.0 and 0.0 < p_x < 1.0 and base_p != 1.0:
            base_odds = base_p / (1.0 - base_p)
            cond_odds = p_x / (1.0 - p_x)
            or_pair = cond_odds / base_odds if base_odds > 0.0 else float("nan")
        else:
            or_pair = float("nan")
        rows.append(
            (
                target,
                "pair",
                base_p,
                p_x,
                delta_pair,
                rr_pair,
                or_pair,
                pair_successes,
                pair_trials,
            )
        )

        if target not in triple_targets:
            continue
        triple_trials, triple_successes = triple_counts.get(target, (0, 0))
        if triple_trials <= 0:
            continue
        p_xy = float(triple_successes) / float(triple_trials)
        delta_triple = p_xy - p_x
        if p_x > 0.0:
            rr_triple = p_xy / p_x
        else:
            rr_triple = float("nan")
        if 0.0 < p_x < 1.0 and 0.0 < p_xy < 1.0 and p_x != 1.0:
            base_odds = p_x / (1.0 - p_x)
            cond_odds = p_xy / (1.0 - p_xy)
            or_triple = cond_odds / base_odds if base_odds > 0.0 else float("nan")
        else:
            or_triple = float("nan")
        rows.append(
            (
                target,
                "triple",
                p_x,
                p_xy,
                delta_triple,
                rr_triple,
                or_triple,
                triple_successes,
                triple_trials,
            )
        )

    return rows


def validate_pairwise_panels(
    left_panel: PanelMetrics,
    right_panel: PanelMetrics,
) -> bool:
    """Return ``True`` when both panels have targets; print otherwise."""

    if not left_panel.targets or not right_panel.targets:
        print("At least one panel had no targets; nothing to plot.")
        return False
    return True


def print_pairwise_effect_summary(
    panel: PanelMetrics,
    *,
    panel_label: str,
    leading_newline: bool = False,
) -> None:
    """Print a per-target K-window summary table for a panel.

    Parameters
    ----------
    panel:
        Per-source panel metrics whose baseline and conditional rates
        are summarised.
    panel_label:
        Human-readable label for the panel, included in the printed
        header (for example, ``\"left panel\"`` or ``\"right panel\"``).
    leading_newline:
        When ``True``, emit a blank line before the header so that
        consecutive tables are visually separated.
    """

    if leading_newline:
        print()

    print(f"Per-target K-window summary ({panel_label}, source={panel.source_id}):")
    print(
        f"{'target':40s} {'type':>8s} "
        f"{'baseline':>10s} {'conditional':>12s} "
        f"{'delta':>10s} {'RR':>10s} {'OR':>10s}"
    )
    for (
        target,
        base,
        cond,
        delta,
        risk_ratio,
        odds_ratio,
    ) in compute_effect_metrics(panel):
        print(
            f"{target:40s} {'pair':>8s} "
            f"{base:10.3f} {cond:12.3f} "
            f"{delta:10.3f} {risk_ratio:10.3f} {odds_ratio:10.3f}"
        )


def build_panel_metrics(
    *,
    output_prefix: Path,
    k: int,
    source_id_raw: str,
    target_ids_raw: Optional[Sequence[str]],
    effect_source: str,
    order_by_effect_size: bool,
) -> PanelMetrics:
    """Return ordered targets and metrics for a single source panel.

    This helper encapsulates the shared logic used by sequential
    per-target plots to:

    * Parse optional role suffixes from ``source_id_raw``.
    * Load per-target baseline and conditional rates from the
      appropriate scope-specific matrices.
    * Combine role-scoped metrics into a single set of target keys.
    * Derive a final target ordering based on CLI preferences.

    Parameters
    ----------
    output_prefix:
        Base path prefix for sequential dynamics CSV tables.
    k:
        Window size K in messages.
    source_id_raw:
        Raw ``--source-id`` value, optionally including a ``:role``
        suffix such as ``grand-significance:assistant``.
    target_ids_raw:
        Optional list of raw ``--target-id`` values, each of which may
        include an optional ``:role`` suffix.
    effect_source:
        Effect-size source for the y-axis (``\"beta\"`` or
        ``\"enrichment\"``).
    order_by_effect_size:
        When ``True``, order targets by absolute effect-size
        difference. When ``False``, preserve CLI order when explicit
        targets are supplied or fall back to alphabetical order.

    Returns
    -------
    source_id:
        Parsed source annotation id without any role suffix.
    targets:
        Ordered list of target keys, each potentially including a
        ``\":role\"`` suffix.
    baseline_means:
        Mapping from target key to global baseline mean.
    baseline_cis:
        Mapping from target key to (low, high) baseline confidence
        interval bounds.
    conditional_means:
        Mapping from target key to conditional mean following the
        source annotation.
    conditional_cis:
        Mapping from target key to (low, high) conditional confidence
        interval bounds.
    """

    raw_source = source_id_raw.strip()
    source_role: Optional[str] = None
    # Support absence-based sources such as ``not:user-suicidal-intent``.
    # Treat an optional trailing ``:role`` suffix as a role filter while
    # preserving the full source id (including ``not:``) for matrix lookups.
    if ":" in raw_source and not raw_source.startswith("not:"):
        src_id_raw, src_role_raw = raw_source.split(":", 1)
        source_id = src_id_raw.strip()
        src_role_token = src_role_raw.strip().lower()
        if not source_id:
            raise ValueError(
                f"Invalid source-id value {source_id_raw!r}; "
                "annotation id must be non-empty.",
            )
        if src_role_token in {"", "all", "any", "both", "auto"}:
            source_role = None
        elif src_role_token in {"user", "assistant"}:
            source_role = src_role_token
        else:
            raise ValueError(
                f"Invalid role {src_role_raw!r} for source-id {source_id!r}; "
                "expected 'user', 'assistant', or a synonym for both.",
            )
    else:
        # Either a bare id or an absence-based id such as ``not:<id>``;
        # role filters, when needed, are specified via --source-role on
        # the compute script rather than as part of the id.
        source_id = raw_source

    if target_ids_raw is None:
        scope_prefix = build_scope_prefix(output_prefix, source_role, None)
        matrix_path = scope_prefix.with_name(f"{scope_prefix.name}_K{k}_matrix.csv")
        (
            targets,
            baseline_means,
            baseline_cis,
            conditional_means,
            conditional_cis,
        ) = load_rows_for_source_and_targets(
            matrix_path,
            k,
            source_id,
            None,
            effect_source,
        )
    else:
        target_specs, roles_by_id = parse_target_specs(target_ids_raw)

        by_role: Dict[Optional[str], List[str]] = {}
        for annotation_id, role in target_specs:
            by_role.setdefault(role, []).append(annotation_id)

        baseline_means = {}
        baseline_cis = {}
        conditional_means = {}
        conditional_cis = {}

        for role, annotation_ids in by_role.items():
            scope_prefix = build_scope_prefix(output_prefix, source_role, role)
            matrix_path = scope_prefix.with_name(f"{scope_prefix.name}_K{k}_matrix.csv")
            (
                _role_targets,
                role_baseline_means,
                role_baseline_cis,
                role_conditional_means,
                role_conditional_cis,
            ) = load_rows_for_source_and_targets(
                matrix_path,
                k,
                source_id,
                annotation_ids,
                effect_source,
            )

            for annotation_id in annotation_ids:
                if annotation_id not in role_baseline_means:
                    continue
                role_token = roles_by_id.get(annotation_id)
                if role_token is None:
                    key = annotation_id
                else:
                    key = f"{annotation_id}:{role_token}"
                if key in baseline_means:
                    raise ValueError(
                        f"Duplicate target label {key!r} encountered while "
                        "combining role-specific matrices.",
                    )
                baseline_means[key] = role_baseline_means[annotation_id]
                baseline_cis[key] = role_baseline_cis[annotation_id]
                conditional_means[key] = role_conditional_means[annotation_id]
                conditional_cis[key] = role_conditional_cis[annotation_id]

        if not baseline_means:
            raise ValueError(
                "No matching rows were found across scope-specific matrices; "
                "nothing to plot.",
            )

        targets = sorted(baseline_means.keys(), key=lambda name: name.lower())

    if not targets:
        raise ValueError(
            "No matching rows were found in the selected matrix file; "
            "nothing to plot.",
        )

    if target_ids_raw is not None and not order_by_effect_size:
        try:
            cli_specs, _roles_by_id = parse_target_specs(target_ids_raw)
        except ValueError:
            cli_specs = []
        ordered: List[str] = []
        for annotation_id, role in cli_specs:
            if role is None:
                key = annotation_id
            else:
                key = f"{annotation_id}:{role}"
            if key in baseline_means:
                ordered.append(key)
        targets = ordered
    elif order_by_effect_size:
        targets = sorted(
            baseline_means.keys(),
            key=lambda name: abs(
                float(conditional_means[name]) - float(baseline_means[name])
            ),
            reverse=True,
        )

    return PanelMetrics(
        source_id=source_id,
        targets=targets,
        baseline_means=baseline_means,
        baseline_cis=baseline_cis,
        conditional_means=conditional_means,
        conditional_cis=conditional_cis,
    )


def format_annotation_display_label(annotation_id: str) -> str:
    """Return a display label for an annotation id.

    Parameters
    ----------
    annotation_id:
        Raw annotation identifier, which may be a bare id such as
        ``assistant-validates-self-harm-feelings``, an id with a role
        suffix of the form ``id:role``, or an absence-based id of the
        form ``not:<id>`` with an optional trailing role.

    Returns
    -------
    str
        Human-readable label suitable for legends and titles. For
        absence-based ids, the label renders as ``\\mathbf{not}`` in
        mathtext followed by the shortened annotation identifier. A
        ``(U)`` or ``(A)`` suffix is appended for user- or
        assistant-scoped roles respectively.
    """

    raw = annotation_id.strip()
    negated = False

    if raw.startswith("not:"):
        negated = True
        remainder = raw[len("not:") :].strip()
    else:
        remainder = raw

    if ":" in remainder:
        base_id, role = remainder.split(":", 1)
    else:
        base_id, role = remainder, ""

    base_label = shorten_annotation_label(base_id)
    if role == "user":
        base_label = f"{base_label} (user)"
    elif role == "assistant":
        base_label = f"{base_label} (bot)"

    if negated:
        return r"$\mathbf{not}$ " + base_label
    return base_label


def plot_per_target_profile_on_axis(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    magnitude_metric: str = "odds",
    add_ylabel: bool = True,
    baseline_color: str = "tab:gray",
    baseline_alpha: float = 1.0,
    conditional_color: Optional[str] = None,
    conditional_alpha: float = 1.0,
    add_arrows: bool = True,
    show_effect_labels: bool = True,
) -> Tuple[plt.Container, plt.Container]:
    """Render a per-target sequential profile onto an existing axis.

    This helper is shared between single-panel and paired-panel plotting
    scripts so that styling and labelling remain consistent.

    Parameters
    ----------
    axis:
        Matplotlib axis onto which the profile is rendered.
    k:
        Window size K in messages used when computing sequential
        dynamics.
    source_id:
        Raw source-id string used for labelling the conditional series
        in the legend.
    targets:
        Ordered target keys to plot. Each key may include a ``\":role\"``
        suffix, for example ``\"grand-significance:assistant\"``.
    baseline_means:
        Mapping from target key to global baseline mean.
    baseline_cis:
        Mapping from target key to (low, high) baseline confidence
        interval bounds.
    conditional_means:
        Mapping from target key to conditional mean following the
        source annotation.
    conditional_cis:
        Mapping from target key to (low, high) conditional confidence
        interval bounds.
    effect_source:
        Effect-size source in use (``\"beta\"`` or ``\"enrichment\"``),
        controlling the y-axis label.
    baseline_color:
        Matplotlib color used for the global baseline series.
    baseline_alpha:
        Alpha value used for the baseline series.
    conditional_color:
        Optional Matplotlib color used for the conditional series. When
        omitted, a label-dependent colour derived from the source id is
        used.
    conditional_alpha:
        Alpha value used for the conditional series.
    add_arrows:
        When ``True``, draw per-target arrows and effect annotations
        between the baseline and conditional series.
    """

    if not panel.targets:
        return None, None

    x_positions = np.arange(len(panel.targets), dtype=float)

    baseline_y = []
    baseline_yerr = [[], []]
    conditional_y = []
    conditional_yerr = [[], []]

    for target in panel.targets:
        base_mean = float(panel.baseline_means[target])
        base_low, base_high = panel.baseline_cis[target]
        cond_mean = float(panel.conditional_means[target])
        cond_low, cond_high = panel.conditional_cis[target]

        baseline_y.append(base_mean)
        baseline_yerr[0].append(base_mean - base_low)
        baseline_yerr[1].append(base_high - base_mean)

        conditional_y.append(cond_mean)
        conditional_yerr[0].append(cond_mean - cond_low)
        conditional_yerr[1].append(cond_high - cond_mean)

    baseline_y_array = np.array(baseline_y, dtype=float)
    baseline_yerr_array = np.array(baseline_yerr, dtype=float)
    conditional_y_array = np.array(conditional_y, dtype=float)
    conditional_yerr_array = np.array(conditional_yerr, dtype=float)

    if conditional_color is None:
        if ":" in panel.source_id:
            base_source_id, _source_role = panel.source_id.split(":", 1)
        else:
            base_source_id = panel.source_id
        conditional_color = annotation_color_for_label(base_source_id)

    baseline_artist = axis.errorbar(
        x_positions,
        baseline_y_array,
        yerr=baseline_yerr_array,
        fmt="o",
        color=baseline_color,
        ecolor=baseline_color,
        alpha=baseline_alpha,
        elinewidth=1.0,
        capsize=3.0,
        zorder=3,
    )

    conditional_artist = axis.errorbar(
        x_positions,
        conditional_y_array,
        yerr=conditional_yerr_array,
        fmt="o",
        color=conditional_color,
        ecolor=conditional_color,
        alpha=conditional_alpha,
        elinewidth=1.0,
        capsize=3.0,
        zorder=4,
    )

    if add_arrows:
        for x_position, base_mean, cond_mean in zip(
            x_positions,
            baseline_y_array,
            conditional_y_array,
        ):
            if not np.isfinite(base_mean) or not np.isfinite(cond_mean):
                continue
            if base_mean == cond_mean:
                continue
            if base_mean <= 0.0:
                continue
            axis.annotate(
                "",
                xy=(x_position, cond_mean),
                xytext=(x_position, base_mean),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "black",
                    "lw": 0.8,
                },
                zorder=2,
            )
            if show_effect_labels:
                magnitude: Optional[float]
                if magnitude_metric == "odds":
                    if (
                        0.0 < base_mean < 1.0
                        and 0.0 < cond_mean < 1.0
                        and base_mean != 1.0
                    ):
                        base_odds = base_mean / (1.0 - base_mean)
                        cond_odds = cond_mean / (1.0 - cond_mean)
                        if base_odds <= 0.0:
                            continue
                        magnitude = cond_odds / base_odds
                    else:
                        continue
                else:
                    if base_mean <= 0.0:
                        continue
                    magnitude = cond_mean / base_mean
                midpoint_y = (base_mean + cond_mean) / 2.0

                is_last = x_position == x_positions[-1]
                x_offset = -0.07 if is_last else 0.05
                ha_val = "right" if is_last else "left"

                axis.text(
                    x_position + x_offset,
                    midpoint_y,
                    f"{magnitude:.1f}x",
                    fontsize=7,
                    ha=ha_val,
                    va="center",
                    color=conditional_color,
                )

    short_labels = []
    for name in panel.targets:
        # Support absence-based targets such as ``not:<id>`` with an
        # optional trailing role suffix. The ``not:`` prefix is treated
        # as a semantic negation marker rather than a role token so
        # that labels render as "not <shortened-id>".
        negated = False
        role = ""
        base_id = name

        if name.startswith("not:"):
            negated = True
            remainder = name[len("not:") :].strip()
            if ":" in remainder:
                base_id, role = remainder.split(":", 1)
            else:
                base_id = remainder
        elif ":" in name:
            base_id, role = name.split(":", 1)

        base_label = shorten_annotation_label(base_id)
        if role == "user":
            base_label = f"{base_label} (user)"
        elif role == "assistant":
            base_label = f"{base_label} (bot)"

        if negated:
            label = r"$\mathbf{not}$ " + base_label
        else:
            label = base_label

        short_labels.append(label.replace("-", "-\n"))
    axis.set_xticks(x_positions)
    axis.set_xticklabels(short_labels, rotation=0, ha="right", fontsize=8)

    if add_ylabel:
        if effect_source == "beta":
            axis.set_ylabel(
                f"P(occurs ≥ 1x in {k} msgs.)",
                fontsize=8,
            )
        else:
            axis.set_ylabel(
                "Per-message rate of Y (per-step within K)",
                fontsize=8,
            )

    return baseline_artist, conditional_artist


__all__ = [
    "PanelMetrics",
    "annotation_color_for_label",
    "parse_target_specs",
    "build_scope_prefix",
    "load_rows_for_source_and_targets",
    "compute_effect_metrics",
    "print_pairwise_effect_summary",
    "build_panel_metrics",
    "plot_per_target_profile_on_axis",
]
