"""Plot sequential annotation dynamics heatmaps from precomputed CSV tables.

This module consumes the X->Y matrix CSVs written by
``compute_sequential_annotation_dynamics.py`` and generates a combined PDF
containing log-enrichment heatmaps for one or more window sizes K.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.cli.sequential_dynamics_cli import (
    add_window_k_argument,
    parse_window_k_arguments,
    read_matrix_header,
)
from analysis.lib.plotting.labels import shorten_annotation_label
from analysis.lib.stats.clustering import cluster_and_order

# Hyperparameters for mapping uncertainty to alpha.
# - BETA_UNCERTAINTY_* control how Beta posterior standard deviations are
#   translated into opacity (smaller SD -> higher alpha).
# - TRIAL_UNCERTAINTY_* control the analogous mapping for enrichment-based
#   effects when we only have trial-count proxies.
BETA_UNCERTAINTY_CENTER = 0.15
BETA_UNCERTAINTY_SCALE = 0.05
BETA_UNCERTAINTY_ALPHA_MIN = 0.1
BETA_UNCERTAINTY_ALPHA_MAX = 1.0

TRIAL_UNCERTAINTY_CENTER = 1.5  # around 10^1.5 ~= 32 trials
TRIAL_UNCERTAINTY_SCALE = 0.5
LOGISTIC_CLAMP = 60.0


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the sequential-dynamics plotting script."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot within-conversation sequential annotation dynamics from "
            "precomputed X->Y matrix CSV tables."
        )
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("analysis") / "data" / "sequential_dynamics" / "base",
        help=(
            "Prefix of sequential dynamics CSV tables produced by "
            "compute_sequential_annotation_dynamics.py. Per-K matrix files "
            "are expected at '<prefix>_K{K}_matrix.csv', for example "
            "'analysis/data/sequential_dynamics/base_K10_matrix.csv'."
        ),
    )
    add_window_k_argument(parser)
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=Path("analysis") / "figures" / "sequential_enrichment_Ks.pdf",
        help=(
            "Destination PDF path for the combined log-enrichment heatmaps "
            "across all requested K values."
        ),
    )
    parser.add_argument(
        "--annotation-id",
        type=str,
        action="append",
        help=(
            "Optional annotation id to include in the heatmaps. May be "
            "provided multiple times; when omitted, all annotations present "
            "in the sequential dynamics matrices are shown."
        ),
    )
    parser.add_argument(
        "--effect-source",
        choices=["beta", "enrichment"],
        default="beta",
        help=(
            "Effect-size source for heatmap colours: 'beta' uses the "
            "Beta-model log-lift (default), while 'enrichment' uses the "
            "original enrichment_K_per_step / enrichment_K columns. The "
            "selected source must be present in the input matrices."
        ),
    )
    parser.add_argument(
        "--no-cluster-order",
        action="store_false",
        dest="cluster_order",
        help=(
            "Disable clustering-based annotation ordering in the heatmaps. "
            "When set, annotations are shown in their canonical sorted order."
        ),
    )
    parser.add_argument(
        "--no-uncertainty-alpha",
        action="store_false",
        dest="uncertainty_alpha",
        help=(
            "Disable mapping Beta-model uncertainty to heatmap alpha. When "
            "omitted, cell opacity reflects the number of Beta trials "
            "supporting each X->Y estimate when available."
        ),
    )
    parser.set_defaults(uncertainty_alpha=True)
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the plotting script."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    args.window_k = parse_window_k_arguments(parser, args.window_k)
    return args


def _load_enrichment_from_matrices(
    prefix: Path,
    ks: Sequence[int],
    effect_source: str,
) -> Tuple[
    List[str],
    Dict[int, Dict[Tuple[str, str], float]],
    Dict[int, Dict[Tuple[str, str], float]],
]:
    """Return ids, per-K enrichment proxies, and Beta trial counts.

    The returned per-K enrichment values are effect-size proxies used for
    log-scale visualisation. When Beta-model columns are available in the
    matrix CSVs, these proxies are derived from the smoothed window-level
    enrichment implied by ``beta_log_lift``. Otherwise, they fall back to the
    original per-step enrichment values.
    """

    annotation_ids: set[str] = set()
    enrichment_by_k: Dict[int, Dict[Tuple[str, str], float]] = {}
    beta_trials_by_k: Dict[int, Dict[Tuple[str, str], float]] = {}

    for k in ks:
        path = prefix.with_name(f"{prefix.name}_K{k}_matrix.csv")
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            print(f"Skipping K={k}: matrix CSV not found at {resolved}")
            continue

        enrichment_for_k: Dict[Tuple[str, str], float] = defaultdict(float)
        trials_for_k: Dict[Tuple[str, str], float] = defaultdict(float)
        with resolved.open("r", encoding="utf-8") as handle:
            _header, indices = read_matrix_header(handle)
            x_index = indices.get("X")
            y_index = indices.get("Y")
            per_step_index = indices.get("enrichment_K_per_step")
            enrichment_index = indices.get("enrichment_K")
            beta_log_lift_index = indices.get("beta_log_lift")
            beta_trials_index = indices.get("beta_trials")
            beta_sd_index = indices.get("beta_p_window_sd")
            if x_index is None or y_index is None:
                raise ValueError(
                    f"Matrix CSV for K={k} is missing required columns at {resolved}"
                )

            if effect_source == "beta":
                if beta_log_lift_index is None:
                    raise ValueError(
                        "Requested effect-source 'beta' but column "
                        "'beta_log_lift' was not found in "
                        f"{resolved}"
                    )
            elif effect_source == "enrichment":
                if per_step_index is None and enrichment_index is None:
                    raise ValueError(
                        "Requested effect-source 'enrichment' but neither "
                        "'enrichment_K_per_step' nor 'enrichment_K' were "
                        f"found in {resolved}"
                    )

            for line in handle:
                parts = line.rstrip("\n").split(",")
                x_value = parts[x_index]
                y_value = parts[y_index]
                annotation_ids.add(x_value)
                annotation_ids.add(y_value)
                enrichment: float
                trials_value: float = 0.0
                sd_value: float = 0.0

                # Select the requested effect-size source for colouring.
                if effect_source == "beta":
                    # Use the smoothed Beta-model log-lift by exponentiating
                    # it to obtain a K-window enrichment ratio.
                    if beta_log_lift_index is None or len(parts) <= beta_log_lift_index:
                        continue
                    beta_log_lift = float(parts[beta_log_lift_index])
                    enrichment = math.exp(beta_log_lift)
                elif effect_source == "enrichment":
                    # Use the original enrichment-based per-step effect size.
                    if per_step_index is not None:
                        if len(parts) <= per_step_index:
                            continue
                        enrichment = float(parts[per_step_index])
                    elif enrichment_index is not None:
                        if len(parts) <= enrichment_index:
                            continue
                        raw_enrichment = float(parts[enrichment_index])
                        effective_k = float(k) if k > 0 else 1.0
                        enrichment = raw_enrichment / effective_k
                    else:
                        continue
                else:
                    enrichment = 0.0

                # Read Beta trial counts and posterior standard deviations
                # when available so that alpha can reflect either Beta-model
                # uncertainty (for effect-source='beta') or a simple
                # trial-count proxy (for effect-source='enrichment').
                if beta_trials_index is not None and len(parts) > beta_trials_index:
                    trials_value = float(parts[beta_trials_index])
                if beta_sd_index is not None and len(parts) > beta_sd_index:
                    sd_value = float(parts[beta_sd_index])

                enrichment_for_k[(x_value, y_value)] = enrichment
                # For beta-sourced effects, use the explicit Beta posterior
                # standard deviation as the uncertainty proxy; when this is
                # zero or missing treat the estimate as fully certain rather
                # than falling back to trial counts. For enrichment-based
                # effects, use the simple trial-count heuristic so that
                # uncertainty is decoupled from the Beta model.
                if effect_source == "beta":
                    trials_for_k[(x_value, y_value)] = sd_value
                else:
                    trials_for_k[(x_value, y_value)] = trials_value

        enrichment_by_k[k] = enrichment_for_k
        beta_trials_by_k[k] = trials_for_k

    ordered_ids = sorted(annotation_ids)
    return ordered_ids, enrichment_by_k, beta_trials_by_k


def _build_log_enrichment_matrix_for_k(
    k: int,
    ids: Sequence[str],
    enrichment_for_k: Mapping[Tuple[str, str], float],
) -> np.ndarray:
    """Return a log2-enrichment matrix for a single K and ordered ids."""

    size = len(ids)
    matrix = np.zeros((size, size), dtype=float)

    for i, source in enumerate(ids):
        for j, target in enumerate(ids):
            enrichment = float(enrichment_for_k.get((source, target), 0.0))
            if enrichment <= 0.0:
                value = 0.0
            else:
                value = math.log(enrichment, 2.0)
            matrix[i, j] = value

    if k == 0 and size > 0:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def _cluster_annotation_order_for_heatmaps(
    ks: Sequence[int],
    annotation_ids: Sequence[str],
    enrichment_by_k: Mapping[int, Mapping[Tuple[str, str], float]],
) -> Tuple[List[str], float]:
    """Return a clustered annotation ordering and global log-scale for heatmaps."""

    ids = list(annotation_ids)
    if not ids:
        return [], 0.0

    features: List[List[float]] = []
    for source in ids:
        row_values: List[float] = []
        for k in ks:
            enrichment_for_k = enrichment_by_k.get(k, {})
            for target in ids:
                enrichment = float(enrichment_for_k.get((source, target), 0.0))
                if enrichment <= 0.0:
                    value = 0.0
                else:
                    value = math.log(enrichment, 2.0)
                row_values.append(value)
        features.append(row_values)

    feature_matrix = np.array(features, dtype=float)
    if feature_matrix.size == 0:
        return ids, 0.0

    max_abs = float(np.max(np.abs(feature_matrix)))
    ordered = cluster_and_order(feature_matrix, ids)
    return ordered, max_abs


def _alpha_from_uncertainty(trials: float) -> float:
    """Return a bounded alpha value derived from Beta posterior uncertainty.

    The input is treated as an uncertainty proxy where larger values indicate
    less certainty. When ``beta_p_window_sd`` is available this value will be
    the posterior standard deviation of the window-level probability; when it
    is not available it falls back to a simple trial-count-based heuristic.
    """

    if trials <= 0.0:
        # Zero uncertainty (or missing) is rendered as fully opaque.
        return 1.0

    # For posterior standard deviations, values tend to lie roughly in
    # [0, 0.25]. Map smaller uncertainty values to higher alpha using the
    # configurable centre and scale.
    x = (BETA_UNCERTAINTY_CENTER - float(trials)) / BETA_UNCERTAINTY_SCALE
    # Clamp x to a safe range to avoid overflow in exp for any unexpected
    # extreme inputs.
    x = max(-LOGISTIC_CLAMP, min(LOGISTIC_CLAMP, x))
    alpha = 1.0 / (1.0 + math.exp(-x))
    return float(
        min(BETA_UNCERTAINTY_ALPHA_MAX, max(BETA_UNCERTAINTY_ALPHA_MIN, alpha))
    )


def _alpha_from_trial_count(trials: float) -> float:
    """Return an alpha value derived from a trial-count proxy.

    This mapping is used when effect sizes are based on enrichment rather than
    the Beta model, mirroring the original behaviour where opacity increased
    with the number of X windows that contributed to each estimate.
    """

    if trials <= 0.0:
        return 0.0

    log_trials = math.log10(trials + 1.0)
    x = (log_trials - TRIAL_UNCERTAINTY_CENTER) / TRIAL_UNCERTAINTY_SCALE
    x = max(-LOGISTIC_CLAMP, min(LOGISTIC_CLAMP, x))
    alpha = 1.0 / (1.0 + math.exp(-x))
    return float(
        min(BETA_UNCERTAINTY_ALPHA_MAX, max(BETA_UNCERTAINTY_ALPHA_MIN, alpha))
    )


def _plot_enrichment_heatmaps_grid(
    output_path: Path,
    ks: Sequence[int],
    annotation_ids: Sequence[str],
    enrichment_by_k: Mapping[int, Mapping[Tuple[str, str], float]],
    *,
    trials_by_k: Optional[Mapping[int, Mapping[Tuple[str, str], float]]] = None,
    ordered_annotations: Optional[Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    use_uncertainty_alpha: bool = True,
    use_beta_uncertainty: bool = True,
    colorbar_label: str = "",
) -> None:
    """Render and save a grid of log-enrichment heatmaps for multiple K values."""

    ids = list(annotation_ids)
    if not ids:
        return
    if ordered_annotations is not None:
        ids = list(ordered_annotations)

    size = len(ids)
    n_panels = len(ks)
    if n_panels <= 0:
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(6.0, 3.0 * float(n_panels)), 6.0),
        constrained_layout=True,
    )
    if n_panels == 1:
        axes_list: Iterable[plt.Axes] = [axes]  # type: ignore[assignment]
    else:
        axes_list = list(axes)  # type: ignore[assignment]

    last_image = None
    short_labels = [shorten_annotation_label(label) for label in ids]

    for index, (k, axis) in enumerate(zip(ks, axes_list)):
        enrichment_for_k = enrichment_by_k.get(k, {})
        matrix = _build_log_enrichment_matrix_for_k(k, ids, enrichment_for_k)
        alphas = None
        if use_uncertainty_alpha and trials_by_k is not None:
            trials_for_k = trials_by_k.get(k, {})
            alphas = np.ones_like(matrix)
            for i, source in enumerate(ids):
                for j, target in enumerate(ids):
                    uncert_value = float(trials_for_k.get((source, target), 0.0))
                    if use_beta_uncertainty:
                        alphas[i, j] = _alpha_from_uncertainty(uncert_value)
                    else:
                        alphas[i, j] = _alpha_from_trial_count(uncert_value)

        if vmin is not None and vmax is not None and vmax > vmin:
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
                alpha=alphas,
            )
        else:
            image = axis.imshow(
                matrix,
                cmap="coolwarm",
                aspect="equal",
                alpha=alphas,
            )
        last_image = image

        axis.set_xticks(range(size))
        axis.set_xticklabels(short_labels, rotation=90, fontsize=5)
        axis.set_yticks(range(size))
        if index == 0:
            axis.set_yticklabels(short_labels, fontsize=5)
        else:
            axis.set_yticklabels([])
        axis.set_xlabel("Target annotation Y")
        if index == 0:
            axis.set_ylabel("Source annotation X")
        else:
            axis.set_ylabel("")
        axis.set_title(f"K = {k}", fontsize=8)

    if last_image is not None:
        color_bar = figure.colorbar(
            last_image,
            ax=axes_list,
            orientation="horizontal",
            pad=0.02,
            shrink=0.6,
        )
        if not colorbar_label:
            colorbar_label = "log2 enrichment"
        color_bar.set_label(colorbar_label, rotation=0, fontsize=7)

    figure.savefig(output_path, format="pdf")
    plt.close(figure)
    print(f"Wrote combined enrichment heatmaps for K={ks} to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for plotting sequential annotation dynamics."""

    args = parse_args(argv)
    ks: List[int] = list(args.window_k)

    (
        annotation_ids,
        enrichment_by_k,
        beta_trials_by_k,
    ) = _load_enrichment_from_matrices(
        args.output_prefix,
        ks,
        args.effect_source,
    )
    if not annotation_ids or not enrichment_by_k:
        print("No enrichment matrices were loaded; nothing to plot.")
        return 1

    if args.annotation_id is not None:
        requested = set(args.annotation_id)
        filtered_ids = [aid for aid in annotation_ids if aid in requested]
        if not filtered_ids:
            print(
                "None of the requested annotation ids were found in the "
                "sequential dynamics matrices; nothing to plot."
            )
            return 1
        annotation_ids = filtered_ids
    else:
        # By default, exclude negated / complement-style annotations whose ids
        # use the ``not:`` prefix (for example, ``not:user-violent-thoughts``).
        # These can still be included explicitly via repeated --annotation-id
        # arguments when needed.
        annotation_ids = [aid for aid in annotation_ids if not aid.startswith("not:")]
        if not annotation_ids:
            print(
                "Only ':not' annotation variants were present in the "
                "sequential dynamics matrices; nothing to plot after "
                "excluding them by default. Re-run with explicit "
                "--annotation-id values to include them."
            )
            return 1

    if args.cluster_order:
        ordered_annotations, max_abs_log = _cluster_annotation_order_for_heatmaps(
            ks,
            annotation_ids,
            enrichment_by_k,
        )
    else:
        # Compute the global log-scale range without altering the canonical
        # annotation ordering.
        _, max_abs_log = _cluster_annotation_order_for_heatmaps(
            ks,
            annotation_ids,
            enrichment_by_k,
        )
        ordered_annotations = None

    have_beta_trials = any(
        any(trial > 0.0 for trial in trials_for_k.values())
        for trials_for_k in beta_trials_by_k.values()
    )
    use_alpha = args.uncertainty_alpha and have_beta_trials
    if args.uncertainty_alpha and not have_beta_trials:
        print(
            "Beta-based uncertainty columns were not found in the input "
            "matrices; rendering heatmaps without alpha scaling."
        )

    if max_abs_log > 0.0:
        vmin = -max_abs_log
        vmax = max_abs_log
    else:
        vmin = None
        vmax = None

    use_beta_uncertainty = args.effect_source == "beta"

    if args.effect_source == "beta":
        colorbar_label = (
            "log2 K-window occurrence lift "
            "(P(Y occurs at least once within K | X) "
            "/ P(Y occurs at least once within K))"
        )
    else:
        colorbar_label = (
            "log2 per-step rate lift "
            "(per-message rate of Y within K after X "
            "/ per-message base rate of Y)"
        )

    _plot_enrichment_heatmaps_grid(
        args.figure_path,
        ks,
        annotation_ids,
        enrichment_by_k,
        trials_by_k=beta_trials_by_k if use_alpha else None,
        ordered_annotations=ordered_annotations,
        vmin=vmin,
        vmax=vmax,
        use_uncertainty_alpha=use_alpha,
        use_beta_uncertainty=use_beta_uncertainty,
        colorbar_label=colorbar_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
