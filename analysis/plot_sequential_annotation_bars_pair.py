"""Plot two per-target sequential dynamics profiles as subplots.

This module renders two per-target sequential dynamics profiles side by
side as subplots in a single figure. It is intended for cases where two
related source annotations should be compared visually, such as
self-harm versus violence or romantic interest versus sentience.

Both panels share the same window size K, output-prefix for the
sequential dynamics matrices, and effect-source configuration, but can
specify different source and target annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt

from analysis.lib.cli.sequential_dynamics_cli import (
    add_hide_effect_annotations_argument,
    add_magnitude_metric_argument,
    add_pairwise_panel_arguments,
    parse_single_window_k_args,
)
from analysis.lib.plotting.plot_effects_utils import save_figure_with_k_placeholder
from analysis.lib.plotting.sequential_bars_utils import (
    PanelMetrics,
    build_panel_metrics,
    compute_triple_effect_rows,
    create_pair_axes,
    format_annotation_display_label,
    init_pair_legend_state,
    iter_triples_rows,
    load_pairwise_window_counts,
    load_triple_beta_stats,
    load_triple_window_counts,
    plot_per_target_profile_on_axis,
    plot_triple_panel_on_axis,
    print_pairwise_effect_summary,
    render_pair_legend,
    set_effect_source_ylabel,
    validate_pairwise_panels,
)


def _build_triple_panel_metrics(
    *,
    output_prefix: Path,
    k: int,
    source_id: str,
    cond_id: str,
    pairwise_targets_raw: Optional[Sequence[str]],
    conditional_targets_raw: Optional[Sequence[str]],
    order_by_effect_size: bool,
) -> Tuple[PanelMetrics, Set[str]]:
    """Return PanelMetrics for a co-window triple configuration."""

    triples_path = output_prefix.with_name(
        f"{output_prefix.name}_K{k}_triples_cowindow.csv",
    )
    triples_path = triples_path.expanduser().resolve()
    if not triples_path.is_file():
        raise ValueError(f"Triples CSV not found at {triples_path}")

    triple_baseline: Dict[str, float] = {}
    triple_conditional: Dict[str, float] = {}

    for row in iter_triples_rows(triples_path, k, source_id, cond_id):
        z_value = row.get("Z")
        if not z_value:
            continue

        try:
            base_p = float(row.get("p_z_within_K_given_X", "0"))
        except (TypeError, ValueError):
            base_p = 0.0
        try:
            cond_p = float(row.get("p_z_within_K_given_XY", "0"))
        except (TypeError, ValueError):
            cond_p = 0.0

        triple_baseline[z_value] = base_p
        triple_conditional[z_value] = cond_p

    if not triple_baseline:
        raise ValueError(
            "No co-window triples were found for the requested "
            f"K={k}, X={source_id!r}, Y={cond_id!r}.",
        )

    z_ids: List[str]
    if conditional_targets_raw:
        z_ids = [
            z for z in conditional_targets_raw if z in triple_baseline and z != cond_id
        ]
    elif pairwise_targets_raw:
        z_ids = [
            z for z in pairwise_targets_raw if z in triple_baseline and z != cond_id
        ]
    else:
        z_ids = sorted(z for z in triple_baseline if z != cond_id)

    if not z_ids:
        raise ValueError(
            "No Z annotations remained after filtering by --target-id; "
            "nothing to plot.",
        )

    pairwise_only: List[str] = []
    if pairwise_targets_raw:
        for target in pairwise_targets_raw:
            if target == cond_id:
                continue
            if target in z_ids:
                continue
            if target not in pairwise_only:
                pairwise_only.append(target)

    matrix_target_ids: List[str] = [cond_id]
    for z in z_ids:
        if z not in matrix_target_ids:
            matrix_target_ids.append(z)
    for target in pairwise_only:
        if target not in matrix_target_ids:
            matrix_target_ids.append(target)

    base_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=source_id,
        target_ids_raw=matrix_target_ids,
        effect_source="beta",
        order_by_effect_size=False,
    )

    baseline_means: Dict[str, float] = dict(base_panel.baseline_means)
    baseline_cis: Dict[str, Tuple[float, float]] = dict(base_panel.baseline_cis)
    conditional_means: Dict[str, float] = dict(base_panel.conditional_means)
    conditional_cis: Dict[str, Tuple[float, float]] = dict(
        base_panel.conditional_cis,
    )

    if order_by_effect_size:
        z_ids = sorted(
            z_ids,
            key=lambda name: abs(
                float(conditional_means.get(name, 0.0))
                - float(baseline_means.get(name, 0.0))
            ),
            reverse=True,
        )
    targets: List[str] = [cond_id]
    targets.extend(pairwise_only)
    targets.extend([z for z in z_ids if z != cond_id])

    panel = PanelMetrics(
        source_id=base_panel.source_id,
        targets=targets,
        baseline_means=baseline_means,
        baseline_cis=baseline_cis,
        conditional_means=conditional_means,
        conditional_cis=conditional_cis,
    )
    return panel, set(z_ids)


def _plot_panel_on_axis(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    magnitude_metric: str,
    add_ylabel: bool,
    cond_id: Optional[str],
    triple_targets: Optional[Set[str]],
    pairwise_counts: Optional[Dict[str, Tuple[int, int]]],
    triple_counts: Optional[Dict[str, Tuple[int, int]]],
    triple_beta: Optional[Dict[str, Tuple[float, float, float]]],
    show_effect_annotations: bool,
) -> Tuple[
    Optional[plt.Container],
    Optional[plt.Container],
    Optional[plt.Container],
]:
    """Render a per-target profile onto an existing Matplotlib axis.

    Returns the baseline, pairwise-conditional, and triple-conditional
    artists (the latter is None in pairwise-only mode) so that the
    caller can construct a shared legend across subplots.
    """

    if cond_id and triple_targets and pairwise_counts is not None and triple_counts:
        (
            baseline_artist,
            conditional_artist_pairwise,
            conditional_artist_triple,
        ) = plot_triple_panel_on_axis(
            axis,
            panel=panel,
            k=k,
            effect_source=effect_source,
            magnitude_metric=magnitude_metric,
            cond_id=cond_id,
            triple_targets=triple_targets,
            pairwise_counts=pairwise_counts,
            triple_counts=triple_counts,
            triple_beta=triple_beta,
            show_effect_annotations=show_effect_annotations,
        )
        if add_ylabel:
            set_effect_source_ylabel(axis, effect_source, k, fontsize=7)
        return baseline_artist, conditional_artist_pairwise, conditional_artist_triple

    baseline_artist, conditional_artist = plot_per_target_profile_on_axis(
        axis,
        panel=panel,
        k=k,
        effect_source=effect_source,
        magnitude_metric=magnitude_metric,
        add_ylabel=add_ylabel,
        add_arrows=True,
        show_effect_labels=show_effect_annotations,
    )
    return baseline_artist, conditional_artist, None


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the paired per-target plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot two per-target sequential annotation profiles side by side "
            "as subplots from precomputed X->Y matrix CSV tables."
        )
    )
    add_pairwise_panel_arguments(
        parser,
        figure_path_default=Path("analysis")
        / "figures"
        / "sequential_enrichment_profile_pair_K{K}.pdf",
        figure_path_help=(
            "Destination PDF path for the paired per-target plot. The "
            "placeholder '{K}' in the default will be replaced with the "
            "selected window size."
        ),
    )
    parser.add_argument(
        "--left-cond-id",
        type=str,
        help=(
            "Optional conditioning annotation id Y for the left subplot. "
            "When provided, the left panel uses the co-window triples table "
            "and treats --left-conditional-target-id values as Z annotations "
            "for which P(Z | X, Y-in-window) is plotted alongside P(Z | X)."
        ),
    )
    parser.add_argument(
        "--left-conditional-target-id",
        type=str,
        action="append",
        help=(
            "Third annotation id Z for the left subplot whose conditional "
            "probability P(Z | X, Y-in-window) should be plotted. When "
            "omitted, any overlapping --left-target-id values that appear in "
            "the triples table are treated as conditional targets."
        ),
    )
    # Extra conditional inputs specific to the paired bar plot.
    parser.add_argument(
        "--right-cond-id",
        type=str,
        help=(
            "Optional conditioning annotation id Y for the right subplot. "
            "When provided, the right panel uses the co-window triples table "
            "and treats --right-conditional-target-id values as Z annotations "
            "for which P(Z | X, Y-in-window) is plotted alongside P(Z | X)."
        ),
    )
    parser.add_argument(
        "--right-conditional-target-id",
        type=str,
        action="append",
        help=(
            "Third annotation id Z for the right subplot whose conditional "
            "probability P(Z | X, Y-in-window) should be plotted. When "
            "omitted, any overlapping --right-target-id values that appear in "
            "the triples table are treated as conditional targets."
        ),
    )
    add_magnitude_metric_argument(parser)
    add_hide_effect_annotations_argument(
        parser,
        help_text=(
            "When set, omit odds/risk ratio annotations and arrows from the "
            "per-target plots in both panels. Printed summary tables are "
            "unaffected."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the paired per-target plotting script."""

    parser = _build_parser()
    return parse_single_window_k_args(parser, argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for paired per-target sequential-dynamics plots."""

    plt.switch_backend("Agg")

    args = parse_args(argv)
    k: int = int(args.window_k[0])

    left_cond_id: Optional[str] = getattr(args, "left_cond_id", None)
    right_cond_id: Optional[str] = getattr(args, "right_cond_id", None)

    left_triple_targets: Optional[Set[str]] = None
    right_triple_targets: Optional[Set[str]] = None

    try:
        if left_cond_id:
            left_panel, left_triple_targets = _build_triple_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id=args.left_source_id,
                cond_id=left_cond_id,
                pairwise_targets_raw=args.left_target_id,
                conditional_targets_raw=getattr(
                    args,
                    "left_conditional_target_id",
                    None,
                ),
                order_by_effect_size=bool(args.order_by_effect_size),
            )
        else:
            left_panel = build_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id_raw=args.left_source_id,
                target_ids_raw=args.left_target_id,
                effect_source=args.effect_source,
                order_by_effect_size=bool(args.order_by_effect_size),
            )

        if right_cond_id:
            right_panel, right_triple_targets = _build_triple_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id=args.right_source_id,
                cond_id=right_cond_id,
                pairwise_targets_raw=args.right_target_id,
                conditional_targets_raw=getattr(
                    args,
                    "right_conditional_target_id",
                    None,
                ),
                order_by_effect_size=bool(args.order_by_effect_size),
            )
        else:
            right_panel = build_panel_metrics(
                output_prefix=args.output_prefix,
                k=k,
                source_id_raw=args.right_source_id,
                target_ids_raw=args.right_target_id,
                effect_source=args.effect_source,
                order_by_effect_size=bool(args.order_by_effect_size),
            )
    except ValueError as exc:
        print(str(exc))
        return 2

    if not validate_pairwise_panels(left_panel, right_panel):
        return 1

    def _load_pairwise_and_triple_counts(
        *,
        output_prefix: Path,
        k_value: int,
        source_id: str,
        targets: Sequence[str],
        cond_id: str,
        triple_targets: Sequence[str],
        include_beta: bool,
    ) -> tuple[
        Dict[str, Tuple[int, int]],
        Dict[str, Tuple[int, int]],
        Optional[Dict[str, Tuple[float, float, float]]],
    ]:
        matrix_path = output_prefix.with_name(
            f"{output_prefix.name}_K{k_value}_matrix.csv",
        )
        pairwise_counts = load_pairwise_window_counts(
            matrix_path,
            k_value,
            source_id,
            targets,
        )
        triples_path = output_prefix.with_name(
            f"{output_prefix.name}_K{k_value}_triples_cowindow.csv",
        )
        triple_counts = load_triple_window_counts(
            triples_path,
            k_value,
            source_id,
            cond_id,
            triple_targets,
        )
        triple_beta = None
        if include_beta and args.effect_source == "beta":
            triple_beta = load_triple_beta_stats(
                triples_path,
                k_value,
                source_id,
                cond_id,
                triple_targets,
            )
        return pairwise_counts, triple_counts, triple_beta

    if left_cond_id and left_triple_targets:
        left_pairwise_counts, left_triple_counts, _ = _load_pairwise_and_triple_counts(
            output_prefix=args.output_prefix,
            k_value=k,
            source_id=left_panel.source_id,
            targets=left_panel.targets,
            cond_id=left_cond_id,
            triple_targets=left_triple_targets,
            include_beta=False,
        )
        for (
            target,
            row_type,
            base,
            cond,
            delta,
            risk_ratio,
            odds_ratio,
            _successes,
            _trials,
        ) in compute_triple_effect_rows(
            left_panel,
            triple_targets=left_triple_targets,
            pairwise_counts=left_pairwise_counts,
            triple_counts=left_triple_counts,
        ):
            print(
                f"{target:40s} {row_type:>8s} "
                f"{base:10.3f} {cond:12.3f} "
                f"{delta:10.3f} {risk_ratio:10.3f} {odds_ratio:10.3f}"
            )
    else:
        print_pairwise_effect_summary(left_panel, panel_label="left panel")

    if right_cond_id and right_triple_targets:
        (
            right_pairwise_counts,
            right_triple_counts,
            _,
        ) = _load_pairwise_and_triple_counts(
            output_prefix=args.output_prefix,
            k_value=k,
            source_id=right_panel.source_id,
            targets=right_panel.targets,
            cond_id=right_cond_id,
            triple_targets=right_triple_targets,
            include_beta=False,
        )
        for (
            target,
            row_type,
            base,
            cond,
            delta,
            risk_ratio,
            odds_ratio,
            _successes,
            _trials,
        ) in compute_triple_effect_rows(
            right_panel,
            triple_targets=right_triple_targets,
            pairwise_counts=right_pairwise_counts,
            triple_counts=right_triple_counts,
        ):
            print(
                f"{target:40s} {row_type:>8s} "
                f"{base:10.3f} {cond:12.3f} "
                f"{delta:10.3f} {risk_ratio:10.3f} {odds_ratio:10.3f}"
            )
    else:
        print_pairwise_effect_summary(
            right_panel,
            panel_label="right panel",
            leading_newline=True,
        )

    figure, axis_left, axis_right = create_pair_axes(
        len(left_panel.targets),
        len(right_panel.targets),
        sharey=False,
    )

    # Preload counts and Beta stats for plotting when conditional
    # triples are requested.
    left_pairwise_counts: Optional[Dict[str, Tuple[int, int]]] = None
    left_triple_counts: Optional[Dict[str, Tuple[int, int]]] = None
    left_triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None
    if left_cond_id and left_triple_targets:
        (
            left_pairwise_counts,
            left_triple_counts,
            left_triple_beta,
        ) = _load_pairwise_and_triple_counts(
            output_prefix=args.output_prefix,
            k_value=k,
            source_id=left_panel.source_id,
            targets=left_panel.targets,
            cond_id=left_cond_id,
            triple_targets=left_triple_targets,
            include_beta=True,
        )

    right_pairwise_counts: Optional[Dict[str, Tuple[int, int]]] = None
    right_triple_counts: Optional[Dict[str, Tuple[int, int]]] = None
    right_triple_beta: Optional[Dict[str, Tuple[float, float, float]]] = None
    if right_cond_id and right_triple_targets:
        (
            right_pairwise_counts,
            right_triple_counts,
            right_triple_beta,
        ) = _load_pairwise_and_triple_counts(
            output_prefix=args.output_prefix,
            k_value=k,
            source_id=right_panel.source_id,
            targets=right_panel.targets,
            cond_id=right_cond_id,
            triple_targets=right_triple_targets,
            include_beta=True,
        )

    (
        left_baseline_artist,
        left_conditional_artist,
        left_triple_artist,
    ) = _plot_panel_on_axis(
        axis_left,
        panel=left_panel,
        k=k,
        effect_source=args.effect_source,
        magnitude_metric=args.magnitude_metric,
        add_ylabel=True,
        cond_id=left_cond_id,
        triple_targets=left_triple_targets,
        pairwise_counts=left_pairwise_counts,
        triple_counts=left_triple_counts,
        triple_beta=left_triple_beta,
        show_effect_annotations=not bool(
            getattr(args, "hide_effect_annotations", False)
        ),
    )
    (
        _,
        right_conditional_artist,
        right_triple_artist,
    ) = _plot_panel_on_axis(
        axis_right,
        panel=right_panel,
        k=k,
        effect_source=args.effect_source,
        magnitude_metric=args.magnitude_metric,
        add_ylabel=False,
        cond_id=right_cond_id,
        triple_targets=right_triple_targets,
        pairwise_counts=right_pairwise_counts,
        triple_counts=right_triple_counts,
        triple_beta=right_triple_beta,
        show_effect_annotations=not bool(
            getattr(args, "hide_effect_annotations", False)
        ),
    )

    legend_handles, legend_labels, seen_labels = init_pair_legend_state(
        axis_left,
        axis_right,
        left_source_id=left_panel.source_id,
        right_source_id=right_panel.source_id,
        fontsize=9,
    )

    if left_baseline_artist is not None:
        label = "Global baseline"
        if label not in seen_labels:
            legend_handles.append(left_baseline_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if left_conditional_artist is not None:
        label = f"Following {format_annotation_display_label(left_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(left_conditional_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_conditional_artist is not None:
        label = f"Following {format_annotation_display_label(right_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(right_conditional_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if left_triple_artist is not None and left_cond_id:
        label = (
            f"Following {format_annotation_display_label(left_panel.source_id)} and "
            f"{format_annotation_display_label(left_cond_id)} in window"
        )
        if label not in seen_labels:
            legend_handles.append(left_triple_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_triple_artist is not None and right_cond_id:
        label = (
            f"Following {format_annotation_display_label(right_panel.source_id)} and "
            f"{format_annotation_display_label(right_cond_id)} in window"
        )
        if label not in seen_labels:
            legend_handles.append(right_triple_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    label_count = render_pair_legend(
        figure,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        compact_threshold=3,
        lower_loc="lower center",
        lower_anchor=(0.5, -0.02),
        upper_loc="upper center",
        upper_anchor=(0.5, 0.18),
        upper_ncol=2,
        fontsize=8,
    )

    figure.tight_layout()

    if label_count >= 3:
        figure.subplots_adjust(left=0.08, right=0.99, bottom=0.4, top=0.85, wspace=0.15)

    save_figure_with_k_placeholder(args.figure_path, figure, k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
