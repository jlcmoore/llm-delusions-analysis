"""Plot two per-target P(Y | X) profiles as histograms.

This module mirrors the structure of
``analysis/plot_sequential_annotation_bars_pair.py`` but focuses on a
single series per target: the conditional probability P(Y | X) for a
selected source annotation X and one or more target annotations Y.

For each source, the script renders a vertical bar-style histogram where
the height of each bar corresponds to P(Y | X) and bars are coloured
according to the source annotation. Global baseline probabilities,
odds/risk ratios, and per-target change annotations are intentionally
omitted so that the figure emphasises the conditional probabilities
alone.

The y-axis label and x-axis tick labels follow the same conventions as
the existing sequential bar plots so that figures remain comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analysis.lib.cli.sequential_dynamics_cli import (
    add_pairwise_panel_arguments,
    parse_single_window_k_args,
)
from analysis.lib.plotting.plot_effects_utils import save_figure_with_k_placeholder
from analysis.lib.plotting.sequential_bars_utils import (
    PanelMetrics,
    build_panel_metrics,
    create_pair_axes,
    format_annotation_display_label,
    format_target_tick_labels,
    init_pair_legend_state,
    print_pairwise_effect_summary,
    render_pair_legend,
    set_effect_source_ylabel,
    validate_pairwise_panels,
)
from analysis.lib.plotting.style import annotation_color_for_label


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the paired P(Y | X) histogram plot.

    The argument structure closely follows
    ``plot_sequential_annotation_bars_pair.py`` so that command-line usage
    remains familiar. Only the pairwise X->Y matrix is consulted; options
    related to triple co-window statistics and magnitude metrics are
    omitted.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot two per-target conditional probability profiles P(Y | X) "
            "side by side as histograms from precomputed X->Y matrix CSV "
            "tables."
        )
    )
    add_pairwise_panel_arguments(
        parser,
        figure_path_default=Path("analysis")
        / "figures"
        / "sequential_enrichment_histogram_pair_K{K}.pdf",
        figure_path_help=(
            "Destination PDF path for the paired per-target histogram. The "
            "placeholder '{K}' in the default will be replaced with the "
            "selected window size."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the paired histogram plotting script.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments to parse instead of
        ``sys.argv[1:]``. This is primarily intended for testing.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``window_k`` normalised to a single-element
        list.
    """

    parser = _build_parser()
    return parse_single_window_k_args(parser, argv)


def _build_panels(
    *,
    output_prefix: Path,
    k: int,
    effect_source: str,
    order_by_effect_size: bool,
    left_source_id: str,
    left_target_ids: Optional[Sequence[str]],
    right_source_id: str,
    right_target_ids: Optional[Sequence[str]],
) -> Tuple[PanelMetrics, PanelMetrics]:
    """Return PanelMetrics for the left and right sources.

    This helper mirrors the setup in the paired bar plot script but only
    relies on pairwise X->Y statistics. Any ValueError raised while
    loading metrics is propagated to the caller so that a clear message
    is printed and a non-zero exit code is returned.
    """

    left_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=left_source_id,
        target_ids_raw=left_target_ids,
        effect_source=effect_source,
        order_by_effect_size=order_by_effect_size,
    )
    right_panel = build_panel_metrics(
        output_prefix=output_prefix,
        k=k,
        source_id_raw=right_source_id,
        target_ids_raw=right_target_ids,
        effect_source=effect_source,
        order_by_effect_size=order_by_effect_size,
    )
    return left_panel, right_panel


def _plot_histogram_panel(
    axis: plt.Axes,
    *,
    panel: PanelMetrics,
    k: int,
    effect_source: str,
    add_ylabel: bool,
) -> Optional[plt.Container]:
    """Render a per-target P(Y | X) histogram onto an axis.

    Bars are drawn for the conditional probabilities P(Y | X) only.
    Global baselines, odds ratios, and per-target difference annotations
    are omitted by design.

    Parameters
    ----------
    axis:
        Matplotlib axis onto which the histogram is rendered.
    panel:
        Per-source per-target metrics loaded from the X->Y matrix.
    k:
        Window size K in messages, used for the y-axis label when
        ``effect_source == "beta"``.
    effect_source:
        Effect-size source in use (``"beta"`` or ``"enrichment"``),
        controlling the y-axis label text.
    add_ylabel:
        When ``True``, set the shared y-axis label on this axis.

    Returns
    -------
    Optional[plt.Container]
        The bar container for the conditional series, or ``None`` when
        no targets are available.
    """

    if not panel.targets:
        return None

    x_positions = np.arange(len(panel.targets), dtype=float)

    conditional_y = []
    conditional_yerr = [[], []]
    for target in panel.targets:
        cond_mean = float(panel.conditional_means[target])
        cond_low, cond_high = panel.conditional_cis[target]
        conditional_y.append(cond_mean)
        conditional_yerr[0].append(cond_mean - cond_low)
        conditional_yerr[1].append(cond_high - cond_mean)

    conditional_y_array = np.asarray(conditional_y, dtype=float)
    conditional_yerr_array = np.asarray(conditional_yerr, dtype=float)

    if ":" in panel.source_id:
        base_source_id, _role = panel.source_id.split(":", 1)
    else:
        base_source_id = panel.source_id
    conditional_color = annotation_color_for_label(base_source_id)

    bar_container = axis.bar(
        x_positions,
        conditional_y_array,
        yerr=conditional_yerr_array,
        align="center",
        width=0.8,
        color=conditional_color,
        ecolor=conditional_color,
        alpha=0.9,
        linewidth=0.0,
        capsize=3.0,
        zorder=3,
    )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        format_target_tick_labels(panel.targets),
        rotation=0,
        ha="right",
        fontsize=8,
    )

    if add_ylabel:
        set_effect_source_ylabel(axis, effect_source, k, fontsize=8)

    return bar_container


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for paired P(Y | X) histogram plots.

    Parameters
    ----------
    argv:
        Optional list of command-line arguments to parse instead of the
        default ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero on success or a non-zero error code when metrics could not
        be loaded or no targets were available for plotting.
    """

    plt.switch_backend("Agg")

    args = parse_args(argv)
    k: int = int(args.window_k[0])

    try:
        left_panel, right_panel = _build_panels(
            output_prefix=args.output_prefix,
            k=k,
            effect_source=args.effect_source,
            order_by_effect_size=bool(args.order_by_effect_size),
            left_source_id=args.left_source_id,
            left_target_ids=args.left_target_id,
            right_source_id=args.right_source_id,
            right_target_ids=args.right_target_id,
        )
    except ValueError as exc:
        print(str(exc))
        return 2

    if not validate_pairwise_panels(left_panel, right_panel):
        return 1

    print_pairwise_effect_summary(left_panel, panel_label="left panel")
    print_pairwise_effect_summary(
        right_panel,
        panel_label="right panel",
        leading_newline=True,
    )

    figure, axis_left, axis_right = create_pair_axes(
        len(left_panel.targets),
        len(right_panel.targets),
        sharey=True,
    )

    left_artist = _plot_histogram_panel(
        axis_left,
        panel=left_panel,
        k=k,
        effect_source=args.effect_source,
        add_ylabel=True,
    )
    right_artist = _plot_histogram_panel(
        axis_right,
        panel=right_panel,
        k=k,
        effect_source=args.effect_source,
        add_ylabel=False,
    )

    legend_handles, legend_labels, seen_labels = init_pair_legend_state(
        axis_left,
        axis_right,
        left_source_id=left_panel.source_id,
        right_source_id=right_panel.source_id,
        fontsize=9,
    )

    if left_artist is not None:
        label = f"Following {format_annotation_display_label(left_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(left_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    if right_artist is not None:
        label = f"Following {format_annotation_display_label(right_panel.source_id)}"
        if label not in seen_labels:
            legend_handles.append(right_artist)
            legend_labels.append(label)
            seen_labels.add(label)

    label_count = render_pair_legend(
        figure,
        legend_handles=legend_handles,
        legend_labels=legend_labels,
        compact_threshold=2,
        lower_loc="lower center",
        lower_anchor=(0.5, -0.02),
        upper_loc="upper center",
        upper_anchor=(0.5, 0.15),
        upper_ncol=2,
        fontsize=8,
    )

    figure.tight_layout()

    if label_count > 2:
        figure.subplots_adjust(
            left=0.08,
            right=0.99,
            bottom=0.45,
            top=0.89,
            wspace=0.15,
        )

    save_figure_with_k_placeholder(args.figure_path, figure, k)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
