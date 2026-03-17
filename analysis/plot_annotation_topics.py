"""Plot topic and annotation-level structure from precomputed artifacts.

This module consumes precomputed topic-modeling artifacts and generates a
small set of standard figures:

* Per-topic term bar charts (one PDF per topic) showing top terms and
  their mean TF-IDF weights.
* An annotation-by-topic heatmap where each cell contains the
  log-enrichment of that topic for the annotation relative to the global
  background.
* A per-annotation bar plot of normalized entropy ordered from most to
  least heterogeneous.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
from tqdm.auto import tqdm

from analysis.lib.plotting.labels import shorten_annotation_label
from utils.cli import add_artifacts_dir_argument


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the topic-plotting script.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Plot global topics and per-annotation enrichment from precomputed "
            "topic-modeling artifacts."
        )
    )
    add_artifacts_dir_argument(
        parser,
        default_dir=Path("analysis") / "data" / "annotation_topics_artifacts",
        help_text=(
            "Directory containing topic-modeling outputs (global_topics.json, "
            "annotation_summary.csv, etc.)."
        ),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("analysis") / "figures",
        help="Base directory where figures will be written.",
    )
    parser.add_argument(
        "--min-messages",
        type=int,
        default=10,
        help=(
            "Minimum number of messages per annotation required to include it "
            "in aggregate plots (heatmap, entropy bar chart)."
        ),
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=15,
        help="Maximum number of top terms to show per topic in term bar plots.",
    )
    parser.add_argument(
        "--max-topic-plots",
        type=int,
        default=None,
        help=(
            "Optional maximum number of topics (sorted by size) for which to "
            "generate term bar-chart PDFs. When omitted, plots are produced "
            "for all topics."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the topic-plotting script.

    Parameters
    ----------
    argv:
        Optional raw argument vector. When omitted, ``sys.argv`` semantics
        are used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace populated with defaults.
    """

    parser = _build_parser()
    return parser.parse_args(argv)


def _load_global_topics(path: Path) -> Mapping[str, object]:
    """Return the parsed contents of ``global_topics.json``."""

    resolved = path.expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_annotation_summary(path: Path) -> pd.DataFrame:
    """Return the annotation summary table as a DataFrame."""

    resolved = path.expanduser().resolve()
    return pd.read_csv(resolved)


def _load_message_topics(path: Path) -> pd.DataFrame:
    """Return the per-message topic assignments as a DataFrame."""

    resolved = path.expanduser().resolve()
    return pd.read_csv(resolved)


def _ensure_dir(path: Path) -> Path:
    """Create a directory when necessary and return its resolved path."""

    resolved = path.expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _plot_topic_term_bars(
    *,
    global_topics: Mapping[str, object],
    figures_dir: Path,
    max_terms: int,
    max_topic_plots: Optional[int],
) -> None:
    """Write one PDF per topic with a horizontal bar chart of top terms.

    Parameters
    ----------
    global_topics:
        Parsed JSON object from ``global_topics.json``.
    figures_dir:
        Destination directory for topic term PDF files.
    max_terms:
        Maximum number of top terms shown per topic.
    max_topic_plots:
        Optional cap on the number of topics to plot, sorted by topic size.
    """

    topic_summaries = list(global_topics.get("topic_summaries", []))
    if not topic_summaries:
        return

    topic_summaries.sort(key=lambda entry: int(entry.get("size", 0)), reverse=True)

    output_dir = _ensure_dir(figures_dir / "topics")

    to_plot = topic_summaries
    if max_topic_plots is not None and max_topic_plots > 0:
        to_plot = topic_summaries[:max_topic_plots]

    for summary in tqdm(to_plot, desc="Topic term plots", unit="topic"):
        _plot_single_topic_term_bars(summary, output_dir, max_terms=max_terms)

    if to_plot:
        print(f"Per-topic term PDFs written under {output_dir}")


def _plot_single_topic_term_bars(
    summary: Mapping[str, object],
    output_dir: Path,
    *,
    max_terms: int,
) -> None:
    """Write a single PDF with a horizontal bar chart of top terms."""

    topic_id = int(summary.get("topic_id", 0))
    size = int(summary.get("size", 0))

    top_terms: List[str] = list(summary.get("top_terms", []))[:max_terms]
    top_scores: List[float] = list(summary.get("top_term_scores", []))[: len(top_terms)]

    if not top_terms:
        return

    if not top_scores:
        # Fall back to a simple rank-based weighting if scores are not
        # present (for example, older artifacts).
        top_scores = list(reversed(range(1, len(top_terms) + 1)))

    indices = np.arange(len(top_terms))

    fig, axis = plt.subplots(figsize=(6, 0.4 * len(top_terms) + 2.0))
    axis.barh(indices, top_scores, align="center")
    axis.set_yticks(indices)
    axis.set_yticklabels(top_terms)
    axis.invert_yaxis()
    axis.set_xlabel("Mean TF-IDF weight")
    axis.set_title(f"Topic {topic_id} (size={size})")
    fig.tight_layout()

    output_path = output_dir / f"topic_{topic_id:03d}_terms.pdf"
    fig.savefig(output_path)
    plt.close(fig)


def _compute_global_topic_frequencies(message_topics: pd.DataFrame) -> np.ndarray:
    """Return an array of global topic counts for each topic identifier."""

    topic_ids = message_topics["topic_id"].to_numpy()
    if topic_ids.size == 0:
        return np.zeros(0, dtype=int)

    max_topic_id = int(topic_ids.max())
    return np.bincount(topic_ids, minlength=max_topic_id + 1)


def _plot_annotation_topic_heatmap(
    *,
    artifacts_dir: Path,
    figures_dir: Path,
    min_messages: int,
) -> None:
    """Plot an annotation-by-topic log-enrichment heatmap.

    Parameters
    ----------
    artifacts_dir:
        Directory containing ``annotation_summary.csv``,
        ``annotation_topic_enrichment.csv``, and ``message_topics.csv``.
    figures_dir:
        Destination directory for the heatmap PDF.
    min_messages:
        Minimum number of messages per annotation required to include it in
        the heatmap.
    """

    summary_path = artifacts_dir / "annotation_summary.csv"
    enrichment_path = artifacts_dir / "annotation_topic_enrichment.csv"
    message_topics_path = artifacts_dir / "message_topics.csv"

    if not summary_path.exists() or not enrichment_path.exists():
        return

    summary = _load_annotation_summary(summary_path)
    enrichment = pd.read_csv(enrichment_path)
    message_topics = _load_message_topics(message_topics_path)

    # Filter annotations for stability and order by normalized entropy
    # (broadest annotations first).
    filtered = summary[summary["num_messages"] >= min_messages].copy()
    if filtered.empty:
        return

    filtered.sort_values("normalized_entropy", ascending=False, inplace=True)
    annotation_ids = filtered["annotation_id"].tolist()

    global_counts = _compute_global_topic_frequencies(message_topics)
    num_topics = global_counts.shape[0]
    total_messages = int(global_counts.sum())
    if total_messages == 0 or num_topics == 0:
        return

    # Map annotation_id -> num_messages for computing proportions.
    ann_size_map: Dict[str, int] = dict(
        zip(filtered["annotation_id"], filtered["num_messages"])
    )

    # Build a sparse mapping of known annotation/topic counts from the
    # enrichment table, then fill in zeros where missing.
    counts_map: Dict[tuple[str, int], int] = {}
    for row in enrichment.itertuples(index=False):
        ann = str(row.annotation_id)
        topic_id = int(row.topic_id)
        counts_map[(ann, topic_id)] = int(row.annotation_count)

    eps = 1e-9
    heatmap = np.zeros((len(annotation_ids), num_topics), dtype=float)

    for i, annotation_id in enumerate(annotation_ids):
        ann_size = int(ann_size_map.get(annotation_id, 0))
        if ann_size <= 0:
            continue

        for topic_id in range(num_topics):
            ann_count = counts_map.get((annotation_id, topic_id), 0)
            ann_prop = float(ann_count) / float(ann_size)

            global_prop = float(global_counts[topic_id] / float(total_messages))

            log_enrichment = np.log((ann_prop + eps) / (global_prop + eps))
            heatmap[i, topic_id] = log_enrichment

    # Compute hierarchical clustering over annotations (rows) and topics
    # (columns) to reveal block structure.
    if len(annotation_ids) > 1:
        row_dist = pdist(heatmap, metric="euclidean")
        row_linkage = linkage(row_dist, method="average")
        row_order = leaves_list(row_linkage)

        col_dist = pdist(heatmap.T, metric="euclidean")
        col_linkage = linkage(col_dist, method="average")
        col_order = leaves_list(col_linkage)

        heatmap = heatmap[row_order, :][:, col_order]
        annotation_ids = [annotation_ids[i] for i in row_order]
        topic_indices = [int(i) for i in col_order]
    else:
        topic_indices = list(range(num_topics))

    short_ann_labels = [shorten_annotation_label(label) for label in annotation_ids]
    topic_labels = [str(i) for i in topic_indices]

    # For plotting, clip extreme log-enrichment values symmetrically around
    # zero so that a small number of very large positive/negative ratios do
    # not compress all other structure. The full, unclipped values remain
    # available in ``annotation_topic_enrichment.csv``; clipping here affects
    # only the colour scale of this figure and should be described in the
    # caption (for example, "log-enrichment values clipped to +/-V for
    # visualisation").
    abs_values = np.abs(heatmap)
    vmax = float(np.percentile(abs_values, 99.0))
    if vmax <= 0.0:
        vmax = 1.0
    heatmap_clipped = np.clip(heatmap, -vmax, vmax)

    _ensure_dir(figures_dir)
    figsize = (0.15 * len(topic_indices) + 4.0, 0.25 * len(annotation_ids) + 4.0)
    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        heatmap_clipped,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    axis.set_yticks(np.arange(len(annotation_ids)))
    axis.set_yticklabels(short_ann_labels)
    axis.set_xticks(np.arange(len(topic_indices)))
    axis.set_xticklabels(topic_labels, rotation=90)

    axis.set_xlabel("Topic")
    axis.set_ylabel("Annotation")
    axis.set_title("Annotation-by-topic log enrichment")

    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("log enrichment")

    fig.tight_layout()

    output_path = figures_dir / "annotation_topic_heatmap.pdf"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Annotation-by-topic heatmap written to {output_path}")


def _plot_annotation_entropy_bar(
    *,
    summary: pd.DataFrame,
    figures_dir: Path,
    min_messages: int,
) -> None:
    """Plot per-annotation normalized entropy ordered from most to least.

    Parameters
    ----------
    summary:
        DataFrame loaded from ``annotation_summary.csv``.
    figures_dir:
        Destination directory for the entropy bar-chart PDF.
    min_messages:
        Minimum number of messages per annotation required to include it in
        the plot.
    """

    filtered = summary[summary["num_messages"] >= min_messages].copy()
    if filtered.empty:
        return

    filtered.sort_values("normalized_entropy", ascending=False, inplace=True)

    annotation_ids = filtered["annotation_id"].tolist()
    entropies = filtered["normalized_entropy"].to_numpy()

    indices = np.arange(len(annotation_ids))

    _ensure_dir(figures_dir)
    fig_width = max(8.0, 0.4 * len(annotation_ids))
    fig_height = 6.0
    fig, axis = plt.subplots(figsize=(fig_width, fig_height))

    axis.bar(indices, entropies)
    axis.set_xticks(indices)
    short_labels = [shorten_annotation_label(label) for label in annotation_ids]
    axis.set_xticklabels(short_labels, rotation=90)
    axis.set_ylabel("Normalized entropy")
    axis.set_xlabel("Annotation (sorted by entropy)")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Per-annotation topic entropy")
    fig.tight_layout()

    output_path = figures_dir / "annotation_entropy_bar.pdf"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Per-annotation entropy bar chart written to {output_path}")


def _plot_participant_topic_heatmap(
    *,
    artifacts_dir: Path,
    figures_dir: Path,
    min_messages: int,
) -> None:
    """Plot a participant-by-topic log-enrichment heatmap.

    Parameters
    ----------
    artifacts_dir:
        Directory containing ``participant_topics.csv`` and
        ``message_topics.csv``.
    figures_dir:
        Destination directory for the heatmap PDF.
    min_messages:
        Minimum number of messages per participant required to include them
        in the heatmap.
    """

    topics_path = artifacts_dir / "participant_topics.csv"
    if not topics_path.exists():
        return

    frame = pd.read_csv(topics_path)
    if frame.empty:
        return

    # Filter participants with enough messages to yield stable proportions.
    valid_participants = (
        frame[frame["participant_total_messages"] >= min_messages]["participant"]
        .drop_duplicates()
        .tolist()
    )
    if not valid_participants:
        return

    filtered = frame[frame["participant"].isin(valid_participants)].copy()

    participants = sorted(filtered["participant"].unique())
    topics = sorted(filtered["topic_id"].unique())
    num_participants = len(participants)
    num_topics = len(topics)

    if num_participants == 0 or num_topics == 0:
        return

    participant_index: Dict[str, int] = {p: i for i, p in enumerate(participants)}
    topic_index: Dict[int, int] = {int(t): j for j, t in enumerate(topics)}

    matrix = np.zeros((num_participants, num_topics), dtype=float)
    for row in filtered.itertuples(index=False):
        participant = str(row.participant)
        topic_id = int(row.topic_id)
        if participant not in participant_index or topic_id not in topic_index:
            continue
        i = participant_index[participant]
        j = topic_index[topic_id]
        matrix[i, j] = float(row.log_enrichment)

    # Cluster participants (rows) and topics (columns) to reveal block
    # structure in the participant-by-topic space.
    if num_participants > 1:
        row_dist = pdist(matrix, metric="euclidean")
        row_linkage = linkage(row_dist, method="average")
        row_order = leaves_list(row_linkage)

        col_dist = pdist(matrix.T, metric="euclidean")
        col_linkage = linkage(col_dist, method="average")
        col_order = leaves_list(col_linkage)

        matrix = matrix[row_order, :][:, col_order]
        participants = [participants[i] for i in row_order]
        topics = [topics[int(i)] for i in col_order]

    # Clip extreme log-enrichment values for visualisation only, keeping the
    # full values in the CSV. This matches the annotation heatmap logic and
    # should be described in any figure caption.
    abs_values = np.abs(matrix)
    vmax = float(np.percentile(abs_values, 99.0))
    if vmax <= 0.0:
        vmax = 1.0
    matrix_clipped = np.clip(matrix, -vmax, vmax)

    _ensure_dir(figures_dir)
    figsize = (0.15 * num_topics + 4.0, 0.25 * num_participants + 4.0)
    fig, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        matrix_clipped,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )

    axis.set_yticks(np.arange(num_participants))
    axis.set_yticklabels(participants)
    axis.set_xticks(np.arange(num_topics))
    axis.set_xticklabels([str(t) for t in topics], rotation=90)

    axis.set_xlabel("Topic")
    axis.set_ylabel("Participant")
    axis.set_title("Participant-by-topic log enrichment")

    cbar = fig.colorbar(image, ax=axis)
    cbar.set_label("log enrichment")

    fig.tight_layout()

    output_path = figures_dir / "participant_topic_heatmap.pdf"
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Participant-by-topic heatmap written to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Program entry point for plotting topic and annotation figures."""

    args = parse_args(argv)

    artifacts_dir = args.artifacts_dir.expanduser().resolve()
    figures_dir = args.figures_dir.expanduser().resolve()

    global_topics_path = artifacts_dir / "global_topics.json"
    summary_path = artifacts_dir / "annotation_summary.csv"

    if not global_topics_path.exists():
        raise SystemExit(f"Missing global topics JSON: {global_topics_path}")
    if not summary_path.exists():
        raise SystemExit(f"Missing annotation summary CSV: {summary_path}")

    global_topics = _load_global_topics(global_topics_path)
    summary = _load_annotation_summary(summary_path)

    print(f"Writing figures under {figures_dir} using artifacts from {artifacts_dir}")

    _plot_topic_term_bars(
        global_topics=global_topics,
        figures_dir=figures_dir,
        max_terms=args.max_terms,
        max_topic_plots=args.max_topic_plots,
    )
    print("Wrote per-topic term PDFs.")

    _plot_annotation_topic_heatmap(
        artifacts_dir=artifacts_dir,
        figures_dir=figures_dir,
        min_messages=args.min_messages,
    )
    print("Wrote annotation-by-topic log-enrichment heatmap.")

    _plot_annotation_entropy_bar(
        summary=summary,
        figures_dir=figures_dir,
        min_messages=args.min_messages,
    )

    _plot_participant_topic_heatmap(
        artifacts_dir=artifacts_dir,
        figures_dir=figures_dir,
        min_messages=args.min_messages,
    )
    print("Wrote per-annotation entropy bar chart.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
