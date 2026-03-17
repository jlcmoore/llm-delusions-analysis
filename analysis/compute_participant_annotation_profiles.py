"""Compute participant-level annotation profiles and onset statistics.

This script aggregates per-participant annotation frequencies from
``classify_chats`` JSONL outputs and produces:

* A wide CSV table with one row per annotation and one column per participant
  containing per-participant annotation rates (proportion of scoped messages
  that are positive).
* Per-annotation onset statistics over participant lifetimes, where the onset
  ``T_p^K(A)`` is defined as the lifetime index of the K-th positive scoped
  message for annotation ``A`` and participant ``p``. The table reports the
  mean and standard error of these onset indices across participants with a
  reliable global ordering.
* A clustered heatmap PDF where each cell is a row-wise z-score indicating how
  much a participant over- or under-uses an annotation relative to peers.
* A dendrogram PDF visualising the hierarchical clustering used to order
  participants or annotations.

Interpreting onset statistics
-----------------------------

Onset statistics are summarised in the CSV columns:

* ``onset_mean``: average message index at which participants cross the
  K-occurrence threshold for an annotation (for example, with
  ``--onset-threshold-k 5``, an ``onset_mean`` of 120 means that, among
  contributing participants with reliable ordering, the fifth positive
  typically appears around their 120th scoped message).
* ``onset_se``: standard error of this mean over contributing participants.
* ``onset_share_of_eligible``: fraction of eligible participants (those with
  at least one scoped message and reliable ordering) who reach the
  K-occurrence threshold for the annotation.
* ``onset_n_eligible``: total number of eligible participants for the
  annotation; together with ``onset_share_of_eligible``, this encodes both the
  support size and the share of participants that ever "turn on" the label.

Clustering behaviour
--------------------

By default, participants are clustered (``cluster_mode=participants``) using
agglomerative hierarchical clustering with Ward linkage on Euclidean distances
between their row-wise z-scored annotation profiles. Each participant is a
point in an "annotation space" whose coordinates are the z-scored rates for
each annotation.

Row-wise z-scoring normalises each annotation across participants so that
values represent deviations from the typical participant for that annotation
(mean 0, unit variance). This emphasises relative over- or under-use rather
than absolute rates and places annotations on a comparable scale for
clustering.

Ward linkage merges clusters to minimise the increase in within-cluster
variance (sum of squared deviations from cluster means) at each step. This
produces compact, variance-based clusters that are well matched to standardised
Euclidean features and works well for discovering participant groups with
similar annotation-usage patterns.

The dendrogram shows participants at the leaves and merge heights that reflect
how dissimilar their profiles are; clusters that merge at low heights have
similar profiles, while large jumps in height indicate distinct groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from llm_delusions_annotations.annotation_metadata import (
    AnnotationMetadata,
    filter_analysis_metadata,
    is_role_in_scope,
)
from scipy.cluster.hierarchy import dendrogram, linkage

from analysis.lib.stats.clustering import cluster_and_order, rowwise_z_scores
from analysis.lib.utils.formatting import round3
from annotation.annotation_jobs import (
    ConversationKey,
    FamilyState,
    run_preprocessed_annotation_job,
)
from annotation.io import ParticipantMessageKey
from utils.cli import add_annotation_metadata_arguments, add_output_path_argument

DEFAULT_HEATMAP_PATH = Path(
    "analysis/figures/participant_profiles_heatmap_participants.pdf"
)
DEFAULT_DENDROGRAM_PATH = Path(
    "analysis/figures/participant_profiles_dendrogram_participants.pdf"
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the participant-profiles script."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute participant-level annotation profiles and clustered "
            "heatmaps from a preprocessed per-message annotation CSV."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "Preprocessed per-message annotation table (Parquet) produced by "
            "preprocess_annotation_family.py."
        ),
    )
    add_annotation_metadata_arguments(parser)
    parser.add_argument(
        "--min-messages-per-participant",
        type=int,
        default=0,
        help=(
            "Minimum number of scoped messages required for a participant to "
            "be included in the profiles (default: 0, keep all participants)."
        ),
    )
    parser.add_argument(
        "--min-nonmissing-per-annotation",
        type=int,
        default=0,
        help=(
            "Minimum number of participants with non-missing rates required "
            "for an annotation to be included (default: 0, keep all "
            "annotations)."
        ),
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["participants", "annotations"],
        default="participants",
        help=(
            "Axis to cluster for the heatmap and dendrogram. "
            "'participants' (default) clusters participants while keeping "
            "annotations in canonical order. 'annotations' clusters "
            "annotations while keeping participants in canonical order."
        ),
    )
    add_output_path_argument(
        parser,
        default_path=Path("analysis/data/participant_annotation_profiles.csv"),
        help_text=(
            "Output CSV path for the participant-level annotation profiles " "table."
        ),
    )
    parser.add_argument(
        "--heatmap-pdf",
        type=Path,
        default=DEFAULT_HEATMAP_PATH,
        help="Output PDF path for the clustered heatmap.",
    )
    parser.add_argument(
        "--dendrogram-pdf",
        type=Path,
        default=DEFAULT_DENDROGRAM_PATH,
        help="Output PDF path for the clustering dendrogram.",
    )
    parser.add_argument(
        "--onset-threshold-k",
        type=int,
        default=5,
        help=(
            "Positive-occurrence threshold K used when computing per-"
            "annotation onset indices over participant lifetimes "
            "(default: 5)."
        ),
    )
    parser.add_argument(
        "--participant-ordering-json",
        type=Path,
        default=Path("analysis") / "participant_ordering.json",
        help=(
            "Path to participant ordering metadata JSON produced by "
            "compute_participant_ordering_and_stats.py; used to restrict "
            "onset calculations to participants with a reliable global "
            "message ordering."
        ),
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the participant-profiles script."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.min_messages_per_participant < 0:
        parser.error("--min-messages-per-participant must be non-negative")
    if args.min_nonmissing_per_annotation < 0:
        parser.error("--min-nonmissing-per-annotation must be non-negative")
    if args.onset_threshold_k <= 0:
        parser.error("--onset-threshold-k must be a positive integer")
    return args


def _collect_participant_annotation_rates(
    message_info: Mapping[ParticipantMessageKey, tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    annotation_ids: Sequence[str],
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
]:
    """Return participants, annotations, and a rate matrix (annotations x participants).

    Each matrix entry is the per-participant rate for an annotation, defined as
    the number of positive scoped messages divided by the number of scoped
    messages where the annotation could apply. When a participant has no scoped
    messages for an annotation, the entry is NaN.
    """

    participants = sorted({key[0] for key in message_info})
    annotation_ids_list = list(annotation_ids)

    # Index mappings for matrix construction.
    participant_to_idx: Dict[str, int] = {pid: i for i, pid in enumerate(participants)}
    annotation_to_idx: Dict[str, int] = {
        aid: i for i, aid in enumerate(annotation_ids_list)
    }

    rates = np.full((len(annotation_ids_list), len(participants)), np.nan, dtype=float)

    # Pre-group message keys by participant for efficient access.
    messages_by_participant: Dict[str, list[ParticipantMessageKey]] = {
        pid: [] for pid in participants
    }
    for message_key, (role, _conv_key) in message_info.items():
        participant = message_key[0]
        if participant in messages_by_participant:
            messages_by_participant[participant].append(message_key)

    # Build rates per annotation and participant.
    for annotation_id in annotation_ids_list:
        meta = metadata_by_id[annotation_id]
        a_idx = annotation_to_idx[annotation_id]
        scope = meta.scope

        for participant in participants:
            p_idx = participant_to_idx[participant]
            messages = messages_by_participant.get(participant, [])
            denom = 0
            numer = 0
            for message_key in messages:
                role, _conv_key = message_info[message_key]
                if not is_role_in_scope(role, scope):
                    continue
                denom += 1
                if annotation_message_positive.get((annotation_id, message_key), False):
                    numer += 1
            if denom <= 0:
                continue
            rates[a_idx, p_idx] = numer / float(denom)

    return participants, annotation_ids_list, rates


def _load_participant_ordering(path: Path) -> Dict[str, str]:
    """Return participant ordering types keyed by participant id.

    Parameters
    ----------
    path:
        JSON file produced by ``compute_participant_ordering_and_stats.py``.

    Returns
    -------
    Dict[str, str]
        Mapping from participant id to ordering type string such as
        ``\"full_dated\"`` or ``\"global_order\"``. When the file does not
        exist or cannot be parsed, an empty mapping is returned.
    """

    resolved = path.expanduser().resolve()
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError:
        return {}
    except json.JSONDecodeError:
        return {}

    ordering: Dict[str, str] = {}
    if not isinstance(data, dict):
        return ordering
    for participant, info in data.items():
        if not isinstance(participant, str):
            continue
        if not isinstance(info, dict):
            continue
        raw_type = info.get("ordering_type")
        if isinstance(raw_type, str) and raw_type:
            ordering[participant] = raw_type
    return ordering


def _compute_annotation_onset_statistics(
    participants: list[str],
    annotation_ids: list[str],
    message_info: Mapping[ParticipantMessageKey, tuple[str, ConversationKey]],
    annotation_message_positive: Mapping[tuple[str, ParticipantMessageKey], bool],
    *,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    participant_ordering: Mapping[str, str],
    threshold_k: int,
) -> Dict[str, Dict[str, object]]:
    """Return onset statistics per annotation over participant lifetimes.

    For each annotation and participant with a reliable global ordering
    (``ordering_type`` equal to ``\"full_dated\"`` or ``\"global_order\"``),
    this helper orders the participant's messages by conversation and message
    indices, counts positive scoped occurrences, and records the lifetime
    index of the K-th positive occurrence when available.

    Parameters
    ----------
    participants:
        Participants included in the profiles table after filtering.
    annotation_ids:
        Annotation identifiers included in the profiles table.
    message_info:
        Mapping from participant message key to (role, conversation key).
    annotation_message_positive:
        Mapping from (annotation id, message key) pairs to a boolean flag
        indicating whether the annotation is positive on that message.
    metadata_by_id:
        Annotation metadata keyed by annotation id.
    participant_ordering:
        Mapping from participant id to ordering type string.
    threshold_k:
        Positive occurrence threshold K used to define onset indices.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Mapping from annotation id to a dictionary of onset summary fields.
    """

    # Only participants with a reliable global ordering contribute to onset
    # statistics. Others remain in the profiles table but are ignored here.
    allowed_ordering_types = {"full_dated", "global_order"}
    eligible_participants = [
        pid
        for pid in participants
        if participant_ordering.get(pid) in allowed_ordering_types
    ]
    if not eligible_participants:
        return {}

    # Pre-group and order messages per eligible participant.
    messages_by_participant: Dict[str, list[ParticipantMessageKey]] = {
        pid: [] for pid in eligible_participants
    }
    for message_key, (role, _conv_key) in message_info.items():
        participant = message_key[0]
        if participant in messages_by_participant:
            messages_by_participant[participant].append(message_key)

    ordered_messages_by_participant: Dict[str, list[ParticipantMessageKey]] = {}
    for participant, message_keys in messages_by_participant.items():
        # ParticipantMessageKey is
        # (participant, source_path, chat_index, message_index). For
        # participants with ordering_type equal to "global_order" or
        # "full_dated", these indices provide a stable ordinal timeline
        # across all messages even when transcripts span multiple files.
        # We therefore order messages lexicographically by source path,
        # then chat index, then message index.
        ordered = sorted(message_keys, key=lambda key: (key[1], key[2], key[3]))
        ordered_messages_by_participant[participant] = ordered

    results: Dict[str, Dict[str, object]] = {}

    for annotation_id in annotation_ids:
        meta = metadata_by_id.get(annotation_id)
        if meta is None:
            continue
        scope = meta.scope

        onset_indices: list[int] = []
        n_eligible_with_scoped_messages = 0

        for participant in eligible_participants:
            ordered_messages = ordered_messages_by_participant.get(participant, [])
            if not ordered_messages:
                continue

            positive_positions: list[int] = []
            lifetime_index = 0
            has_scoped_messages = False

            for message_key in ordered_messages:
                role, _conv_key = message_info[message_key]
                if not is_role_in_scope(role, scope):
                    continue
                has_scoped_messages = True
                lifetime_index += 1
                if annotation_message_positive.get((annotation_id, message_key), False):
                    positive_positions.append(lifetime_index)

            if not has_scoped_messages:
                continue
            n_eligible_with_scoped_messages += 1
            if len(positive_positions) >= threshold_k:
                onset_indices.append(positive_positions[threshold_k - 1])

        if not onset_indices or n_eligible_with_scoped_messages <= 0:
            continue

        n_onset = len(onset_indices)
        mean_onset = mean(onset_indices)
        std_onset = pstdev(onset_indices) if n_onset > 1 else 0.0
        se_onset = std_onset / math.sqrt(float(n_onset)) if n_onset > 0 else 0.0
        share = n_onset / float(n_eligible_with_scoped_messages)

        results[annotation_id] = {
            "onset_mean": round3(mean_onset),
            "onset_se": round3(se_onset),
            "onset_share_of_eligible": round3(share),
            "onset_n_eligible": n_eligible_with_scoped_messages,
        }

    return results


def _apply_filters(
    annotation_ids: list[str],
    participants: list[str],
    rates: np.ndarray,
    *,
    min_messages_per_participant: int,
    min_nonmissing_per_annotation: int,
) -> tuple[list[str], list[str], np.ndarray]:
    """Return filtered ids and rate matrix according to minimum thresholds."""

    # Participant filter: total number of non-NaN entries across annotations
    # is used as a proxy for having enough scoped messages.
    keep_participants: list[int] = []
    for j, _pid in enumerate(participants):
        non_missing = int(np.isfinite(rates[:, j]).sum())
        if non_missing >= min_messages_per_participant:
            keep_participants.append(j)

    if not keep_participants:
        return [], [], np.empty((0, 0), dtype=float)

    rates = rates[:, keep_participants]
    participants = [participants[j] for j in keep_participants]

    # Annotation filter: number of participants with non-NaN rates.
    keep_annotations: list[int] = []
    for i, _aid in enumerate(annotation_ids):
        non_missing = int(np.isfinite(rates[i, :]).sum())
        if non_missing >= min_nonmissing_per_annotation:
            keep_annotations.append(i)

    if not keep_annotations:
        return [], [], np.empty((0, 0), dtype=float)

    rates = rates[keep_annotations, :]
    annotation_ids = [annotation_ids[i] for i in keep_annotations]

    return participants, annotation_ids, rates


def _plot_dendrogram(
    z_matrix: np.ndarray,
    labels: list[str],
    *,
    axis: str,
    output_path: Path,
) -> None:
    """Render and save a dendrogram for the clustered axis as a PDF."""

    if z_matrix.size == 0 or len(labels) <= 1:
        return

    if axis == "participants":
        features = z_matrix.T
    else:
        features = z_matrix

    linkage_matrix = linkage(features, method="ward", metric="euclidean")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90)
    ax.set_ylabel("Cluster distance (Ward linkage)")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Wrote clustering dendrogram PDF to {output_path}")


def _plot_heatmap(
    z_matrix: np.ndarray,
    annotation_ids: list[str],
    participants: list[str],
    *,
    output_path: Path,
) -> None:
    """Render and save the z-scored annotation-by-participant heatmap as a PDF."""

    if z_matrix.size == 0:
        return

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6, len(participants) * 0.3), 8))
    im = ax.imshow(z_matrix, aspect="auto", cmap="coolwarm", vmin=-3.0, vmax=3.0)
    ax.set_yticks(range(len(annotation_ids)))
    ax.set_yticklabels(annotation_ids, fontsize=6)
    ax.set_xticks(range(len(participants)))
    ax.set_xticklabels(participants, rotation=90, fontsize=6)
    ax.set_xlabel("Participants")
    ax.set_ylabel("Annotations")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise z-score of participant rates")
    fig.tight_layout()
    fig.savefig(output_path, format="pdf")
    plt.close(fig)
    print(f"Wrote clustered heatmap PDF to {output_path}")


def _write_profiles_csv(
    output_csv: Path,
    annotation_ids: list[str],
    participants: list[str],
    rates: np.ndarray,
    onset_by_annotation: Mapping[str, Mapping[str, object]],
) -> None:
    """Write the participant-level profiles table to ``output_csv``."""
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "annotation_id",
        "onset_mean",
        "onset_se",
        "onset_share_of_eligible",
        "onset_n_eligible",
    ] + [f"rate__{pid}" for pid in participants]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, annotation_id in enumerate(annotation_ids):
            row: Dict[str, object] = {"annotation_id": annotation_id}
            onset_stats = onset_by_annotation.get(annotation_id, {})
            row["onset_mean"] = onset_stats.get("onset_mean", "")
            row["onset_se"] = onset_stats.get("onset_se", "")
            row["onset_share_of_eligible"] = onset_stats.get(
                "onset_share_of_eligible", ""
            )
            row["onset_n_eligible"] = onset_stats.get("onset_n_eligible", "")
            for j, participant in enumerate(participants):
                value = rates[i, j]
                row[f"rate__{participant}"] = (
                    round3(float(value)) if math.isfinite(value) else ""
                )
            writer.writerow(row)
    print(f"Wrote participant profiles CSV to {output_csv}")


def _run_profiles_analysis(
    _family_files: Sequence[Path],
    family_state: FamilyState,
    metadata_by_id: Mapping[str, AnnotationMetadata],
    cutoffs_by_id: Mapping[str, int],
    args: argparse.Namespace,
) -> int:
    """Callback used with run_annotation_job to compute profiles."""

    message_info, _conversation_messages, annotation_message_positive = family_state

    # Apply shared analysis filters so that excluded annotations (for example,
    # test-category or synthetic ids) do not participate in profile summaries.
    metadata_by_id = filter_analysis_metadata(metadata_by_id)

    # Restrict to annotations that both have LLM score cutoffs and appear in
    # the current job family so that synthetic or test labels without usable
    # outputs are excluded from downstream summaries.
    present_ids = {aid for (aid, _mkey) in annotation_message_positive.keys()}
    annotation_ids_active = [
        aid for aid in metadata_by_id if aid in present_ids and aid in cutoffs_by_id
    ]
    if not annotation_ids_active:
        print(
            "No annotations with both metrics and outputs were found for the "
            "selected job family."
        )
        return 0

    participants, annotation_ids, rates = _collect_participant_annotation_rates(
        message_info,
        annotation_message_positive,
        metadata_by_id=metadata_by_id,
        annotation_ids=annotation_ids_active,
    )
    if rates.size == 0:
        print("No participant-level rates could be computed.")
        return 0

    participants, annotation_ids, rates = _apply_filters(
        annotation_ids,
        participants,
        rates,
        min_messages_per_participant=args.min_messages_per_participant,
        min_nonmissing_per_annotation=args.min_nonmissing_per_annotation,
    )
    if rates.size == 0:
        print("All participants or annotations were filtered out.")
        return 0

    # Compute onset statistics over participant lifetimes for annotations
    # included in the filtered profiles table.
    participant_ordering = _load_participant_ordering(args.participant_ordering_json)
    onset_by_annotation = _compute_annotation_onset_statistics(
        participants,
        annotation_ids,
        message_info,
        annotation_message_positive,
        metadata_by_id=metadata_by_id,
        participant_ordering=participant_ordering,
        threshold_k=args.onset_threshold_k,
    )

    # Compute row-wise z-scores for clustering and heatmap colouring.
    z = rowwise_z_scores(rates)

    # Determine ordering based on the selected clustering axis.
    if args.cluster_mode == "participants":
        ordered_participants = cluster_and_order(z.T, participants)
        ordered_annotations = annotation_ids
    else:
        ordered_annotations = cluster_and_order(z, annotation_ids)
        ordered_participants = participants

    # Reindex rates and z-matrix according to the chosen orderings.
    pid_to_idx = {pid: i for i, pid in enumerate(participants)}
    aid_to_idx = {aid: i for i, aid in enumerate(annotation_ids)}
    ordered_rate_matrix = np.zeros(
        (len(ordered_annotations), len(ordered_participants)), dtype=float
    )
    ordered_rate_matrix.fill(np.nan)
    for i, aid in enumerate(ordered_annotations):
        src_i = aid_to_idx[aid]
        for j, pid in enumerate(ordered_participants):
            src_j = pid_to_idx[pid]
            ordered_rate_matrix[i, j] = rates[src_i, src_j]

    ordered_z = rowwise_z_scores(ordered_rate_matrix)

    # Write CSV table, heatmap, and dendrogram.
    _write_profiles_csv(
        args.output,
        ordered_annotations,
        ordered_participants,
        ordered_rate_matrix,
        onset_by_annotation,
    )
    _plot_heatmap(
        ordered_z,
        ordered_annotations,
        ordered_participants,
        output_path=args.heatmap_pdf,
    )
    _plot_dendrogram(
        ordered_z,
        (
            ordered_participants
            if args.cluster_mode == "participants"
            else ordered_annotations
        ),
        axis=args.cluster_mode,
        output_path=args.dendrogram_pdf,
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for computing participant-level profiles."""

    args = parse_args(argv)
    return run_preprocessed_annotation_job(args, _run_profiles_analysis)


if __name__ == "__main__":
    raise SystemExit(main())
