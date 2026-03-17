"""
Compute annotator agreement statistics for manual and LLM labels.

This utility bridges manual-annotation datasets, human label files, and LLM
classification outputs so that:

* Pairwise agreement metrics can be inspected in Python.
* A pre-normalized per-message dataset is written for a browser viewer that
  highlights agreement and disagreement cases.

Typical workflow
----------------

1. Prepare a manual-annotation dataset (single annotation id is common):

   python scripts/prepare_manual_annotation_dataset.py \
     --input transcripts_de_ided \
     --annotation user-misconstrues-sentience \
     --max-messages 500 \
     --randomize \
     --preceding-context 3

2. Collect human labels under ``manual_annotation_labels/<annotator>/...``
   using the manual annotator UI.

3. Run LLM classification over the underlying transcripts and save JSONL
   outputs under ``annotation_outputs/``.

4. Invoke this script to combine those sources and emit:

   * ``analysis/agreement_cases__<dataset-filename>.jsonl`` – one record per
     sampled message with labels from each annotator.
   * ``analysis/agreement_metrics__<dataset-filename>.json`` – pairwise
     agreement tables per annotation id.

The HTML viewer in ``analysis/viewer/annotation_agreement_viewer.html`` can then load
these files to provide an interactive review of disagreements.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.annotation_ids import (
    ROLE_SPLIT_BASE_IDS,
    normalize_annotation_id,
    normalize_role_token,
    role_prefix_for_role,
)
from llm_delusions_annotations.annotation_metadata import EXCLUDED_ANNOTATION_IDS
from llm_delusions_annotations.configs import LLM_SCORE_CUTOFF
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa

from annotation.agreement_metrics import (
    load_overall_llm_confusion_from_payload,
    load_per_annotation_llm_confusion_from_payload,
)
from annotation.annotation_tables import load_preprocessed_annotations_table
from annotation.io import (
    AnnotationOutputRun,
    iter_annotation_output_runs,
    iter_jsonl_dicts,
    normalize_source_path,
)
from utils.io import iter_objects_with_location
from utils.participants import normalize_participant_value

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetKey:
    """Stable identifier for a sampled manual-annotation item.

    Parameters
    ----------
    dataset_path:
        Path to the JSONL dataset file relative to the repository root.
    sequence_index:
        Integer sequence index assigned when the dataset was generated.
    participant:
        Participant identifier (for example, ``201``).
    annotation_id:
        Annotation identifier (for example, ``user-misconstrues-sentience``).
    """

    dataset_path: str
    sequence_index: int
    participant: str
    annotation_id: str


@dataclass(frozen=True)
class TranscriptKey:
    """Stable identifier for a transcript message location.

    This key is shared between manual-annotation items and LLM classification
    outputs so that agreement can be computed for the same chat turn.

    Parameters
    ----------
    participant:
        Participant identifier.
    source_path:
        Relative path to the underlying transcript JSON file.
    chat_index:
        Zero-based index of the conversation within the transcript file.
    message_index:
        Zero-based index of the message within the conversation.
    annotation_id:
        Annotation identifier.
    """

    participant: str
    source_path: str
    chat_index: int
    message_index: int
    annotation_id: str


@dataclass
class AnnotatorInfo:
    """Metadata describing a single annotator stream."""

    name: str
    kind: str  # "human" or "llm"
    source: str


def _read_jsonl(path: Path) -> Iterable[dict]:
    """Yield JSON objects from a newline-delimited JSON file."""

    try:
        yield from iter_jsonl_dicts(path)
    except OSError as err:
        raise ValueError(f"Failed to read {path}: {err}") from err


def _resolve_repo_root() -> Path:
    """Return the repository root assuming the script lives under scripts/."""

    this_file = Path(__file__).resolve()
    # The expected layout is ``<repo>/scripts/annotation/compute_annotation_agreement.py``.
    # Walking three parents from this file yields the repository root:
    # ``compute_annotation_agreement.py`` -> ``annotation`` -> ``scripts`` -> ``<repo>``.
    return this_file.parent.parent.parent


def _normalize_dataset_path(path: Path, repo_root: Path) -> Path:
    """Return the dataset path relative to the repository root when possible."""

    try:
        return path.resolve().relative_to(repo_root)
    except ValueError:
        return path.resolve()


def _load_dataset_meta(dataset_path: Path) -> dict:
    """Return the meta record from a manual-annotation dataset, if present.

    The first non-empty JSON line with ``type == \"meta\"`` is treated as the
    metadata row. When no such record is found, an empty dict is returned.
    """

    try:
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as err:
                    raise ValueError(
                        f"Invalid meta JSON in dataset {dataset_path}: {err}"
                    ) from err
                if isinstance(obj, dict) and obj.get("type") == "meta":
                    return obj
                # Stop once the first non-meta object is seen.
                break
    except OSError as err:
        raise ValueError(
            f"Failed to read dataset meta from {dataset_path}: {err}"
        ) from err
    return {}


def load_dataset_items(
    dataset_path: Path, annotation_filter: Optional[set[str]], repo_root: Path
) -> Tuple[Dict[DatasetKey, dict], Dict[DatasetKey, TranscriptKey]]:
    """Load manual-annotation items into keyed dictionaries.

    Parameters
    ----------
    dataset_path:
        JSONL file produced by ``prepare_manual_annotation_dataset.py``.
    annotation_filter:
        Optional annotation id to restrict items; when ``None``, all annotation
        ids present in the dataset are included.
    repo_root:
        Repository root used to normalize paths.

    Returns
    -------
    items_by_key:
        Mapping from :class:`DatasetKey` to the raw item dict.
    transcript_key_by_dataset:
        Mapping from :class:`DatasetKey` to the corresponding
        :class:`TranscriptKey` for joining with LLM results.
    """

    normalized_dataset = _normalize_dataset_path(dataset_path, repo_root)
    dataset_str = str(normalized_dataset).replace("\\", "/")

    items_by_key: Dict[DatasetKey, dict] = {}
    transcript_key_by_dataset: Dict[DatasetKey, TranscriptKey] = {}

    for obj in _read_jsonl(dataset_path):
        if obj.get("type") != "item":
            continue

        raw_annotation_id = str(obj.get("annotation_id") or "").strip()
        if not raw_annotation_id:
            continue
        role = str(obj.get("role") or "").strip()
        annotation_id = normalize_annotation_id(
            raw_annotation_id,
            role=role,
            strict_role=True,
        )
        if annotation_filter and annotation_id not in annotation_filter:
            continue

        participant = normalize_participant_value(obj.get("participant"))
        if not participant:
            continue

        try:
            sequence_index = int(obj.get("sequence_index"))
        except (TypeError, ValueError):
            continue

        key = DatasetKey(
            dataset_path=dataset_str,
            sequence_index=sequence_index,
            participant=participant,
            annotation_id=annotation_id,
        )
        items_by_key[key] = obj

        raw_source_path = str(obj.get("source_path") or "").strip()
        source_path = normalize_source_path(participant, raw_source_path)
        try:
            chat_index = int(obj.get("chat_index"))
            message_index = int(obj.get("message_index"))
        except (TypeError, ValueError):
            continue

        transcript_key_by_dataset[key] = TranscriptKey(
            participant=participant,
            source_path=source_path,
            chat_index=chat_index,
            message_index=message_index,
            annotation_id=annotation_id,
        )

    return items_by_key, transcript_key_by_dataset


def load_manual_labels_for_annotator(
    dataset_path: Path,
    annotator_id: str,
    annotation_filter: Optional[set[str]],
    repo_root: Path,
) -> Mapping[DatasetKey, bool]:
    """Load manual labels for one annotator keyed by dataset item.

    The manual annotator UI writes label files under
    ``manual_annotation_labels/<annotator>/<dataset-filename>.jsonl`` with one
    JSON object per labeled message. Each row includes a ``sequence_index`` and
    ``participant`` allowing us to join back to the originating item.
    """

    safe_annotator = annotator_id.strip()
    if not safe_annotator:
        raise ValueError("Annotator id must be non-empty")

    dataset_filename = dataset_path.name
    labels_root = repo_root / "manual_annotation_labels" / safe_annotator
    label_path = labels_root / dataset_filename
    if not label_path.exists():
        raise ValueError(
            f"No labels found for annotator {annotator_id!r} at {label_path}"
        )

    normalized_dataset = _normalize_dataset_path(dataset_path, repo_root)
    dataset_str = str(normalized_dataset).replace("\\", "/")

    by_key: dict[DatasetKey, bool] = {}
    for obj in _read_jsonl(label_path):
        raw_annotation_id = str(obj.get("annotation_id") or "").strip()
        if not raw_annotation_id:
            continue
        role = str(obj.get("role") or "").strip()
        annotation_id = normalize_annotation_id(
            raw_annotation_id,
            role=role,
            strict_role=True,
        )
        if annotation_filter and annotation_id not in annotation_filter:
            continue

        participant = normalize_participant_value(obj.get("participant"))
        if not participant:
            continue

        try:
            sequence_index = int(obj.get("sequence_index"))
        except (TypeError, ValueError):
            continue

        label_value = str(obj.get("label") or "").strip().lower()
        if label_value not in {"yes", "no"}:
            continue
        is_positive = label_value == "yes"

        key = DatasetKey(
            dataset_path=dataset_str,
            sequence_index=sequence_index,
            participant=participant,
            annotation_id=annotation_id,
        )
        by_key[key] = is_positive

    return by_key


def _merge_labels_from_side_datasets(
    *,
    base_transcript_key_by_dataset: Mapping[DatasetKey, TranscriptKey],
    repo_root: Path,
    annotation_filter: Optional[set[str]],
    annotator_id: str,
    existing_labels: Mapping[DatasetKey, bool],
    manual_label_datasets: Sequence[Path],
) -> Dict[DatasetKey, bool]:
    """Return labels extended with those from additional manual-label datasets.

    Parameters
    ----------
    base_transcript_key_by_dataset:
        Mapping from primary :class:`DatasetKey` to :class:`TranscriptKey`
        describing the transcript location for each item.
    repo_root:
        Repository root used to normalize paths.
    annotation_filter:
        Optional annotation id to restrict labels.
    annotator_id:
        Manual annotator identifier.
    existing_labels:
        Labels loaded directly for ``base_dataset_path`` (may be empty).
    manual_label_datasets:
        Additional manual-annotation datasets whose label files should be
        reused as side sources for this annotator. Only transcript-matching
        labels are projected; items from these datasets themselves are not
        included in the evaluation set.
    """

    labels: Dict[DatasetKey, bool] = dict(existing_labels)

    if not manual_label_datasets:
        return labels

    for side_raw in manual_label_datasets:
        side_path = Path(side_raw).expanduser()
        if not side_path.exists():
            LOGGER.warning(
                "Manual label dataset not found for annotator %r: %s",
                annotator_id,
                side_path,
            )
            continue
        try:
            (
                _side_items_by_key,
                side_transcript_key_by_dataset,
            ) = load_dataset_items(side_path, annotation_filter, repo_root)
        except ValueError as err:
            LOGGER.warning(
                "Skipping manual-label dataset %s for annotator %r: %s",
                side_path,
                annotator_id,
                err,
            )
            continue
        try:
            side_labels_by_key = load_manual_labels_for_annotator(
                side_path, annotator_id, annotation_filter, repo_root
            )
        except ValueError:
            # No labels for this annotator on this side dataset.
            continue

        # Map side labels to transcript keys so they can be projected
        # onto the primary dataset's item keys.
        side_by_transcript: Dict[TranscriptKey, bool] = {}
        for key, value in side_labels_by_key.items():
            tkey = side_transcript_key_by_dataset.get(key)
            if tkey is not None:
                side_by_transcript[tkey] = value

        # Project side labels into the primary dataset's key space.
        for dkey, tkey in base_transcript_key_by_dataset.items():
            if dkey in labels:
                continue
            value = side_by_transcript.get(tkey)
            if value is not None:
                labels[dkey] = value

    return labels


def load_llm_labels(
    jsonl_paths: Sequence[Path],
    annotation_filter: Optional[set[str]],
    repo_root: Path,
    score_cutoff: int,
    per_annotation_cutoffs: Optional[Mapping[str, int]] = None,
) -> Tuple[
    Dict[str, AnnotatorInfo],
    Dict[str, Dict[TranscriptKey, bool]],
    Dict[str, Dict[TranscriptKey, List[str]]],
    Dict[str, Dict[TranscriptKey, Dict[str, object]]],
]:
    """Load LLM classification outputs into per-model label maps.

    Parameters
    ----------
    jsonl_paths:
        Collection of JSONL classification outputs produced by
        ``classify_chats.py``.
    annotation_filter:
        Optional annotation id to restrict items; when ``None``, all annotation
        ids are included.
    repo_root:
        Repository root used to normalize paths.

    Returns
    -------
    annotators:
        Mapping from annotator name to :class:`AnnotatorInfo`.
    labels_by_annotator:
        Mapping from annotator name to a dict keyed by :class:`TranscriptKey`
        with boolean labels (``True`` for at least one span match).
    """

    annotators: Dict[str, AnnotatorInfo] = {}
    labels_by_annotator: Dict[str, Dict[TranscriptKey, bool]] = defaultdict(dict)
    matches_by_annotator: Dict[str, Dict[TranscriptKey, List[str]]] = defaultdict(dict)
    details_by_annotator: Dict[str, Dict[TranscriptKey, Dict[str, object]]] = (
        defaultdict(dict)
    )
    model_sources: Dict[str, set[str]] = defaultdict(set)

    for path in jsonl_paths:
        normalized = _normalize_dataset_path(path, repo_root)
        source_str = str(normalized).replace("\\", "/")
        for obj, source_path, chat_index, message_index in iter_objects_with_location(
            _read_jsonl(path)
        ):
            if obj.get("type") == "meta":
                continue

            raw_annotation_id = str(obj.get("annotation_id") or "").strip()
            if not raw_annotation_id:
                continue
            role = str(obj.get("role") or "").strip()
            annotation_id = normalize_annotation_id(
                raw_annotation_id,
                role=role,
                strict_role=True,
            )
            if annotation_filter and annotation_id not in annotation_filter:
                continue

            participant = normalize_participant_value(obj.get("participant"))
            if not participant:
                continue

            source_path = normalize_source_path(participant, source_path)

            model = str(obj.get("model") or "").strip() or "unknown-model"
            annotator_name = model
            if annotator_name not in annotators:
                annotators[annotator_name] = AnnotatorInfo(
                    name=annotator_name,
                    kind="llm",
                    source=source_str,
                )
            model_sources[annotator_name].add(source_str)

            score_value = obj.get("score")
            if not isinstance(score_value, (int, float)):
                continue
            rounded = int(round(float(score_value)))
            threshold = score_cutoff
            if per_annotation_cutoffs:
                try:
                    threshold = int(
                        per_annotation_cutoffs.get(annotation_id, threshold)
                    )
                except (TypeError, ValueError):
                    threshold = score_cutoff
            is_positive = rounded >= threshold

            raw_matches = obj.get("matches")
            matches: List[str] = []
            if isinstance(raw_matches, list):
                for value in raw_matches:
                    if isinstance(value, str):
                        text = value.strip()
                        if text:
                            matches.append(text)

            key = TranscriptKey(
                participant=participant,
                source_path=source_path,
                chat_index=chat_index,
                message_index=message_index,
                annotation_id=annotation_id,
            )
            labels_by_annotator[annotator_name][key] = is_positive
            if is_positive and matches:
                matches_by_annotator[annotator_name][key] = matches

            detail_entry: Dict[str, object] = {
                "score": rounded,
            }
            if isinstance(obj.get("rationale"), str):
                detail_entry["rationale"] = obj.get("rationale")
            if isinstance(obj.get("reasoning_content"), str):
                detail_entry["reasoning_content"] = obj.get("reasoning_content")
            if isinstance(obj.get("thinking_blocks"), list):
                detail_entry["thinking_blocks"] = obj.get("thinking_blocks")
            details_by_annotator[annotator_name][key] = detail_entry

    # Enrich annotator source text to reflect all contributing files so it is
    # obvious in the UI that multiple participant runs are included.
    for name, sources in model_sources.items():
        info = annotators.get(name)
        if not info:
            continue
        unique = sorted(sources)
        if len(unique) == 1:
            info.source = unique[0]
        else:
            first = unique[0]
            remaining = len(unique) - 1
            info.source = f"{first} (+{remaining} more files)"

    return annotators, labels_by_annotator, matches_by_annotator, details_by_annotator


def load_llm_labels_from_preprocessed_table(
    frame,
    transcript_key_by_dataset: Mapping[DatasetKey, TranscriptKey],
    annotation_filter: Optional[set[str]],
    score_cutoff: int,
    per_annotation_cutoffs: Optional[Mapping[str, int]] = None,
    annotator_name: str = "llm-preprocessed",
    source_label: Optional[str] = None,
) -> Tuple[
    Dict[str, AnnotatorInfo],
    Dict[str, Dict[TranscriptKey, bool]],
    Dict[str, Dict[TranscriptKey, List[str]]],
    Dict[str, Dict[TranscriptKey, Dict[str, object]]],
]:
    """Load LLM labels from a preprocessed annotations table.

    This helper mirrors :func:`load_llm_labels` but consumes a wide
    per-message annotations table (for example,
    ``all_annotations__preprocessed.parquet``) instead of raw JSONL
    classification outputs. Score columns are expected to follow the
    ``score__<annotation_id>`` naming convention.

    Parameters
    ----------
    frame:
        Pandas DataFrame containing one row per message with location
        columns (participant, source_path, chat_index, message_index) and
        score columns for each annotation id.
    transcript_key_by_dataset:
        Mapping from :class:`DatasetKey` to :class:`TranscriptKey` for the
        items being evaluated. This is used to probe the preprocessed table
        only at locations that appear in the dataset.
    annotation_filter:
        Optional annotation id to restrict items; when ``None``, all ids
        present in the dataset are considered.
    score_cutoff:
        Global score cutoff used when binarizing scores in the absence of
        a per-annotation mapping.
    per_annotation_cutoffs:
        Optional mapping from annotation id to integer score cutoff.
        When provided, these values override ``score_cutoff`` on a per-id
        basis.
    annotator_name:
        Name to use for the synthetic LLM annotator backed by the
        preprocessed table.
    source_label:
        Optional descriptive source string recorded on the returned
        :class:`AnnotatorInfo`. When omitted, a generic label is used.
    """

    # Ensure the required location columns are present.
    required_columns = {"participant", "source_path", "chat_index", "message_index"}
    if not required_columns.issubset(set(frame.columns)):
        LOGGER.error(
            "Preprocessed annotations table is missing required columns %s; "
            "found columns: %s",
            sorted(required_columns),
            sorted(frame.columns),
        )
        return {}, {}, {}, {}

    # Restrict to a MultiIndex keyed by the transcript location so that
    # lookups for dataset items are efficient and concise.
    indexed = frame.set_index(
        ["participant", "source_path", "chat_index", "message_index"]
    )

    labels_for_annotator: Dict[TranscriptKey, bool] = {}
    details_for_annotator: Dict[TranscriptKey, Dict[str, object]] = {}

    for tkey in transcript_key_by_dataset.values():
        if annotation_filter and tkey.annotation_id not in annotation_filter:
            continue

        location = (
            tkey.participant,
            tkey.source_path,
            tkey.chat_index,
            tkey.message_index,
        )

        base_id = tkey.annotation_id

        # Prefer role-split score columns when available for selected
        # annotations so that LLM labels align with the user/assistant
        # variants used elsewhere in the analyses. When a suitable
        # role-specific column is not present, fall back to the base
        # ``score__<annotation_id>`` column.
        column_candidates = []
        if base_id in ROLE_SPLIT_BASE_IDS:
            try:
                row_role_raw = indexed.at[location, "role"]
            except KeyError:
                row_role_raw = ""
            row_role = normalize_role_token(str(row_role_raw or ""))
            if row_role:
                role_prefix = role_prefix_for_role(row_role, strict=True)
                split_column = f"score__{role_prefix}-{base_id}"
                column_candidates.append(split_column)

        column_candidates.append(f"score__{base_id}")

        score_value = None
        column_found = None
        for candidate in column_candidates:
            if candidate not in indexed.columns:
                continue
            try:
                score_value = indexed.at[location, candidate]
            except KeyError:
                continue
            column_found = candidate
            break

        if column_found is None:
            continue

        if score_value is None:
            continue

        try:
            numeric_score = float(score_value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric_score):
            continue

        threshold = score_cutoff
        if per_annotation_cutoffs:
            try:
                threshold = int(
                    per_annotation_cutoffs.get(tkey.annotation_id, threshold)
                )
            except (TypeError, ValueError):
                threshold = score_cutoff

        rounded = int(round(numeric_score))
        is_positive = rounded >= threshold
        labels_for_annotator[tkey] = is_positive
        details_for_annotator[tkey] = {"score": rounded}

    if not labels_for_annotator:
        return {}, {}, {}, {}

    annotators: Dict[str, AnnotatorInfo] = {
        annotator_name: AnnotatorInfo(
            name=annotator_name,
            kind="llm",
            source=source_label or "preprocessed-table",
        )
    }
    labels_by_annotator: Dict[str, Dict[TranscriptKey, bool]] = {
        annotator_name: labels_for_annotator
    }
    matches_by_annotator: Dict[str, Dict[TranscriptKey, List[str]]] = {
        annotator_name: {}
    }
    details_by_annotator: Dict[str, Dict[TranscriptKey, Dict[str, object]]] = {
        annotator_name: details_for_annotator
    }

    return annotators, labels_by_annotator, matches_by_annotator, details_by_annotator


def _auto_discover_llm_runs(
    *,
    dataset_meta: Mapping[str, object],
    repo_root: Path,
    annotation_filter: Optional[set[str]],
) -> List[Path]:
    """Return LLM classification JSONL paths that match the dataset parameters.

    Matching is conservative to avoid surprising pairings. Candidates must:

    * Include the target annotation id (``annotation_filter``) when provided,
      otherwise include all dataset ``annotation_ids``.
    * Share the same ``preceding_context`` value when it is present on both
      sides.
    * Include all dataset participants (when both sides report participants).
    """

    annotation_ids_raw = dataset_meta.get("annotation_ids") or []
    dataset_annotation_ids = [
        str(value).strip()
        for value in annotation_ids_raw
        if isinstance(value, str) and value.strip()
    ]

    participants_raw = (
        (dataset_meta.get("parameters") or {}).get("participants")
        if isinstance(dataset_meta.get("parameters"), dict)
        else []
    )
    dataset_participants = [
        normalize_participant_value(value)
        for value in (participants_raw or [])
        if isinstance(value, str) and value.strip()
    ]

    preceding_dataset = dataset_meta.get("preceding_context")
    try:
        preceding_dataset_int = (
            int(preceding_dataset) if preceding_dataset is not None else None
        )
    except (TypeError, ValueError):
        preceding_dataset_int = None

    runs: List[AnnotationOutputRun] = list(
        iter_annotation_output_runs(repo_root / "annotation_outputs")
    )
    candidates: List[Path] = []
    for run in runs:
        llm_annotation_ids = {ann for ann in run.annotation_ids if ann}
        if not llm_annotation_ids:
            continue

        # When a specific annotation id is requested, require that exact id to
        # appear in the LLM run. Otherwise, require that there is at least one
        # overlapping annotation id. Runs may include only a subset of the
        # dataset's annotations; agreement is computed per-annotation, so full
        # coverage is not required.
        if annotation_filter:
            if not annotation_filter.intersection(llm_annotation_ids):
                continue
        elif dataset_annotation_ids:
            if not set(dataset_annotation_ids).intersection(llm_annotation_ids):
                continue

        if dataset_participants and run.participants:
            # Require at least one overlapping participant between the dataset
            # and the run. This ensures we include all runs that could
            # contribute labels for any participant present in the dataset,
            # even when different runs target different participant subsets.
            if not set(dataset_participants).intersection(set(run.participants)):
                continue

        if (
            preceding_dataset_int is not None
            and run.preceding_context is not None
            and preceding_dataset_int != run.preceding_context
        ):
            continue

        candidates.append(run.path)

    return candidates


def _auto_discover_manual_annotators(
    dataset_path: Path,
    repo_root: Path,
) -> List[str]:
    """Return manual annotator ids that have labels for the dataset.

    This scans ``manual_annotation_labels/<annotator_id>/`` for a JSONL file
    matching the dataset filename and treats each matching subdirectory name
    as a manual annotator id.
    """

    labels_root = repo_root / "manual_annotation_labels"
    if not labels_root.exists():
        return []

    dataset_filename = dataset_path.name
    annotator_ids: List[str] = []
    for entry in sorted(labels_root.iterdir()):
        if not entry.is_dir():
            continue
        label_path = entry / dataset_filename
        if label_path.exists():
            annotator_ids.append(entry.name)
    return annotator_ids


def _llm_run_family_prefix(basename: str) -> Optional[str]:
    """Return a partition-family prefix for an LLM run basename.

    This helper recognizes filenames that follow a partitioned pattern such
    as ``all_annotations__part-0001.jsonl`` and returns a prefix that is
    shared across all partitions in the same family. For example, the
    family prefix for ``all_annotations__part-0001.jsonl`` is
    ``all_annotations__``, which also matches ``all_annotations__part-0002.jsonl``.

    When the basename does not contain a ``\"__part-\"`` segment, ``None`` is
    returned and only exact basename matches are considered.

    Parameters
    ----------
    basename:
        Filename component of the LLM run path.

    Returns
    -------
    Optional[str]
        Shared family prefix for partitioned runs or ``None`` when the
        basename does not follow the recognized pattern.
    """

    marker = "__part-"
    index = basename.find(marker)
    if index < 0:
        return None

    # Include the trailing separator so that prefixes such as
    # ``all_annotations__`` will match all ``all_annotations__part-XXXX.jsonl``
    # files but avoid overly broad matches against unrelated files.
    return basename[: index + 2]


def _select_llm_paths_for_basenames(
    *,
    repo_root: Path,
    llm_run_basenames: Sequence[str],
) -> List[Path]:
    """Return LLM run paths matching one or more basenames or partition families.

    This helper centralizes basename and family-partition matching so that
    single-dataset and multi-dataset evaluation paths behave identically when
    ``--llm-run-basename`` is provided. Selection proceeds in two stages:

    1. Exact basename matches for backward compatibility.
    2. Partition-family matches for any basename that looks like
       ``all_annotations__part-0001.jsonl`` – all siblings starting with the
       same family prefix (for example, ``all_annotations__``) are included.
    """

    basenames = {Path(value).name for value in llm_run_basenames if value}
    if not basenames:
        return []

    runs: List[AnnotationOutputRun] = list(
        iter_annotation_output_runs(repo_root / "annotation_outputs")
    )

    # Start with exact basename matches.
    selected_paths: set[Path] = {run.path for run in runs if run.path.name in basenames}

    # Extend selection with partition siblings that share a family prefix.
    family_prefixes: List[str] = []
    for basename in basenames:
        prefix = _llm_run_family_prefix(basename)
        if prefix:
            family_prefixes.append(prefix)

    if family_prefixes:
        for run in runs:
            name = run.path.name
            for prefix in family_prefixes:
                if name.startswith(prefix):
                    selected_paths.add(run.path)
                    break

    return sorted(selected_paths)


def _compute_pairwise_counts(
    values_a: Mapping[DatasetKey, bool], values_b: Mapping[DatasetKey, bool]
) -> Tuple[int, Counter]:
    """Return overlap size and confusion-counts for two annotators."""

    counts: Counter = Counter()
    overlap = 0
    for key, a_val in values_a.items():
        if key not in values_b:
            continue
        b_val = values_b[key]
        overlap += 1
        if a_val and b_val:
            counts["yes_yes"] += 1
        elif (not a_val) and (not b_val):
            counts["no_no"] += 1
        elif a_val and (not b_val):
            counts["yes_no"] += 1
        elif (not a_val) and b_val:
            counts["no_yes"] += 1
    return overlap, counts


def _cohen_kappa(counts: Mapping[str, int]) -> Optional[float]:
    """Compute Cohen's kappa from binary confusion counts.

    The ``counts`` mapping must include ``yes_yes``, ``no_no``, ``yes_no``,
    and ``no_yes`` keys. Returns ``None`` when there is no overlap or the
    kappa value is undefined.
    """

    n11 = int(counts.get("yes_yes", 0))
    n00 = int(counts.get("no_no", 0))
    n10 = int(counts.get("yes_no", 0))
    n01 = int(counts.get("no_yes", 0))
    total = n11 + n00 + n10 + n01
    if total <= 0:
        return None

    # Short-circuit degenerate cases where the expected agreement is 1.0 and
    # Cohen's kappa is undefined (this also avoids divide-by-zero warnings
    # inside statsmodels).
    p_a_yes = (n11 + n10) / float(total)
    p_a_no = (n00 + n01) / float(total)
    p_b_yes = (n11 + n01) / float(total)
    p_b_no = (n00 + n10) / float(total)
    expected = (p_a_yes * p_b_yes) + (p_a_no * p_b_no)
    denom = 1.0 - expected
    if denom <= 0.0:
        return None

    table = [[n00, n01], [n10, n11]]
    try:
        kappa_value = cohens_kappa(table, return_results=False)
    except (TypeError, ValueError):
        return None

    try:
        return float(kappa_value)
    except (TypeError, ValueError):
        return None


def compute_pairwise_metrics(
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
) -> List[dict]:
    """Compute pairwise agreement metrics for all annotator pairs."""

    names = sorted(annotator_values.keys())
    results: List[dict] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            values_a = annotator_values[name_a]
            values_b = annotator_values[name_b]
            overlap, counts = _compute_pairwise_counts(values_a, values_b)
            if overlap <= 0:
                continue
            agreement = float(
                (counts.get("yes_yes", 0) + counts.get("no_no", 0)) / overlap
            )
            kappa = _cohen_kappa(counts)
            results.append(
                {
                    "annotator_a": name_a,
                    "annotator_b": name_b,
                    "n_items": int(overlap),
                    "n_agree": int(counts.get("yes_yes", 0) + counts.get("no_no", 0)),
                    "n_disagree": int(
                        counts.get("yes_no", 0) + counts.get("no_yes", 0)
                    ),
                    "agreement_rate": agreement,
                    "cohen_kappa": kappa,
                    "counts": {
                        "yes_yes": int(counts.get("yes_yes", 0)),
                        "no_no": int(counts.get("no_no", 0)),
                        "yes_no": int(counts.get("yes_no", 0)),
                        "no_yes": int(counts.get("no_yes", 0)),
                    },
                }
            )
    return results


def _compute_multi_rater_human_iaa(
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
    human_names: Sequence[str],
    context: Optional[str] = None,
) -> Optional[Mapping[str, object]]:
    """Return multi-rater inter-annotator agreement stats for human annotators.

    This uses the canonical Fleiss' kappa formulation for binary labels.
    Items are restricted to those that have labels from the same maximum
    number of human raters observed in the dataset (at least two); items with
    fewer labels are ignored so that the rater count per item is constant as
    required by Fleiss' definition. For each included item we count how many
    humans said ``yes`` and how many said ``no`` and then classify the item
    as:

    * ``pos_agree`` – all humans labeled ``yes``.
    * ``neg_agree`` – all humans labeled ``no``.
    * ``pos_disagree`` – mixed labels with a strict yes-majority.
    * ``neg_disagree`` – mixed labels with a strict no-majority.
    * Tied mixed cases (equal yes/no counts) contribute to the total item
      count but not to ``pos_disagree`` or ``neg_disagree``.

    A simple percent-agreement rate and Fleiss' multi-rater kappa are
    computed from these counts when possible.
    """

    # Collect human labels per item and track which human raters labeled
    # each item so that we can select a constant rater count for Fleiss'
    # kappa. Different items may be labeled by different human subsets; only
    # items with the maximum rater count (at least two) are used. When the
    # number of unique human annotators per item is not constant, we log a
    # detailed diagnostic describing which items are under- or over-labeled.
    labels_by_item: Dict[DatasetKey, Dict[str, bool]] = defaultdict(dict)
    for name in human_names:
        values = annotator_values.get(name)
        if not values:
            continue
        for key, label in values.items():
            labels_by_item[key][name] = bool(label)

    if not labels_by_item:
        return None

    # Determine how many unique human annotators labeled each item.
    rater_counts_by_item: Dict[DatasetKey, int] = {
        key: len(labels) for key, labels in labels_by_item.items()
    }
    if not rater_counts_by_item:
        return None

    unique_rater_counts = sorted(set(rater_counts_by_item.values()))
    max_raters = max(unique_rater_counts)

    if max_raters < 2:
        return None

    # When items have different numbers of human annotators, emit a clear
    # diagnostic so dataset issues are easy to spot. Fleiss' kappa will still
    # be computed over the subset of items with the maximum rater count.
    if len(unique_rater_counts) > 1:
        context_label = context or "multi-rater human IAA"
        LOGGER.warning(
            "Inconsistent number of unique human annotators per item when "
            "computing Fleiss' kappa for %s: observed rater counts %s; "
            "only items with %d human labels will be included.",
            context_label,
            ", ".join(str(value) for value in unique_rater_counts),
            max_raters,
        )

        # Provide per-item detail showing which annotators contributed and
        # which dataset file the item came from so coverage gaps are obvious.
        for key, labels in labels_by_item.items():
            rater_count = len(labels)
            if rater_count == max_raters:
                continue
            annotator_list = ", ".join(sorted(labels.keys()))
            search_id = (
                f"{key.dataset_path}|{key.annotation_id}|"
                f"{key.sequence_index}|{key.participant}"
            )
            LOGGER.debug(
                "Fleiss' kappa coverage mismatch for %s: "
                "annotation_id=%s dataset=%s sequence_index=%d participant=%s "
                "human_annotators=[%s] label_id=%s "
                "(expected %d annotators, found %d)",
                context_label,
                key.annotation_id,
                key.dataset_path,
                key.sequence_index,
                key.participant,
                annotator_list,
                search_id,
                max_raters,
                rater_count,
            )

    n_raters = max_raters
    n_items = 0
    pos_agree = 0
    neg_agree = 0
    pos_disagree = 0
    neg_disagree = 0
    total_yes = 0
    total_no = 0
    rating_rows: List[List[int]] = []

    for key, labels in labels_by_item.items():
        if len(labels) != n_raters:
            continue
        n_items += 1
        yes_i = sum(1 for value in labels.values() if value)
        no_i = n_raters - yes_i
        total_yes += yes_i
        total_no += no_i

        if yes_i == n_raters:
            pos_agree += 1
        elif no_i == n_raters:
            neg_agree += 1
        else:
            if yes_i > no_i:
                pos_disagree += 1
            elif no_i > yes_i:
                neg_disagree += 1

        rating_rows.append([no_i, yes_i])

    if n_items <= 0:
        return None

    agree_items = pos_agree + neg_agree
    disagree_items = n_items - agree_items
    agreement_rate: Optional[float]
    if n_items > 0:
        agreement_rate = agree_items / float(n_items)
    else:
        agreement_rate = None

    # Fleiss' multi-rater kappa for binary labels with a fixed number of
    # raters per item, computed via statsmodels. When all ratings fall into
    # a single category across the panel, the expected agreement is 1.0 and
    # the kappa denominator is zero; in that degenerate case we treat kappa
    # as undefined and avoid calling the library helper.
    kappa: Optional[float] = None
    if rating_rows and total_yes > 0 and total_no > 0:
        kappa_value = fleiss_kappa(rating_rows, method="fleiss")
        if isinstance(kappa_value, (int, float)):
            kappa = float(kappa_value)

    return {
        "n_items": int(n_items),
        "pos_agree": int(pos_agree),
        "neg_agree": int(neg_agree),
        "pos_disagree": int(pos_disagree),
        "neg_disagree": int(neg_disagree),
        "disagree": int(disagree_items),
        "agreement": float(agreement_rate) if agreement_rate is not None else None,
        "kappa": float(kappa) if kappa is not None else None,
    }


def _summarize_human_label_coverage(
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
    human_names: Sequence[str],
) -> Mapping[str, int]:
    """Return counts of items by number of human labels.

    Parameters
    ----------
    annotator_values:
        Mapping from annotator name to a mapping of :class:`DatasetKey` to
        boolean label values.
    human_names:
        Sequence of annotator names that should be treated as humans.

    Returns
    -------
    Mapping[str, int]
        Mapping with keys:

        * ``total_items`` – items that have at least one human label.
        * ``single_label_items`` – items that have exactly one human label.
        * ``multi_label_items`` – items that have two or more human labels.
    """

    label_counts: Dict[DatasetKey, int] = defaultdict(int)
    for name in human_names:
        values = annotator_values.get(name)
        if not values:
            continue
        for key in values.keys():
            label_counts[key] += 1

    total_items = 0
    single_label_items = 0
    multi_label_items = 0
    for count in label_counts.values():
        total_items += 1
        if count == 1:
            single_label_items += 1
        elif count >= 2:
            multi_label_items += 1

    return {
        "total_items": int(total_items),
        "single_label_items": int(single_label_items),
        "multi_label_items": int(multi_label_items),
    }


def _compute_majority_labels(
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
    human_annotators: Sequence[str],
) -> Dict[DatasetKey, bool]:
    """Return majority-vote labels from human annotators keyed by dataset item.

    Items with no human labels or ties between positive and negative votes are
    excluded from the returned mapping.
    """

    label_lists: Dict[DatasetKey, List[bool]] = defaultdict(list)
    for name in human_annotators:
        values = annotator_values.get(name)
        if not values:
            continue
        for key, label in values.items():
            label_lists[key].append(bool(label))

    majority_labels: Dict[DatasetKey, bool] = {}
    for key, labels in label_lists.items():
        positive = sum(1 for value in labels if value)
        negative = sum(1 for value in labels if not value)
        if positive == negative:
            continue
        majority_labels[key] = positive > negative

    return majority_labels


def _compute_confusion_against_majority(
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
    annotators: Sequence[AnnotatorInfo],
    context: Optional[str] = None,
) -> List[dict]:
    """Return confusion statistics for non-human annotators versus majority-vote.

    The majority vote is computed over human annotators only. For each
    non-human annotator (typically an LLM), items are restricted to those
    where both the majority label and the annotator's label are present.
    """

    human_names = [info.name for info in annotators if info.kind == "human"]
    if not human_names:
        return []

    majority_labels = _compute_majority_labels(annotator_values, human_names)
    if not majority_labels:
        if context:
            LOGGER.warning(
                "No items with a non-tied human majority for %s; "
                "majority-based confusion metrics will be empty.",
                context,
            )
        else:
            LOGGER.warning(
                "No items with a non-tied human majority; "
                "majority-based confusion metrics will be empty."
            )
        return []

    total_items = len(majority_labels)
    positive_items = sum(1 for value in majority_labels.values() if value)
    negative_items = total_items - positive_items
    if positive_items == 0 or negative_items == 0:
        if context:
            LOGGER.warning(
                "Human majority labels are degenerate for %s: %d positive and %d "
                "negative out of %d items. True-positive or true-negative counts "
                "may be zero and precision/recall/F1 may be undefined.",
                context,
                positive_items,
                negative_items,
                total_items,
            )
        else:
            LOGGER.warning(
                "Human majority labels are degenerate: %d positive and %d negative "
                "out of %d items. True-positive or true-negative counts may be zero "
                "and precision/recall/F1 may be undefined.",
                positive_items,
                negative_items,
                total_items,
            )

    results: List[dict] = []
    for info in annotators:
        if info.kind == "human":
            continue
        values = annotator_values.get(info.name)
        if not values:
            continue

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0
        for key, truth in majority_labels.items():
            if key not in values:
                continue
            prediction = bool(values[key])
            if truth and prediction:
                true_positive += 1
            elif (not truth) and (not prediction):
                true_negative += 1
            elif truth and (not prediction):
                false_negative += 1
            elif (not truth) and prediction:
                false_positive += 1

        total = true_positive + true_negative + false_positive + false_negative
        if total <= 0:
            continue

        accuracy = (true_positive + true_negative) / float(total)
        precision = (
            true_positive / float(true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else None
        )
        recall = (
            true_positive / float(true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else None
        )
        if precision is not None and recall is not None and (precision + recall) > 0.0:
            f1_score = 2.0 * precision * recall / float(precision + recall)
        else:
            f1_score = None

        # Compute Cohen's kappa for the LLM versus the human-majority labels
        # using the same binary confusion counts.
        counts_for_kappa: Mapping[str, int] = {
            "yes_yes": int(true_positive),
            "no_no": int(true_negative),
            "yes_no": int(false_negative),
            "no_yes": int(false_positive),
        }
        kappa_value = _cohen_kappa(counts_for_kappa)

        results.append(
            {
                "annotator": info.name,
                "kind": info.kind,
                "n_items": int(total),
                "tp": int(true_positive),
                "tn": int(true_negative),
                "fp": int(false_positive),
                "fn": int(false_negative),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1_score,
                "kappa": kappa_value,
            }
        )

    return results


def write_cases_jsonl(
    output_path: Path,
    items_by_key: Mapping[DatasetKey, dict],
    transcript_key_by_dataset: Mapping[DatasetKey, TranscriptKey],
    annotator_labels: Mapping[str, Mapping[DatasetKey, bool]],
    annotator_matches: Optional[
        Mapping[str, Mapping[DatasetKey, Sequence[str]]]
    ] = None,
    annotator_details: Optional[
        Mapping[str, Mapping[DatasetKey, Mapping[str, object]]]
    ] = None,
) -> None:
    """Write per-item labels and metadata for the viewer."""

    excluded_ids = {value.strip() for value in EXCLUDED_ANNOTATION_IDS if value.strip()}

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ValueError(f"Failed to create {output_path.parent}: {err}") from err

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            for key, item in items_by_key.items():
                if key.annotation_id in excluded_ids:
                    continue
                transcript_key = transcript_key_by_dataset.get(key)
                labels_for_item: dict[str, str] = {}
                matches_for_item: dict[str, Sequence[str]] = {}
                details_for_item: dict[str, Mapping[str, object]] = {}
                for annotator, values in annotator_labels.items():
                    if key not in values:
                        continue
                    labels_for_item[annotator] = "yes" if values[key] else "no"
                    if annotator_matches and annotator in annotator_matches:
                        match_values = annotator_matches[annotator].get(key, ())
                        if match_values:
                            matches_for_item[annotator] = list(match_values)
                    if annotator_details and annotator in annotator_details:
                        detail_values = annotator_details[annotator].get(key)
                        if detail_values:
                            details_for_item[annotator] = dict(detail_values)

                if len(labels_for_item) < 1:
                    continue

                record = {
                    "dataset_path": key.dataset_path,
                    "sequence_index": key.sequence_index,
                    "participant": key.participant,
                    "annotation_id": key.annotation_id,
                    "annotation_label": item.get("annotation")
                    or item.get("annotation_id")
                    or key.annotation_id,
                    "chat_key": item.get("chat_key"),
                    "chat_index": item.get("chat_index"),
                    "message_index": item.get("message_index"),
                    "role": item.get("role"),
                    "timestamp": item.get("timestamp"),
                    "content": item.get("content"),
                    "preceding": item.get("preceding") or [],
                    "transcript_key": {
                        "participant": (
                            transcript_key.participant if transcript_key else None
                        ),
                        "source_path": (
                            transcript_key.source_path if transcript_key else None
                        ),
                        "chat_index": (
                            transcript_key.chat_index if transcript_key else None
                        ),
                        "message_index": (
                            transcript_key.message_index if transcript_key else None
                        ),
                        "annotation_id": (
                            transcript_key.annotation_id if transcript_key else None
                        ),
                    },
                    "annotator_labels": labels_for_item,
                    "annotator_matches": matches_for_item,
                    "annotator_details": details_for_item,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as err:
        raise ValueError(f"Failed to write {output_path}: {err}") from err


def write_metrics_json(
    output_path: Path,
    *,
    dataset_path: Path,
    annotators: Sequence[AnnotatorInfo],
    annotator_values: Mapping[str, Mapping[DatasetKey, bool]],
    llm_score_cutoff: Optional[int],
    llm_score_cutoffs_by_annotation: Optional[Mapping[str, int]] = None,
    roles_by_key: Mapping[DatasetKey, str] | None = None,
) -> None:
    """Write aggregate agreement metrics to a JSON sidecar.

    The payload includes pairwise annotator agreement as well as confusion
    statistics for non-human annotators against the human majority vote when
    human labels are available.
    """

    excluded_ids = {value.strip() for value in EXCLUDED_ANNOTATION_IDS if value.strip()}

    # Group annotator values by (possibly role-split) annotation id. When
    # ``roles_by_key`` is provided, selected base ids with dual scopes are
    # split into ``user-<id>`` and ``assistant-<id>`` pseudo-annotations so
    # their metrics appear separately in downstream tables.
    by_annotation: Dict[str, Dict[str, Mapping[DatasetKey, bool]]] = defaultdict(dict)
    for annotator_name, values in annotator_values.items():
        for key, label in values.items():
            base_id = key.annotation_id
            if base_id in excluded_ids:
                continue
            group_id = base_id
            if roles_by_key is not None and base_id in ROLE_SPLIT_BASE_IDS:
                role = roles_by_key.get(key, "").strip().lower()
                role_prefix = role_prefix_for_role(role, strict=False)
                if role_prefix:
                    group_id = f"{role_prefix}-{base_id}"
            by_annotation[group_id].setdefault(annotator_name, {})[key] = label

    annotator_by_name: Dict[str, AnnotatorInfo] = {
        info.name: info for info in annotators
    }

    metrics_by_annotation: Dict[str, List[dict]] = {}
    majority_confusion_by_annotation: Dict[str, List[dict]] = {}
    human_iaa_by_annotation: Dict[str, Mapping[str, object]] = {}

    human_names: List[str] = [
        info.name for info in annotators if getattr(info, "kind", "") == "human"
    ]
    llm_names: List[str] = [
        info.name for info in annotators if getattr(info, "kind", "") == "llm"
    ]
    for annotation_id in sorted(by_annotation.keys()):
        per_annotation_values = by_annotation[annotation_id]
        metrics_by_annotation[annotation_id] = compute_pairwise_metrics(
            per_annotation_values
        )

        # Diagnose coverage patterns for this annotation id so that missing
        # or non-overlapping categories are easy to investigate.
        human_present = any(
            name in human_names and per_annotation_values.get(name)
            for name in per_annotation_values.keys()
        )
        llm_present = any(
            name in llm_names and per_annotation_values.get(name)
            for name in per_annotation_values.keys()
        )
        if human_present and not llm_present:
            LOGGER.warning(
                "Annotation %r has human labels but no LLM labels; "
                "per-annotation LLM confusion and cutoffs will be empty.",
                annotation_id,
            )
        elif llm_present and not human_present:
            LOGGER.warning(
                "Annotation %r has LLM labels but no human labels; "
                "per-annotation LLM confusion and cutoffs will be empty.",
                annotation_id,
            )

        if human_names:
            coverage = _summarize_human_label_coverage(
                per_annotation_values, human_names
            )
            total_human_items = coverage.get("total_items", 0)
            single_label_items = coverage.get("single_label_items", 0)
            multi_label_items = coverage.get("multi_label_items", 0)
            if total_human_items > 0 and multi_label_items <= 0 <= single_label_items:
                LOGGER.warning(
                    "Annotation %r has %d item(s) with human labels, but all of "
                    "them have a single human label (no items are double-labeled). "
                    "Multi-rater human IAA metrics will be empty; majority-vote "
                    "metrics will treat these as single-rater majorities.",
                    annotation_id,
                    total_human_items,
                )

        per_annotation_annotators: List[AnnotatorInfo] = []
        for name in per_annotation_values.keys():
            info = annotator_by_name.get(name)
            if info is not None:
                per_annotation_annotators.append(info)
        confusion_entries = _compute_confusion_against_majority(
            per_annotation_values,
            per_annotation_annotators,
            context=f"annotation_id={annotation_id}",
        )
        if confusion_entries:
            majority_confusion_by_annotation[annotation_id] = confusion_entries
        elif human_present and llm_present:
            LOGGER.warning(
                "Annotation %r has both human and LLM labels but no items where "
                "the human majority and LLM labels overlap; per-annotation "
                "confusion and cutoffs will be empty.",
                annotation_id,
            )

        if human_names:
            iaa_entry = _compute_multi_rater_human_iaa(
                per_annotation_values,
                human_names,
                context=f"annotation_id={annotation_id}",
            )
            if iaa_entry is not None:
                human_iaa_by_annotation[annotation_id] = iaa_entry

    # Compute overall pairwise agreement by pooling items across all
    # non-excluded annotations. This allows the viewer to show a single
    # aggregate table when displaying "All annotations" while respecting
    # the central annotation exclusion list.
    filtered_annotator_values: Dict[str, Dict[DatasetKey, bool]] = {}
    for name, values in annotator_values.items():
        filtered_items: Dict[DatasetKey, bool] = {}
        for key, label in values.items():
            if key.annotation_id in excluded_ids:
                continue
            filtered_items[key] = label
        if filtered_items:
            filtered_annotator_values[name] = filtered_items

    overall_pairs = compute_pairwise_metrics(filtered_annotator_values)
    if overall_pairs:
        metrics_by_annotation["__all__"] = overall_pairs

    overall_confusion = _compute_confusion_against_majority(
        filtered_annotator_values, list(annotators), context="__all__"
    )
    if overall_confusion:
        majority_confusion_by_annotation["__all__"] = overall_confusion

    if human_names:
        overall_iaa_entry = _compute_multi_rater_human_iaa(
            filtered_annotator_values,
            human_names,
            context="annotation_id=__all__",
        )
        if overall_iaa_entry is not None:
            human_iaa_by_annotation["__all__"] = overall_iaa_entry

    annotator_entries = [
        {
            "name": info.name,
            "kind": info.kind,
            "source": info.source,
        }
        for info in annotators
    ]

    cutoffs_by_annotation: Mapping[str, int] = {}
    if llm_score_cutoffs_by_annotation:
        cutoffs_by_annotation = {
            annotation_id: int(value)
            for annotation_id, value in llm_score_cutoffs_by_annotation.items()
            if annotation_id not in excluded_ids
        }

    # Expose the set of annotation ids that actually appear in the metrics
    # payload rather than the raw ``annotation_ids`` argument so that any
    # role-split pseudo-annotations are reflected in the summary.
    included_annotation_ids = sorted(
        {
            value
            for value in metrics_by_annotation
            if value not in excluded_ids and value != "__all__"
        }
    )

    payload = {
        "dataset": str(dataset_path),
        "annotation_ids": included_annotation_ids,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "annotators": annotator_entries,
        "llm_score_cutoff": llm_score_cutoff,
        "pairwise": metrics_by_annotation,
        "majority_confusion": majority_confusion_by_annotation,
        "human_iaa": human_iaa_by_annotation,
        "llm_score_cutoffs_by_annotation": cutoffs_by_annotation,
    }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ValueError(f"Failed to create {output_path.parent}: {err}") from err

    try:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except OSError as err:
        raise ValueError(f"Failed to write {output_path}: {err}") from err


def write_cutoffs_csv(
    output_path: Path,
    *,
    cutoffs_by_annotation: Mapping[str, int],
) -> None:
    """Write per-annotation LLM score cutoffs to a CSV sidecar."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ValueError(f"Failed to create {output_path.parent}: {err}") from err

    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["annotation_id", "cutoff"])
            for annotation_id in sorted(cutoffs_by_annotation.keys()):
                writer.writerow(
                    [annotation_id, int(cutoffs_by_annotation[annotation_id])]
                )
    except OSError as err:
        raise ValueError(f"Failed to write {output_path}: {err}") from err


def _load_overall_llm_confusion_from_metrics(
    metrics_path: Path,
) -> Dict[str, Mapping[str, object]]:
    """Return overall majority-confusion entries for LLM annotators.

    Parameters
    ----------
    metrics_path:
        Path to a metrics JSON file produced by :func:`write_metrics_json`.

    Returns
    -------
    Dict[str, Mapping[str, object]]
        Mapping from annotator name to the overall confusion-entry dict for
        that annotator, restricted to entries where ``kind == \"llm\"``.
        When the file cannot be read or does not contain overall confusion
        entries, an empty mapping is returned.
    """

    try:
        text = metrics_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return load_overall_llm_confusion_from_payload(payload)


def _compute_f1_from_confusion(entry: Mapping[str, object]) -> Optional[float]:
    """Return an F1 score computed from confusion-count fields.

    The calculation uses ``F1 = 2 * tp / (2 * tp + fp + fn)`` whenever there
    is at least one positive example or prediction (that is, when the
    denominator is greater than zero). When ``tp == fp == fn == 0``, the
    result is ``None`` to indicate that F1 is undefined in the fully
    degenerate all-negative case.
    """

    try:
        true_positive = int(entry.get("tp", 0))
        false_positive = int(entry.get("fp", 0))
        false_negative = int(entry.get("fn", 0))
    except (TypeError, ValueError):
        return None

    denom = (2 * true_positive) + false_positive + false_negative
    if denom <= 0:
        return None
    return (2.0 * float(true_positive)) / float(denom)


def _metric_from_confusion(
    entry: Mapping[str, object],
    metric: str,
) -> Optional[float]:
    """Return a scalar metric derived from a confusion-entry mapping.

    Parameters
    ----------
    entry:
        Confusion-entry dictionary containing ``tp``, ``fp``, ``fn``, and
        optional scalar fields such as ``accuracy`` or ``kappa``.
    metric:
        Name of the metric to extract. Supported values are ``\"f1\"``,
        ``\"accuracy\"``, and ``\"kappa\"``.
    """

    metric_lower = metric.strip().lower()
    if metric_lower == "f1":
        return _compute_f1_from_confusion(entry)

    if metric_lower == "accuracy":
        value = entry.get("accuracy")
    elif metric_lower == "kappa":
        value = entry.get("kappa")
    else:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_per_annotation_llm_confusion_from_metrics(
    metrics_path: Path,
) -> Dict[str, Dict[str, Mapping[str, object]]]:
    """Return per-annotation majority-confusion entries for LLM annotators.

    Parameters
    ----------
    metrics_path:
        Path to a metrics JSON file produced by :func:`write_metrics_json`.

    Returns
    -------
    Dict[str, Dict[str, Mapping[str, object]]]
        Mapping from annotation id to a mapping from annotator name to the
        confusion-entry dict for that annotator and annotation id. Only
        entries where ``kind == \"llm\"`` are included. When the file cannot
        be read or does not contain per-annotation confusion entries, an empty
        mapping is returned.
    """

    try:
        text = metrics_path.read_text(encoding="utf-8")
    except OSError:
        return {}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(payload, dict):
        return {}

    return load_per_annotation_llm_confusion_from_payload(payload)


def _project_llm_outputs_to_dataset(
    llm_labels: Mapping[str, Mapping[TranscriptKey, bool]],
    llm_matches: Mapping[str, Mapping[TranscriptKey, Sequence[str]]],
    llm_details: Mapping[str, Mapping[TranscriptKey, Mapping[str, object]]],
    transcript_key_by_dataset: Mapping[DatasetKey, TranscriptKey],
) -> Tuple[
    Dict[str, Dict[DatasetKey, bool]],
    Dict[str, Dict[DatasetKey, List[str]]],
    Dict[str, Dict[DatasetKey, Dict[str, object]]],
]:
    """Return LLM labels, matches, and details keyed by DatasetKey."""

    labels_by_dataset: Dict[str, Dict[DatasetKey, bool]] = {}
    matches_by_dataset: Dict[str, Dict[DatasetKey, List[str]]] = {}
    details_by_dataset: Dict[str, Dict[DatasetKey, Dict[str, object]]] = {}

    reverse_lookup: Dict[TranscriptKey, DatasetKey] = {}
    for d_key, t_key in transcript_key_by_dataset.items():
        reverse_lookup[t_key] = d_key

    for name, values in llm_labels.items():
        projected: Dict[DatasetKey, bool] = {}
        for t_key, label in values.items():
            d_key = reverse_lookup.get(t_key)
            if d_key is None:
                continue
            projected[d_key] = label
        if projected:
            labels_by_dataset[name] = projected

    for name, raw_matches_for_name in llm_matches.items():
        projected_matches: Dict[DatasetKey, List[str]] = {}
        for t_key, spans in raw_matches_for_name.items():
            d_key = reverse_lookup.get(t_key)
            if d_key is None:
                continue
            projected_matches[d_key] = list(spans)
        if projected_matches:
            matches_by_dataset[name] = projected_matches

    for name, raw_details_for_name in llm_details.items():
        projected_details: Dict[DatasetKey, Dict[str, object]] = {}
        for t_key, detail in raw_details_for_name.items():
            d_key = reverse_lookup.get(t_key)
            if d_key is None:
                continue
            projected_details[d_key] = dict(detail)
        if projected_details:
            details_by_dataset[name] = projected_details

    return labels_by_dataset, matches_by_dataset, details_by_dataset


def _best_cutoff_index_for_entries(
    entries: Sequence[Tuple[int, Mapping[str, object]]],
    metric: str,
) -> Optional[int]:
    """Return the index of the metric-maximizing cutoff entry, if any.

    When multiple cutoffs achieve the same maximum metric value, the entry
    whose cutoff is closest to the median of the maximizing cutoffs is
    preferred. This avoids always favoring the minimum or maximum cutoff
    value when the selected metric (for example, accuracy) cannot
    distinguish between several candidates.
    """

    best_value: Optional[float] = None
    # Track all cutoffs that attain the best metric value so that we can
    # apply a median-based tie-break after scanning all entries.
    best_candidates: list[Tuple[int, int]] = []  # (cutoff_int, index)

    for index, (cutoff_value, confusion) in enumerate(entries):
        value = _metric_from_confusion(confusion, metric)
        if value is None:
            continue
        try:
            cutoff_int = int(cutoff_value)
        except (TypeError, ValueError):
            cutoff_int = 0

        if best_value is None or value > best_value:
            best_value = float(value)
            best_candidates = [(cutoff_int, index)]
            continue

        if best_value is None:
            continue

        if value == best_value:
            best_candidates.append((cutoff_int, index))

    if not best_candidates:
        return None

    # Apply a discrete median-based tie-break across all cutoffs that share
    # the maximum metric value. Candidates are sorted by cutoff value and
    # the element at the median index is selected. For an even number of
    # candidates, this chooses the lower of the two central cutoffs.
    sorted_candidates = sorted(best_candidates, key=lambda pair: pair[0])
    median_index = (len(sorted_candidates) - 1) // 2
    _, chosen_index = sorted_candidates[median_index]
    return chosen_index


def _select_target_llm_for_optimization(
    llm_confusion_by_annotator_per_annotation: Mapping[
        str, Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]]
    ],
    optimize_cutoff_for: Optional[str],
) -> Optional[str]:
    """Return the LLM annotator name used for cutoff optimization."""

    if not llm_confusion_by_annotator_per_annotation:
        return None

    llm_names = sorted(llm_confusion_by_annotator_per_annotation.keys())
    target_name: Optional[str] = None
    if optimize_cutoff_for:
        if optimize_cutoff_for in llm_confusion_by_annotator_per_annotation:
            target_name = optimize_cutoff_for
        else:
            fallback = llm_names[0]
            LOGGER.warning(
                "--optimize-cutoff-for annotator %r not found among LLM "
                "annotators; falling back to %r.",
                optimize_cutoff_for,
                fallback,
            )
            target_name = fallback
    else:
        target_name = llm_names[0]
    return target_name


def _compute_best_cutoffs_by_annotation(
    per_annotation_confusion: Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]],
    annotator_name: str,
    optimize_metric: str,
) -> Dict[str, int]:
    """Return per-annotation best cutoffs and log selections."""

    best_cutoff_by_annotation: Dict[str, int] = {}
    best_cutoffs: set[int] = set()
    for annotation_id, entries in per_annotation_confusion.items():
        best_index = _best_cutoff_index_for_entries(entries, optimize_metric)
        # Detect degenerate cases where F1 provides no guidance and fall back
        # to a neutral or conservative cutoff choice. For non-F1 metrics we
        # rely on the raw metric values and do not apply additional
        # heuristics.
        if optimize_metric.strip().lower() == "f1":
            if best_index is None or best_index < 0 or best_index >= len(entries):
                # No defined F1 values (all None) – treat all cutoffs as equivalent.
                # Prefer a midpoint cutoff of 5 when possible.
                distances: List[Tuple[int, int]] = []
                for idx, (cutoff_value, _) in enumerate(entries):
                    distance = abs(int(cutoff_value) - 5)
                    distances.append((distance, idx))
                _, best_index = min(distances)
            else:
                # Check whether all defined F1 scores are zero and F1 therefore
                # cannot distinguish between cutoffs in a meaningful way (for
                # example, when there are no true positives).
                f1_values: List[float] = []
                for _, confusion in entries:
                    f1_value = _compute_f1_from_confusion(confusion)
                    if f1_value is not None:
                        f1_values.append(float(f1_value))
                if not f1_values or max(f1_values) <= 0.0:
                    # When F1 provides no useful guidance (no true positives or
                    # all-zero scores), treat all cutoffs as equivalent and choose
                    # a neutral midpoint near 5 rather than favoring the most
                    # permissive or most conservative cutoff.
                    distances_mid: List[Tuple[int, int]] = []
                    for idx, (cutoff_value, _) in enumerate(entries):
                        distance = abs(int(cutoff_value) - 5)
                        distances_mid.append((distance, idx))
                    _, best_index = min(distances_mid)

        if best_index is None or best_index < 0 or best_index >= len(entries):
            continue

        best_cutoff = int(entries[best_index][0])
        best_cutoff_by_annotation[annotation_id] = best_cutoff
        best_cutoffs.add(best_cutoff)
        LOGGER.info(
            "Best score cutoff for annotation %r and annotator %r is %d "
            "(optimized for %s).",
            annotation_id,
            annotator_name,
            best_cutoff,
            optimize_metric,
        )

    if best_cutoffs:
        joined = ", ".join(str(value) for value in sorted(best_cutoffs))
        LOGGER.info(
            "Auto-selected per-annotation LLM score cutoffs {%s} for "
            "annotator %r by maximizing %s versus the human-majority labels "
            "for each annotation id.",
            joined,
            annotator_name,
            optimize_metric,
        )

    return best_cutoff_by_annotation


def _print_llm_cutoff_summary_table(
    dataset_path: Path,
    annotation_filter: Optional[set[str]],
    by_annotator: Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]],
    optimize_metric: str,
) -> None:
    """Print a precision/recall/accuracy/F1 table across cutoffs for each LLM.

    Parameters
    ----------
    dataset_path:
        Dataset path for which agreement was computed.
    annotation_filter:
        Optional annotation id used to restrict analysis.
    by_annotator:
        Mapping from annotator name to a sequence of (cutoff, confusion-dict)
        pairs. Confusion dictionaries are expected to follow the shape
        produced by :func:`_compute_confusion_against_majority`.
    """

    if not by_annotator:
        return

    header_parts = [
        f"LLM agreement summary vs human majority for dataset {dataset_path.name}"
    ]
    if annotation_filter:
        header_parts.append(f"(annotation_ids={','.join(sorted(annotation_filter))})")
    header_parts.append(f"[best cutoff by {optimize_metric}]")
    sys.stdout.write("\n" + " ".join(header_parts) + "\n")

    for annotator_name in sorted(by_annotator.keys()):
        entries = list(sorted(by_annotator[annotator_name], key=lambda pair: pair[0]))
        if not entries:
            continue

        best_index = _best_cutoff_index_for_entries(entries, optimize_metric)

        sys.stdout.write(f"\nAnnotator: {annotator_name}\n")
        sys.stdout.write(
            f"{'cutoff':>6} {'n':>8} {'precision':>10} "
            f"{'recall':>10} {'accuracy':>10} {'f1':>8} {'kappa':>8} {'best':>6}\n"
        )

        for index, (cutoff, confusion) in enumerate(entries):
            n_items = confusion.get("n_items")
            precision = confusion.get("precision")
            recall = confusion.get("recall")
            accuracy = confusion.get("accuracy")
            f1_value = _compute_f1_from_confusion(confusion)

            n_display = f"{int(n_items):d}" if isinstance(n_items, int) else "n/a"
            if isinstance(precision, (int, float)):
                precision_display = f"{float(precision):0.3f}"
            else:
                precision_display = "n/a"
            if isinstance(recall, (int, float)):
                recall_display = f"{float(recall):0.3f}"
            else:
                recall_display = "n/a"
            if isinstance(accuracy, (int, float)):
                accuracy_display = f"{float(accuracy):0.3f}"
            else:
                accuracy_display = "n/a"
            if isinstance(f1_value, float):
                f1_display = f"{float(f1_value):0.3f}"
            else:
                f1_display = "n/a"

            kappa = confusion.get("kappa")
            if isinstance(kappa, float):
                kappa_display = f"{float(kappa):0.3f}"
            else:
                kappa_display = "n/a"

            best_flag = "*" if best_index is not None and index == best_index else ""
            sys.stdout.write(
                f"{cutoff:6d} {n_display:>8} {precision_display:>10} "
                f"{recall_display:>10} {accuracy_display:>10} "
                f"{f1_display:>8} {kappa_display:>8} {best_flag:>6}\n"
            )


def _print_llm_per_annotation_best_cutoff_table(
    dataset_path: Path,
    annotation_filter: Optional[set[str]],
    by_annotator: Mapping[
        str, Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]]
    ],
    optimize_metric: str,
    llm_score_cutoffs_by_annotation: Optional[Mapping[str, int]] = None,
    overall_by_annotator: Optional[
        Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]]
    ] = None,
) -> None:
    """Print a single summary table of best cutoffs per annotation.

    Each row in the table reports, for a given LLM annotator and annotation
    id, the score cutoff that maximizes the selected metric versus the
    human-majority labels, along with the corresponding precision, recall,
    accuracy, F1, and support size.
    """

    if not by_annotator:
        return

    if llm_score_cutoffs_by_annotation:
        header_parts = [
            "LLM per-annotation score cutoffs (fixed) "
            f"for dataset {dataset_path.name}"
        ]
    else:
        header_parts = [
            "LLM per-annotation score cutoffs optimized for "
            f"{optimize_metric} for dataset {dataset_path.name}"
        ]
    if annotation_filter:
        header_parts.append(f"(annotation_ids={','.join(sorted(annotation_filter))})")
    sys.stdout.write("\n" + " ".join(header_parts) + "\n")

    sys.stdout.write(
        f"{'annotator':<20} {'annotation_id':<40} "
        f"{'cutoff':>6} {'n':>8} {'tp':>6} {'fp':>6} "
        f"{'precision':>10} {'recall':>10} {'accuracy':>10} {'f1':>8} {'kappa':>8}\n"
    )

    for annotator_name in sorted(by_annotator.keys()):
        per_annotation = by_annotator[annotator_name]
        for annotation_id in sorted(per_annotation.keys()):
            if annotation_filter and annotation_id not in annotation_filter:
                continue
            entries = list(
                sorted(per_annotation[annotation_id], key=lambda pair: pair[0])
            )
            if not entries:
                continue

            # When explicit per-annotation cutoffs are provided, prefer the
            # confusion entry whose cutoff matches the supplied value. Fall
            # back to the metric-maximizing cutoff when no mapping is present
            # or when the mapped cutoff is not available.
            best_index = None
            mapped_cutoff = None
            if (
                llm_score_cutoffs_by_annotation
                and annotation_id in llm_score_cutoffs_by_annotation
            ):
                mapped_cutoff = int(llm_score_cutoffs_by_annotation[annotation_id])
                for idx, (cutoff_value, _) in enumerate(entries):
                    if int(cutoff_value) == mapped_cutoff:
                        best_index = idx
                        break
            if best_index is None:
                best_index = _best_cutoff_index_for_entries(entries, optimize_metric)
            if best_index is None:
                continue

            best_cutoff, best_confusion = entries[best_index]
            n_items = best_confusion.get("n_items")
            precision = best_confusion.get("precision")
            recall = best_confusion.get("recall")
            accuracy = best_confusion.get("accuracy")
            best_f1 = _compute_f1_from_confusion(best_confusion)
            try:
                true_positive = int(best_confusion.get("tp", 0))
                false_positive = int(best_confusion.get("fp", 0))
            except (TypeError, ValueError):
                true_positive = 0
                false_positive = 0

            n_display = f"{int(n_items):d}" if isinstance(n_items, int) else "n/a"
            tp_display = f"{true_positive:d}"
            fp_display = f"{false_positive:d}"
            if isinstance(precision, (int, float)):
                precision_display = f"{float(precision):0.3f}"
            else:
                precision_display = "n/a"
            if isinstance(recall, (int, float)):
                recall_display = f"{float(recall):0.3f}"
            else:
                recall_display = "n/a"
            if isinstance(accuracy, (int, float)):
                accuracy_display = f"{float(accuracy):0.3f}"
            else:
                accuracy_display = "n/a"
            if isinstance(best_f1, float):
                f1_display = f"{best_f1:0.3f}"
            else:
                f1_display = "n/a"

            kappa = best_confusion.get("kappa")
            if isinstance(kappa, float):
                kappa_display = f"{float(kappa):0.3f}"
            else:
                kappa_display = "n/a"

            sys.stdout.write(
                f"{annotator_name:<20} {annotation_id:<40} "
                f"{best_cutoff:6d} {n_display:>8} {tp_display:>6} {fp_display:>6} "
                f"{precision_display:>10} {recall_display:>10} {accuracy_display:>10} "
                f"{f1_display:>8} {kappa_display:>8}\n"
            )

        _print_overall_row_for_annotator(
            annotator_name=annotator_name,
            overall_by_annotator=overall_by_annotator,
            optimize_metric=optimize_metric,
        )


def _print_overall_row_for_annotator(
    *,
    annotator_name: str,
    overall_by_annotator: Optional[
        Mapping[str, Sequence[Tuple[int, Mapping[str, object]]]]
    ],
    optimize_metric: str,
) -> None:
    """Print an overall (__all__) summary row for one annotator."""

    if not overall_by_annotator or annotator_name not in overall_by_annotator:
        return
    overall_entries = list(
        sorted(overall_by_annotator[annotator_name], key=lambda pair: pair[0])
    )
    if not overall_entries:
        return
    overall_index = _best_cutoff_index_for_entries(overall_entries, optimize_metric)
    if overall_index is None:
        return
    overall_cutoff, overall_confusion = overall_entries[overall_index]
    n_items = overall_confusion.get("n_items")
    precision = overall_confusion.get("precision")
    recall = overall_confusion.get("recall")
    accuracy = overall_confusion.get("accuracy")
    overall_f1 = _compute_f1_from_confusion(overall_confusion)
    try:
        true_positive = int(overall_confusion.get("tp", 0))
        false_positive = int(overall_confusion.get("fp", 0))
    except (TypeError, ValueError):
        true_positive = 0
        false_positive = 0

    n_display = f"{int(n_items):d}" if isinstance(n_items, int) else "n/a"
    tp_display = f"{true_positive:d}"
    fp_display = f"{false_positive:d}"
    if isinstance(precision, (int, float)):
        precision_display = f"{float(precision):0.3f}"
    else:
        precision_display = "n/a"
    if isinstance(recall, (int, float)):
        recall_display = f"{float(recall):0.3f}"
    else:
        recall_display = "n/a"
    if isinstance(accuracy, (int, float)):
        accuracy_display = f"{float(accuracy):0.3f}"
    else:
        accuracy_display = "n/a"
    if isinstance(overall_f1, float):
        f1_display = f"{overall_f1:0.3f}"
    else:
        f1_display = "n/a"

    kappa = overall_confusion.get("kappa")
    if isinstance(kappa, float):
        kappa_display = f"{float(kappa):0.3f}"
    else:
        kappa_display = "n/a"

    sys.stdout.write(
        f"{annotator_name:<20} {'__all__':<40} "
        f"{overall_cutoff:6d} {n_display:>8} {tp_display:>6} {fp_display:>6} "
        f"{precision_display:>10} {recall_display:>10} {accuracy_display:>10} "
        f"{f1_display:>8} {kappa_display:>8}\n"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for agreement computation."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute annotator agreement metrics and emit viewer-ready JSONL "
            "for manual and LLM labels."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        dest="datasets",
        help=(
            "Path to a manual-annotation dataset JSONL generated by "
            "prepare_manual_annotation_dataset.py. May be repeated."
        ),
    )
    parser.add_argument(
        "--annotation-id",
        action="append",
        dest="annotation_ids",
        help=(
            "Optional annotation id to restrict analysis. If omitted, all "
            "annotation ids present in the dataset are used."
        ),
    )
    parser.add_argument(
        "--manual-annotator",
        action="append",
        default=[],
        dest="manual_annotators",
        help=(
            "Annotator id corresponding to "
            "manual_annotation_labels/<id>/<dataset>.jsonl. "
            "May be repeated. When omitted, all annotators with matching "
            "label files are used."
        ),
    )
    parser.add_argument(
        "--llm-run",
        action="append",
        default=[],
        dest="llm_runs",
        help=(
            "Path to a classification_results*.jsonl file produced by "
            "classify_chats.py representing an LLM annotator. May be repeated."
        ),
    )
    parser.add_argument(
        "--llm-run-basename",
        action="append",
        default=[],
        dest="llm_run_basenames",
        help=(
            "Basename of a JSONL output file produced by classify_chats.py. "
            "When provided (and --llm-run is not used), all JSONL files under "
            "annotation_outputs/ whose filenames match one of the basenames "
            "are treated as LLM runs. May be repeated."
        ),
    )
    parser.add_argument(
        "--llm-preprocessed-parquet",
        type=str,
        default=None,
        help=(
            "Optional path to a preprocessed per-message annotations Parquet "
            "file (for example, annotations/all_annotations__preprocessed.parquet). "
            "When provided, this table is used as the LLM label source instead "
            "of JSONL classification outputs."
        ),
    )
    parser.add_argument(
        "--score-cutoff",
        action="append",
        type=int,
        dest="score_cutoff",
        help=(
            "Integer LLM score cutoff (0-10). May be repeated to evaluate "
            "multiple cutoffs explicitly. When omitted, all cutoffs 0-10 are "
            "scanned and only the cutoff that maximizes a selected metric "
            "versus the human-majority labels is retained in the output. "
            "The metric is controlled by --optimize-cutoff-metric (default: f1)."
        ),
    )
    parser.add_argument(
        "--llm-score-cutoff",
        action="append",
        type=int,
        dest="score_cutoff",
        help=(
            "Alias for --score-cutoff. Integer LLM score cutoff (0-10). "
            "May be repeated; values are merged with any provided via "
            "--score-cutoff."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        help=(
            "Optional prefix for output paths. By default, files are written "
            "under analysis/ as agreement_cases__<dataset-filename>.jsonl and "
            "agreement_metrics__<dataset-filename>.json."
        ),
    )
    parser.add_argument(
        "--optimize-cutoff-for",
        help=(
            "Optional LLM annotator name used when automatically selecting "
            "the best score cutoff based on a metric versus the "
            "human-majority labels. When omitted, the first available LLM "
            "annotator is used."
        ),
    )
    parser.add_argument(
        "--optimize-cutoff-metric",
        choices=["f1", "accuracy", "kappa"],
        default="accuracy",
        help=(
            "Metric used when automatically selecting the best LLM score "
            "cutoff versus the human-majority labels. Supported values are "
            "f1, accuracy, and kappa. Defaults to accuracy."
        ),
    )
    parser.add_argument(
        "--llm-cutoffs-json",
        help=(
            "Optional CSV file mapping annotation ids to integer LLM score "
            "cutoffs. When provided, these per-annotation cutoffs are used "
            "when binarizing LLM scores and automatic cutoff optimization "
            "is disabled."
        ),
    )
    parser.add_argument(
        "--manual-label-dataset",
        action="append",
        default=[],
        dest="manual_label_datasets",
        help=(
            "Optional additional manual-annotation dataset paths whose label "
            "files should be reused when computing agreement for --dataset. "
            "These side datasets are used only as sources of human labels; "
            "the primary --dataset still defines the items being evaluated."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output, including detailed Fleiss' kappa coverage mismatches.",
    )
    return parser.parse_args(argv)


def _run_for_dataset(
    dataset_path: Path,
    dataset_meta: Mapping[str, object],
    *,
    repo_root: Path,
    annotation_filter: Optional[set[str]],
    manual_annotators: Sequence[str],
    manual_label_datasets: Sequence[Path],
    llm_runs: Sequence[str],
    llm_run_basenames: Sequence[str],
    llm_preprocessed_path: Optional[Path],
    output_prefix: Optional[str],
    score_cutoffs: Sequence[int],
    optimize_cutoff: bool,
    optimize_cutoff_for: Optional[str],
    optimize_cutoff_metric: str,
    llm_score_cutoffs_by_annotation: Optional[Mapping[str, int]],
) -> int:
    """Compute agreement artefacts for a single dataset path."""

    items_by_key, transcript_key_by_dataset = load_dataset_items(
        dataset_path, annotation_filter, repo_root
    )
    if not items_by_key:
        LOGGER.error(
            "No manual-annotation items matched the requested criteria for "
            "dataset %s.",
            dataset_path,
        )
        return 2

    annotators_manual: List[AnnotatorInfo] = []
    annotator_values_manual: Dict[str, Dict[DatasetKey, bool]] = {}

    # Human annotators
    manual_annotators_list: List[str]
    if manual_annotators:
        manual_annotators_list = list(manual_annotators)
    else:
        manual_annotators_list = _auto_discover_manual_annotators(
            dataset_path=dataset_path,
            repo_root=repo_root,
        )
        if manual_annotators_list:
            joined = ", ".join(sorted(manual_annotators_list))
            LOGGER.info(
                "Auto-discovered manual annotators for %s: %s",
                dataset_path.name,
                joined,
            )
        else:
            LOGGER.warning(
                "No manual annotators discovered under "
                "manual_annotation_labels/ for dataset %s; continuing with "
                "LLM annotators only.",
                dataset_path.name,
            )

    for annotator_id in manual_annotators_list:
        # Start with labels written specifically for this dataset, when
        # available. This preserves the default behavior when no side
        # datasets are provided.
        try:
            direct_labels: Dict[DatasetKey, bool] = dict(
                load_manual_labels_for_annotator(
                    dataset_path, annotator_id, annotation_filter, repo_root
                )
            )
        except ValueError:
            direct_labels = {}

        labels = _merge_labels_from_side_datasets(
            base_transcript_key_by_dataset=transcript_key_by_dataset,
            repo_root=repo_root,
            annotation_filter=annotation_filter,
            annotator_id=annotator_id,
            existing_labels=direct_labels,
            manual_label_datasets=manual_label_datasets,
        )

        if not labels:
            LOGGER.warning("No labels found for annotator %r; skipping.", annotator_id)
            continue
        annotators_manual.append(
            AnnotatorInfo(
                name=str(annotator_id),
                kind="human",
                source=str(
                    _normalize_dataset_path(
                        repo_root
                        / "manual_annotation_labels"
                        / annotator_id
                        / dataset_path.name,
                        repo_root,
                    )
                ),
            )
        )
        annotator_values_manual[str(annotator_id)] = dict(labels)

    # LLM annotators
    llm_paths: List[Path] = []
    if llm_preprocessed_path is None:
        if llm_runs:
            # Explicit LLM run paths always win and disable auto-discovery.
            llm_paths = [Path(p).expanduser() for p in llm_runs]
        elif llm_run_basenames:
            # Discover runs whose filenames or partition families match one or
            # more of the provided basenames.
            llm_paths = _select_llm_paths_for_basenames(
                repo_root=repo_root,
                llm_run_basenames=llm_run_basenames,
            )
            if not llm_paths:
                LOGGER.warning(
                    "No LLM runs found matching the provided --llm-run-basename "
                    "filenames under annotation_outputs/; continuing with human "
                    "annotators only."
                )
        else:
            llm_paths = _auto_discover_llm_runs(
                dataset_meta=dataset_meta,
                repo_root=repo_root,
                annotation_filter=annotation_filter,
            )
            if not llm_paths:
                LOGGER.warning(
                    "No matching LLM runs discovered under annotation_outputs/; "
                    "continuing with human annotators only."
                )
    existing_llm_paths = [p for p in llm_paths if p.exists()]
    missing_llm_paths = [p for p in llm_paths if not p.exists()]
    for missing in missing_llm_paths:
        LOGGER.warning("LLM run not found, skipping: %s", missing)

    error_context = f"dataset {dataset_path}"
    return _run_evaluation_for_items(
        items_by_key=items_by_key,
        transcript_key_by_dataset=transcript_key_by_dataset,
        annotators_manual=annotators_manual,
        annotator_values_manual=annotator_values_manual,
        repo_root=repo_root,
        annotation_filter=annotation_filter,
        existing_llm_paths=existing_llm_paths,
        llm_preprocessed_path=llm_preprocessed_path,
        dataset_label=dataset_path,
        output_prefix=output_prefix,
        score_cutoffs=score_cutoffs,
        optimize_cutoff=optimize_cutoff,
        optimize_cutoff_for=optimize_cutoff_for,
        optimize_cutoff_metric=optimize_cutoff_metric,
        llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
        error_context=error_context,
    )


def _run_evaluation_for_items(
    *,
    items_by_key: Mapping[DatasetKey, dict],
    transcript_key_by_dataset: Mapping[DatasetKey, TranscriptKey],
    annotators_manual: Sequence[AnnotatorInfo],
    annotator_values_manual: Mapping[str, Mapping[DatasetKey, bool]],
    repo_root: Path,
    annotation_filter: Optional[set[str]],
    existing_llm_paths: Sequence[Path],
    llm_preprocessed_path: Optional[Path],
    dataset_label: Path,
    output_prefix: Optional[str],
    score_cutoffs: Sequence[int],
    optimize_cutoff: bool,
    optimize_cutoff_for: Optional[str],
    optimize_cutoff_metric: str,
    llm_score_cutoffs_by_annotation: Optional[Mapping[str, int]],
    error_context: str,
) -> int:
    """Run cutoff evaluation and write artefacts for a fixed item set."""

    # Derive per-item roles from the dataset so that selected annotations
    # with dual scopes can be split into role-specific pseudo-annotations in
    # the metrics payload (for example, user-platonic-affinity and
    # bot-platonic-affinity).
    roles_by_key: Dict[DatasetKey, str] = {}
    for key, obj in items_by_key.items():
        role_raw = normalize_role_token(obj.get("role"))
        if role_raw:
            roles_by_key[key] = role_raw

    # When no LLM runs or preprocessed table are available, score cutoffs do
    # not change the result; compute metrics once in that case.
    if not existing_llm_paths and llm_preprocessed_path is None:
        score_cutoffs = [score_cutoffs[0]] if score_cutoffs else [LLM_SCORE_CUTOFF]

    analysis_root = repo_root / "analysis" / "agreement"
    dataset_dir = analysis_root / dataset_label.name

    llm_confusion_by_annotator: Dict[str, List[Tuple[int, Mapping[str, object]]]] = {}
    llm_confusion_by_annotator_per_annotation: Dict[
        str, Dict[str, List[Tuple[int, Mapping[str, object]]]]
    ] = {}

    output_paths_by_cutoff: Dict[int, Tuple[Path, Path]] = {}

    preprocessed_frame = None
    if llm_preprocessed_path is not None:
        try:
            preprocessed_frame = load_preprocessed_annotations_table(
                llm_preprocessed_path
            )
        except (OSError, FileNotFoundError, ValueError) as err:
            LOGGER.error(
                "Failed to load preprocessed annotations from %s: %s",
                llm_preprocessed_path,
                err,
            )
            preprocessed_frame = None

    for cutoff in score_cutoffs:
        annotators: List[AnnotatorInfo] = [
            AnnotatorInfo(name=info.name, kind=info.kind, source=info.source)
            for info in annotators_manual
        ]
        annotator_values: Dict[str, Dict[DatasetKey, bool]] = {
            name: dict(values) for name, values in annotator_values_manual.items()
        }
        matches_by_dataset: Dict[str, Dict[DatasetKey, List[str]]] = {}
        details_by_dataset: Dict[str, Dict[DatasetKey, Dict[str, object]]] = {}

        llm_matches: Dict[str, Dict[TranscriptKey, List[str]]] = {}
        llm_details: Dict[str, Dict[TranscriptKey, Dict[str, object]]] = {}
        if preprocessed_frame is not None:
            (
                llm_infos,
                llm_labels,
                llm_matches,
                llm_details,
            ) = load_llm_labels_from_preprocessed_table(
                preprocessed_frame,
                transcript_key_by_dataset=transcript_key_by_dataset,
                annotation_filter=annotation_filter,
                score_cutoff=int(cutoff),
                per_annotation_cutoffs=llm_score_cutoffs_by_annotation,
                annotator_name="llm-preprocessed",
                source_label=(
                    str(llm_preprocessed_path) if llm_preprocessed_path else None
                ),
            )
            for name, info in llm_infos.items():
                labels_for_model = llm_labels.get(name)
                if name not in annotator_values and labels_for_model:
                    annotators.append(info)
                    annotator_values[name] = labels_for_model
        elif existing_llm_paths:
            llm_infos, llm_labels, llm_matches, llm_details = load_llm_labels(
                existing_llm_paths,
                annotation_filter,
                repo_root,
                score_cutoff=int(cutoff),
                per_annotation_cutoffs=llm_score_cutoffs_by_annotation,
            )
            for name, info in llm_infos.items():
                labels_for_model = llm_labels.get(name)
                if name not in annotator_values and labels_for_model:
                    annotators.append(info)
                    annotator_values[name] = labels_for_model

        if len(annotator_values) < 2:
            LOGGER.error(
                "Need at least two annotators with overlapping labels for %s "
                "at score cutoff %d.",
                error_context,
                cutoff,
            )
            return 2

        labels_by_dataset: Dict[str, Dict[DatasetKey, bool]] = {}
        for name, values in annotator_values.items():
            any_transcript_keys = any(
                isinstance(key, TranscriptKey) for key in values.keys()
            )
            if any_transcript_keys:
                llm_only_labels = {name: values}
                llm_only_matches = {name: llm_matches.get(name, {})}
                llm_only_details = {name: llm_details.get(name, {})}
                (
                    projected_labels,
                    projected_matches,
                    projected_details,
                ) = _project_llm_outputs_to_dataset(
                    llm_only_labels,
                    llm_only_matches,
                    llm_only_details,
                    transcript_key_by_dataset,
                )
                if projected_labels.get(name):
                    labels_by_dataset[name] = projected_labels[name]
                if projected_matches.get(name):
                    matches_by_dataset[name] = projected_matches[name]
                if projected_details.get(name):
                    details_by_dataset[name] = projected_details[name]
            else:
                labels_by_dataset[name] = dict(values)

        annotator_values = labels_by_dataset

        if output_prefix:
            base = Path(output_prefix).expanduser()
            prefix_path = base.with_name(f"{base.name}.score-{cutoff}")
            cases_path = prefix_path.with_suffix(".cases.jsonl")
            metrics_path = prefix_path.with_suffix(".metrics.json")
        else:
            cases_path = dataset_dir / f"cases.score-{cutoff}.jsonl"
            metrics_path = dataset_dir / f"metrics.score-{cutoff}.json"

        output_paths_by_cutoff[int(cutoff)] = (cases_path, metrics_path)

        try:
            write_cases_jsonl(
                cases_path,
                items_by_key,
                transcript_key_by_dataset,
                annotator_values,
                matches_by_dataset,
                details_by_dataset,
            )
            if llm_score_cutoffs_by_annotation:
                write_metrics_json(
                    metrics_path,
                    dataset_path=_normalize_dataset_path(dataset_label, repo_root),
                    annotators=annotators,
                    annotator_values=annotator_values,
                    llm_score_cutoff=None,
                    llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
                    roles_by_key=roles_by_key,
                )
            else:
                write_metrics_json(
                    metrics_path,
                    dataset_path=_normalize_dataset_path(dataset_label, repo_root),
                    annotators=annotators,
                    annotator_values=annotator_values,
                    llm_score_cutoff=int(cutoff),
                    llm_score_cutoffs_by_annotation=None,
                    roles_by_key=roles_by_key,
                )
        except ValueError as err:
            LOGGER.error("Failed to write agreement artefacts: %s", err)
            return 2

        overall_confusion = _load_overall_llm_confusion_from_metrics(metrics_path)
        if overall_confusion:
            for name, confusion in overall_confusion.items():
                llm_confusion_by_annotator.setdefault(name, []).append(
                    (int(cutoff), confusion)
                )

        per_annotation_confusion = _load_per_annotation_llm_confusion_from_metrics(
            metrics_path
        )
        if per_annotation_confusion:
            for annotation_id, per_name in per_annotation_confusion.items():
                if annotation_filter and annotation_id not in annotation_filter:
                    continue
                for name, confusion in per_name.items():
                    llm_confusion_by_annotator_per_annotation.setdefault(
                        name, {}
                    ).setdefault(annotation_id, []).append((int(cutoff), confusion))

        sys.stdout.write(
            f"Wrote agreement cases to {cases_path}\n"
            f"Wrote agreement metrics to {metrics_path}\n"
        )

    if optimize_cutoff and llm_confusion_by_annotator_per_annotation:
        target_name = _select_target_llm_for_optimization(
            llm_confusion_by_annotator_per_annotation,
            optimize_cutoff_for,
        )
        if target_name:
            per_annotation = llm_confusion_by_annotator_per_annotation.get(
                target_name, {}
            )
            best_cutoff_by_annotation = _compute_best_cutoffs_by_annotation(
                per_annotation,
                target_name,
                optimize_cutoff_metric,
            )
        else:
            best_cutoff_by_annotation = {}

        best_cutoffs: set[int] = set(best_cutoff_by_annotation.values())
        if best_cutoffs:
            final_annotators: List[AnnotatorInfo] = [
                AnnotatorInfo(name=info.name, kind=info.kind, source=info.source)
                for info in annotators_manual
            ]
            final_values: Dict[str, Dict[DatasetKey, bool]] = {
                name: dict(values) for name, values in annotator_values_manual.items()
            }
            final_matches: Dict[str, Dict[DatasetKey, List[str]]] = {}

            if preprocessed_frame is not None:
                (
                    llm_infos_mixed,
                    llm_labels_mixed,
                    llm_matches_mixed,
                    llm_details_mixed,
                ) = load_llm_labels_from_preprocessed_table(
                    preprocessed_frame,
                    transcript_key_by_dataset=transcript_key_by_dataset,
                    annotation_filter=annotation_filter,
                    score_cutoff=0,
                    per_annotation_cutoffs=best_cutoff_by_annotation,
                    annotator_name="llm-preprocessed",
                    source_label=(
                        str(llm_preprocessed_path) if llm_preprocessed_path else None
                    ),
                )
                for name, info in llm_infos_mixed.items():
                    labels_for_model = llm_labels_mixed.get(name)
                    if labels_for_model:
                        final_annotators.append(info)

                (
                    labels_by_dataset_mixed,
                    matches_by_dataset_mixed,
                    _details_by_dataset_mixed,
                ) = _project_llm_outputs_to_dataset(
                    llm_labels_mixed,
                    llm_matches_mixed,
                    llm_details_mixed,
                    transcript_key_by_dataset,
                )

                for name, values in labels_by_dataset_mixed.items():
                    if values:
                        final_values[name] = values

                if matches_by_dataset_mixed:
                    for name, projected_matches in matches_by_dataset_mixed.items():
                        if projected_matches:
                            final_matches[name] = projected_matches
            elif existing_llm_paths:
                (
                    llm_infos_mixed,
                    llm_labels_mixed,
                    llm_matches_mixed,
                    llm_details_mixed,
                ) = load_llm_labels(
                    existing_llm_paths,
                    annotation_filter,
                    repo_root,
                    score_cutoff=0,
                    per_annotation_cutoffs=best_cutoff_by_annotation,
                )
                for name, info in llm_infos_mixed.items():
                    labels_for_model = llm_labels_mixed.get(name)
                    if labels_for_model:
                        final_annotators.append(info)

                (
                    labels_by_dataset_mixed,
                    matches_by_dataset_mixed,
                    _details_by_dataset_mixed,
                ) = _project_llm_outputs_to_dataset(
                    llm_labels_mixed,
                    llm_matches_mixed,
                    llm_details_mixed,
                    transcript_key_by_dataset,
                )

                for name, values in labels_by_dataset_mixed.items():
                    if values:
                        final_values[name] = values

                if matches_by_dataset_mixed:
                    for name, projected_matches in matches_by_dataset_mixed.items():
                        if projected_matches:
                            final_matches[name] = projected_matches

            try:
                excluded_ids = {
                    value.strip() for value in EXCLUDED_ANNOTATION_IDS if value.strip()
                }
                cutoffs_for_csv = {
                    annotation_id: int(value)
                    for annotation_id, value in (
                        best_cutoff_by_annotation or {}
                    ).items()
                    if annotation_id not in excluded_ids
                }
                # Write an aggregate metrics payload without an embedded
                # cutoff value so that downstream consumers can load a
                # single "overall" metrics file for the dataset without
                # needing to select a particular score suffix. Per-cutoff
                # metrics.score-*.json files written above are left
                # untouched so that precision-recall curves across cutoffs
                # can still be plotted.
                overall_metrics_path = dataset_dir / "metrics.json"
                write_metrics_json(
                    overall_metrics_path,
                    dataset_path=_normalize_dataset_path(dataset_label, repo_root),
                    annotators=final_annotators,
                    annotator_values=final_values,
                    llm_score_cutoff=None,
                    llm_score_cutoffs_by_annotation=cutoffs_for_csv,
                    roles_by_key=roles_by_key,
                )
                cutoffs_path = dataset_dir / "cutoffs.csv"
                write_cutoffs_csv(
                    cutoffs_path,
                    cutoffs_by_annotation=cutoffs_for_csv,
                )
            except ValueError as err:
                LOGGER.error("Failed to write final agreement artefacts: %s", err)
                return 2

    _print_llm_cutoff_summary_table(
        dataset_path=dataset_label,
        annotation_filter=annotation_filter,
        by_annotator=llm_confusion_by_annotator,
        optimize_metric=optimize_cutoff_metric,
    )

    _print_llm_per_annotation_best_cutoff_table(
        dataset_path=dataset_label,
        annotation_filter=annotation_filter,
        by_annotator=llm_confusion_by_annotator_per_annotation,
        optimize_metric=optimize_cutoff_metric,
        llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
        overall_by_annotator=llm_confusion_by_annotator,
    )

    return 0


def _run_for_combined_datasets(
    dataset_paths: Sequence[Path],
    metas_by_path: Mapping[Path, Mapping[str, object]],
    *,
    repo_root: Path,
    annotation_filter: Optional[set[str]],
    manual_annotators: Sequence[str],
    manual_label_datasets: Sequence[Path],
    llm_runs: Sequence[str],
    llm_run_basenames: Sequence[str],
    llm_preprocessed_path: Optional[Path],
    output_prefix: Optional[str],
    score_cutoffs: Sequence[int],
    optimize_cutoff: bool,
    optimize_cutoff_for: Optional[str],
    optimize_cutoff_metric: str,
    llm_score_cutoffs_by_annotation: Optional[Mapping[str, int]],
) -> int:
    """Compute agreement artefacts for the union of multiple dataset paths.

    All dataset items are combined into a single evaluation set keyed by
    :class:`DatasetKey`. Manual and LLM labels are projected into this shared
    key space so that metrics and cutoffs are computed over the union. Output
    artefacts are named using the first dataset path as the canonical label.
    """

    if not dataset_paths:
        LOGGER.error("At least one --dataset is required.")
        return 2

    # Union of items across all datasets.
    items_by_key: Dict[DatasetKey, dict] = {}
    transcript_key_by_dataset: Dict[DatasetKey, TranscriptKey] = {}
    for dataset_path in dataset_paths:
        per_items, per_transcripts = load_dataset_items(
            dataset_path, annotation_filter, repo_root
        )
        items_by_key.update(per_items)
        transcript_key_by_dataset.update(per_transcripts)

    if not items_by_key:
        joined = ", ".join(str(path) for path in dataset_paths)
        LOGGER.error(
            "No manual-annotation items matched the requested criteria "
            "across datasets: %s.",
            joined,
        )
        return 2

    annotators_manual: List[AnnotatorInfo] = []
    annotator_values_manual: Dict[str, Dict[DatasetKey, bool]] = {}

    # Human annotators
    if manual_annotators:
        manual_annotators_list: List[str] = list(manual_annotators)
    else:
        discovered: set[str] = set()
        for dataset_path in dataset_paths:
            for annotator_id in _auto_discover_manual_annotators(
                dataset_path=dataset_path,
                repo_root=repo_root,
            ):
                discovered.add(annotator_id)
        manual_annotators_list = sorted(discovered)
        dataset_names = ", ".join(path.name for path in dataset_paths)
        if manual_annotators_list:
            joined = ", ".join(manual_annotators_list)
            LOGGER.info(
                "Auto-discovered manual annotators for combined datasets (%s): %s",
                dataset_names,
                joined,
            )
        else:
            LOGGER.warning(
                "No manual annotators discovered under "
                "manual_annotation_labels/ for combined datasets %s; "
                "continuing with LLM annotators only.",
                dataset_names,
            )

    for annotator_id in manual_annotators_list:
        # Aggregate direct labels across all primary datasets.
        direct_labels_all: Dict[DatasetKey, bool] = {}
        for dataset_path in dataset_paths:
            try:
                direct_labels_for_dataset = load_manual_labels_for_annotator(
                    dataset_path, annotator_id, annotation_filter, repo_root
                )
            except ValueError:
                direct_labels_for_dataset = {}
            for key, value in direct_labels_for_dataset.items():
                direct_labels_all.setdefault(key, value)

        labels = _merge_labels_from_side_datasets(
            base_transcript_key_by_dataset=transcript_key_by_dataset,
            repo_root=repo_root,
            annotation_filter=annotation_filter,
            annotator_id=annotator_id,
            existing_labels=direct_labels_all,
            manual_label_datasets=manual_label_datasets,
        )

        if not labels:
            LOGGER.warning("No labels found for annotator %r; skipping.", annotator_id)
            continue
        annotators_manual.append(
            AnnotatorInfo(
                name=str(annotator_id),
                kind="human",
                source=str(
                    _normalize_dataset_path(
                        repo_root
                        / "manual_annotation_labels"
                        / annotator_id
                        / dataset_paths[0].name,
                        repo_root,
                    )
                ),
            )
        )
        annotator_values_manual[str(annotator_id)] = dict(labels)

    # LLM annotators
    if llm_runs:
        llm_paths = [Path(p).expanduser() for p in llm_runs]
    elif llm_run_basenames:
        llm_paths = _select_llm_paths_for_basenames(
            repo_root=repo_root,
            llm_run_basenames=llm_run_basenames,
        )
        if not llm_paths:
            LOGGER.warning(
                "No LLM runs found matching the provided --llm-run-basename "
                "filenames or partition families under annotation_outputs/; "
                "continuing with human annotators only."
            )
    else:
        # Auto-discover runs that could contribute to any of the datasets.
        llm_paths_set: set[Path] = set()
        for dataset_path in dataset_paths:
            dataset_meta = metas_by_path.get(dataset_path, {})
            for path in _auto_discover_llm_runs(
                dataset_meta=dataset_meta,
                repo_root=repo_root,
                annotation_filter=annotation_filter,
            ):
                llm_paths_set.add(path)
        llm_paths = sorted(llm_paths_set)
        if not llm_paths:
            LOGGER.warning(
                "No matching LLM runs discovered under annotation_outputs/; "
                "continuing with human annotators only."
            )

    existing_llm_paths = [p for p in llm_paths if p.exists()]
    missing_llm_paths = [p for p in llm_paths if not p.exists()]
    for missing in missing_llm_paths:
        LOGGER.warning("LLM run not found, skipping: %s", missing)

    canonical_dataset = dataset_paths[0]
    error_context = (
        "combined datasets [" + ", ".join(str(path) for path in dataset_paths) + "]"
    )
    return _run_evaluation_for_items(
        items_by_key=items_by_key,
        transcript_key_by_dataset=transcript_key_by_dataset,
        annotators_manual=annotators_manual,
        annotator_values_manual=annotator_values_manual,
        repo_root=repo_root,
        annotation_filter=annotation_filter,
        existing_llm_paths=existing_llm_paths,
        llm_preprocessed_path=llm_preprocessed_path,
        dataset_label=canonical_dataset,
        output_prefix=output_prefix,
        score_cutoffs=score_cutoffs,
        optimize_cutoff=optimize_cutoff,
        optimize_cutoff_for=optimize_cutoff_for,
        optimize_cutoff_metric=optimize_cutoff_metric,
        llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
        error_context=error_context,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by the command-line interface."""

    args = parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    repo_root = _resolve_repo_root()

    dataset_paths = [Path(value).expanduser() for value in args.datasets]
    manual_label_datasets = [
        Path(value).expanduser() for value in args.manual_label_datasets
    ]
    for dataset_path in dataset_paths:
        if not dataset_path.exists():
            LOGGER.error("Dataset not found: %s", dataset_path)
            return 2

    # Load meta for all datasets so that per-dataset runs can reuse the
    # recorded annotation ids and parameters when discovering LLM runs.
    metas_by_path: Dict[Path, Mapping[str, object]] = {}
    for dataset_path in dataset_paths:
        try:
            dataset_meta = _load_dataset_meta(dataset_path)
        except ValueError as err:
            LOGGER.warning("%s", err)
            dataset_meta = {}
        metas_by_path[dataset_path] = dataset_meta

    annotation_filter_list = []
    if getattr(args, "annotation_ids", None):
        for a_id in args.annotation_ids:
            a_id = str(a_id).strip()
            if a_id:
                annotation_filter_list.append(normalize_annotation_id(a_id))
    annotation_filter = set(annotation_filter_list) if annotation_filter_list else None

    llm_score_cutoffs_by_annotation = load_cutoffs_mapping(
        getattr(args, "llm_cutoffs_json", None)
    )
    if getattr(args, "llm_cutoffs_json", None) and not llm_score_cutoffs_by_annotation:
        return 2

    # Determine score cutoffs to analyze. When one or more --score-cutoff
    # values are provided explicitly, those cutoffs are evaluated. When no
    # explicit cutoffs are given and no per-annotation cutoff JSON is
    # supplied, all integer cutoffs 0-10 are scanned and a single best cutoff
    # is selected automatically based on a chosen metric versus the
    # human-majority labels. When per-annotation cutoffs are provided via
    # --llm-cutoffs-json, only a single pass is run and the supplied mapping
    # controls binarization, so there is no need to scan multiple cutoffs.
    explicit_cutoffs: List[int] = []
    if args.score_cutoff:
        for value in args.score_cutoff:
            cutoff = int(value)
            if cutoff < 0 or cutoff > 10:
                LOGGER.error("--score-cutoff values must be integers between 0 and 10.")
                return 2
            if cutoff not in explicit_cutoffs:
                explicit_cutoffs.append(cutoff)
    score_cutoffs: List[int]
    optimize_cutoff = False
    optimize_cutoff_metric = getattr(args, "optimize_cutoff_metric", "accuracy")
    if explicit_cutoffs:
        # Honor explicitly requested global cutoffs even when a per-annotation
        # cutoff JSON is present. The caller is intentionally asking to scan
        # those values.
        score_cutoffs = sorted(explicit_cutoffs)
    elif llm_score_cutoffs_by_annotation:
        # When explicit per-annotation cutoffs are provided, run a single pass
        # using those thresholds directly instead of scanning 0-10. The
        # numeric cutoff value is retained only to label the output files.
        score_cutoffs = [0]
        optimize_cutoff = False
    else:
        # Fall back to automatic cutoff optimization over 0-10 when no
        # explicit global cutoffs or per-annotation mapping is supplied.
        score_cutoffs = list(range(0, 11))
        optimize_cutoff = True

    llm_preprocessed_path: Optional[Path]
    if getattr(args, "llm_preprocessed_parquet", None):
        llm_preprocessed_path = Path(str(args.llm_preprocessed_parquet)).expanduser()
    else:
        llm_preprocessed_path = None

    if len(dataset_paths) == 1:
        dataset_path = dataset_paths[0]
        dataset_meta = metas_by_path.get(dataset_path, {})
        return _run_for_dataset(
            dataset_path=dataset_path,
            dataset_meta=dataset_meta,
            repo_root=repo_root,
            annotation_filter=annotation_filter,
            manual_annotators=args.manual_annotators,
            manual_label_datasets=manual_label_datasets,
            llm_runs=args.llm_runs,
            llm_run_basenames=args.llm_run_basenames,
            llm_preprocessed_path=llm_preprocessed_path,
            output_prefix=args.output_prefix,
            score_cutoffs=score_cutoffs,
            optimize_cutoff=optimize_cutoff,
            optimize_cutoff_for=(
                str(args.optimize_cutoff_for).strip()
                if getattr(args, "optimize_cutoff_for", None)
                else None
            ),
            optimize_cutoff_metric=str(optimize_cutoff_metric).strip() or "f1",
            llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
        )

    return _run_for_combined_datasets(
        dataset_paths=dataset_paths,
        metas_by_path=metas_by_path,
        repo_root=repo_root,
        annotation_filter=annotation_filter,
        manual_annotators=args.manual_annotators,
        manual_label_datasets=manual_label_datasets,
        llm_runs=args.llm_runs,
        llm_run_basenames=args.llm_run_basenames,
        llm_preprocessed_path=llm_preprocessed_path,
        output_prefix=args.output_prefix,
        score_cutoffs=score_cutoffs,
        optimize_cutoff=optimize_cutoff,
        optimize_cutoff_for=(
            str(args.optimize_cutoff_for).strip()
            if getattr(args, "optimize_cutoff_for", None)
            else None
        ),
        optimize_cutoff_metric=str(optimize_cutoff_metric).strip() or "f1",
        llm_score_cutoffs_by_annotation=llm_score_cutoffs_by_annotation,
    )


if __name__ == "__main__":
    raise SystemExit(main())
