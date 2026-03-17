"""Dataset metadata for shared Parquet assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata describing a dataset file.

    Parameters
    ----------
    name:
        Identifier used to reference the dataset.
    relpath:
        Path relative to the repository root.
    description:
        Human-readable description of the dataset contents.
    generator:
        Script path that produces the dataset, or ``None`` when unknown.
    kind:
        File format label such as ``"parquet"``.
    """

    name: str
    relpath: Path
    description: str
    generator: Optional[str]
    kind: str


DATASETS: Dict[str, DatasetSpec] = {
    "annotations_preprocessed": DatasetSpec(
        name="annotations_preprocessed",
        relpath=Path("annotations") / "all_annotations__preprocessed.parquet",
        description="Preprocessed per-message annotations table.",
        generator="scripts/annotation/prepare_manual_annotation_dataset.py",
        kind="parquet",
    ),
    "annotations_matches": DatasetSpec(
        name="annotations_matches",
        relpath=Path("annotations") / "all_annotations__matches.parquet",
        description="Matches table with validated quote spans.",
        generator="analysis/preprocess_annotation_family.py",
        kind="parquet",
    ),
    "transcripts": DatasetSpec(
        name="transcripts",
        relpath=Path("transcripts_data") / "transcripts.parquet",
        description="Full transcripts table including message content.",
        generator="scripts/parse/export_transcripts_parquet.py",
        kind="parquet",
    ),
    "transcripts_index": DatasetSpec(
        name="transcripts_index",
        relpath=Path("transcripts_data") / "transcripts_index.parquet",
        description="Metadata-only transcripts index without content.",
        generator="scripts/parse/export_transcripts_parquet.py",
        kind="parquet",
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    """Return the dataset spec matching ``name``.

    Parameters
    ----------
    name:
        Dataset identifier, such as ``"transcripts"``.

    Returns
    -------
    DatasetSpec
        Matching dataset metadata.

    Raises
    ------
    KeyError
        If the dataset name is not known.
    """

    if name in DATASETS:
        return DATASETS[name]
    available = ", ".join(sorted(DATASETS))
    raise KeyError(f"Unknown dataset {name!r}. Available: {available}")


def get_dataset_relpath(name: str) -> Path:
    """Return the dataset path relative to the repository root.

    Parameters
    ----------
    name:
        Dataset identifier.

    Returns
    -------
    pathlib.Path
        Relative dataset path.
    """

    return get_dataset(name).relpath
