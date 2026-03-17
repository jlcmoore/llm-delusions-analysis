"""Demonstrate joining annotations to transcript indices and content.

This temporary helper script shows how to:

* Load the wide per-message annotations table from Parquet.
* Load the transcript index table without content.
* Join them on the shared location key.
* Fetch the full message content for a single joined row on demand.

It is intended as an executable example corresponding to the README
snippets; analysis code can copy or adapt the patterns as needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from llm_delusions_data import get_dataset_relpath
from llm_delusions_data.loaders import (
    load_annotations_preprocessed_parquet,
    load_transcripts_index_parquet,
    load_transcripts_parquet,
)
from utils.cli import add_annotations_parquet_argument, add_transcripts_parquet_argument


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the demo script.

    Parameters
    ----------
    argv:
        Optional raw argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Demo: join annotations to transcript indices and fetch a single "
            "message's content from transcripts_data/."
        )
    )
    add_annotations_parquet_argument(parser)
    parser.add_argument(
        "--index-parquet",
        type=Path,
        default=get_dataset_relpath("transcripts_index"),
        help=(
            "Transcript index table without content "
            "(default: transcripts_data/transcripts_index.parquet)."
        ),
    )
    add_transcripts_parquet_argument(parser)
    parser.add_argument(
        "--participant",
        type=str,
        default="",
        help=(
            "Optional participant id to restrict the demo join. When omitted, "
            "the script selects the first participant found in the "
            "annotations table."
        ),
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=5000,
        help=(
            "Optional cap on the number of annotation rows to include in the "
            "demo join (default: 5000)."
        ),
    )
    return parser.parse_args(argv)


def _select_demo_subset(
    annotations: pd.DataFrame,
    participant: str,
    max_messages: int,
) -> pd.DataFrame:
    """Return a small subset of the annotations table for the demo join.

    Parameters
    ----------
    annotations:
        Full per-message annotations DataFrame.
    participant:
        Optional participant id to filter to. When empty, the first
        participant in the table is used.
    max_messages:
        Maximum number of rows to include in the subset.

    Returns
    -------
    pandas.DataFrame
        Subsetted annotations frame suitable for a quick join.
    """

    if annotations.empty:
        return annotations

    if not participant:
        participant = str(annotations["participant"].iloc[0])
    subset = annotations[annotations["participant"].astype(str) == participant]

    if 0 < max_messages < len(subset):
        subset = subset.iloc[:max_messages].copy()
    return subset


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point for the annotations/transcripts join demo.

    Parameters
    ----------
    argv:
        Optional raw argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """

    args = parse_args(argv)

    annotations_path = args.annotations_parquet.expanduser().resolve()
    index_path = args.index_parquet.expanduser().resolve()
    transcripts_path = args.transcripts_parquet.expanduser().resolve()

    if not annotations_path.exists():
        print(f"Annotations parquet not found at {annotations_path}")
        return 1
    if not index_path.exists():
        print(f"Transcript index parquet not found at {index_path}")
        return 1
    if not transcripts_path.exists():
        print(f"Transcripts parquet not found at {transcripts_path}")
        return 1

    print(f"Loading annotations from {annotations_path} ...")
    annotations = load_annotations_preprocessed_parquet(annotations_path)
    if annotations.empty:
        print("Annotations table is empty; nothing to join.")
        return 0

    demo_annotations = _select_demo_subset(
        annotations,
        participant=args.participant,
        max_messages=args.max_messages,
    )
    if demo_annotations.empty:
        print("No annotations found for the requested participant.")
        return 0

    print(f"Selected {len(demo_annotations)} annotation rows for the demo join.")

    print(f"Loading transcript index from {index_path} ...")
    transcript_index = load_transcripts_index_parquet(index_path)
    if transcript_index.empty:
        print("Transcript index table is empty; nothing to join.")
        return 0

    joined = demo_annotations.merge(
        transcript_index,
        on=["participant", "source_path", "chat_index", "message_index"],
        how="inner",
        suffixes=("_ann", "_tx"),
    )
    if joined.empty:
        print("Join produced no rows; check that inputs are aligned.")
        return 0

    print(f"Joined table has {len(joined)} rows.")

    # Fetch content for a single joined row using Parquet filters so we do
    # not need to read the entire transcripts table into memory.
    sample = joined.iloc[0]
    filters = [
        ("participant", "=", str(sample["participant"])),
        ("source_path", "=", str(sample["source_path"])),
        ("chat_index", "=", int(sample["chat_index"])),
        ("message_index", "=", int(sample["message_index"])),
    ]

    print(
        "Fetching content for one joined message using filters on "
        f"{transcripts_path} ..."
    )
    try:
        content_frame = load_transcripts_parquet(
            transcripts_path,
            filters=filters,
        )
    except (OSError, ValueError) as err:
        print(f"Failed to load content for the sample message: {err}")
        return 1

    if content_frame.empty:
        print("No matching transcript row found for the sample message.")
        return 0

    print("Sample message content:\n")
    print(content_frame["content"].iloc[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
