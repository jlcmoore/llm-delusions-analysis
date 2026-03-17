"""Export ML-ready splits from preprocessed annotations.

This script takes the preprocessed annotations Parquet file and produces a
single Parquet file with a consistent train/test split across all annotations.
It binarizes the numeric scores based on a provided global cutoff or a
per-annotation cutoffs CSV file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from llm_delusions_annotations.cutoffs import load_cutoffs_mapping
from sklearn.model_selection import train_test_split

from utils.cli import (
    add_annotations_parquet_argument,
    add_optional_llm_cutoffs_argument,
    add_output_path_argument,
    add_score_cutoff_argument,
)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for exporting ML splits."""
    parser = argparse.ArgumentParser(
        description="Export ML-ready splits from preprocessed annotations."
    )
    add_annotations_parquet_argument(parser)
    add_score_cutoff_argument(
        parser,
        help_text="Global numeric score cutoff for binarizing labels.",
    )
    add_optional_llm_cutoffs_argument(
        parser,
        help_text="Path to a CSV mapping annotation_id to cutoff.",
    )
    add_output_path_argument(
        parser,
        default_path=Path("annotations/ml_splits.parquet"),
        help_text="Output Parquet path for the ML splits.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the train/test split (default: 42).",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.score_cutoff is None and args.llm_cutoffs_json is None:
        parser.error(
            "Must provide either --llm-score-cutoff or --llm-cutoffs-json to "
            "binarize the annotations."
        )

    return args


def _binarize_scores(
    df: pd.DataFrame,
    cutoffs_mapping: dict[str, int],
    global_cutoff: Optional[int],
) -> None:
    """In-place binarization of score__ columns to label__ columns."""
    score_columns = [col for col in df.columns if col.startswith("score__")]

    for score_col in score_columns:
        annotation_id = score_col[len("score__") :]
        cutoff = cutoffs_mapping.get(annotation_id, global_cutoff)

        if cutoff is None:
            print(
                f"Warning: No cutoff found for {annotation_id}, skipping binarization."
            )
            continue

        label_col = f"label__{annotation_id}"

        # Binarize keeping NaN as NaN using pandas boolean Dtype
        mask_notna = df[score_col].notna()
        # Ensure we use the nullable boolean type so missing values map to pd.NA
        df[label_col] = pd.Series(index=df.index, dtype="boolean")
        df.loc[mask_notna, label_col] = df.loc[mask_notna, score_col] >= cutoff


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Script entry point."""
    args = parse_args(argv)

    # Load the preprocessed annotations.
    input_path = Path(args.annotations_parquet).expanduser().resolve()
    print(f"Loading annotations from {input_path}...")
    df = pd.read_parquet(input_path)

    # Load per-annotation cutoffs if provided.
    cutoffs_mapping = {}
    if getattr(args, "llm_cutoffs_json", None):
        cutoffs_mapping = load_cutoffs_mapping(args.llm_cutoffs_json)
        print(f"Loaded cutoffs for {len(cutoffs_mapping)} annotations.")

    _binarize_scores(df, cutoffs_mapping, getattr(args, "score_cutoff", None))

    keep_cols = [
        "participant",
        "source_path",
        "chat_index",
        "message_index",
        "role",
        "timestamp",
        "chat_key",
        "chat_date",
    ]
    # Ensure all keep_cols exist
    keep_cols = [col for col in keep_cols if col in df.columns]

    label_columns = [col for col in df.columns if col.startswith("label__")]

    out_df = df[keep_cols + label_columns].copy()

    # Create stratified train/test split based on role
    print(
        f"Splitting data with test_size={args.test_size} "
        f"and random_state={args.random_state}..."
    )

    # We use train_test_split with stratify=out_df["role"] to ensure exact
    # proportional splits for both users and assistants.
    _, test_idx = train_test_split(
        out_df.index,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=out_df["role"],
    )

    out_df["split"] = "train"
    out_df.loc[test_idx, "split"] = "test"

    # Reorder columns to put 'split' earlier for convenience
    final_cols = keep_cols + ["split"] + label_columns
    out_df = out_df[final_cols]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing ML splits to {output_path} ({len(out_df)} rows)...")
    out_df.to_parquet(output_path, index=False)
    print("Done.")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
