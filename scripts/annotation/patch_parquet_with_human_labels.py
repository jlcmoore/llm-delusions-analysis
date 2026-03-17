"""
Patch preprocessed annotation Parquet tables with human validation labels.

This script reads a preprocessed per-message Parquet table and a cases JSONL
produced by the annotation agreement script. It identifies human majority
labels for reviewed messages and adds role-specific 'human_validated__<id>'
boolean columns to the Parquet, where True/False represent the human
consensus and NaN represents messages that were not reviewed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Parquet patching script."""
    parser = argparse.ArgumentParser(
        description="Patch preprocessed Parquet with human validation labels."
    )
    parser.add_argument(
        "--parquet", required=True, help="Path to original preprocessed.parquet"
    )
    parser.add_argument(
        "--cases-jsonl",
        required=True,
        help="Path to cases.score-X.jsonl from agreement script",
    )
    parser.add_argument("--output", required=True, help="Path to output parquet")
    return parser.parse_args()


def is_human_annotator(name: str) -> bool:
    """Return True if the provided annotator name represents a human rater."""
    llm_prefixes = ("gpt-", "claude-", "gemini-", "vertex_ai-")
    if name == "llm-preprocessed":
        return False
    for prefix in llm_prefixes:
        if name.startswith(prefix):
            return False
    return True


def _get_human_majority(labels: Dict[str, str]) -> Optional[bool]:
    """Return the boolean majority of human labels, or None if no consensus."""
    human_yes = 0
    human_no = 0

    for annotator, label in labels.items():
        if is_human_annotator(annotator):
            if label == "yes":
                human_yes += 1
            elif label == "no":
                human_no += 1

    if human_yes == 0 and human_no == 0:
        return None
    if human_yes == human_no:
        return None
    return human_yes > human_no


def _collect_updates(
    cases_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, bool]]:
    """Yield update records and unique column names from the cases file."""
    updates = []
    human_columns = {}
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            case = json.loads(line)
            labels = case.get("annotator_labels", {})
            val = _get_human_majority(labels)

            if val is None:
                continue

            annotation_id = case["annotation_id"]
            col_name = f"human_validated__{annotation_id}"
            human_columns[col_name] = True

            tkey = case.get("transcript_key")
            if tkey and tkey.get("participant"):
                updates.append(
                    {
                        "participant": tkey["participant"],
                        "source_path": tkey["source_path"],
                        "chat_index": tkey["chat_index"],
                        "message_index": tkey["message_index"],
                        "col_name": col_name,
                        "val": val,
                    }
                )
    return updates, human_columns


def main() -> int:
    """Script entry point for patching Parquet tables with human labels."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parquet_path = Path(args.parquet)
    cases_path = Path(args.cases_jsonl)
    output_path = Path(args.output)

    if not parquet_path.exists():
        LOGGER.error("Parquet file not found: %s", parquet_path)
        return 1
    if not cases_path.exists():
        LOGGER.error("Cases JSONL not found: %s", cases_path)
        return 1

    LOGGER.info("Loading parquet from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    updates, human_columns = _collect_updates(cases_path)

    if not updates:
        LOGGER.info("No human validations found to apply.")
    else:
        for col_name in human_columns:
            # Clear existing column (if any) or create a fresh one to guarantee
            # previous partial runs do not leak into the updated table.
            df[col_name] = pd.NA
            df[col_name] = df[col_name].astype("boolean")

        LOGGER.info(
            "Applying %d human validations across %d IDs.",
            len(updates),
            len(human_columns),
        )

        idx_cols = ["participant", "source_path", "chat_index", "message_index"]
        df.set_index(idx_cols, inplace=True)
        success_count = 0
        for upd in updates:
            idx = (
                upd["participant"],
                upd["source_path"],
                upd["chat_index"],
                upd["message_index"],
            )
            if idx in df.index:
                df.at[idx, upd["col_name"]] = upd["val"]
                success_count += 1
        df.reset_index(inplace=True)
        LOGGER.info("Successfully updated %d/%d rows.", success_count, len(updates))

    LOGGER.info("Saving patched parquet to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
