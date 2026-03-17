"""
Build a curated annotations CSV from the raw Google Sheet export.

This script trims the raw CSV to the columns used by the codebase and
normalizes annotation ids to canonical bot/user prefixes. Role-split base
ids are expanded into separate bot/user rows.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List

from llm_delusions_annotations.annotation_ids import (
    expand_annotation_row,
    normalize_role_token,
)

OUTPUT_COLUMNS = [
    "id",
    "name",
    "description",
    "scope",
    "positive-examples",
    "negative-examples",
    "category",
]


def _normalize_scope_for_output(raw_scope: str) -> str:
    """Return a canonical scope string for output."""

    if not raw_scope:
        return ""
    tokens = [chunk.strip() for chunk in raw_scope.replace(";", ",").split(",")]
    normalized = []
    for token in tokens:
        role = normalize_role_token(token)
        if not role:
            continue
        normalized.append("bot" if role == "assistant" else "user")
    return ", ".join(normalized)


def _iter_rows(raw_path: Path) -> Iterable[Dict[str, str]]:
    """Yield curated rows from the raw annotations CSV."""

    with raw_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row:
                continue
            annotation_id = (row.get("id") or "").strip()
            if not annotation_id:
                continue
            scope_raw = (row.get("scope") or "").strip()
            scope_tokens = [
                chunk.strip() for chunk in scope_raw.replace(";", ",").split(",")
            ]
            for expanded_id, expanded_scope in expand_annotation_row(
                annotation_id,
                scope_tokens,
            ):
                scope_output = ", ".join(
                    "bot" if token == "assistant" else "user"
                    for token in expanded_scope
                )
                yield {
                    "id": expanded_id,
                    "name": (row.get("name") or "").strip(),
                    "description": (row.get("description") or "").strip(),
                    "scope": scope_output or _normalize_scope_for_output(scope_raw),
                    "positive-examples": (row.get("positive-examples") or "").strip(),
                    "negative-examples": (row.get("negative-examples") or "").strip(),
                    "category": (row.get("category") or "").strip(),
                }


def write_curated_csv(raw_path: Path, output_path: Path) -> None:
    """Write a curated annotations CSV."""

    rows: List[Dict[str, str]] = list(_iter_rows(raw_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for building a curated annotations CSV."""

    parser = argparse.ArgumentParser(
        description="Build a curated annotations.csv from a raw export.",
    )
    parser.add_argument(
        "--raw",
        required=True,
        help="Path to the raw annotations CSV export.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the curated annotations CSV.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    raw_path = Path(args.raw).expanduser()
    output_path = Path(args.output).expanduser()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw annotations CSV not found: {raw_path}")

    write_curated_csv(raw_path, output_path)
    print(f"Wrote curated annotations CSV to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
