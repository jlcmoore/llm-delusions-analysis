"""Convert Qualtrics CSV data into per-response JSON documents."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    """Parse command-line arguments for the Qualtrics converter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="qualtrics.csv",
        help="Path to the Qualtrics CSV export.",
    )
    parser.add_argument(
        "--metadata",
        default="transcripts/metadata.csv",
        help="CSV file containing identifier mappings.",
    )
    parser.add_argument(
        "--output",
        default="surveys",
        help="Directory that will receive one JSON file per survey response.",
    )
    return parser.parse_args(argv)


def load_header(reader: csv.reader) -> List[str]:
    """Return the header row, using the second CSV row as the header."""
    try:
        next(reader)  # discard the first Qualtrics-generated header row
        header = next(reader)
    except StopIteration as exc:
        raise ValueError("The CSV file must contain at least two rows.") from exc
    return header


def identifier_index(header: List[str]) -> int | None:
    """Return the index of the identifier column or None if absent."""
    for idx, column_name in enumerate(header):
        if column_name.strip().lower() == "response id":
            return idx
    return None


def sanitize_identifier(raw_identifier: str) -> str:
    """Create a filesystem-safe identifier slug."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_identifier.strip())
    cleaned = cleaned.lstrip(".")
    return cleaned


def row_to_record(
    header: List[str], row: List[str], excluded_columns: Set[str] | None = None
) -> dict[str, str]:
    """Combine the supplied header and row into a mapping, omitting excluded columns."""
    if len(row) < len(header):
        row = row + [""] * (len(header) - len(row))
    elif len(row) > len(header):
        row = row[: len(header)]

    excluded_columns = excluded_columns or set()
    record: dict[str, str] = {}
    for idx, column_name in enumerate(header):
        if column_name in excluded_columns:
            continue
        record[column_name] = row[idx]
    return record


def load_identifier_map(metadata_path: Path) -> Dict[str, str]:
    """Build a mapping from Qualtrics Response ID to Identifier."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    mapping: Dict[str, str] = {}
    with metadata_path.open(newline="", encoding="utf-8") as meta_file:
        reader = csv.DictReader(meta_file)
        if (
            "Identifier" not in reader.fieldnames
            or "Qualtrics Response ID" not in reader.fieldnames
        ):
            raise ValueError(
                "Metadata CSV must contain 'Identifier' and 'Qualtrics Response ID' columns."
            )

        for row_number, row in enumerate(reader, start=2):
            identifier = (row.get("Identifier") or "").strip()
            response_id = (row.get("Qualtrics Response ID") or "").strip()
            if not identifier or not response_id:
                continue

            sanitized = sanitize_identifier(identifier)
            if not sanitized:
                continue

            if response_id in mapping and mapping[response_id] != sanitized:
                raise ValueError(
                    f"Conflicting identifiers for response '{response_id}' (line {row_number})."
                )

            mapping[response_id] = sanitized

    return mapping


def build_transcript_mapping(header: List[str]) -> Dict[str, str]:
    """Derive transcript-related columns and map them to shorter keys."""
    prefix = "Please share your chatbot transcripts"
    suffix_map = {
        "- Id": "id",
        "- Name": "name",
        "- Size": "size",
        "- Type": "type",
    }

    mapping: Dict[str, str] = {}
    for column_name in header:
        normalized = column_name.strip()
        if not normalized.startswith(prefix):
            continue
        for suffix, key in suffix_map.items():
            if normalized.endswith(suffix):
                mapping[column_name] = key
                break
    return mapping


def convert(csv_path: Path, metadata_path: Path, output_dir: Path) -> None:
    """Load the CSV files and write one JSON file per identifier."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV input not found: {csv_path}")

    response_to_identifier = load_identifier_map(metadata_path)
    if not response_to_identifier:
        print(
            "No identifier mappings found in metadata; no survey files were written.",
            file=sys.stderr,
        )
        return

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file)
        header = load_header(reader)

        id_idx = identifier_index(header)
        if id_idx is None:
            print(
                "No 'Response ID' column found in Qualtrics CSV; no survey files were written.",
                file=sys.stderr,
            )
            return

        transcripts_mapping = build_transcript_mapping(header)
        column_indices = {name: idx for idx, name in enumerate(header)}

        output_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        unmatched = 0
        contact_column = (
            "If you would like to be interviewed, how should we contact you to provide "
            "more information and to coordinate?"
        )
        excluded_columns = {
            "Response ID",
            contact_column,
        }.union(transcripts_mapping.keys())
        for row_number, row in enumerate(reader, start=3):
            if not row:
                continue

            response_id = row[id_idx].strip() if len(row) > id_idx else ""
            if not response_id:
                continue

            identifier = response_to_identifier.get(response_id)
            if not identifier:
                unmatched += 1
                continue

            record = row_to_record(header, row, excluded_columns)
            if transcripts_mapping:
                transcript_payload: Dict[str, str] = {}
                for column_name, subkey in transcripts_mapping.items():
                    idx = column_indices.get(column_name)
                    if idx is None or idx >= len(row):
                        value = ""
                    else:
                        value = row[idx].strip()
                    transcript_payload[subkey] = value
                record["transcripts"] = transcript_payload

            target_path = output_dir / f"{identifier}.json"

            if target_path.exists():
                print(
                    f"Skipping row {row_number}: duplicate identifier '{identifier}'.",
                    file=sys.stderr,
                )
                continue

            target_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            written += 1

    print(f"Wrote {written} survey file(s) to {output_dir}")
    if unmatched:
        print(
            f"Ignored {unmatched} row(s) without matching identifier.", file=sys.stderr
        )


def main(argv: Iterable[str] | None = None) -> None:
    """Script entry point."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    convert(Path(args.csv_path), Path(args.metadata), Path(args.output))


if __name__ == "__main__":
    main()
