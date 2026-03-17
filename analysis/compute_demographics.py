"""Compute participant demographics from survey JSON files.

This module aggregates basic demographic statistics such as age and gender
from IRB survey JSON files and can write a CSV that is easy to plug into
paper tables.

Typical usage:

    python analysis/compute_demographics.py \\
        --surveys-dir surveys \\
        --output analysis/demographics.csv

The JSON schema is expected to match the current IRB survey exports, with
age recorded under the key "What is your age?" and gender under
"What is your gender? - Selected Choice".
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Tuple

from utils.cli import add_participants_argument
from utils.demographics import AGE_KEY, GENDER_KEY
from utils.participants import is_excluded_participant, normalize_participant_value


def load_survey_files(
    surveys_dir: Path,
    participants: Optional[List[str]] = None,
) -> List[Path]:
    """Return sorted list of survey JSON paths under the given directory.

    Parameters
    ----------
    surveys_dir:
        Directory containing survey JSON files, typically with names like
        ``1*.json``.
    participants:
        Optional list of participant IDs to restrict processing to.

    Returns
    -------
    list of pathlib.Path
        Sorted list of JSON file paths.
    """
    patterns = ["1*.json", "irb_*.json"]
    selected: Dict[str, Path] = {}

    # Normalize requested participants
    allowed_participants = None
    if participants:
        allowed_participants = {normalize_participant_value(p) for p in participants}

    for pattern in patterns:
        for match in sorted(glob.glob(str(surveys_dir / pattern))):
            path = Path(match)
            participant_id = normalize_participant_value(path.stem)
            if not participant_id:
                continue
            if is_excluded_participant(participant_id):
                continue
            if allowed_participants and participant_id not in allowed_participants:
                continue
            if participant_id not in selected:
                selected[participant_id] = path
    return [selected[key] for key in sorted(selected.keys())]


def extract_age_and_gender(file_paths: Iterable[Path]) -> Tuple[List[int], List[str]]:
    """Extract age and gender values from survey JSON files.

    Parameters
    ----------
    file_paths:
        Iterable of JSON file paths to read.

    Returns
    -------
    tuple
        Two-element tuple ``(ages, genders)`` where ``ages`` is a list of
        integer ages and ``genders`` is a list of gender strings.
    """
    ages: List[int] = []
    genders: List[str] = []

    for file_path in file_paths:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        age_value = data.get(AGE_KEY, "")
        if str(age_value).strip():
            try:
                ages.append(int(age_value))
            except ValueError:
                print(
                    f"Warning: Invalid age value '{age_value}' in file {file_path}",
                )

        gender_value = data.get(GENDER_KEY, "")
        if str(gender_value).strip():
            genders.append(str(gender_value))
        else:
            print(f"Warning: Missing gender in file {file_path}")

    return ages, genders


def compute_age_stats(ages: List[int]) -> Dict[str, float]:
    """Compute summary statistics for a list of ages.

    Parameters
    ----------
    ages:
        List of integer ages.

    Returns
    -------
    dict
        Dictionary with summary statistics such as ``count``, ``min``, ``max``,
        ``mean`` and ``median``. Returns an empty dictionary when no ages are
        provided.
    """
    if not ages:
        return {}

    sorted_ages = sorted(ages)
    return {
        "count": float(len(sorted_ages)),
        "min": float(sorted_ages[0]),
        "max": float(sorted_ages[-1]),
        "mean": float(mean(sorted_ages)),
        "median": float(median(sorted_ages)),
    }


def compute_gender_stats(genders: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute gender counts and percentages.

    Parameters
    ----------
    genders:
        List of gender labels.

    Returns
    -------
    dict
        Nested dictionary mapping each gender label to ``count`` and
        ``percentage`` keys. Returns an empty dictionary when no genders are
        provided.
    """
    if not genders:
        return {}

    counts = Counter(genders)
    total = float(len(genders))
    return {
        label: {
            "count": float(count),
            "percentage": (count / total) * 100.0,
        }
        for label, count in counts.items()
    }


def compute_demographics(
    surveys_dir: Path,
    participants: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Compute overall demographics from survey JSON files.

    Parameters
    ----------
    surveys_dir:
        Directory containing survey JSON files.
    participants:
        Optional list of participant IDs to restrict processing to.

    Returns
    -------
    dict
        Dictionary with keys ``total_files``, ``age`` and ``gender`` holding
        summary statistics.
    """
    file_paths = load_survey_files(surveys_dir, participants=participants)
    ages, genders = extract_age_and_gender(file_paths)

    stats = {
        "total_files": len(file_paths),
        "age": compute_age_stats(ages),
        "gender": compute_gender_stats(genders),
    }
    return stats


def write_demographics_csv(stats: Dict[str, object], output_path: Path) -> None:
    """Write demographics summary to a long-format CSV file.

    Parameters
    ----------
    stats:
        Demographic statistics as returned by :func:`compute_demographics`.
    output_path:
        Path to the CSV file to write.

    Returns
    -------
    None
    """
    fieldnames = ["field", "category", "n", "pct", "note"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        age_stats = stats.get("age") or {}
        if age_stats:
            writer.writerow(
                {
                    "field": "age",
                    "category": "summary",
                    "n": int(age_stats["count"]),
                    "pct": "",
                    "note": (
                        f"min={age_stats['min']:.0f}; "
                        f"max={age_stats['max']:.0f}; "
                        f"mean={age_stats['mean']:.1f}; "
                        f"median={age_stats['median']:.0f}"
                    ),
                },
            )

        gender_stats = stats.get("gender") or {}
        for label, values in sorted(gender_stats.items()):
            writer.writerow(
                {
                    "field": "gender",
                    "category": label,
                    "n": int(values["count"]),
                    "pct": f"{values['percentage']:.1f}",
                    "note": "",
                },
            )


def print_demographics(stats: Dict[str, object]) -> None:
    """Print demographic statistics in a formatted way."""
    print("DEMOGRAPHIC STATISTICS")
    print("=" * 50)
    print()

    age_stats = stats.get("age") or {}
    if age_stats:
        print("AGE:")
        print(f"  Total responses: {int(age_stats['count'])}")
        print(
            f"  Range: {age_stats['min']:.0f} - {age_stats['max']:.0f} years",
        )
        print(f"  Mean: {age_stats['mean']:.1f} years")
        print(f"  Median: {age_stats['median']:.0f} years")
        print()
    else:
        print("No age data found")
        print()

    gender_stats = stats.get("gender") or {}
    if gender_stats:
        print("GENDER:")
        total = int(
            sum(int(values["count"]) for values in gender_stats.values()),
        )
        print(f"  Total responses: {total}")
        for label, values in sorted(gender_stats.items()):
            print(
                f"  {label}: {int(values['count'])} " f"({values['percentage']:.1f}%)",
            )
        print()
    else:
        print("No gender data found")
        print()

    print(f"Total survey files processed: {stats['total_files']}")
    print()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute basic demographics from survey JSON files.",
    )
    parser.add_argument(
        "--surveys-dir",
        type=Path,
        default=Path("surveys"),
        help="Directory containing 1*.json survey files (default: surveys).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis") / "data" / "demographics.csv",
        help=(
            "CSV output path. A long-format summary CSV is written to this "
            "location in addition to console output "
            "(default: analysis/data/demographics.csv)."
        ),
    )
    add_participants_argument(parser)
    return parser.parse_args()


def main() -> None:
    """Entry point for command line usage."""
    args = _parse_args()
    stats = compute_demographics(
        args.surveys_dir,
        participants=getattr(args, "participants", None),
    )
    print_demographics(stats)

    write_demographics_csv(stats, args.output)


if __name__ == "__main__":
    main()
