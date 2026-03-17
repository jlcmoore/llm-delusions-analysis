"""Find duplicated content strings across JSON transcript files.

This script scans JSON files under one or more target directories, extracts all
string values from keys named "content", and reports any content
strings that appear in more than one file.

Usage:
    python scripts/parse/find_duplicate_contents.py \
        --root transcripts_de_ided \
        --root tmp_anon \
        --min-overlap-percent 90 \
        --min-shared-lines 100

Parameters
----------
root : str
    A directory containing JSON files to scan. Repeatable.

Returns
-------
int
    Exit code 0 on success, non-zero on error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def iter_content_values(node: Any) -> Iterable[str]:
    """Yield all string values associated with a key named "content".

    Parameters
    ----------
    node : Any
        Parsed JSON value to traverse recursively.

    Returns
    -------
    Iterable[str]
        An iterator over all discovered content strings.
    """

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "content" and isinstance(value, str):
                yield value
            else:
                yield from iter_content_values(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_content_values(item)


def build_content_index(
    roots: Iterable[Path],
) -> Tuple[Dict[str, List[Tuple[Path, int]]], Dict[Path, int]]:
    """Build an index of content strings to the files where they appear.

    Parameters
    ----------
    roots : Iterable[Path]
        Directories to recurse into looking for JSON files.

    Returns
    -------
    Tuple[Dict[str, List[Tuple[Path, int]]], Dict[Path, int]]
        A tuple of:
        - Mapping from content string to a list of (path, count_in_file).
        - Mapping from file path to total number of content entries.
    """

    index: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    file_line_counts: Dict[Path, int] = defaultdict(int)

    for root in roots:
        for path in sorted(root.rglob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, json.JSONDecodeError) as error:
                print(f"Warning: failed to read {path}: {error}", file=sys.stderr)
                continue

            counts_in_file: Dict[str, int] = defaultdict(int)
            total_lines = 0
            for content in iter_content_values(data):
                total_lines += 1
                counts_in_file[content] += 1

            file_line_counts[path] += total_lines

            for content, count in counts_in_file.items():
                index[content].append((path, count))

    return index, file_line_counts


def print_duplicates(index: Dict[str, List[Tuple[Path, int]]]) -> None:
    """Print groups of identical content strings that span multiple files.

    Parameters
    ----------
    index : Dict[str, List[Tuple[Path, int]]]
        Mapping from content string to file occurrences.

    Notes
    -----
    This function does not print full content strings, only a short
    preview and metadata, to keep the output manageable.
    """

    group_index = 0

    for content, occurrences in index.items():
        if len(occurrences) < 2:
            continue

        group_index += 1
        preview = content.replace("\n", "\\n")
        if len(preview) > 120:
            preview = preview[:117] + "..."

        print("=" * 80)
        print(f"GROUP {group_index}")
        print(f"content_length={len(content)}")
        print(f"preview={preview}")
        print("FILES:")
        for path, count in occurrences:
            print(f"- {path} (count_in_file={count})")


def print_pairwise_overlap(
    index: Dict[str, List[Tuple[Path, int]]],
    file_line_counts: Dict[Path, int],
    *,
    min_overlap_percent: float,
    min_shared_lines: int,
) -> None:
    """Print pairwise overlap statistics between files.

    For each pair of files that share at least one content string, this
    reports:
    - number of overlapping content entries (lines)
    - percent overlap, defined as:
      shared_lines / min(total_lines_in_file_a, total_lines_in_file_b)

    Parameters
    ----------
    index : Dict[str, List[Tuple[Path, int]]]
        Mapping from content string to file occurrences.
    file_line_counts : Dict[Path, int]
        Mapping from file path to total number of content entries.
    """

    pair_overlap: Dict[Tuple[Path, Path], int] = defaultdict(int)

    for occurrences in index.values():
        if len(occurrences) < 2:
            continue

        # Each content contributes overlap equal to the minimum count
        # between any two files that contain it.
        for index_i, (path_i, count_i) in enumerate(occurrences):
            for path_j, count_j in occurrences[index_i + 1 :]:
                shared_count = min(count_i, count_j)
                if shared_count <= 0:
                    continue

                if str(path_i) <= str(path_j):
                    key = (path_i, path_j)
                else:
                    key = (path_j, path_i)
                pair_overlap[key] += shared_count

    if not pair_overlap:
        return

    print("=" * 80)
    print("PAIRWISE_OVERLAP_SUMMARY")
    overlap_threshold = min_overlap_percent
    for (path_a, path_b), shared_lines in sorted(
        pair_overlap.items(),
        key=lambda item: (-item[1], str(item[0][0]), str(item[0][1])),
    ):
        if shared_lines < min_shared_lines:
            continue

        total_a = file_line_counts.get(path_a, 0)
        total_b = file_line_counts.get(path_b, 0)
        smaller_total = min(total_a, total_b) if min(total_a, total_b) > 0 else 0
        if smaller_total > 0:
            percent = 100.0 * float(shared_lines) / float(smaller_total)
        else:
            percent = 0.0

        if percent < overlap_threshold:
            continue

        print(f"- FILE_A={path_a}")
        print(f"  FILE_B={path_b}")
        print(f"  total_lines_A={total_a}")
        print(f"  total_lines_B={total_b}")
        print(f"  shared_lines={shared_lines}")
        print(f"  percent_overlap_of_smaller={percent:.2f}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : List[str]
        Argument vector without the program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description="Find duplicated 'content' values across JSON files."
    )
    parser.add_argument(
        "--root",
        type=str,
        action="append",
        required=True,
        help="Root directory to scan for JSON files. Repeatable.",
    )
    parser.add_argument(
        "--min-overlap-percent",
        type=float,
        default=50.0,
        help=(
            "Minimum percent overlap (of the smaller file) to report a pair "
            "in the overlap summary (default: 50)."
        ),
    )
    parser.add_argument(
        "--min-shared-lines",
        type=int,
        default=1,
        help=(
            "Minimum shared content entries required to report a pair " "(default: 1)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Entry point for the duplicate content finder.

    Parameters
    ----------
    argv : List[str] | None
        Optional command-line arguments, excluding the program name.

    Returns
    -------
    int
        Exit code.
    """

    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    roots = [Path(value) for value in (args.root or [])]
    for root in roots:
        if not root.exists() or not root.is_dir():
            print(
                "Error: root directory does not exist or is not a directory: " f"{root}"
            )
            return 1

    index, file_line_counts = build_content_index(roots)
    print_duplicates(index)
    print_pairwise_overlap(
        index,
        file_line_counts,
        min_overlap_percent=float(args.min_overlap_percent),
        min_shared_lines=int(args.min_shared_lines),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
