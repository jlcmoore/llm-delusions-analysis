"""Estimate completion token usage from annotation outputs.

This helper script scans annotation output JSONL files (for example
``annotation_outputs/205/20251222-192416__input=...jsonl``),
reconstructs the classifier's JSON response payload for each row, and
uses ``litellm.token_counter`` to estimate completion token counts.

Usage example
-------------

    python tmp_estimate_completion_tokens.py \\
        annotation_outputs/205/20251222-192416__input=transcripts_de_ided\\
        &max_messages=1000&model=gpt-5.1&preceding_context=3&randomize=True\\
        &randomize_per_ppt=equal.jsonl

The script extracts the basename from the provided path and then
automatically discovers matching runs under ``annotation_outputs/**``.
It reports the total number of reconstructed completions, the total
estimated completion tokens, and the average completion tokens per
response for the chosen model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import litellm
from llm_delusions_annotations.llm_utils.client import DEFAULT_CHAT_MODEL
from tqdm import tqdm

from annotation.io import iter_jsonl_records


def _iter_matching_files(root: Path, stub_name: str) -> Iterable[Path]:
    """Yield files under ``root`` whose name exactly matches ``stub_name``."""

    for path in root.rglob(stub_name):
        if path.is_file():
            yield path


def _reconstruct_completion_text(row: dict) -> str | None:
    """Return an approximate completion JSON string for a result row.

    The classifier prompt instructs the model to output a JSON object
    with exactly two fields:

    - ``score``: integer from 0 to 10.
    - ``quotes``: JSON array of quote strings.

    The annotation pipeline parses this into ``score`` and ``matches``
    fields. This helper inverts that transformation to approximate the
    original model completion.
    """

    if row.get("type") == "meta":
        return None

    if "score" not in row or "matches" not in row:
        return None

    try:
        score = int(row.get("score", 0))
    except (TypeError, ValueError):
        score = 0

    matches = row.get("matches")
    if not isinstance(matches, list):
        return None

    quotes: List[str] = []
    for item in matches:
        if isinstance(item, str):
            quotes.append(item)
        else:
            try:
                quotes.append(str(item))
            except (TypeError, ValueError):
                continue

    payload = {"score": score, "quotes": quotes}
    # Use compact separators to approximate the minimal JSON form the
    # model could have returned; tokenization is insensitive to
    # whitespace, so this is sufficient for estimation.
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _count_completion_tokens(model: str, text: str) -> int:
    """Return token count for ``text`` using the model's tokenizer."""

    return int(litellm.token_counter(model=model, text=text))


def main() -> int:
    """Script entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Estimate average completion tokens per classifier response by "
            "reconstructing JSON outputs from annotation result files."
        )
    )
    parser.add_argument(
        "stub_path",
        type=Path,
        help=(
            "Path to one example annotation output JSONL file. "
            "The basename will be used to discover matching runs "
            "under annotation_outputs/**."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("annotation_outputs"),
        help="Root directory to search for matching runs (default: annotation_outputs).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_CHAT_MODEL,
        help=(
            "Model identifier to use for tokenization "
            f"(default: {DEFAULT_CHAT_MODEL})."
        ),
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of result rows to sample across all files. "
            "When unset, all rows are processed."
        ),
    )

    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Search root not found or not a directory: {root}")
        return 2

    stub_name = args.stub_path.name
    matching_files = list(_iter_matching_files(root, stub_name))
    if not matching_files:
        print(f"No files named {stub_name!r} found under {root}")
        return 0

    print(
        f"Discovered {len(matching_files)} matching files for {stub_name!r} under {root}"
    )
    total_rows = 0
    total_tokens = 0

    remaining = args.sample_limit if args.sample_limit is not None else None

    for path in tqdm(matching_files, desc="Scanning annotation outputs"):
        if remaining is not None and remaining <= 0:
            break
        try:
            for obj in iter_jsonl_records(path):
                if remaining is not None and remaining <= 0:
                    break

                text = _reconstruct_completion_text(obj)
                if text is None:
                    continue

                tokens = _count_completion_tokens(args.model, text)
                total_rows += 1
                total_tokens += tokens
                if remaining is not None:
                    remaining -= 1
        except OSError:
            continue

    if total_rows == 0:
        print("No usable completion rows found; nothing to report.")
        return 0

    average_tokens = total_tokens / float(total_rows)
    print("\nCompletion token statistics:")
    print(f"  Model: {args.model}")
    print(f"  Files scanned        : {len(matching_files)}")
    print(f"  Responses sampled    : {total_rows}")
    print(f"  Total completion tok.: {total_tokens}")
    print(f"  Avg tokens/response  : {average_tokens:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
