"""Parse files according to a plan CSV, or generate one.

Usage:
  - Generate plan:
      env-delusions/bin/python scripts/parse_plan.py \
        --input DIR --generate-plan plan.csv

  - Parse by plan:
      env-delusions/bin/python scripts/parse_plan.py \
        --input DIR --plan plan.csv --output-dir OUT [--strict]

  - Parse by plan for selected participants:
      env-delusions/bin/python scripts/parse_plan.py \
        --input DIR --plan plan.csv --output-dir OUT \
        --participant 201 --participant 102
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

from chatlog_processing_pipeline.processor import _process_one_file
from chatlog_processing_pipeline.util import (
    ensure_dir,
    find_json_snippet,
    html_contains_snippet,
    write_parsed_output,
)
from utils.cli import add_participants_argument
from utils.participants import normalize_participant_id


def read_plan(plan_path: Path) -> Dict[str, Dict[str, object]]:
    """Read a plan CSV into a lookup table.

    Parameters:
    - plan_path: Absolute path to a CSV file with headers
      `rel_path,method,role_labels,conv_separator,skip`.

    Returns:
    - Dictionary keyed by `rel_path` containing parsed configuration for each
      file, including the normalized method, role label list, optional
      conversation separator, and skip flag.
    """
    table: Dict[str, Dict[str, object]] = {}
    with plan_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel = (row.get("rel_path") or "").strip()
            if not rel:
                continue
            method = (row.get("method") or "").strip().lower() or "auto"
            role_labels_raw = (row.get("role_labels") or "").strip()
            labels = (
                [s.strip() for s in role_labels_raw.split("|") if s.strip()]
                if role_labels_raw
                else []
            )
            skip = (row.get("skip") or "").strip().lower()
            conv_sep = (row.get("conv_separator") or "").strip()
            table[rel] = {
                "method": method,
                "role_labels": labels,
                "skip": skip,
                "conv_separator": conv_sep,
            }
    return table


def _default_method_for(path: Path) -> str:
    """Infer a reasonable default parsing method for a file path.

    Parameters:
    - path: File system path to inspect.

    Returns:
    - A short method string such as `auto`, `docx_titles`, `chatgpt_html`,
      or `chatgpt_json` that can be placed in the plan CSV.
    """
    ext = path.suffix.lower()
    name = path.name.lower()
    if ext == ".pdf":
        return "auto"
    if ext == ".docx":
        return "docx_titles"
    if ext in {".html", ".htm"} or name.endswith("chat.html"):
        return "chatgpt_html"
    if ext == ".json" or name.endswith("conversations.json"):
        return "chatgpt_json"
    return "auto"


def _is_image(path: Path) -> bool:
    """Return True if the path looks like an image file by extension."""
    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".heic",
        ".heif",
        ".svg",
    }


def _is_probably_binary(path: Path) -> bool:
    """Heuristic check for binary-like content in small header read."""
    try:
        with path.open("rb") as handle:
            chunk = handle.read(2048)
    except OSError:
        return False
    if not chunk:
        return False
    if b"\x00" in chunk:
        return True
    text_bytes = sum(1 for b in chunk if 9 <= b <= 13 or 32 <= b <= 126)
    ratio = text_bytes / max(1, len(chunk))
    return ratio < 0.7


def _is_ignored(path: Path) -> bool:
    """Return True if the file should be omitted from plan generation."""
    name = path.name
    if name == ".DS_Store":
        return True
    if path.suffix.lower() in {".wav", ".mp4"}:
        return True
    if _is_image(path):
        return True
    if path.suffix == "" and _is_probably_binary(path):
        return True
    return False


def _compute_skip_html_rels(
    plan: Dict[str, Dict[str, object]], in_root: Path
) -> set[str]:
    """Compute which plan rows for chat.html should be skipped in favor of JSON.

    Detects sibling pairs of `chat.html` and `conversations.json` and performs a
    light equivalence check before deciding to prefer the JSON.
    """
    by_dir: Dict[Path, Dict[str, str]] = {}
    for rel in plan:
        p = in_root / rel
        name = p.name.lower()
        if name in {"chat.html", "conversations.json"}:
            bucket = by_dir.setdefault(p.parent, {})
            bucket[name] = rel

    skip_html_rels: set[str] = set()
    for names in by_dir.values():
        chat_rel = names.get("chat.html")
        conv_rel = names.get("conversations.json")
        if not chat_rel or not conv_rel:
            continue
        chat_path = in_root / chat_rel
        conv_path = in_root / conv_rel
        snippet = find_json_snippet(conv_path)
        same = False
        if snippet:
            same = html_contains_snippet(chat_path, snippet)
        else:
            try:
                same = chat_path.stat().st_size > 0 and conv_path.stat().st_size > 0
            except OSError:
                same = False
        if same:
            skip_html_rels.add(chat_rel)
            print(
                f"[SKIP-HTML-DUPLICATE] {chat_rel} (using conversations.json in same folder)"
            )
    return skip_html_rels


def _rel_matches_any_participant(rel: str, participants: Iterable[str]) -> bool:
    """Return True if rel_path contains one of the participant ids as a component.

    Parameters:
    - rel: Relative path from the plan CSV, using POSIX-style separators.
    - participants: Iterable of participant identifiers such as ``105`` or
      ``201``. Matching is case-insensitive and uses individual path
      components only (for example, ``201/file.pdf`` matches ``201``).

    Returns:
    - True if any participant identifier appears as a path component within
      ``rel``; False otherwise.
    """
    wanted = {
        normalize_participant_id(p.strip()).casefold()
        for p in participants
        if p.strip()
    }
    if not wanted:
        return False
    parts = [
        normalize_participant_id(part.strip()).casefold()
        for part in Path(rel).parts
        if part.strip()
    ]
    return any(part in wanted for part in parts)


def _generate_plan(in_root: Path, out_csv: Path) -> None:
    """Walk input tree and write a plan CSV with default methods."""
    rows: list[dict[str, str]] = []
    for dirpath, _dirs, filenames in os.walk(in_root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if _is_ignored(p):
                continue
            rows.append(
                {
                    "rel_path": str(p.relative_to(in_root)),
                    "method": _default_method_for(p),
                    "role_labels": "",
                    "conv_separator": "",
                    "skip": "",
                }
            )
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "rel_path",
                "method",
                "role_labels",
                "conv_separator",
                "skip",
            ],
        )
        writer.writeheader()
        for r in sorted(rows, key=lambda r: r["rel_path"].lower()):
            writer.writerow(r)
    print(f"Plan written: {out_csv}")


def _process_plan(
    *,
    in_root: Path,
    out_root: Path,
    plan: Dict[str, Dict[str, object]],
    strict: bool,
    participants: Optional[Iterable[str]] = None,
) -> None:
    """Process inputs listed in a plan, honoring duplicate skip policy.

    Parameters:
    - in_root: Input root directory corresponding to the plan ``rel_path``
      entries.
    - out_root: Destination directory for parsed JSON outputs.
    - plan: Parsed plan table keyed by ``rel_path``.
    - strict: When True, enable stricter parsing behavior in the underlying
      chat processing pipeline.
    - participants: Optional iterable of participant identifiers (ppt ids). If
      provided, only plan rows whose ``rel_path`` includes one of these ids as
      a path component are processed.
    """
    ok = 0
    fail = 0
    participants_list = list(participants or [])
    skip_html_rels = _compute_skip_html_rels(plan, in_root)
    for rel, cfg in plan.items():
        if participants_list and not _rel_matches_any_participant(
            rel, participants_list
        ):
            continue
        if rel in skip_html_rels:
            continue
        src = in_root / rel
        if not src.exists():
            print(f"[MISSING] {rel}")
            fail += 1
            continue
        if (cfg.get("skip") or "").lower() in {"1", "y", "yes", "true"}:
            print(f"[SKIP] {rel}")
            continue
        method = cfg.get("method") or "auto"
        roles: Optional[Iterable[str]] = cfg.get("role_labels")
        conv_sep = (cfg.get("conv_separator") or "").strip() or None
        try:
            meta, out = _process_one_file(
                src=src,
                in_root=in_root,
                out_root=out_root,
                verbose=True,
                strict_parsing=bool(strict),
                forced_method=str(method) if method else None,
                role_labels=roles,
                conv_separator=conv_sep,
            )
        except (OSError, RuntimeError, ValueError) as e:
            print(f"[CRASH] {rel}: {e}")
            fail += 1
            continue
        if meta.ok and out is not None:
            out_path = write_parsed_output(out_root, meta, out)
            ok += 1
            print(f"[OK] {rel} -> {out_path}")
        elif meta.ok and out is None:
            print(
                f"[OK] {rel} (pass-through: {meta.source_guess}, count={meta.message_count})"
            )
        else:
            fail += 1
            method_str = meta.source_guess if hasattr(meta, "source_guess") else ""
            if method_str:
                print(f"[FAIL] {rel}: method={method_str}; error={meta.error}")
            else:
                print(f"[FAIL] {rel}: {meta.error}")
    print(f"Done. Plan processed. ok={ok}, fail={fail}, total={ok+fail}")


def main() -> None:
    """Generate a plan or process one while skipping duplicated chat.html."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--plan")
    ap.add_argument("--output-dir")
    ap.add_argument("--generate-plan")
    ap.add_argument("--strict", action="store_true")
    add_participants_argument(
        ap,
        help_text=(
            "Restrict processing to plan rows whose rel_path contains one of "
            "these participant identifiers (ppt ids) as a path component, "
            "for example 201 or 102. Repeatable."
        ),
    )
    args = ap.parse_args()

    in_root = Path(args.input).expanduser().resolve()

    if args.generate_plan:
        _generate_plan(in_root, Path(args.generate_plan).expanduser().resolve())
        return

    if not args.plan or not args.output_dir:
        raise SystemExit("Provide --plan and --output-dir or use --generate-plan")

    out_root = Path(args.output_dir).expanduser().resolve()
    ensure_dir(out_root)
    plan = read_plan(Path(args.plan).expanduser().resolve())
    _process_plan(
        in_root=in_root,
        out_root=out_root,
        plan=plan,
        strict=args.strict,
        participants=args.participants,
    )


if __name__ == "__main__":
    main()
