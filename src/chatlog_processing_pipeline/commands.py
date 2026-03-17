"""CLI entry points for chat log parsing and anonymization.

This module wires together the parsing pipeline and optional anonymization
using Presidio. Behavior is unchanged; the updates add documentation and
minor style cleanups (imports/docstrings) only.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from .processor import ParseMeta, _process_one_file
from .redactor import run_redaction
from .textloaders import LoadError
from .util import (
    ensure_dir,
    find_json_snippet,
    html_contains_snippet,
    write_parsed_output,
)


def main() -> None:
    """Main CLI dispatcher.

    Parses arguments and executes requested subcommands or pipeline phases.
    """
    parser = argparse.ArgumentParser(
        prog="convparse",
        description="Parse chat transcripts and/or anonymize them with Presidio",
    )

    sub = parser.add_subparsers(dest="cmd")

    # ---------------- main pipeline ----------------
    p_main = parser  # top-level flags apply directly
    p_main.add_argument("--input", required=True, help="Input directory to process")
    p_main.add_argument(
        "-o",
        "--output-dir",
        help="Output directory for parsed results (default: INPUT_DIR + '_parsed')",
    )
    p_main.add_argument(
        "-j", "--jobs", type=int, default=os.cpu_count(), help="Parallel workers"
    )
    p_main.add_argument(
        "--single-thread",
        action="store_true",
        help="Force single-threaded parsing (no multiprocessing). Use for GUI/Windows.",
    )
    p_main.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_main.add_argument("--log-file", help="Write a detailed log under the output dir")
    p_main.add_argument(
        "--no-progress", action="store_true", help="Disable parse progress bar"
    )
    p_main.add_argument(
        "--overwrite",
        action="store_true",
        help="Process even if output file already exists (default: skip existing)",
    )

    p_main.add_argument("--parse", action="store_true", help="Run parsing step")
    p_main.add_argument(
        "--method",
        dest="method",
        default=None,
        choices=[
            "auto",
            "pdf_highlight",
            "pdf_boxes",
            "pdf_text",
            "docx_titles",
            "docx_text",
            "chatgpt_html",
            "chatgpt_json",
        ],
        help=(
            "Force a parser method for all files. "
            "For PDFs: pdf_highlight | pdf_boxes | pdf_text. "
            "For DOCX: docx_titles | docx_text. "
            "Use 'auto' for best effort (default)."
        ),
    )
    p_main.add_argument("--anon", action="store_true", help="Run anonymization step")
    p_main.add_argument(
        "--anon-output",
        help="Output directory for anonymized results (default: [source]_anonymised)",
    )
    p_main.add_argument(
        "--strict-parsing",
        action="store_true",
        help="Require first turn = user and strict alternation; default off (lenient)",
    )
    p_main.add_argument(
        "--conv-separator",
        dest="conv_separator",
        default=None,
        help=(
            "Regex used to split multiple conversations within a single file. "
            "Example for lines of dashes: '(?m)^\\s*---+\\s*$'"
        ),
    )
    p_main.add_argument(
        "--role-labels",
        dest="role_labels",
        default=None,
        help=(
            "Pipe-separated role labels to recognize in transcripts where turns "
            "are written on one line (e.g., 'Player: Hello'). Example: "
            "'Player:|Gemini:' (first label maps to user, others to assistant)."
        ),
    )

    _add_anon_args(p_main)

    # ---------------- validate ----------------
    p_val = sub.add_parser("validate", help="Validate parsed JSONs")
    p_val.add_argument("parsed_dir")

    args = parser.parse_args()

    if args.cmd == "validate":
        cmd_validate(args)
        return

    if not args.parse and not args.anon:
        raise SystemExit("Nothing to do: set at least one of --parse or --anon")

    # run steps
    parse_out: Optional[Path] = None
    if args.parse:
        parse_out = cmd_parse(args)

    if args.anon:
        src_for_anon = (
            parse_out if parse_out else Path(args.input).expanduser().resolve()
        )
        # Allow --output-dir as a fallback target for --anon when --anon-output
        # is not provided, to mirror the parse step UX.
        anon_out = None
        if args.anon_output:
            anon_out = Path(args.anon_output).expanduser().resolve()
        elif args.output_dir:
            anon_out = Path(args.output_dir).expanduser().resolve()
        else:
            anon_out = src_for_anon.with_name(src_for_anon.name + "_anonymised")
        counts = run_redaction(
            in_dir=src_for_anon,
            out_dir=anon_out,
            jobs=args.jobs,
            lang=args.lang,
            entities=args.entities,
            score_threshold=args.threshold,
            operator=args.operator,
            replace_with=args.replace_with,
            mask_char=args.mask_char,
            mask_chars_to_mask=args.mask_chars_to_mask,
            mask_from_end=args.mask_from_end,
            allow_list=args.allow_list,
            allow_list_match=args.allow_list_match,
            name_entities=args.name_entities,
            name_threshold=args.name_threshold,
            name_operator=args.name_operator,
            name_replace_with=args.name_replace_with,
            name_mask_char=args.name_mask_char,
            name_mask_chars_to_mask=args.name_mask_chars_to_mask,
            name_mask_from_end=args.name_mask_from_end,
            name_allow_list=args.name_allow_list,
            name_allow_list_match=args.name_allow_list_match,
            chunk_size=args.chunk_size,
            chunk_break_window=args.chunk_break_window,
            spacy_max_length=args.spacy_max_length,
            include_all=args.include_all,
            skip_nontext=args.skip_nontext,
            overwrite=args.overwrite,
            names_only=args.names_only,
            content_only=args.content_only,
            faker_locale=args.faker_locale,
            generic_json_strings=(not args.preserve_generic_json),
            dry_run=args.dry_run,
            no_progress=args.no_progress,
            verbose=args.verbose,
        )
        print(
            f"Anonymization done. Text processed: {counts['text']}, "
            f"Binary copied: {counts['bin']}, Failed: {counts['failed']}, "
            f"Total: {counts['total']}. Output: {anon_out}"
        )


def _add_anon_args(p: argparse.ArgumentParser) -> None:
    """Add anonymization flags to an ArgumentParser without altering behavior."""
    # Content
    p.add_argument(
        "--lang", default="en", help="Language code for Presidio spaCy pipeline"
    )
    p.add_argument(
        "--entities",
        nargs="*",
        default=None,
        help="PII entities to detect in content (default: all)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Score threshold for content detection",
    )
    p.add_argument(
        "--operator",
        default="replace",  # default to replace with <REDACTED>
        choices=["redact", "replace", "mask", "hash", "faker"],
        help="Anonymizer operator for content",
    )
    p.add_argument(
        "--replace-with",
        default="<REDACTED>",
        help="Replacement value when operator=replace (content)",
    )
    p.add_argument(
        "--mask-char", default="*", help="Masking char for operator=mask (content)"
    )
    p.add_argument(
        "--mask-chars-to-mask",
        type=int,
        default=4,
        help="Chars to mask with operator=mask (content)",
    )
    p.add_argument(
        "--mask-from-end",
        action="store_true",
        help="Mask from end when operator=mask (content)",
    )
    p.add_argument(
        "--faker-locale",
        default="en_US",
        help="Locale to use when operator=faker (content/names).",
    )
    p.add_argument(
        "--allow-list", nargs="*", default=None, help="Terms/regex to allow in content"
    )
    p.add_argument(
        "--allow-list-match",
        choices=["exact", "regex"],
        default="exact",
        help="Interpretation of allow-list entries",
    )
    # Names
    p.add_argument(
        "--name-entities",
        nargs="*",
        default=None,
        help="PII entities to detect in names",
    )
    p.add_argument(
        "--name-threshold",
        type=float,
        default=0.2,
        help="Score threshold for name detection",
    )
    p.add_argument(
        "--name-operator",
        default="replace",
        choices=["replace", "redact", "mask", "hash", "faker"],
        help="Anonymizer operator for names",
    )
    p.add_argument(
        "--name-replace-with",
        default="REDACTED",
        help="Replacement value when operator=replace (names)",
    )
    p.add_argument(
        "--name-mask-char", default="_", help="Masking char for operator=mask (names)"
    )
    p.add_argument(
        "--name-mask-chars-to-mask",
        type=int,
        default=6,
        help="Chars to mask with operator=mask (names)",
    )
    p.add_argument(
        "--name-mask-from-end",
        action="store_true",
        help="Mask from end when operator=mask (names)",
    )
    p.add_argument(
        "--name-allow-list",
        nargs="*",
        default=None,
        help="Terms/regex to allow in names (not anonymized)",
    )
    p.add_argument(
        "--name-allow-list-match",
        choices=["exact", "regex"],
        default="exact",
        help="How to interpret name allow-list entries",
    )
    # Chunking & perf
    p.add_argument(
        "--chunk-size", type=int, default=200_000, help="Max characters per chunk"
    )
    p.add_argument(
        "--chunk-break-window",
        type=int,
        default=2000,
        help="Look-back window near chunk limit",
    )
    p.add_argument(
        "--spacy-max-length",
        type=int,
        default=None,
        help="Override for spaCy nlp.max_length",
    )
    # Behavior
    p.add_argument(
        "--include-all",
        action="store_true",
        help="Heuristically treat non-binary as text",
    )
    p.add_argument(
        "--skip-nontext",
        action="store_true",
        help="Skip non-text files instead of copying",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print actions without writing"
    )
    p.add_argument(
        "--names-only",
        action="store_true",
        help="Only de-identify directory/file names, leave content unchanged",
    )
    p.add_argument(
        "--content-only",
        action="store_true",
        help="Only de-identify content, keep original names (sanitized)",
    )
    p.add_argument(
        "--preserve-generic-json",
        action="store_true",
        help=(
            "Leave non-parser JSON unchanged; default is to anonymize string "
            "values in generic JSON (structure-safe)"
        ),
    )


def cmd_parse(args) -> Path:
    """Parse all files under the input root and write JSON outputs.

    Returns the output directory path containing parsed JSONs.
    """
    in_root = Path(args.input).expanduser().resolve()
    if not in_root.is_dir():
        raise FileNotFoundError(in_root)

    out_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else in_root.with_name(in_root.name + "_parsed")
    )
    ensure_dir(out_root)

    # logging
    logger = logging.getLogger("convparse")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if args.verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    if args.log_file:
        lf_path = out_root / args.log_file
        ensure_dir(lf_path.parent)
        fh = logging.FileHandler(str(lf_path), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)

    # Decide execution mode
    use_single_thread = bool(getattr(args, "single_thread", False)) or (
        getattr(args, "jobs", 1) in (0, 1)
    )

    # Gather files
    files = []
    for dirpath, _, filenames in os.walk(in_root):
        for fn in filenames:
            # Skip MS Office lock files (e.g., '~$file.docx')
            if fn.startswith("~$"):
                continue
            files.append(Path(dirpath) / fn)

    # Prefer conversations.json over chat.html when both exist in the same folder.
    # Perform a light sanity check that the JSON appears to match the HTML by
    # looking for a known snippet (title or first content) from the JSON in the HTML.
    skip_html: set[Path] = set()

    # Build a quick index of directory -> {chat.html, conversations.json}
    by_dir: dict[Path, dict[str, Path]] = {}
    for p in files:
        parent = p.parent
        name = p.name.lower()
        if name in {"chat.html", "conversations.json"}:
            d = by_dir.setdefault(parent, {})
            d[name] = p

    for parent, names in by_dir.items():
        chat = names.get("chat.html")
        conv = names.get("conversations.json")
        if not chat or not conv:
            continue
        # Light sanity check: look for a JSON snippet in the HTML
        snippet = find_json_snippet(conv)
        same = False
        if snippet:
            same = html_contains_snippet(chat, snippet)
        else:
            # Fallback: if both files are reasonably sized, assume equivalence
            try:
                size_ok = chat.stat().st_size > 0 and conv.stat().st_size > 0
            except OSError:
                size_ok = False
            same = size_ok

        if same:
            skip_html.add(chat)
            logger.info(
                "[SKIP-HTML-DUPLICATE] %s (using conversations.json in same folder)",
                chat.relative_to(in_root),
            )

    # Remove redundant chat.html entries from the worklist
    if skip_html:
        files = [p for p in files if p not in skip_html]

    total = len(files)
    ok = 0
    fail = 0

    def _expected_output_path(src_path: Path) -> Path:
        rel = src_path.relative_to(in_root)
        if src_path.suffix.lower() == ".json":
            return out_root / rel
        base = out_root / rel
        return base.with_name(base.name + ".json")

    def _parse_role_labels() -> Optional[list[str]]:
        role_labels_text = getattr(args, "role_labels", None)
        if not role_labels_text:
            return None
        return [value.strip() for value in role_labels_text.split("|") if value.strip()]

    def _should_skip_existing(src_path: Path) -> bool:
        nonlocal ok
        if src_path.suffix.lower() == ".zip":
            return False
        expected_output = _expected_output_path(src_path)
        if expected_output.exists() and not args.overwrite:
            logger.info(
                "[SKIP-EXISTS] %s -> %s",
                src_path.relative_to(in_root),
                expected_output,
            )
            ok += 1
            return True
        return False

    def _handle_parsed_output(meta: ParseMeta, out: Optional[Dict[str, Any]]) -> None:
        nonlocal ok, fail
        if meta.file_ext == ".zip":
            ok += 1
            logger.info("[ZIP] %s -> entries: %s", meta.rel_path, meta.message_count)
            return
        if meta.file_ext == ".json":
            ok += 1
            logger.info("[JSON] %s (pass-through)", meta.rel_path)
            return

        if meta.ok and out is not None:
            write_parsed_output(out_root, meta, out)
            ok += 1
            logger.info("[OK] %s", meta.rel_path)
            return

        fail += 1
        logger.warning("[FAIL] %s: %s", meta.rel_path, meta.error)

    if use_single_thread:
        # ---- Serial path (GUI-friendly, no multiprocessing) ----
        for src in files:
            # Skip if output already exists and not overwriting (non-zip only)
            if _should_skip_existing(src):
                continue
            try:
                # Parse optional role labels
                role_labels = _parse_role_labels()

                meta, out = _process_one_file(
                    src,
                    in_root,
                    out_root,
                    args.verbose,
                    args.strict_parsing,
                    forced_method=args.method,
                    role_labels=role_labels,
                    conv_separator=args.conv_separator,
                )
            except (OSError, zipfile.BadZipFile, LoadError) as e:
                fail += 1
                logger.error("[CRASH] %s: %s", src, e)
                continue

            _handle_parsed_output(meta, out)
    else:
        # ---- Parallel path (original behavior) ----
        futures: Dict = {}
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for src in files:
                # Skip if output already exists and not overwriting (non-zip only)
                if _should_skip_existing(src):
                    continue
                # Parse optional role labels (picklable list of strings)
                role_labels = _parse_role_labels()
                fut = pool.submit(
                    _process_one_file,
                    src,
                    in_root,
                    out_root,
                    args.verbose,
                    args.strict_parsing,
                    args.method,
                    role_labels,
                    args.conv_separator,
                )
                futures[fut] = src

            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    meta, out = fut.result()
                except (OSError, zipfile.BadZipFile, LoadError) as e:
                    fail += 1
                    logger.error("[CRASH] %s: %s", src, e)
                    continue

                _handle_parsed_output(meta, out)

    print(f"Done. Parsed outputs under: {out_root}")
    print(f"Summary: ok={ok}, fail={fail}, total={total}")
    return out_root


def cmd_validate(args) -> None:
    """Validate parsed JSON files for basic structural correctness."""
    parsed_dir = Path(args.parsed_dir).expanduser().resolve()
    if not parsed_dir.is_dir():
        raise FileNotFoundError(parsed_dir)

    ok = 0
    bad = 0
    for dirpath, _, filenames in os.walk(parsed_dir):
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            p = Path(dirpath) / fn
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
                print(f"[INVALID JSON] {p}: {e}")
                bad += 1
                continue

            def _validate_one(msgs, file_path=p) -> bool:
                if not isinstance(msgs, list) or len(msgs) < 2:
                    print(f"[INVALID] {file_path}: require >= 2 messages")
                    return False
                roles = [m.get("role") for m in msgs]
                if roles[0] != "user":
                    print(f"[INVALID] {file_path}: first role must be user")
                    return False
                for i in range(1, len(roles)):
                    if roles[i] == roles[i - 1]:
                        print(
                            f"[INVALID] {file_path}: non-alternating at {i} ({roles[i]})"
                        )
                        return False
                    if roles[i] not in {"user", "assistant"}:
                        print(f"[INVALID] {file_path}: bad role {roles[i]}")
                        return False
                if any(not isinstance(m.get("content"), str) for m in msgs):
                    print(f"[INVALID] {file_path}: content must be strings")
                    return False
                return True

            valid = False
            if isinstance(obj.get("messages"), list):
                valid = _validate_one(obj["messages"])  # single-conversation shape
            elif isinstance(obj.get("conversations"), list):
                # consider valid only if all conversations are valid
                convs = obj["conversations"]
                if convs:
                    valid = True
                    for conv in convs:
                        msgs = conv.get("messages") if isinstance(conv, dict) else None
                        if not _validate_one(msgs):
                            valid = False
                            break
            if valid:
                ok += 1
            else:
                bad += 1

    print(f"Validation summary: ok={ok}, bad={bad}, total={ok + bad}")
