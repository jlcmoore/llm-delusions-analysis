"""Core processing helpers to parse files into chat message JSONs.

This module reads input files, detects likely source interfaces, attempts to
parse them into a normalized message list, and writes outputs. Zip containers
are handled via a dedicated helper that expands them and processes members.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from docx.opc.exceptions import PackageNotFoundError

from .detectors import guess_source_interface
from .doc_titles import parse_docx_by_fonts
from .parsers import ParseFailed, parse_with_labels, try_parse_any
from .pdf_highlight_roles import parse_pdf_with_highlight_roles
from .pdf_rule_boxes import parse_pdf_by_horizontal_rules
from .textloaders import LoadError, load_text_from_file
from .util import (
    ensure_dir,
    is_probably_text_by_content,
    normalize_meta_dict,
    write_parsed_output,
)

SUPPORTED_EXTS = {
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".odt",
    ".rtf",
    ".zip",
    ".html",
    ".htm",
}


@dataclass
class ParseMeta:
    """
    A dataclass for parsing metadata.
    """

    src_path: str
    rel_path: str
    file_ext: str
    source_guess: str
    ok: bool
    error: Optional[str]
    message_count: int


def _process_one_file(
    src: Path,
    in_root: Path,
    out_root: Path,
    verbose: bool,
    strict_parsing: bool,
    forced_method: Optional[str] = None,
    role_labels: Optional[Iterable[str]] = None,
    conv_separator: Optional[str] = None,
) -> Tuple[ParseMeta, Optional[Dict[str, Any]]]:
    """Parse a single file and return metadata plus parsed output.

    The returned output is a dict with keys: meta, messages, notes; or None on
    failure. Zip files are forwarded to _process_zip for recursive handling.
    """
    rel = src.relative_to(in_root)
    ext = src.suffix.lower()

    # Pass-through for pre-parsed JSON files: copy without modification.
    if ext == ".json":
        rel = src.relative_to(in_root)
        dest = out_root / rel
        ensure_dir(dest.parent)
        shutil.copy2(src, dest)
        return (
            ParseMeta(str(src), str(rel), ext, "json-pass-through", True, None, 0),
            None,
        )

    if ext not in SUPPORTED_EXTS:
        if is_probably_text_by_content(src):
            ext = ".txt"
        else:
            return (
                ParseMeta(
                    str(src),
                    str(rel),
                    ext,
                    "unknown",
                    False,
                    "unsupported extension",
                    0,
                ),
                None,
            )

    if ext == ".zip":
        return _process_zip(src, in_root, out_root, verbose, strict_parsing)

    def _try_labels_first(
        text: str,
        *,
        source_guess: str,
    ) -> Tuple[ParseMeta, Optional[Dict[str, Any]]]:
        """Try label-colon parsing first when role_labels are provided.

        Returns a successful (meta, out) tuple if messages were parsed; otherwise
        raises ParseFailed to let caller fall back to generic parsing.
        """
        if not role_labels:
            raise ParseFailed("no role labels provided")
        parsed_local = parse_with_labels(
            text, labels=role_labels, strict=strict_parsing
        )
        msgs_lbl = (
            parsed_local.get("messages")
            if isinstance(parsed_local.get("messages"), list)
            else None
        )
        if msgs_lbl is None:
            raise ParseFailed("label-colon parser did not return messages")
        meta_local = ParseMeta(
            str(src),
            str(src.relative_to(in_root)),
            ext,
            source_guess,
            True,
            None,
            len(msgs_lbl),
        )
        out_local = {
            "meta": normalize_meta_dict(meta_local),
            "messages": msgs_lbl,
            "notes": parsed_local.get("notes", ""),
        }
        return meta_local, out_local

    def _parse_text_with_fallback(
        text: str,
        *,
        source_guess: str,
    ) -> Optional[Tuple[ParseMeta, Dict[str, Any]]]:
        """Parse text with role-label fallback and generic parsers."""

        if role_labels:
            try:
                return _try_labels_first(text, source_guess=source_guess)
            except ParseFailed:
                pass
        parsed = try_parse_any(text, source_guess, strict=strict_parsing)
        if "conversations" in parsed and isinstance(parsed["conversations"], list):
            convs = parsed["conversations"]
            meta_local = ParseMeta(
                str(src),
                str(src.relative_to(in_root)),
                ext,
                source_guess,
                True,
                None,
                sum(len(c.get("messages", [])) for c in convs),
            )
            out = {
                "meta": normalize_meta_dict(meta_local),
                "conversations": convs,
                "notes": parsed.get("notes", ""),
            }
            return meta_local, out
        messages = (
            parsed.get("messages") if isinstance(parsed.get("messages"), list) else None
        )
        if messages is None:
            return None
        meta_local = ParseMeta(
            str(src),
            str(src.relative_to(in_root)),
            ext,
            source_guess,
            True,
            None,
            len(messages),
        )
        out = {
            "meta": normalize_meta_dict(meta_local),
            "messages": messages,
            "notes": parsed.get("notes", ""),
        }
        return meta_local, out

    # Forced method handling
    if forced_method and forced_method != "auto":
        method = forced_method.lower()
        try:
            if method == "pdf_highlight" and ext == ".pdf":
                hl = parse_pdf_with_highlight_roles(src)
                msgs = hl.get("messages") if isinstance(hl, dict) else None
                if isinstance(msgs, list):
                    meta_local = ParseMeta(
                        str(src),
                        str(src.relative_to(in_root)),
                        ext,
                        "pdf-highlight-roles",
                        True,
                        None,
                        len(msgs),
                    )
                    out = {
                        "meta": normalize_meta_dict(meta_local),
                        "messages": msgs,
                        "notes": hl.get("notes", ""),
                    }
                    return meta_local, out
            if method == "pdf_text" and ext == ".pdf":
                text = load_text_from_file(src)
                source_guess = guess_source_interface(text, ext)
                parsed_output = _parse_text_with_fallback(
                    text, source_guess=source_guess
                )
                if parsed_output is not None:
                    return parsed_output
            if method == "docx_text" and ext == ".docx":
                # Force DOCX through plain-text extraction + generic parsers
                text = load_text_from_file(src)
                source_guess = guess_source_interface(text, ext)
                parsed_output = _parse_text_with_fallback(
                    text, source_guess=source_guess
                )
                if parsed_output is not None:
                    return parsed_output
            if method == "pdf_boxes" and ext == ".pdf":
                rule_parsed = parse_pdf_by_horizontal_rules(src, strict=strict_parsing)
                convs = (
                    rule_parsed.get("conversations")
                    if isinstance(rule_parsed, dict)
                    else None
                )
                if isinstance(convs, list) and convs:
                    out = {
                        "meta": {
                            "filename": src.name,
                            "rel_path": str(src.relative_to(in_root)),
                            "file_ext": ext,
                            "source_guess": "pdf-rules",
                        },
                        "conversations": convs,
                        "notes": rule_parsed.get("notes", ""),
                    }
                    return (
                        ParseMeta(
                            str(src),
                            str(src.relative_to(in_root)),
                            ext,
                            "pdf-rules",
                            True,
                            None,
                            sum(len(c.get("messages", [])) for c in convs),
                        ),
                        out,
                    )
            if method == "docx_titles" and ext == ".docx":
                titled = parse_docx_by_fonts(
                    src, strict=strict_parsing, role_labels=role_labels
                )
                convs = (
                    titled.get("conversations") if isinstance(titled, dict) else None
                )
                if isinstance(convs, list) and convs:
                    out = {
                        "meta": {
                            "filename": src.name,
                            "rel_path": str(src.relative_to(in_root)),
                            "file_ext": ext,
                            "source_guess": "docx-titles-fonts",
                        },
                        "conversations": convs,
                        "notes": titled.get("notes", ""),
                    }
                    return (
                        ParseMeta(
                            str(src),
                            str(src.relative_to(in_root)),
                            ext,
                            "docx-titles-fonts",
                            True,
                            None,
                            sum(len(c.get("messages", [])) for c in convs),
                        ),
                        out,
                    )
            # Fall back to default flow below if forced method didn't match
        except (RuntimeError, OSError, ValueError, ParseFailed) as e:
            # On failure of forced method, return failure with detailed context
            detail = f"forced method {method} failed: {type(e).__name__}: {e}"
            if verbose:
                # Provide additional context for debugging without stopping execution
                print(
                    f"[FORCED-METHOD-FAIL] {src} ({ext}): {detail}",
                    flush=True,
                )
            return (
                ParseMeta(
                    str(src),
                    str(src.relative_to(in_root)),
                    ext,
                    method,
                    False,
                    detail,
                    0,
                ),
                None,
            )

    # For PDFs, prefer highlights first; then boxes; then text parsing.
    if ext == ".pdf":
        # 1) Highlights path (alternating highlighted lines detected)
        try:
            hl = parse_pdf_with_highlight_roles(src)
            hl_messages = hl.get("messages") if isinstance(hl, dict) else None
            if isinstance(hl_messages, list) and len(hl_messages) >= 2:
                chat_date = hl.get("chat_date") if isinstance(hl, dict) else None
                out = {
                    "meta": {
                        "filename": src.name,
                        "rel_path": str(rel),
                        "file_ext": ext,
                        "source_guess": "pdf-highlight-roles",
                        "chat_date": chat_date,
                    },
                    "messages": hl_messages,
                    "notes": hl.get("notes", ""),
                }
                return (
                    ParseMeta(
                        str(src),
                        str(rel),
                        ext,
                        "pdf-highlight-roles",
                        True,
                        None,
                        len(hl_messages),
                    ),
                    out,
                )
        except (RuntimeError, ValueError, OSError):
            pass

        # 2) Boxes (horizontal rule) segmentation
        rule_parsed = parse_pdf_by_horizontal_rules(src, strict=strict_parsing)
        convs = (
            rule_parsed.get("conversations") if isinstance(rule_parsed, dict) else None
        )
        if isinstance(convs, list) and convs:
            msg_count = sum(len(c.get("messages", [])) for c in convs)
            meta_local = ParseMeta(
                str(src), str(rel), ext, "pdf-rules", True, None, msg_count
            )
            out = {
                "meta": normalize_meta_dict(meta_local),
                "conversations": convs,
                "notes": rule_parsed.get("notes", ""),
            }
            return meta_local, out

    # DOCX special handling: segment by visible titles before first role turn
    if ext == ".docx":
        try:
            # Prefer style/size-based titles directly from the .docx
            titled_fonts = parse_docx_by_fonts(
                src, strict=strict_parsing, role_labels=role_labels
            )
            convs = (
                titled_fonts.get("conversations")
                if isinstance(titled_fonts, dict)
                else None
            )
            if isinstance(convs, list) and convs:
                msg_count = sum(len(c.get("messages", [])) for c in convs)
                meta_local = ParseMeta(
                    str(src),
                    str(rel),
                    ext,
                    "docx-titles-fonts",
                    True,
                    None,
                    msg_count,
                )
                out = {
                    "meta": normalize_meta_dict(meta_local),
                    "conversations": convs,
                    "notes": titled_fonts.get("notes", ""),
                }
                return meta_local, out
        except ParseFailed as e_docx:
            if verbose:
                print(f"[DOCX-TITLES-NO-TITLES] {src}: {e_docx}")
        except (
            OSError,
            ValueError,
            RuntimeError,
            zipfile.BadZipFile,
            PackageNotFoundError,
        ) as e:
            # If python-docx fails on this file, fall back to plain text
            if verbose:
                print(f"[DOCX-TITLES-ERROR] {src}: {e}")
            return (
                ParseMeta(
                    str(src),
                    str(rel),
                    ext,
                    "docx-titles-fonts",
                    False,
                    f"docx parse failed: {e}",
                    0,
                ),
                None,
            )
    # Fallback: load text and use content-based parsers
    try:
        text = load_text_from_file(src)
    except LoadError as e:
        return (
            ParseMeta(
                str(src), str(rel), ext, "unknown", False, f"load failed: {e}", 0
            ),
            None,
        )

    source = guess_source_interface(text, ext)

    # If a conversation separator is provided, split text and parse segments
    if conv_separator:
        parts = re.split(conv_separator, text)
        conversations = []
        for part in parts:
            seg = part.strip()
            if not seg:
                continue
            try:
                if role_labels:
                    try:
                        parsed_seg = parse_with_labels(
                            seg, labels=role_labels, strict=strict_parsing
                        )
                    except ParseFailed:
                        parsed_seg = try_parse_any(seg, source, strict=strict_parsing)
                else:
                    parsed_seg = try_parse_any(seg, source, strict=strict_parsing)
            except ParseFailed:
                conversations.append(
                    {"messages": [{"role": "assistant", "content": seg}]}
                )
                continue
            if "messages" in parsed_seg and isinstance(parsed_seg["messages"], list):
                conversations.append({"messages": parsed_seg["messages"]})
            elif "conversations" in parsed_seg and isinstance(
                parsed_seg["conversations"], list
            ):
                conversations.extend(parsed_seg["conversations"])
        if conversations:
            msg_count = sum(len(c.get("messages", [])) for c in conversations)
            meta_local = ParseMeta(
                str(src), str(rel), ext, source, True, None, msg_count
            )
            out = {
                "meta": normalize_meta_dict(meta_local),
                "conversations": conversations,
                "notes": "text_split_by_separator",
            }
            return meta_local, out

    try:
        if role_labels:
            try:
                parsed = parse_with_labels(
                    text, labels=role_labels, strict=strict_parsing
                )
            except ParseFailed:
                parsed = try_parse_any(text, source, strict=strict_parsing)
        else:
            parsed = try_parse_any(text, source, strict=strict_parsing)
    except ParseFailed as e:
        return (
            ParseMeta(str(src), str(rel), ext, source, False, f"parse failed: {e}", 0),
            None,
        )

    # The PDF-specific paths were attempted above; proceed with parsed content

    # If we reach here, either not a PDF or no PDF-special segmentation applied.

    if "conversations" in parsed and isinstance(parsed["conversations"], list):
        convs = parsed["conversations"]
        msg_count = sum(len(c.get("messages", [])) for c in convs)
        meta_local = ParseMeta(str(src), str(rel), ext, source, True, None, msg_count)
        out = {
            "meta": normalize_meta_dict(meta_local),
            "conversations": convs,
            "notes": parsed.get("notes", ""),
        }
        return meta_local, out
    messages = parsed["messages"]
    meta_local = ParseMeta(str(src), str(rel), ext, source, True, None, len(messages))
    out = {
        "meta": normalize_meta_dict(meta_local),
        "messages": messages,
        "notes": parsed.get("notes", ""),
    }
    return meta_local, out


def _process_zip(
    src_zip: Path,
    in_root: Path,
    out_root: Path,
    verbose: bool,
    strict_parsing: bool,
) -> Tuple[ParseMeta, Optional[Dict[str, Any]]]:
    """Process a zip archive by extracting and parsing its members."""
    rel = src_zip.relative_to(in_root)
    sub_out_root = out_root / rel
    ensure_dir(sub_out_root)
    if verbose:
        print(f"[UNZIP] {src_zip} -> {sub_out_root}", flush=True)

    ok_count = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(src_zip) as zf:
            zf.extractall(tmpdir)
        extracted_root = Path(tmpdir)

        for root, _, files in os.walk(extracted_root):
            for fn in files:
                sp = Path(root) / fn
                meta, out = _process_one_file(
                    sp, extracted_root, sub_out_root, verbose, strict_parsing
                )
                # Write child outputs here so the parent CLI does not need to.
                if meta.ok and out is not None and meta.file_ext != ".zip":
                    write_parsed_output(sub_out_root, meta, out)
                    ok_count += 1
                elif meta.file_ext == ".zip":
                    ok_count += 1

    return (
        ParseMeta(str(src_zip), str(rel), ".zip", "archive", True, None, ok_count),
        None,
    )
