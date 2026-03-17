"""Utilities to extract plain text from common document formats.

Supported extensions: .txt, .md, .pdf, .docx, .odt, .rtf. Zip archives are
handled at the processor layer (for recursion); .odt is handled inline.
"""

from __future__ import annotations

import logging
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

from docx import Document  # python-docx
from pdfminer.high_level import extract_text as pdf_extract_text
from striprtf.striprtf import rtf_to_text

# pdfminer is chatty; suppress warnings like "FontBBox ... 4 floats"
for name in ("pdfminer", "pdfminer.pdfinterp", "pdfminer.psparser", "pdfminer.pdffont"):
    logging.getLogger(name).setLevel(logging.ERROR)

# --- Public API ----------------------------------------------------


class LoadError(Exception):
    """Raised when a loader fails to extract text from a file."""


def load_text_from_file(path: Path) -> str:
    """
    Extract plain UTF-8 text from supported container formats.
    Supported: .txt, .md, .html, .htm, .pdf, .docx, .odt, .rtf
    Zips are handled one level up (processor) since they imply recursion.
    """
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".html", ".htm"}:
        return read_text_best_effort(path)
    if ext == ".pdf":
        return load_pdf_text(path)
    if ext == ".docx":
        return load_docx_text(path)
    if ext == ".odt":
        return load_odt_text(path)
    if ext == ".rtf":
        return load_rtf_text(path)
    raise LoadError(f"Unsupported extension: {ext}")


# --- Helpers -------------------------------------------------------


def read_text_best_effort(path: Path) -> str:
    """Read raw bytes and decode using a best-effort set of encodings.

    Tries UTF BOM-aware decoders and several common encodings before falling
    back to UTF-8 with replacement for undecodable bytes.
    """
    raw = path.read_bytes()
    # UTF BOMs
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    for enc in ("utf-8", "utf-16", "utf-32", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_pdf_text(path: Path) -> str:
    """Extract text from a PDF via pdfminer (no OCR)."""
    # No OCR here by design. If the PDF has no extractable text, this returns ''.
    try:
        txt = pdf_extract_text(str(path)) or ""
    except Exception as e:
        raise LoadError(f"pdfminer failed: {e}") from e
    cleaned = remove_repeating_page_margins(txt)
    # Heuristic: strip trailing whitespace noise
    cleaned = normalize_pdf_cids(cleaned)
    return normalize_text(cleaned)


def load_docx_text(path: Path) -> str:
    """Extract text from a .docx file, including simple tables."""
    try:
        doc = Document(str(path))
    except Exception as e:
        raise LoadError(f"python-docx failed: {e}") from e
    parts = []
    for p in doc.paragraphs:
        parts.append(p.text)
    # Include simple table text too
    for tbl in getattr(doc, "tables", []):
        for row in tbl.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return normalize_text("\n".join(parts))


def load_odt_text(path: Path) -> str:
    """Extract text from an .odt file by reading content.xml and stripping tags."""
    # ODT is a zip. Extract content.xml and collapse text nodes.
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("content.xml") as fh:
                xml = fh.read().decode("utf-8", errors="replace")
    except Exception as e:
        raise LoadError(f"ODT read failed: {e}") from e
    # extremely simple XML text extraction—strip tags but preserve paragraph-ish breaks
    # Replace common paragraph tags with newlines
    xml = re.sub(r"</text:p>", "\n", xml)
    xml = re.sub(r"</text:h>", "\n", xml)
    # Remove remaining tags
    txt = re.sub(r"<[^>]+>", "", xml)
    return normalize_text(txt)


def load_rtf_text(path: Path) -> str:
    """Extract text from an RTF file using striprtf."""
    try:
        txt = rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        raise LoadError(f"RTF parse failed: {e}") from e
    return normalize_text(txt)


def normalize_text(s: str) -> str:
    """Normalize newlines (CRLF -> LF) and collapse excessive blank lines."""
    s = s.replace("\f", "\n\n")
    # Normalize CRLF -> LF; collapse excessive blank lines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_pdf_cids(text: str) -> str:
    """Normalize pdfminer CID artifacts in extracted PDF text.

    Behavior:
    - Replace the inline ligature marker (cid:431) with the ASCII sequence "ff".
    - Remove all other (cid:<number>) markers when they are not inside a word
      (i.e., when surrounded only by non-word characters). This drops decorative
      glyph runs that precede headings without altering real words.

    Parameters
    ----------
    text:
        Raw text extracted from a PDF via pdfminer.

    Returns
    -------
    str
        Text with CID artifacts normalized.
    """

    def _replace_ligature(match: re.Match[str]) -> str:
        cid = match.group(1)
        if cid == "431":
            return "ff"
        return match.group(0)

    # First, handle the known ligature CID everywhere it appears.
    text = re.sub(r"\(cid:(\d+)\)", _replace_ligature, text)

    # Then, drop any remaining CID markers that are not embedded in words.
    # Match a non-word (or start), the CID, then a non-word (or end), and
    # return only the surrounding context without the CID itself.

    def _drop_decorative(match: re.Match[str]) -> str:
        before = match.group(1) or ""
        after = match.group(3) or ""
        return before + after

    text = re.sub(r"(^|[^\w])\(cid:(\d+)\)([^\w]|$)", _drop_decorative, text)
    return text


def remove_repeating_page_margins(text: str) -> str:
    """Drop recurring PDF header/footer lines that repeat on most pages."""
    pages = text.split("\f")
    if len(pages) <= 1:
        return text

    header_infos: list[list[tuple[int, str, str]]] = []
    footer_infos: list[list[tuple[int, str, str]]] = []
    header_counts: Counter[str] = Counter()
    footer_counts: Counter[str] = Counter()
    lines_per_page = []
    limit = 3

    for page in pages:
        raw_lines = page.splitlines()
        lines_per_page.append(raw_lines)
        header_info: list[tuple[int, str, str]] = []
        footer_info: list[tuple[int, str, str]] = []

        for idx, line in enumerate(raw_lines):
            stripped = line.strip()
            if not stripped:
                continue
            key = stripped.lower()
            norm = _normalize_margin_key(key)
            header_info.append((idx, key, norm))
            if len(header_info) >= limit:
                break
        for idx in range(len(raw_lines) - 1, -1, -1):
            line = raw_lines[idx]
            stripped = line.strip()
            if not stripped:
                continue
            key = stripped.lower()
            norm = _normalize_margin_key(key)
            footer_info.append((idx, key, norm))
            if len(footer_info) >= limit:
                break

        header_infos.append(header_info)
        footer_infos.append(footer_info)
        header_counts.update(info[2] for info in header_info)
        footer_counts.update(info[2] for info in footer_info)

    threshold = max(2, math.ceil(len(pages) * 0.6))
    header_norms = {norm for norm, count in header_counts.items() if count >= threshold}
    footer_norms = {norm for norm, count in footer_counts.items() if count >= threshold}

    cleaned_pages: list[str] = []
    for raw_lines, header_info, footer_info in zip(
        lines_per_page, header_infos, footer_infos
    ):
        drop_indices = set()
        for idx, key, norm in header_info:
            if norm in header_norms or _is_margin_noise(key):
                drop_indices.add(idx)
        for idx, key, norm in footer_info:
            if norm in footer_norms or _is_margin_noise(key):
                drop_indices.add(idx)
        filtered_lines = [
            line for idx, line in enumerate(raw_lines) if idx not in drop_indices
        ]
        page_text = "\n".join(filtered_lines).strip("\n")
        if page_text:
            cleaned_pages.append(page_text)

    if not cleaned_pages:
        return ""
    return "\n\n".join(cleaned_pages)


def _normalize_margin_key(key: str) -> str:
    """Normalize a header/footer line for comparison across pages."""
    key = re.sub(r"\s+", " ", key.strip())
    return re.sub(r"\d+", "<num>", key)


def _is_margin_noise(key: str) -> bool:
    """Return True if the line looks like a generic header/footer artifact."""
    lowered = key.lower()
    normalized = _normalize_margin_key(lowered)
    if lowered.startswith(("file://", "http://", "https://")):
        return True
    if normalized.startswith(("file://", "http://", "https://")):
        return True
    if re.match(r"^(?:page\s*)?<num>(?:\s*(?:/|of)\s*<num>)?$", normalized):
        return True
    if re.match(
        r"^<num>/<num>/<num>,\s*<num>:<num>\s*(?:am|pm)$",
        normalized,
    ):
        return True
    return False
