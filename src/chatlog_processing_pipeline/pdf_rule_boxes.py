"""Segment ChatGPT export PDFs by horizontal rule separators using PyMuPDF.

Some ChatGPT PDF exports draw two thin horizontal lines between conversations.
This module detects those lines per page and splits the text stream at each
separator. It then attempts to parse each segment into role-labelled messages,
falling back to a single assistant message if needed. Titles are detected from
the trailing standalone line of the final assistant turn.
"""

from __future__ import annotations

from typing import List, Optional

import fitz  # PyMuPDF

from .parsers import ParseFailed, try_parse_any
from .textloaders import _is_margin_noise


def _looks_like_title(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3 or len(s) > 120:
        return False
    if s.strip().lower() in {"user", "assistant", "system", "you", "model"}:
        return False
    if any(ch.isdigit() for ch in s):
        return False
    if s[-1] in ".!?;:":
        return False
    words = [w for w in s.split() if w]
    if len(words) <= 6 and all(w.isalpha() for w in words):
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= max(2, int(0.5 * len(words)))


def _page_separators(page: fitz.Page, y_min: float, y_max: float) -> List[float]:
    """Return y-positions of horizontal rule separators on a page.

    Looks for very thin filled rectangles spanning most of the content width,
    then clusters nearby duplicates (double-line motif) into one separator.
    """
    page_width = float(page.rect.width)
    drawings = page.get_drawings()
    rules: List[float] = []
    # Identify thin horizontal filled rectangles
    for d in drawings:
        r = d.get("rect")
        if not r:
            continue
        w = float(r.width)
        h = float(r.height)
        if h <= 1.6 and w >= 0.6 * page_width:
            rules.append(float((r.y0 + r.y1) / 2.0))
    if not rules:
        return []
    rules.sort()
    # Cluster close rules (e.g., the "double line" is two rules within ~24pt)
    clustered: List[float] = []
    group: List[float] = []
    for y in rules:
        if not group:
            group = [y]
        elif abs(y - group[-1]) <= 24.0:
            group.append(y)
        else:
            if len(group) >= 2:
                clustered.append(sum(group) / len(group))
            group = [y]
    if group:
        if len(group) >= 2:
            clustered.append(sum(group) / len(group))
    # Keep only separators within the content band
    banded = [y for y in clustered if (y_min + 20.0) <= y <= (y_max - 20.0)]
    return banded


def _is_margin_block(
    page: fitz.Page, block: tuple[float, float, float, float, str]
) -> bool:
    """Return True if a text block looks like a repeating page header/footer.

    This uses a combination of vertical position (near the top or bottom
    margin) and the existing margin-noise heuristics from textloaders
    (_is_margin_noise) applied line-by-line. Blocks are removed only when
    *all* non-empty lines in the block look like generic margins such as
    timestamps, page numbers, or file:// URLs.
    """
    x0, y0, x1, y1, text = block
    del x0, x1  # unused
    page_top = float(page.rect.y0)
    page_bottom = float(page.rect.y1)
    # Allow a generous ~60pt band at top and bottom for headers/footers.
    header_band = page_top + 60.0
    footer_band = page_bottom - 60.0

    # Only consider blocks that live entirely within the header/footer bands.
    in_header = float(y1) <= header_band
    in_footer = float(y0) >= footer_band
    if not (in_header or in_footer):
        return False

    text_str = str(text)
    lines = [ln.strip() for ln in text_str.splitlines() if ln.strip()]
    if not lines:
        return False
    # ChatGPT PDF exports commonly include a two-line header with a timestamp
    # and the literal string "ChatGPT Data Export". Treat that header as
    # margin noise outright when it appears in the top band.
    if in_header and "chatgpt data export" in text_str.lower():
        return True
    return all(_is_margin_noise(line) for line in lines)


def parse_pdf_by_horizontal_rules(
    path,
    *,
    strict: bool = False,
) -> dict:
    """Split a PDF into conversations using horizontal rule separators.

    Returns {"conversations": [...], "notes": str}.
    """

    doc = fitz.open(str(path))
    conversations: List[dict] = []
    carry_text_parts: List[str] = []

    for page in doc:
        # Build aligned text blocks for this page
        raw_blocks = [
            b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()
        ]
        if not raw_blocks:
            continue
        # Convert to a normalized tuple form and drop margin header/footer blocks
        blocks = []
        for b in raw_blocks:
            block = (
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
                str(b[4]).rstrip(),
            )
            if _is_margin_block(page, block):
                continue
            blocks.append(block)
        if not blocks:
            continue
        # Sort blocks top-to-bottom
        blocks.sort(key=lambda b: (b[1], b[0]))  # y0 asc, then x0

        # Compute a broad content band from all raw blocks (pre-filter) to avoid
        # over-shrinking when only a subset of blocks survive margin filtering
        rb_y0s = [float(b[1]) for b in raw_blocks]
        rb_y1s = [float(b[3]) for b in raw_blocks]
        band_min = min(rb_y0s) if rb_y0s else float(page.rect.y0)
        band_max = max(rb_y1s) if rb_y1s else float(page.rect.y1)
        separators = _page_separators(page, band_min, band_max)
        sep_idx = 0
        next_sep = separators[sep_idx] if separators else None

        for _x0, y0, _x1, _y1, text in blocks:
            # If we passed a separator, close out a conversation BEFORE adding this block
            while next_sep is not None and y0 >= next_sep - 0.1:
                full_text = "\n".join(carry_text_parts).strip()
                if full_text:
                    conversations.append(_segment_to_conversation(full_text, strict))
                carry_text_parts = []
                sep_idx += 1
                next_sep = separators[sep_idx] if sep_idx < len(separators) else None
            # Append block text to the current segment
            carry_text_parts.append(text)

    # Flush any remaining text as the last conversation
    full_text = "\n".join(carry_text_parts).strip()
    if full_text:
        conversations.append(_segment_to_conversation(full_text, strict))

    return {"conversations": conversations, "notes": "pdf_horizontal_rules"}


def _segment_to_conversation(text: str, strict: bool) -> dict:
    """Convert a text segment into a conversation dict with optional title.

    Title is taken as the last non-empty line immediately before the first
    role label (User/Assistant/ChatGPT/System/You). If no such line looks like
    a title, the conversation has no title.
    """
    lines = text.splitlines()
    # Find first role label line
    role_labels = {"user", "assistant", "chatgpt", "system", "you"}
    first_role_idx: Optional[int] = None
    for idx, raw in enumerate(lines):
        if raw.strip().lower() in role_labels:
            first_role_idx = idx
            break
    title: Optional[str] = None
    body_start = 0
    if first_role_idx is not None:
        for j in range(first_role_idx - 1, -1, -1):
            if lines[j].strip():
                cand = lines[j].strip()
                if _looks_like_title(cand):
                    title = cand
                    body_start = j + 1
                break
    text_body = "\n".join(lines[body_start:]).lstrip()

    try:
        parsed = try_parse_any(text_body, "chatgpt", strict=strict)
        if isinstance(parsed, dict) and isinstance(parsed.get("messages"), list):
            messages = parsed["messages"]
        else:
            messages = None
    except ParseFailed:
        messages = None

    if not messages:
        messages = [{"role": "assistant", "content": text_body}]

    conv = {"messages": messages}
    if title:
        conv["title"] = title
    return conv
