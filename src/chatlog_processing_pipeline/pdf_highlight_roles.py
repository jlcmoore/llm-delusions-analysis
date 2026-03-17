"""PDF fallback: infer chat turns from alternating highlighted lines.

This module provides a best-effort parser that examines the page layout of a
PDF and detects horizontal rectangles behind text lines (as produced by some
Google Docs exports). When present and alternating across contiguous blocks,
these background rectangles can be used as a proxy for speaker role: highlighted
blocks are treated as "user" and non-highlighted blocks as "assistant".

This is intended as a last-resort fallback and should only be used when the
regular text-based parsers fail and highlight-like rectangles are detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

from pdfminer.high_level import extract_pages
from pdfminer.layout import (
    LTChar,
    LTPage,
    LTRect,
    LTTextBoxHorizontal,
    LTTextLineHorizontal,
)


@dataclass
class LineSeg:
    """A single text line with geometry and highlight flag."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    highlighted: bool
    color: tuple | None


def _iter_layout(objs: Iterable) -> Iterable:
    """Yield all layout objects recursively."""

    for obj in objs:
        yield obj
        for child in getattr(obj, "_objs", []) or []:
            yield from _iter_layout([child])


def _rects_on_page(page: LTPage) -> List[LTRect]:
    """Collect rectangle objects for a page that likely correspond to highlights.

    We filter out full-page or nearly white rectangles that serve as generic
    backgrounds and keep smaller, colored strips that more plausibly represent
    user-applied highlights behind text lines.
    """

    rects: List[LTRect] = []
    page_height = float(getattr(page, "height", (page.bbox[3] - page.bbox[1])))
    for obj in _iter_layout(page):
        if isinstance(obj, LTRect):
            w = abs(obj.x1 - obj.x0)
            h = abs(obj.y1 - obj.y0)
            if w <= 5 or h <= 2:
                # Avoid hairlines and tiny artifacts
                continue
            # Skip very tall rectangles (likely page backgrounds or large panels)
            if h >= 0.5 * page_height:
                continue
            color = getattr(obj, "non_stroking_color", None)
            if isinstance(color, (tuple, list)) and len(color) >= 3:
                # Drop rectangles that are effectively white backgrounds
                if all(c >= 0.97 for c in color[:3]):
                    continue
            rects.append(obj)
    return rects


def _lines_on_page(page: LTPage) -> List[LTTextLineHorizontal]:
    """Collect text lines for a page, de-duplicating overlaps.

    Some PDFs render the same text line multiple times (for example, once
    as part of a text box and again as a standalone line) with identical
    geometry. This helper normalizes by text content and bounding box so
    that each visual line appears only once.
    """

    lines: List[LTTextLineHorizontal] = []
    seen: set[tuple] = set()

    def _maybe_add(line: LTTextLineHorizontal) -> None:
        text = line.get_text().strip("\n")
        if not text:
            return
        key = (
            round(float(line.x0), 2),
            round(float(line.y0), 2),
            round(float(line.x1), 2),
            round(float(line.y1), 2),
            text,
        )
        if key in seen:
            return
        seen.add(key)
        lines.append(line)

    for obj in _iter_layout(page):
        if isinstance(obj, LTTextLineHorizontal):
            _maybe_add(obj)
        elif isinstance(obj, LTTextBoxHorizontal):
            # Some versions wrap lines inside boxes only
            for child in getattr(obj, "_objs", []) or []:
                if isinstance(child, LTTextLineHorizontal):
                    _maybe_add(child)
    return lines


def _overlap_area(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    """Compute area of intersection between two (x0,y0,x1,y1) boxes."""

    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _is_line_highlighted(line: LTTextLineHorizontal, rects: List[LTRect]) -> bool:
    """Determine if a line has a backing rectangle covering a meaningful area."""

    lx0, ly0, lx1, ly1 = line.x0, line.y0, line.x1, line.y1
    larea = max(1.0, (lx1 - lx0) * (ly1 - ly0))
    # Generous threshold: at least 25% of line box is covered by some rect
    for r in rects:
        rbox = (r.x0, r.y0, r.x1, r.y1)
        if _overlap_area((lx0, ly0, lx1, ly1), rbox) >= 0.25 * larea:
            return True
    return False


def _color_to_rgb(c) -> tuple | None:
    """Normalize a pdfminer color value to an RGB tuple in 0..1 or None."""
    if c is None:
        return None
    if isinstance(c, (int, float)):
        v = float(c)
        if v < 0:
            v = 0.0
        if v > 1:
            v = 1.0
        return (v, v, v)
    if isinstance(c, tuple):
        if len(c) >= 3:
            return tuple(float(max(0, min(1, x))) for x in c[:3])
        if len(c) == 1:
            v = float(max(0, min(1, c[0])))
            return (v, v, v)
    return None


def _dominant_line_color(line: LTTextLineHorizontal) -> tuple | None:
    """Return the most common LTChar fill color on this line (approx)."""
    counts: dict[tuple, int] = {}
    for ch in getattr(line, "_objs", []) or []:
        if isinstance(ch, LTChar):
            rgb = _color_to_rgb(
                getattr(getattr(ch, "graphicstate", None), "ncolor", None)
            )
            if rgb is None:
                continue
            # bucket by rounded color to reduce noise
            key = tuple(round(c * 20) / 20.0 for c in rgb)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def parse_pdf_with_highlight_roles(path: Path) -> dict:
    """Parse a PDF by inferring user/assistant turns from alternated highlights.

    Returns a dict compatible with the pipeline: {messages: [...], notes: str}.
    Raises RuntimeError if highlights are not detectable or signals are weak.
    """

    lines: List[LineSeg] = []
    any_rects = False
    for page_num, page in enumerate(extract_pages(str(path)), start=1):
        if not isinstance(page, LTPage):
            continue
        rects = _rects_on_page(page)
        any_rects = any_rects or bool(rects)
        tl = _lines_on_page(page)
        for ln in tl:
            text = ln.get_text().strip("\n")
            if not text:
                continue
            hl = _is_line_highlighted(ln, rects)
            color = _dominant_line_color(ln)
            lines.append(
                LineSeg(
                    page=page_num,
                    x0=ln.x0,
                    y0=ln.y0,
                    x1=ln.x1,
                    y1=ln.y1,
                    text=text,
                    highlighted=hl,
                    color=color,
                )
            )

    if not lines:
        raise RuntimeError("no usable lines detected")
    if not any_rects:
        raise RuntimeError("no highlight rectangles detected")

    # Sort: top-to-bottom within pages, then by x
    lines.sort(key=lambda l: (l.page, -l.y1, l.x0))

    # Group consecutive lines by highlight flag: highlighted -> user, unhighlighted -> assistant.
    messages: List[dict] = []
    cur_role: str | None = None
    cur_parts: List[str] = []
    chat_date: str | None = None

    def _maybe_extract_chat_date(text: str) -> str | None:
        """Return ISO-like chat date string if text matches a CHAT DATE header."""
        prefix = "CHAT DATE"
        if not text.upper().startswith(prefix):
            return None
        remainder = text[len(prefix) :].strip()
        # Expect a short date such as 04/02/25 or 4/2/25
        for fmt in ("%m/%d/%y", "%m/%d/%Y"):
            try:
                dt = datetime.strptime(remainder, fmt)
            except ValueError:
                continue
            # Normalize to date-only ISO string; downstream can add timezone if needed.
            return dt.date().isoformat()
        return None

    for ln in lines:
        text = ln.text.strip()
        if not text:
            continue
        if chat_date is None:
            maybe_date = _maybe_extract_chat_date(text)
            if maybe_date is not None:
                chat_date = maybe_date
                # Do not treat the header as a conversational turn.
                continue
        role = "user" if ln.highlighted else "assistant"
        if cur_role is None:
            cur_role = role
            cur_parts = [text]
            continue
        if role == cur_role:
            cur_parts.append(text)
        else:
            if cur_parts:
                messages.append({"role": cur_role, "content": "\n".join(cur_parts)})
            cur_role = role
            cur_parts = [text]

    if cur_role is not None and cur_parts:
        messages.append({"role": cur_role, "content": "\n".join(cur_parts)})

    # Require at least one user and one assistant segment; otherwise fall back.
    roles_present = {m["role"] for m in messages}
    if "user" not in roles_present or "assistant" not in roles_present:
        raise RuntimeError("highlight roles not separable into user/assistant")

    # Merge any accidental empty or whitespace-only messages (should be rare).
    messages = [
        m
        for m in messages
        if isinstance(m.get("content"), str) and m["content"].strip()
    ]
    if len(messages) < 2:
        raise RuntimeError("insufficient messages after highlight grouping")

    notes = (
        "Parsed by highlight-fallback: roles inferred from highlighted vs. "
        "non-highlighted line groups (highlighted=user)."
    )
    result: dict = {"messages": messages, "notes": notes}
    if chat_date is not None:
        result["chat_date"] = chat_date
    return result
