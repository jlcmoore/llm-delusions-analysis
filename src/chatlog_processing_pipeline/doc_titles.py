"""Segment plain text or .docx into conversations using titles.

This utility scans a block of text for standalone title lines that precede the
first role turn (User/Assistant/ChatGPT/System/You). It uses these titles as
conversation boundaries and parses each segment with the existing content
parsers. Titles are not hardcoded and are taken from the text itself.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Set

from docx import Document  # python-docx

from .parsers import ParseFailed, try_parse_any

ROLE_LABELS = {"user", "assistant", "chatgpt", "system", "you"}


def _is_role_label_with_set(line: str, role_labels: Set[str]) -> bool:
    s = line.strip().lower()
    if s in role_labels:
        return True
    if re.match(r"^(you|chatgpt)\s+said:\s*$", s):
        return True
    if re.match(r"^(user|assistant|chatgpt|system|you)\s*:\s*$", s):
        return True
    return False


def parse_docx_by_fonts(
    path, *, strict: bool = False, role_labels: Optional[Iterable[str]] = None
) -> dict:
    """Parse a .docx by detecting title paragraphs via styles/fonts.

    Heuristics:
    - Paragraphs with style names like 'Title' or 'Heading' are titles.
    - Otherwise, a paragraph is considered a title if its max run font size is
      significantly larger than the median body font size and it is short.
    - Titles mark conversation boundaries; each segment is parsed via content parser.
    """
    doc = Document(str(path))
    paras = list(doc.paragraphs)
    # Collect font sizes across body text to establish a median
    all_sizes: List[float] = []
    for p in paras:
        for run in p.runs:
            if run.font.size:
                try:
                    all_sizes.append(float(run.font.size.pt))
                except (AttributeError, TypeError, ValueError):
                    continue
    body_median = 11.0
    if all_sizes:
        sorted_sizes = sorted(all_sizes)
        body_median = sorted_sizes[len(sorted_sizes) // 2]

    def para_max_size(p) -> float:
        sizes: List[float] = []
        for r in p.runs:
            if r.font.size:
                try:
                    sizes.append(float(r.font.size.pt))
                except (AttributeError, TypeError, ValueError):
                    continue
        return max(sizes) if sizes else 0.0

    # Decide which paragraphs are titles
    title_idxs: List[int] = []
    rl_set = set(ROLE_LABELS)
    if role_labels:
        rl_set.update(s.strip().lower() for s in role_labels if s)
    for i, p in enumerate(paras):
        text = p.text.strip()
        if not text:
            continue
        # Skip obvious role labels
        if _is_role_label_with_set(text, rl_set):
            continue
        style_name = (p.style.name or "").lower() if p.style else ""
        is_heading_style = "heading" in style_name or style_name in {
            "title",
            "heading",
            "heading 1",
            "heading 2",
        }
        size = para_max_size(p)
        looks_big = size and size >= body_median * 1.2
        is_short = len(text) <= 100
        if is_heading_style or (looks_big and is_short):
            # Also require that a role label appears within a few paragraphs ahead to
            # ensure this is a conversation title, not a section header
            found_role_ahead = False
            for j in range(i + 1, min(i + 12, len(paras))):
                t2 = paras[j].text.strip()
                if t2 and _is_role_label_with_set(t2, rl_set):
                    found_role_ahead = True
                    break
            if found_role_ahead:
                title_idxs.append(i)

    if not title_idxs:
        raise ParseFailed("docx: no title paragraphs detected")

    # Build segments from titles
    conversations: List[dict] = []
    for k, ti in enumerate(title_idxs):
        title_text = paras[ti].text.strip()
        end = title_idxs[k + 1] if k + 1 < len(title_idxs) else len(paras)
        body_lines: List[str] = []
        for p in paras[ti + 1 : end]:
            t = p.text
            if t is not None:
                body_lines.append(t)
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        try:
            parsed = try_parse_any(body, "unknown", strict=strict)
        except ParseFailed:
            conversations.append(
                {
                    "title": title_text,
                    "messages": [{"role": "assistant", "content": body}],
                }
            )
            continue
        if "messages" in parsed and isinstance(parsed["messages"], list):
            conversations.append({"title": title_text, "messages": parsed["messages"]})
        elif "conversations" in parsed and isinstance(parsed["conversations"], list):
            convs = parsed["conversations"]
            if convs:
                if isinstance(convs[0], dict):
                    convs[0]["title"] = title_text
            conversations.extend(convs)

    if not conversations:
        raise ParseFailed("docx: title segmentation produced no conversations")

    return {"conversations": conversations, "notes": "docx_titles_fonts"}
