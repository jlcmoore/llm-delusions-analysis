"""Parser dispatch utilities for chat log text sources."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .parser_chatgpt_html import parse as _parse_chatgpt_html
from .parser_chatgpt_md import ParseFailed
from .parser_chatgpt_md import parse as _parse_chatgpt_md
from .parser_label_colon import parse as _parse_label_colon

Parsed = Dict[str, Any]


def try_parse_any(text: str, _source_hint: str, *, strict: bool) -> Parsed:
    """Attempt to parse text using a suitable parser given the source hint.

    Heuristics:
    - If the text looks like a ChatGPT HTML export (contains `var jsonData =`),
      use the HTML parser.
    - Otherwise, fall back to the markdown parser tolerant of headings/bold.
    """

    if "jsonData" in text and "<html" in text.lower():
        return _parse_chatgpt_html(text, strict=strict)
    return _parse_chatgpt_md(text, strict=strict)


def parse_with_labels(text: str, *, labels: Iterable[str], strict: bool) -> Parsed:
    """Parse using inline label prefixes like "User: content".

    Delegates to the label-colon parser. Raises ParseFailed on failure.
    """
    return _parse_label_colon(text, labels=labels, strict=strict)
