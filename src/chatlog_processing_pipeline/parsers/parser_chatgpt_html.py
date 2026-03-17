"""Pass-through parser for ChatGPT HTML exports embedding `jsonData`.

The HTML export includes a JavaScript variable `jsonData` that contains an
array of conversation objects. This parser now extracts that payload and
returns it unchanged under a top-level `conversations` key, behaving like a
JSON pass-through. Older behavior that attempted to linearize messages is no
longer supported.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any, Dict

from .parser_chatgpt_md import ParseFailed

Parsed = Dict[str, Any]


def _extract_json_data_blob(html_text: str) -> list[dict] | None:
    """Extract `jsonData` from a ChatGPT HTML export robustly.

    Supports formats like:
      - var jsonData = [ ... ];
      - const jsonData = [ ... ];
      - window.jsonData = [ ... ];
      - jsonData = JSON.parse("...");
    """

    # Case 1: jsonData assigned via JSON.parse("...") with an escaped JSON string
    m = re.search(
        r"\bjsonData\s*=\s*JSON\.parse\(\s*([\'\"])((?:.|\n)*?)\1\s*\)", html_text, re.S
    )
    if m:
        raw = m.group(2)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Case 2: jsonData assigned to an array literal; find balanced brackets safely
    m = re.search(r"\bjsonData\s*=\s*\[", html_text)
    if not m:
        return None

    start = m.end() - 1  # points to the '['
    i = start
    n = len(html_text)
    depth_bracket = 0
    depth_brace = 0
    in_string = False
    str_q = ""
    esc = False
    while i < n:
        ch = html_text[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == str_q:
                in_string = False
        else:
            if ch in ('"', "'"):
                in_string = True
                str_q = ch
            elif ch == "[":
                depth_bracket += 1
            elif ch == "]":
                depth_bracket -= 1
                if depth_bracket == 0 and depth_brace == 0:
                    blob = html_text[start : i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        return None
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                depth_brace -= 1
        i += 1

    return None


def parse(text: str, *, strict: bool = True) -> Parsed:
    """Extract and pass through the embedded `jsonData` array.

    The returned object has a top-level "conversations" key that is exactly the
    extracted array from the HTML. No normalization or linearization is
    performed. The `strict` flag is accepted for signature compatibility but is
    ignored.
    """

    data = _extract_json_data_blob(text)
    if not isinstance(data, list):
        raise ParseFailed("html export: jsonData not found")

    def _unescape_html(value: Any) -> Any:
        """Recursively unescape HTML entities in parsed JSON data."""

        if isinstance(value, str):
            return html.unescape(value)
        if isinstance(value, list):
            return [(_unescape_html(item)) for item in value]
        if isinstance(value, dict):
            return {key: _unescape_html(item) for key, item in value.items()}
        return value

    unescaped = _unescape_html(data)

    return {
        "conversations": unescaped,
        "notes": "chatgpt_html_export_passthrough",
    }
