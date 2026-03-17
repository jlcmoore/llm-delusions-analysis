"""
Format parsed chat logs into Google-Docs-friendly documents, one output file per
input JSON. The script walks the entire input directory tree, detects JSON files
that match the expected chat schema, and writes a corresponding output file in a
mirrored directory tree under the output directory.

Input format (produced by chatlog_processing_pipeline):
  {
    "meta": { ... },
    "messages": [ {"role": "user"|"assistant", "content": str}, ... ],
    "notes": str
  }

Output:
  - HTML (default): Each chat becomes a standalone HTML document suitable for
    uploading to Google Drive and opening with Google Docs. Roles are styled and
    messages are clearly separated. A small footer marks the end of each chat.
  - TXT: Plain text with clear separators.

Usage:
  python scripts/parse/format_chats_html.py INPUT_DIR OUTPUT_DIR \
      --format html --include-notes --title-prefix "Chat"

Notes:
  - This script assumes Python 3.11 and a virtual environment.
  - The script avoids broad exception catches and only handles specific errors.
  - All code and generated markup use ASCII-only syntax, while preserving any
    Unicode content found in input messages.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import List

from llm_delusions_annotations.chat import (
    Chat,
    iter_chat_json_files,
    load_chats_for_file,
    load_chats_from_directory,
    parse_date_label,
)

_DEFAULT_GDOCS_CSS = (
    "body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, "
    "Helvetica, Arial, sans-serif; line-height: 1.45; color: #111; }\n"
)

# -------------------------- Data structures ---------------------------


@lru_cache(maxsize=1)
def _load_order_toggle_js() -> str:
    """Load and cache the JavaScript used for toggling chat order."""

    js_path = Path(__file__).with_name("gdocs_order_toggle.js")
    try:
        return js_path.read_text(encoding="utf-8")
    except OSError as err:
        sys.stderr.write(
            "[WARN] Failed to read order toggle script " f"{js_path}: {err}\n"
        )
        return ""


@lru_cache(maxsize=1)
def _load_gdocs_css() -> str:
    """Load and cache the shared CSS used for Google Docs exports."""

    css_path = Path(__file__).with_name("gdocs_styles.css")
    try:
        return css_path.read_text(encoding="utf-8")
    except OSError as err:
        sys.stderr.write(
            "[WARN] Failed to read gdocs style sheet " f"{css_path}: {err}\n"
        )
        return _DEFAULT_GDOCS_CSS


# --------------------------- Render: HTML ------------------------------


def _escape_text_to_html(text: str) -> str:
    """Escape text for safe HTML rendering and preserve newlines as <br>.

    Args:
        text: Raw text to escape.

    Returns:
        Escaped HTML string with newline breaks.
    """

    return html.escape(text).replace("\n", "<br>\n")


def _render_role_filter_controls() -> str:
    """Return HTML controls for toggling message visibility by role."""

    role_rows: List[str] = []
    for role_slug, label in (
        ("user", "User"),
        ("assistant", "Assistant"),
        ("tool", "Tool"),
    ):
        role_rows.append(
            '<div class="role-filter-row">'
            + '<span class="role-filter-role">'
            + label
            + "</span>"
            + '<label><input type="radio" name="role-'
            + role_slug
            + '" value="show" checked> Show</label>'
            + '<label><input type="radio" name="role-'
            + role_slug
            + '" value="hide"> Hide</label>'
            + "</div>"
        )
    return (
        '<fieldset class="role-filter-group">'
        '<legend class="role-filter-legend">Roles</legend>'
        + "".join(role_rows)
        + "</fieldset>\n"
    )


def _format_message_count_label(count: int) -> str:
    """Return a short label for a message count.

    Args:
        count: Total number of messages.

    Returns:
        Label such as "1 message" or "3 messages".
    """

    noun = "msg" if count == 1 else "msgs"
    return f"{count} {noun}"


def _sort_chats_by_date(chats: List[Chat]) -> List[Chat]:
    """Return chats sorted by parsed date (newest first).

    If no chats have a usable date label, preserve the original ordering
    provided by the loader instead of falling back to alphabetical keys.
    """

    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

    # Detect whether all chats have a parsed date. If any are missing dates,
    # keep the input order to avoid reordering based on titles or keys.
    has_all_dates = all(parse_date_label(c.date_label) is not None for c in chats)
    if not has_all_dates:
        return list(chats)

    def sort_key(chat: Chat) -> tuple[bool, float, str]:
        dt_val = parse_date_label(chat.date_label)
        if dt_val is None:
            return (False, epoch.timestamp(), chat.key.lower())
        timestamp = dt_val.timestamp()
        return (True, timestamp, chat.key.lower())

    return sorted(chats, key=sort_key, reverse=True)


def _render_message_html(idx: int, role: str, content: str) -> str:
    """Render a single message block in HTML.

    Args:
        idx: Message index (0-based).
        role: Message role (user, assistant, or other).
        content: Message content.

    Returns:
        HTML snippet for the message.
    """

    normalized_role = (role or "").strip().lower()
    role_label = {
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }.get(normalized_role, role.title() if role else "Unknown")
    role_class = {
        "user": "msg-user",
        "assistant": "msg-assistant",
        "tool": "msg-tool",
    }.get(normalized_role, "msg-other")
    role_slug = re.sub(r"[^a-z0-9]+", "-", normalized_role) or "unknown"
    body = _escape_text_to_html(content)
    return (
        '<div class="message '
        + role_class
        + " msg-role-"
        + role_slug
        + '" data-role="'
        + role_slug
        + '" data-manual-state="visible">\n      <div class="meta">'
        + str(idx + 1)
        + ". "
        + role_label
        + '</div>\n      <div class="message-controls">'
        + '<button type="button" class="message-toggle" data-state="visible">'
        + "Hide message"
        + "</button>"
        + '<span class="filter-indicator" hidden></span>'
        + '</div>\n      <div class="content">'
        + body
        + "</div>\n    </div>\n"
    )


# ------------------------ Render: Aggregate ---------------------------


def render_html_aggregate(chats: List[Chat], title: str, include_notes: bool) -> str:
    """Render many chats in a single HTML document with a TOC and headings.

    Args:
        chats: List of Chat objects.
        title: Document title for the aggregate export.
        include_notes: Whether to include parser notes under each chat header.

    Returns:
        A complete HTML document string containing all chats.
    """

    css = _load_gdocs_css()
    style_block = "<style>\n" + css + "\n</style>\n" if css else ""
    head = (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        '<head>\n  <meta charset="utf-8">\n  '
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n  '
        + "<title>"
        + html.escape(title)
        + "</title>\n"
        + style_block
        + "</head>\n<body>\n"
    )

    header = (
        '<div class="doc-title">'
        + html.escape(title)
        + "</div>\n"
        + '<div class="subtitle">'
        + "Each chat is a section with a page break following it."
        + "</div>\n"
    )

    ordered_chats = _sort_chats_by_date(chats)

    has_multiple = len(ordered_chats) > 1
    if has_multiple:
        order_button = (
            '<button id="order-toggle" type="button" class="order-toggle">'
            "Show Oldest First"
            "</button>"
        )
    else:
        order_button = ""
    expand_all_button = (
        '<button id="expand-collapse-all" type="button" class="order-toggle">'
        "Expand All Chats"
        "</button>"
    )
    order_note_text = (
        "Chat ## labels follow the current newest-first ordering. "
        "Use the controls above each chat to change order or navigate quickly."
    )
    order_note = '<div class="order-note">' + order_note_text + "</div>"
    role_filter_controls = _render_role_filter_controls()
    order_controls = (
        '<div class="order-controls">'
        + order_button
        + expand_all_button
        + role_filter_controls
        + order_note
        + "</div>\n"
    )

    toc_items: List[str] = []
    for i, c in enumerate(ordered_chats):
        anchor = "chat-" + str(i + 1)
        label = str(i + 1).zfill(2)
        msg_label = _format_message_count_label(len(c.messages))
        toc_items.append(
            '<li class="toc-item" data-index="'
            + str(i)
            + '"><a href="#'
            + anchor
            + '"><span class="chat-label">Chat '
            + label
            + "</span> ("
            + msg_label
            + "): "
            + html.escape(c.key)
            + "</a></li>"
        )
    toc_html = "\n".join(toc_items)

    parts: List[str] = [
        head,
        header,
        order_controls,
        (
            '<div class="toc"><h2>Contents</h2><ol id="toc-list">'
            + toc_html
            + "</ol></div>\n"
        ),
        '<div id="chat-container">\n',
    ]

    for i, c in enumerate(ordered_chats):
        anchor = "chat-" + str(i + 1)
        label = str(i + 1).zfill(2)
        msg_label = _format_message_count_label(len(c.messages))
        h = (
            '<h2 id="'
            + anchor
            + '" class="chat-heading" tabindex="-1" data-title="'
            + html.escape(c.key)
            + '"><span class="chat-label">Chat '
            + label
            + "</span> ("
            + msg_label
            + "): "
            + html.escape(c.key)
            + "</h2>\n"
        )
        d = (
            '<div class="meta-line">Date: ' + html.escape(c.date_label) + "</div>\n"
            if c.date_label
            else ""
        )
        n = (
            '<div class="notes">Notes: ' + html.escape(c.notes) + "</div>\n"
            if (include_notes and c.notes)
            else ""
        )
        msgs = "\n".join(
            _render_message_html(mi, m.get("role", ""), m.get("content", ""))
            for mi, m in enumerate(c.messages)
        )
        sep = '<hr class="chat-sep">\n<div class="page-break"></div>\n'
        prev_disabled = ' disabled="disabled"' if i == 0 else ""
        next_disabled = ' disabled="disabled"' if i == len(ordered_chats) - 1 else ""
        actions = (
            '<div class="chat-actions">'
            '<button type="button" class="chat-nav chat-nav-prev" data-dir="-1"'
            + prev_disabled
            + ">Previous</button>"
            '<button type="button" class="chat-toggle" data-state="expanded">'
            "Hide Chat"
            "</button>"
            '<button type="button" class="chat-nav chat-nav-next" data-dir="1"'
            + next_disabled
            + ">Next</button>"
            "</div>\n"
        )
        body = (
            '<div class="chat-body" data-collapsed="false">'
            + d
            + n
            + msgs
            + sep
            + "</div>\n"
        )
        parts.append(
            '<section class="chat" data-index="'
            + str(i)
            + '">'
            + h
            + actions
            + body
            + "</section>\n"
        )

    parts.append("</div>\n")
    parts.append(
        '<button type="button" id="scroll-to-chat-start" '
        'class="floating-scroll-button" hidden>Jump to chat start</button>\n'
    )
    js_code = _load_order_toggle_js().rstrip("\n")
    if js_code:
        parts.append("<script>\n")
        parts.append(js_code)
        parts.append("\n</script>\n")

    parts.append("</body></html>\n")
    return "".join(parts)


def render_txt_aggregate(chats: List[Chat], include_notes: bool) -> str:
    """Render many chats in a single plain text document.

    Args:
        chats: List of Chat objects.
        include_notes: Whether to include parser notes under each chat header.

    Returns:
        A plain text document string containing all chats.
    """

    sorted_chats = _sort_chats_by_date(chats)

    out: List[str] = []
    for i, c in enumerate(sorted_chats):
        out.append("===== Chat " + str(i + 1) + ": " + c.key + " =====")
        if c.date_label:
            out.append("[date] " + c.date_label)
        if include_notes and c.notes:
            out.append("[notes] " + c.notes)
        for mi, m in enumerate(c.messages):
            out.append("")
            role = m.get("role", "").upper() or "UNKNOWN"
            out.append("(" + str(mi + 1) + ") " + role + ":")
            out.append(m.get("content", ""))
        out.append("")
        out.append("----- END OF CHAT -----")
        out.append("")
    return "\n".join(out)


# ------------------------------- Main ---------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the exporter.

    Args:
        argv: Optional list of arguments to parse.

    Returns:
        Parsed arguments namespace.
    """

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", help="Directory containing parsed chat JSONs")
    parser.add_argument(
        "output",
        help="Output directory (per-file mode) or output file (aggregate mode)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["html", "txt"],
        default="html",
        help="Output format",
    )
    parser.add_argument(
        "--mode",
        choices=["per-file", "aggregate"],
        default="per-file",
        help="Export mode: per-file (default) or single aggregate file",
    )
    parser.add_argument(
        "--include-notes",
        action="store_true",
        help="Include parser notes in output",
    )
    parser.add_argument(
        "--title-prefix",
        default="Chat",
        help="Title prefix for HTML documents",
    )
    parser.add_argument(
        "--max-chats",
        type=int,
        default=None,
        help="Limit number of chats processed",
    )
    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symlinked directories when scanning input",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    """Program entry: export chats one-to-one from input to output tree.

    The function validates the input directory, creates the output directory,
    walks all JSON files, and writes one output file per input chat JSON while
    mirroring the directory structure.

    Args:
        argv: Optional list of arguments (useful for testing).

    Returns:
        Exit code: 0 on success; non-zero on errors.
    """

    args = parse_args(argv)
    in_dir = Path(args.input).expanduser().resolve()
    if not in_dir.exists() or not in_dir.is_dir():
        sys.stderr.write(f"Input directory not found: {in_dir}\n")
        return 2

    if args.mode == "aggregate":
        chats = load_chats_from_directory(
            in_dir, followlinks=args.follow_links, limit=args.max_chats
        )
        if not chats:
            sys.stderr.write("No chats found (no .json files with messages).\n")
            return 1
        out_file = Path(args.output).expanduser().resolve()
        try:
            out_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            sys.stderr.write(f"Failed to create directory {out_file.parent}: {err}\n")
            return 2
        if args.format == "html":
            agg = render_html_aggregate(chats, args.title_prefix, args.include_notes)
        else:
            agg = render_txt_aggregate(chats, args.include_notes)
        try:
            out_file.write_text(agg, encoding="utf-8")
        except OSError as err:
            sys.stderr.write(f"Failed to write output file {out_file}: {err}\n")
            return 2
        print("Wrote aggregate:", str(out_file))
        return 0

    out_dir = Path(args.output).expanduser().resolve()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        sys.stderr.write(f"Failed to create output directory {out_dir}: {err}\n")
        return 2

    files = list(iter_chat_json_files(in_dir, followlinks=args.follow_links))
    written = 0
    for fp in files:
        rel = fp.relative_to(in_dir)
        chats_for_file = load_chats_for_file(fp)
        if not chats_for_file:
            continue
        out_rel = rel.with_suffix(".html" if args.format == "html" else ".txt")
        out_path = out_dir / out_rel
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            sys.stderr.write(
                f"[WARN] Could not create directory {out_path.parent}: {err}\n"
            )
            continue

        if args.format == "html":
            # Combine all conversations in this file into one doc with sections
            title = f"{args.title_prefix}: {rel}"
            doc = render_html_aggregate(chats_for_file, title, args.include_notes)
        else:
            doc = render_txt_aggregate(chats_for_file, args.include_notes)
        try:
            out_path.write_text(doc, encoding="utf-8")
            written += 1
        except OSError as err:
            sys.stderr.write(f"[WARN] Failed to write {out_path}: {err}\n")

    print(f"Processed files: {written} (from {len(files)} JSON files)")
    print(f"Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
