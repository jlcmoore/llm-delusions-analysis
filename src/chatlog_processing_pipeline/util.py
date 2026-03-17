"""Utility helpers for filesystem, JSON serialization, and parsed outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from llm_delusions_annotations.chat import load_chats_for_file


def ensure_dir(p: Path) -> None:
    """Create directory `p` and all parents if they do not exist."""

    p.mkdir(parents=True, exist_ok=True)


def _sanitize(obj):
    """Recursively coerce strings to valid UTF-8 for safe JSON writing."""

    if isinstance(obj, str):
        # Replace invalid surrogates with U+FFFD to keep JSON valid
        return obj.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    return obj


def write_json(path: Path, obj) -> None:
    """Write an object as pretty-printed UTF-8 JSON after sanitizing strings."""

    ensure_dir(path.parent)
    clean = _sanitize(obj)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)


def normalize_meta_dict(meta: Any) -> Dict[str, Any]:
    """Return a canonical ``meta`` mapping from a ParseMeta-like object.

    Parameters:
    - meta: Object with ``src_path``, ``rel_path``, ``file_ext``,
      ``source_guess``, and ``message_count`` attributes.

    Returns:
    - Dictionary suitable for JSON serialization with keys:
      ``filename``, ``rel_path``, ``full_path``, ``file_ext``,
      ``source_guess``, and ``message_count``.
    """

    src = Path(meta.src_path)
    return {
        "filename": src.name,
        "rel_path": meta.rel_path,
        "full_path": str(src.resolve()),
        "file_ext": meta.file_ext,
        "source_guess": meta.source_guess,
        "message_count": meta.message_count,
    }


def is_probably_text_by_content(path: Path, sample_size: int = 4096) -> bool:
    """Heuristically decide if a file is likely text by sampling bytes."""

    data = path.read_bytes()[:sample_size]
    if not data:
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def write_parsed_output(out_root: Path, meta, out: Dict[str, Any]) -> Path:
    """Write a parsed conversation payload under ``out_root``.

    Parameters:
    - out_root: Root directory for parsed JSON outputs.
    - meta: Parse metadata object with ``src_path``, ``rel_path``, ``file_ext``,
      ``source_guess``, and ``message_count`` attributes.
    - out: Parsed payload containing either ``messages`` or ``conversations``
      and optional ``notes``.

    Returns:
    - Path to the JSON file that was written.
    """

    base = out_root / meta.rel_path
    out_path = base.with_name(base.name + ".json")
    payload: Dict[str, Any] = {
        "meta": normalize_meta_dict(meta),
        "notes": out.get("notes", ""),
    }
    if isinstance(out, dict) and "conversations" in out:
        payload["conversations"] = out["conversations"]
    else:
        payload["messages"] = out.get("messages", [])
    write_json(out_path, payload)
    return out_path


def find_json_snippet(json_path: Path) -> Optional[str]:
    """Extract a short representative snippet from a chat JSON file."""

    chats = load_chats_for_file(json_path)
    for chat_obj in chats:
        if isinstance(chat_obj.key, str) and len(chat_obj.key.strip()) >= 5:
            return chat_obj.key.strip()[:80]
        for msg in chat_obj.messages:
            content = msg.get("content")
            if isinstance(content, str):
                txt = content.strip()
                if len(txt) >= 20:
                    return txt[:120]
    return None


def html_contains_snippet(html_path: Path, snippet: str) -> bool:
    """Return True if ``snippet`` appears in ``html_path`` (case-insensitive)."""

    try:
        text = html_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return snippet.lower() in text.lower()


def looks_like_parsed_chat_json(obj: Any) -> bool:
    """Return True if ``obj`` matches the parsed chat JSON schema.

    Supported shapes:
    - Single: {"messages": [ {"role": str, "content": str}, ... ], ...}
    - Multi:  {"conversations": [ {"messages": [...]}, ... ], ...}
    """
    if not isinstance(obj, dict):
        return False
    if isinstance(obj.get("messages"), list):
        for message in obj["messages"]:
            if isinstance(message, dict) and "role" in message and "content" in message:
                return True
    if isinstance(obj.get("conversations"), list):
        for conv in obj["conversations"]:
            messages = conv.get("messages") if isinstance(conv, dict) else None
            if isinstance(messages, list):
                for message in messages:
                    if (
                        isinstance(message, dict)
                        and "role" in message
                        and "content" in message
                    ):
                        return True
    return False


def iter_chatgpt_messages(conversations):
    """Yield message dictionaries from ChatGPT mapping-style conversations."""
    if not isinstance(conversations, list):
        return
    for conv in conversations:
        if not isinstance(conv, dict):
            continue
        mapping = conv.get("mapping")
        if not isinstance(mapping, dict):
            continue
        for node in mapping.values():
            if not isinstance(node, dict):
                continue
            message = node.get("message")
            if isinstance(message, dict):
                yield message


def looks_like_chatgpt_mapping(obj: Any) -> bool:
    """Detect ChatGPT conversation exports with mapping-based structure."""
    if not isinstance(obj, (dict, list)):
        return False
    conversations = obj.get("conversations") if isinstance(obj, dict) else obj
    if not isinstance(conversations, list):
        return False
    for message in iter_chatgpt_messages(conversations):
        content = message.get("content")
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if isinstance(parts, list):
            return True
    return False
