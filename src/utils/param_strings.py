"""Helpers for encoding and decoding small parameter dictionaries.

These utilities convert between dictionaries and filesystem-safe
``key=value`` strings that use ``&`` as a separator. They are used by
multiple tools (including the analysis viewer) and are intentionally
kept free of any LLM client or annotation dependencies so they can be
imported in lightweight environments.
"""

from __future__ import annotations

from typing import Any, Dict
from urllib.parse import quote, unquote


def escape_string(value: str) -> str:
    """Return a filesystem-safe escaped representation of ``value``.

    Parameters
    ----------
    value: str
        String to escape for safe filesystem usage.

    Returns
    -------
    str
        Escaped variant of ``value`` that does not introduce reserved
        ``&`` or ``=`` characters before decoding.
    """

    if not isinstance(value, str):
        raise TypeError("escape_string expects a string value.")
    return quote(value, safe="-_.")


def unescape_string(value: str) -> str:
    """Decode strings produced by :func:`escape_string`.

    Parameters
    ----------
    value: str
        Escaped string produced by :func:`escape_string`.

    Returns
    -------
    str
        Original unescaped string.
    """

    if not isinstance(value, str):
        raise TypeError("unescape_string expects a string value.")
    return unquote(value)


def dict_to_string(d: Dict[str, Any]) -> str:
    """Return a compact ``key=value`` ``&``-joined representation of ``d``.

    The keys are sorted and values are escaped so the output is safe for
    filesystem usage. Only integers, booleans, strings, and ``None`` are
    permitted. Strings must not contain ``&`` or ``=`` before escaping.

    Parameters
    ----------
    d: Dict[str, Any]
        Dictionary to convert into a string representation.

    Returns
    -------
    str
        Escaped string representing ``d``.

    Raises
    ------
    TypeError
        If a value is not an int, bool, str, or ``None``.
    ValueError
        If a key or value contains ``&`` or ``=`` before escaping.
    """

    for key, value in d.items():
        if not isinstance(value, (int, bool, str, type(None))):
            raise TypeError(f"Unsupported type for key '{key}': {type(value).__name__}")
        key_str = str(key)
        value_str = str(value)
        if "&" in value_str or "=" in value_str or "&" in key_str or "=" in key_str:
            raise ValueError("Cannot pass in values or keys with = or &")

    sorted_items = sorted(d.items(), key=lambda item: str(item[0]))
    kv_pairs = [
        f"{str(key)}={escape_string(str(value)) if value is not None else 'None'}"
        for key, value in sorted_items
    ]
    return "&".join(kv_pairs)


def string_to_dict(s: str) -> Dict[str, Any]:
    """Convert a compact ``key=value`` string back into a dictionary.

    The input must be in the format produced by :func:`dict_to_string`.
    Integer, boolean, string, and ``None`` value types are restored.

    Parameters
    ----------
    s: str
        Encoded key-value string.

    Returns
    -------
    Dict[str, Any]
        Reconstructed dictionary with typed values.
    """

    if not s:
        return {}

    def convert_value(value: str) -> Any:
        unescaped = unescape_string(value)
        lowered = unescaped.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if unescaped == "None":
            return None
        if unescaped.isdigit() or (
            unescaped.startswith("-") and unescaped[1:].isdigit()
        ):
            return int(unescaped)
        return unescaped

    result: Dict[str, Any] = {}
    for pair in s.split("&"):
        if not pair:
            continue
        key, value = pair.split("=", 1)
        result[key] = convert_value(value)
    return result
