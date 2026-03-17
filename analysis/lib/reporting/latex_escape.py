"""Shared LaTeX escaping helpers backed by ``pylatexenc``.

This module provides a single helper function that scripts can use
to escape text for inclusion in LaTeX documents.
"""

from __future__ import annotations

from typing import Optional

from pylatexenc.latexencode import unicode_to_latex


def escape_latex(text: Optional[str]) -> str:
    """Escape text for LaTeX using ``pylatexenc``.

    Parameters
    ----------
    text:
        Input text to be escaped. ``None`` is treated as an empty string.

    Returns
    -------
    str
        LaTeX-escaped text suitable for general table and text usage.
    """
    if text is None:
        return ""
    return unicode_to_latex(str(text))
