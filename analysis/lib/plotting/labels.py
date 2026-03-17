"""Label-formatting helpers shared across analysis scripts.

This module centralizes simple label transformations that are reused in
multiple analysis and plotting scripts, such as shortening long annotation
identifiers for axis tick labels.
"""

from __future__ import annotations

from typing import List, Sequence

from llm_delusions_annotations.annotation_ids import ROLE_SPLIT_BASE_IDS


def shorten_annotation_label(label: str) -> str:
    """Return a shortened annotation label for plotting.

    Parameters
    ----------
    label:
        Full annotation identifier, typically prefixed with ``assistant-``
        or ``user-``.

    Returns
    -------
    str
    Shortened identifier suitable for compact axis labels. Special
    cases include:

        * ``theme-awakening-consciousness`` -> ``bot-metaphysical-themes``.
        * ``grand-significance`` -> ``bot-grand-significance``.
        * Substrings ``intent`` -> ``thoughts``.
        * Role-prefixed ids are kept as-is (``bot-``/``user-``).
    """

    # Map variants of the awakening/consciousness theme to compact
    # metaphysical-themes labels while preserving role when available. This
    # captures ids such as ``theme-awakening-consciousness``,
    # ``assistant-theme-awakening-consciousness``, and
    # ``user-theme-awakening-consciousness``.
    if "theme-awakening-consciousness" in label:
        if label.startswith("user-"):
            return "user-metaphysical-themes"
        return "bot-metaphysical-themes"

    if label == "grand-significance":
        label = "bot-grand-significance"

    if "intent" in label:
        label = label.replace("intent", "thoughts")

    return label


def filter_annotation_ids_for_display(
    annotation_ids: Sequence[str],
) -> List[str]:
    """Return annotation ids filtered for plotting and tables.

    This helper centralises small, presentation-oriented decisions about
    which annotation identifiers should be hidden when both canonical and
    role-split variants are present. In particular, when any of
    :data:`ROLE_SPLIT_BASE_IDS` (for example, ``platonic-affinity``) have
    corresponding ``user-<id>`` or ``assistant-<id>`` variants in
    ``annotation_ids``, the base id is omitted from the returned list so
    that plots and tables only show the role-specific series.
    """

    present = {annotation_id for annotation_id in annotation_ids if annotation_id}
    hidden_bases: set[str] = set()

    for base_id in ROLE_SPLIT_BASE_IDS:
        if base_id not in present:
            continue
        has_role_split = any(f"{role}-{base_id}" in present for role in ("user", "bot"))
        if has_role_split:
            hidden_bases.add(base_id)

    return [
        annotation_id
        for annotation_id in annotation_ids
        if annotation_id not in hidden_bases
    ]


__all__ = [
    "ROLE_SPLIT_BASE_IDS",
    "filter_annotation_ids_for_display",
    "shorten_annotation_label",
]
