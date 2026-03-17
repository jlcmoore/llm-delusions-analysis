"""Utility helper package for shared script and analysis helpers.

Submodules
----------
cli
    Shared argparse configuration helpers for CLI scripts.
io
    Shared JSONL and transcript I/O helpers used by scripts and analysis
    modules.
participants
    Participant-level utilities shared across scripts and analyses.
transcripts
    Normalization helpers for transcript message dictionaries.
"""

from __future__ import annotations

__all__ = [
    "cli",
    "io",
    "participants",
    "transcripts",
]
