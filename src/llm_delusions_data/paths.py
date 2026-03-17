"""Path utilities for shared dataset assets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llm_delusions_data.datasets import get_dataset


def resolve_root(root: Optional[Path | str]) -> Path:
    """Return a resolved root directory for dataset paths.

    Parameters
    ----------
    root:
        Optional root override. When ``None``, the current working
        directory is used.

    Returns
    -------
    pathlib.Path
        Resolved root directory.
    """

    if root is None:
        return Path.cwd()
    return Path(root).expanduser()


def get_path(
    name: str, *, root: Optional[Path | str] = None, must_exist: bool = True
) -> Path:
    """Return the resolved path for a dataset.

    Parameters
    ----------
    name:
        Dataset identifier.
    root:
        Optional root override. When ``None``, the current working directory
        is used.
    must_exist:
        When True, raise ``FileNotFoundError`` if the resolved path is missing.

    Returns
    -------
    pathlib.Path
        Resolved dataset path.
    """

    dataset = get_dataset(name)
    resolved_root = resolve_root(root)
    path = resolved_root / dataset.relpath
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Dataset {name!r} not found at {path}.")
    return path
