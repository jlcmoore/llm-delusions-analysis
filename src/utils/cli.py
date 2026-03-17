"""CLI helper utilities for shared argparse patterns.

This module centralizes common command-line argument definitions used across
multiple scripts so that:

- Repeated argument groups (model selection, participants, score cutoffs)
  remain consistent across tools.
- Pylint duplicate-code warnings are reduced by keeping the shared logic in
  one place.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import threading
import time
from pathlib import Path
from typing import Mapping, Optional

from llm_delusions_annotations.annotation_prompts import (
    ANNOTATIONS_FILE,
    add_preceding_context_argument,
)
from llm_delusions_annotations.cutoffs import CUTOFFS_FILE

from llm_delusions_data import get_dataset_relpath
from utils.io import get_default_transcripts_root
from utils.utils import extract_non_default_arguments


def add_model_argument(parser: argparse.ArgumentParser, *, default_model: str) -> None:
    """Add a ``--model/-m`` argument understood by LiteLLM-style tools.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_model:
        Default model name to advertise in the help text.
    """

    parser.add_argument(
        "--model",
        "-m",
        default=default_model,
        help=("Model name understood by LiteLLM " f"(default: {default_model})."),
    )


def add_subset_input_argument(
    parser: argparse.ArgumentParser,
    *,
    flag: str,
    default_input_dir: Path | str,
) -> None:
    """Add a shared ``-i`` subset input argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    flag:
        Long-form flag name, for example ``\"--input\"`` or ``\"--input-file\"``.
    default_input_dir:
        Default input path.
    """

    parser.add_argument(
        flag,
        "-i",
        type=Path,
        default=Path(default_input_dir),
        help=(
            "Directory containing subset JSON files " f"(default: {default_input_dir})."
        ),
    )


def add_chat_io_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path | str,
    output_help: str,
) -> None:
    """Add shared ``--input`` and ``--output`` chat directory arguments.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_output_dir:
        Default directory where annotation outputs will be written.
    output_help:
        Help text describing the output directory semantics.
    """

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory containing chat JSON exports.",
    )

    parser.add_argument(
        "--output",
        "-o",
        dest="output_dir",
        default=default_output_dir,
        help=output_help,
    )


def add_participants_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add a shared ``--participant/-p`` argument for participant filters.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Optional custom help string. When omitted, a generic description is
        used that applies to most participant-filtering scripts.
    """

    parser.add_argument(
        "--participant",
        "-p",
        action="append",
        dest="participants",
        help=help_text
        or (
            "Restrict processing to these participant ids (repeatable). "
            "Defaults to all participants when omitted."
        ),
    )


def add_randomize_per_ppt_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--randomize-per-ppt`` participant allocation argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    """

    parser.add_argument(
        "--randomize-per-ppt",
        choices=["proportional", "equal"],
        default="proportional",
        help=(
            "When using --randomize, control how sampled items are "
            "distributed across participants (ppts): 'proportional' samples "
            "in proportion to each participant's available items; 'equal' "
            "aims for an even per-participant allocation."
        ),
    )


def add_chat_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    max_messages_help: str,
) -> None:
    """Add shared chat sampling and ordering arguments.

    Parameters
    ----------
    parser:
        Target argument parser.
    max_messages_help:
        Help text describing the ``--max-messages`` cap semantics for the
        calling script.
    """

    parser.add_argument(
        "--max-messages",
        type=int,
        default=0,
        help=max_messages_help,
    )

    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomly sample messages when used with --max-messages.",
    )

    add_randomize_per_ppt_argument(parser)

    parser.add_argument(
        "--randomize-conversations",
        action="store_true",
        help=(
            "Randomly sample entire conversations within each participant (ppt) "
            "when used with --max-messages."
        ),
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=0,
        help=(
            "Limit processing to the first N conversations per participant (ppt). "
            "Set to 0 to process all conversations."
        ),
    )
    parser.add_argument(
        "--reverse-conversations",
        action="store_true",
        help=(
            "Process conversations in reverse message order (latest turns first). "
            "Prefiltering will then consider the last turns."
        ),
    )
    add_preceding_context_argument(parser)


def add_annotations_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """Add a shared ``--annotation/-a`` filter argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Help string describing how the annotation filter is applied.
    """

    parser.add_argument(
        "--annotation",
        "-a",
        action="append",
        dest="annotations",
        help=help_text,
    )


def add_score_cutoff_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add a shared global score-cutoff argument for positive thresholds.

    Parameters
    ----------
    parser:
        Target argument parser.
    """

    parser.add_argument(
        "--llm-score-cutoff",
        dest="score_cutoff",
        type=int,
        default=None,
        help=help_text
        or (
            "Optional minimum numeric score required for a record to count as "
            "positive. When omitted, any record with one or more matches is "
            "treated as positive."
        ),
    )


def add_follow_links_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--follow-links`` argument for chat scanners."""

    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links while scanning for chat files.",
    )


def add_preprocessed_input_csv_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = (
        "Preprocessed per-message annotation table (Parquet) produced by "
        "preprocess_annotation_family.py."
    ),
) -> None:
    """Add a shared positional ``input_csv`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Help string describing the expected CSV, typically pointing to the
        output of ``analysis/preprocess_annotation_family.py``.
    """

    parser.add_argument(
        "input_csv",
        type=Path,
        help=help_text,
    )

    parser.add_argument(
        "--use-human-validation",
        action="store_true",
        help=(
            "If set, read human_validated__* columns from the input parquet and "
            "use them to override LLM scores where present."
        ),
    )


def add_transcripts_parquet_argument(
    parser: argparse.ArgumentParser,
    *,
    default_path: Path = get_dataset_relpath("transcripts"),
    help_text: str = (
        "Full transcripts Parquet table with message content "
        "(default: transcripts_data/transcripts.parquet)."
    ),
) -> None:
    """Add a shared ``--transcripts-parquet`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_path:
        Default location of the transcripts Parquet file.
    help_text:
        Help string describing the transcripts table and its default path.
    """

    parser.add_argument(
        "--transcripts-parquet",
        type=Path,
        default=default_path,
        help=help_text,
    )


def add_annotations_parquet_argument(
    parser: argparse.ArgumentParser,
    *,
    default_path: Path = get_dataset_relpath("annotations_preprocessed"),
    help_text: str = (
        "Per-message annotations table in Parquet format "
        "(default: annotations/all_annotations__preprocessed.parquet)."
    ),
) -> None:
    """Add a shared ``--annotations-parquet`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_path:
        Default location of the preprocessed annotations Parquet file.
    help_text:
        Help string describing the annotations table and its default path.
    """

    parser.add_argument(
        "--annotations-parquet",
        type=Path,
        default=default_path,
        help=help_text,
    )


def add_optional_llm_cutoffs_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str,
) -> None:
    """Add an optional ``--llm-cutoffs-json`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Help string describing how the per-annotation LLM score cutoffs are
        used by the calling script.
    """

    parser.add_argument(
        "--llm-cutoffs-json",
        type=str,
        default=CUTOFFS_FILE,
        help=help_text,
    )


def add_transcripts_root_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = (
        "Root directory containing de-identified transcript JSON files "
        "(default: transcripts_de_ided)."
    ),
) -> None:
    """Add a shared ``--transcripts-root`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Help string describing the transcripts root directory. The default
        targets de-identified transcript JSON files.
    """

    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=get_default_transcripts_root(),
        help=help_text,
    )


class Spinner:
    """Lightweight terminal spinner for long-running operations.

    The spinner writes a single animated line to stderr when attached to a
    TTY. It is a no-op when stderr is not a TTY (for example, during tests
    or when output is redirected).
    """

    def __init__(self, message: str, interval: float = 0.2) -> None:
        """Initialise a spinner with a message and refresh interval."""

        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        """Start animating the spinner if stderr is a TTY."""

        if not sys.stderr.isatty():
            return self

        def _run() -> None:
            cycle = itertools.cycle("|/-\\")
            while not self._stop_event.is_set():
                frame = next(cycle)
                sys.stderr.write(f"\r{self.message} {frame}")
                sys.stderr.flush()
                time.sleep(self.interval)

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Stop the spinner and clear the line."""

        if not sys.stderr.isatty():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        clear_line = "\r" + " " * (len(self.message) + 2) + "\r"
        sys.stderr.write(clear_line)
        sys.stderr.flush()


def add_output_path_argument(
    parser: argparse.ArgumentParser,
    *,
    default_path: Path | str,
    help_text: str,
) -> None:
    """Add a shared ``--output/-o`` path argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_path:
        Default file or directory path for the output.
        help_text:
        Help string describing the output target.
    """

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(default_path),
        help=help_text,
    )


def add_artifacts_dir_argument(
    parser: argparse.ArgumentParser,
    *,
    default_dir: Path | str,
    help_text: str,
) -> None:
    """Add a shared ``--artifacts-dir`` directory argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    default_dir:
        Default directory path for artifacts written by analysis scripts.
    help_text:
        Help string describing the artifacts directory.
    """

    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(default_dir),
        help=help_text,
    )


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--log-level`` argument for logging verbosity.

    Parameters
    ----------
    parser:
        Target argument parser.
    """

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )


def add_annotation_outputs_arguments(
    parser: argparse.ArgumentParser,
    *,
    file_help: str,
    outputs_root_default: Path | str = Path("annotation_outputs"),
) -> None:
    """Add shared ``file`` and ``--outputs-root`` arguments for annotation jobs.

    Parameters
    ----------
    parser:
        Target argument parser.
    file_help:
        Help text describing the required JSONL reference file.
    outputs_root_default:
        Default root directory containing annotation outputs.
    """

    parser.add_argument(
        "file",
        type=Path,
        help=file_help,
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path(outputs_root_default),
        help=(
            "Root directory containing annotation outputs "
            f"(default: {outputs_root_default})."
        ),
    )


def add_annotation_metadata_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared LLM cutoff and annotation-metadata arguments.

    These arguments are used by analysis scripts that consume annotation
    outputs and need both per-annotation LLM score cutoffs and metadata from
    ``src/llm_delusions_annotations/data/annotations.csv``. Callers must provide either
    ``--llm-cutoffs-json`` or ``--llm-score-cutoff``; when both are given,
    per-annotation cutoffs from the CSV file take precedence and the global
    cutoff may be used as a fallback for any missing annotation ids.
    """

    parser.add_argument(
        "--llm-cutoffs-json",
        type=str,
        default=CUTOFFS_FILE,
        help=(
            "Path to a CSV file containing per-annotation LLM score cutoffs "
            "with columns annotation_id and cutoff."
        ),
    )
    parser.add_argument(
        "--llm-score-cutoff",
        type=int,
        help=(
            "Global minimum LLM score applied to all annotations when "
            "per-annotation cutoffs are not available."
        ),
    )
    add_annotations_csv_argument(parser)


def add_annotations_csv_argument(parser: argparse.ArgumentParser) -> None:
    """Add a shared ``--annotations-csv`` argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    """

    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=ANNOTATIONS_FILE,
        help="CSV file containing annotation metadata (id, scope, category).",
    )


def add_classify_chats_family_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_metadata: bool,
) -> None:
    """Add shared family-IO arguments for classify_chats-derived jobs.

    This helper is used by analysis scripts that operate on JSONL outputs
    produced by ``scripts/annotation/classify_chats.py``. It standardises the
    required ``file`` positional argument and ``--outputs-root`` while
    optionally attaching annotation-metadata arguments for tools that need
    score cutoffs and scopes.
    """

    add_annotation_outputs_arguments(
        parser,
        file_help=(
            "Path to a single JSONL output file produced by classify_chats.py. "
            "All sibling files with the same basename under the outputs root "
            "will be included in the aggregation."
        ),
    )
    if include_metadata:
        add_annotation_metadata_arguments(parser)


def add_annotation_id_argument(
    parser: argparse.ArgumentParser,
    *,
    help_text: Optional[str] = None,
) -> None:
    """Add a shared ``--annotation-id`` filter argument.

    Parameters
    ----------
    parser:
        Target argument parser.
    help_text:
        Optional custom help string.
    """

    parser.add_argument(
        "--annotation-id",
        "--label",
        "-l",
        action="append",
        dest="annotation_ids",
        help=help_text
        or (
            "Restrict processing to specific annotation ids or labels "
            "(repeatable). Defaults to all when omitted."
        ),
    )


def extract_non_default_arguments_with_model(
    args: argparse.Namespace,
) -> Mapping[str, object]:
    """Return a mapping of non-default CLI arguments for manifest recording.

    This helper extracts arguments that differ from their default values and
    ensures that the ``model`` argument is always included in the result.

    Parameters
    ----------
    args:
        Parsed CLI arguments, which must have a ``_defaults`` attribute
        containing the original default values.

    Returns
    -------
    Mapping[str, object]
        Mapping from argument name to normalized non-default value.
    """

    defaults = getattr(args, "_defaults", {})
    non_default_arguments = extract_non_default_arguments(args, defaults)
    non_default_arguments = dict(non_default_arguments)
    non_default_arguments["model"] = getattr(args, "model", None)
    return non_default_arguments


__all__ = [
    "add_model_argument",
    "extract_non_default_arguments_with_model",
    "add_subset_input_argument",
    "add_chat_io_arguments",
    "add_participants_argument",
    "add_randomize_per_ppt_argument",
    "add_chat_sampling_arguments",
    "add_score_cutoff_argument",
    "add_output_path_argument",
    "add_log_level_argument",
    "add_annotation_outputs_arguments",
    "add_annotation_metadata_arguments",
    "add_annotations_csv_argument",
    "add_classify_chats_family_arguments",
    "add_follow_links_argument",
    "add_annotations_argument",
    "add_annotation_id_argument",
    "add_optional_llm_cutoffs_argument",
    "add_preprocessed_input_csv_argument",
    "add_transcripts_root_argument",
    "add_transcripts_parquet_argument",
    "add_annotations_parquet_argument",
    "Spinner",
]
