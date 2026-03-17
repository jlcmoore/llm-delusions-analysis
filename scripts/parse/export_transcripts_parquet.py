"""Export flattened transcript messages to Parquet tables.

This script scans parsed chat JSON transcripts under a root directory
(``--transcripts-root``, default ``transcripts_de_ided``), flattens every
message into a row keyed by:

* participant
* source_path (path relative to the transcripts root)
* chat_index
* message_index

For each message it also records lightweight metadata such as role,
timestamp, conversation key, and conversation date. The script then writes
two Parquet files under an output directory:

* ``transcripts_index.parquet`` – metadata-only table without ``content``.
* ``transcripts.parquet`` – full table including the message ``content``.

These tables are intended as a query-friendly backing store for analysis
code and downstream tools so that raw JSON under ``transcripts_de_ided/``
does not need to be reparsed for each experiment.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional

import pandas as pd
from llm_delusions_annotations.chat import (
    iter_loaded_chats,
    parse_date_label,
    resolve_bucket_and_rel_path,
)

from utils.cli import add_transcripts_root_argument
from utils.participants import normalize_participant_id
from utils.transcripts import normalise_message_fields

LOGGER = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the export helper.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser instance with transcripts root and output options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Flatten parsed chat transcripts into Parquet tables with one row "
            "per message."
        )
    )
    add_transcripts_root_argument(
        parser,
        help_text=(
            "Root directory containing parsed chat JSON transcripts "
            "(default: transcripts_de_ided)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("transcripts_data"),
        help=(
            "Directory where transcripts_index.parquet and transcripts.parquet "
            "will be written (default: transcripts_data)."
        ),
    )
    parser.add_argument(
        "--follow-links",
        action="store_true",
        help="Follow symbolic links while scanning transcript JSON files.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity (default: INFO).",
    )
    return parser


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the export script.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments namespace.
    """

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return args


def iter_transcript_rows(
    transcripts_root: Path,
    *,
    followlinks: bool,
) -> Iterator[Dict[str, object]]:
    """Yield flattened transcript message rows from a transcripts root.

    Parameters
    ----------
    transcripts_root:
        Root directory containing parsed chat JSON transcripts.
    followlinks:
        When True, follow symbolic links while scanning ``transcripts_root``.

    Yields
    ------
    Dict[str, object]
        Row dictionaries with key and metadata fields including a ``content``
        string for each message.
    """

    root = transcripts_root.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        LOGGER.error("Transcripts root not found or not a directory: %s", root)
        return

    for file_path, chats in iter_loaded_chats(root, followlinks=followlinks):
        bucket, rel_path = resolve_bucket_and_rel_path(file_path, root)
        if not bucket or rel_path is None:
            continue
        participant = normalize_participant_id(bucket)
        rel_path_str = str(rel_path)
        for chat_index, chat in enumerate(chats):
            for message_index, message in enumerate(chat.messages):
                fields = normalise_message_fields(message)
                if fields is None:
                    continue

                row: Dict[str, object] = {
                    "participant": participant,
                    "source_path": rel_path_str,
                    "chat_index": chat_index,
                    "message_index": message_index,
                    "role": fields["role"],
                    "timestamp": fields["timestamp"],
                    "chat_key": chat.key,
                    "chat_date": chat.date_label,
                    "model_id": fields.get("model_id"),
                    "content": fields["content"],
                }
                yield row


def _materialize_rows(
    rows: Iterable[Mapping[str, object]],
) -> pd.DataFrame:
    """Return a pandas DataFrame constructed from transcript rows.

    Parameters
    ----------
    rows:
        Iterable of row dictionaries produced by :func:`iter_transcript_rows`.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing all provided rows. When no rows are produced,
        an empty DataFrame with no columns is returned.
    """

    data = list(rows)
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data)


def _normalise_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a datetime ``timestamp`` column when present.

    Parameters
    ----------
    df:
        Source DataFrame containing the flattened transcript rows.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` where the ``timestamp`` column entries have been
        parsed into naive UTC datetimes when possible, preserving seconds
        precision for epoch-based labels and richer formats. Rows with
        unparseable or missing timestamps retain ``None`` in the column.
    """

    if "timestamp" not in df.columns:
        return df

    result = df.copy()

    def _parse_timestamp(value: object) -> Optional[object]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return parse_date_label(text)

    result["timestamp"] = result["timestamp"].map(_parse_timestamp)
    return result


def _write_parquet_tables(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write index-only and full-content Parquet tables from a DataFrame.

    Parameters
    ----------
    df:
        DataFrame containing at least the columns emitted by
        :func:`iter_transcript_rows`.
    output_dir:
        Directory where ``transcripts_index.parquet`` and ``transcripts.parquet``
        will be written. The directory is created when it does not exist.
    """

    output_root = output_dir.expanduser().resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        LOGGER.error("Failed to create output directory %s: %s", output_root, err)
        return

    if df.empty:
        LOGGER.warning("No transcript rows discovered; nothing to write.")
        return

    index_cols = [col for col in df.columns if col != "content"]
    df_index = df[index_cols].copy()

    index_path = output_root / "transcripts_index.parquet"
    content_path = output_root / "transcripts.parquet"

    try:
        df_index.to_parquet(index_path)
        LOGGER.info("Wrote transcripts index to %s", index_path)
    except (OSError, ValueError) as err:
        LOGGER.error("Failed to write transcripts index parquet: %s", err)
        return

    try:
        df.to_parquet(content_path)
        LOGGER.info("Wrote transcripts with content to %s", content_path)
    except (OSError, ValueError) as err:
        LOGGER.error("Failed to write transcripts content parquet: %s", err)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Script entry point for exporting transcripts to Parquet.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    int
        Exit code suitable for ``sys.exit``.
    """

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    transcripts_root: Path = args.transcripts_root
    LOGGER.info("Scanning transcripts under %s", transcripts_root)
    df = _materialize_rows(
        iter_transcript_rows(
            transcripts_root=transcripts_root,
            followlinks=bool(args.follow_links),
        )
    )
    if df.empty:
        LOGGER.warning("No transcript messages found; exiting without outputs.")
        return 0

    df = _normalise_timestamp_column(df)
    _write_parquet_tables(df, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
