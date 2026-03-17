"""Compute participant ordering categories and transcript-level statistics.

This script performs two related tasks that are used by both static analysis
and interactive dashboards:

- It scans parsed transcript JSON files under a transcripts root directory
  and assigns each participant to an ordering category describing how well
  their messages can be placed on a global timeline (for example, fully
  timestamped versus only ordered within conversations). The result is
  written to a JSON file that downstream tools should read instead of
  recomputing the classification.

- It computes high-level descriptive statistics per participant that do not
  depend on annotation outputs (for example, number of conversations, total
  messages, and average message lengths). These are written to a CSV table
  for inclusion in descriptive summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from llm_delusions_annotations.chat import parse_date_label, resolve_bucket_and_rel_path
from llm_delusions_annotations.chat.chat_utils import iter_loaded_chats

from annotation.io import ParticipantOrderingInfo, resolve_ordering_or_unknown
from utils.cli import add_transcripts_root_argument
from utils.demographics import AGE_KEY, GENDER_KEY, bin_age
from utils.io import write_dicts_to_csv
from utils.participants import is_excluded_participant, normalize_participant_id


@dataclass
class ParticipantTranscriptStats:
    """Aggregated transcript statistics for a single participant.

    Parameters
    ----------
    participant:
        Canonical participant identifier (for example, ``"205"``).
    num_conversations:
        Total number of conversations found in the transcripts root.
    total_messages:
        Total number of user and assistant messages across conversations.
    total_user_messages:
        Number of user messages across all conversations.
    total_user_words:
        Sum of whitespace-delimited word counts for all user messages.
    total_assistant_words:
        Sum of whitespace-delimited word counts for all assistant messages.
    conversation_lengths:
        List of per-conversation message counts in processing order.
    earliest_timestamp:
        Earliest message- or conversation-level timestamp when available.
    latest_timestamp:
        Latest message- or conversation-level timestamp when available.
    gender:
        Optional self-reported gender from survey data when available.
    age:
        Optional self-reported age (in years) from survey data when available.
    age_bin:
        Optional coarse age bin label (for example, ``"18-29"`` or ``"30-39"``)
        derived from the ``age`` field.
    """

    participant: str
    num_conversations: int = 0
    total_messages: int = 0
    total_user_messages: int = 0
    total_user_words: int = 0
    total_assistant_words: int = 0
    conversation_lengths: List[int] = field(default_factory=list)
    earliest_timestamp: Optional[datetime] = None
    latest_timestamp: Optional[datetime] = None
    transcript_files: set[Path] = field(default_factory=set)
    transcript_file_types: set[str] = field(default_factory=set)
    models: Dict[str, int] = field(default_factory=dict)
    gender: Optional[str] = None
    age: Optional[int] = None
    age_bin: Optional[str] = None


def _should_include_transcript(bucket: str, file_path: Path) -> bool:
    """Return True when a transcript file should be included in stats.

    This helper centralizes small, participant-specific overrides for which
    transcript files participate in ordering and descriptive statistics.

    Manual adjustments:

    - ``205`` (legacy ``hl_05``) is restricted to the longitudinal JLI
      conversation captured in ``JLI July 26-27 Chat.pdf.json`` and ignores
      other ad hoc files under the same bucket.

    Parameters
    ----------
    bucket:
        Participant identifier inferred from the transcripts root.
    file_path:
        Path to a JSON transcript file under the transcripts root.

    Returns
    -------
    bool
        ``True`` when the transcript should contribute to ordering and
        statistics; ``False`` when it should be ignored.
    """

    participant = normalize_participant_id(bucket)
    if participant == "205":
        return file_path.name == "JLI July 26-27 Chat.pdf.json"
    return True


def _load_demographics_from_surveys(
    surveys_dir: Path,
) -> Dict[str, Dict[str, Optional[object]]]:
    """Load per-participant demographics from survey JSON files.

    This helper expects one JSON file per participant under ``surveys_dir``,
    named ``<participant>.json`` (for example, ``105.json``). It reads
    self-reported age and gender using the same keys as the standalone
    demographics script:

    - ``\"What is your age?\"``
    - ``\"What is your gender? - Selected Choice\"``

    Parameters
    ----------
    surveys_dir:
        Directory containing survey JSON files.

    Returns
    -------
    dict
        Mapping from participant id to a dictionary with optional ``age``
        and ``gender`` entries.
    """

    if not surveys_dir.exists() or not surveys_dir.is_dir():
        return {}

    demographics: Dict[str, Dict[str, Optional[object]]] = {}

    for json_path in sorted(surveys_dir.glob("*.json")):
        participant_id = normalize_participant_id(json_path.stem)
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        raw_age = data.get(AGE_KEY, "")
        age_value: Optional[int]
        if str(raw_age).strip():
            try:
                age_value = int(raw_age)
            except ValueError:
                print(
                    "Warning: invalid age value "
                    f"'{raw_age}' in {json_path}, ignoring.",
                )
                age_value = None
        else:
            age_value = None

        raw_gender = data.get(GENDER_KEY, "")
        gender_value: Optional[str] = (
            str(raw_gender).strip() if str(raw_gender).strip() else None
        )
        if gender_value is None:
            print(f"Warning: missing gender in {json_path}")

        demographics[participant_id] = {
            "age": age_value,
            "gender": gender_value,
        }

    return demographics


def _attach_demographics_from_surveys(
    stats_by_participant: Dict[str, ParticipantTranscriptStats],
    surveys_dir: Optional[Path],
) -> None:
    """Augment participant transcript stats with demographics from surveys.

    When ``surveys_dir`` is provided and contains survey JSON files, this
    helper looks for participant-matching files and adds ``gender``, ``age``,
    and coarse ``age_bin`` labels to the in-memory statistics. Participants
    without matching survey entries are left unchanged.
    """

    if surveys_dir is None:
        return

    resolved_dir = surveys_dir.expanduser().resolve()
    demographics_by_participant = _load_demographics_from_surveys(resolved_dir)
    if not demographics_by_participant:
        return

    for participant, stats in stats_by_participant.items():
        demo = demographics_by_participant.get(participant)
        if demo is None:
            continue

        age_value = demo.get("age")
        gender_value = demo.get("gender")

        stats.age = int(age_value) if isinstance(age_value, int) else None
        stats.gender = str(gender_value) if isinstance(gender_value, str) else None
        stats.age_bin = bin_age(stats.age)


def _compute_transcript_stats(
    transcripts_root: Path,
) -> Dict[str, ParticipantTranscriptStats]:
    """Return transcript-level statistics keyed by participant id."""

    resolved_root = transcripts_root.expanduser().resolve()
    stats_by_participant: Dict[str, ParticipantTranscriptStats] = {}

    if not resolved_root.exists() or not resolved_root.is_dir():
        return stats_by_participant

    for file_path, chats in iter_loaded_chats(resolved_root):
        bucket, _rel_path = resolve_bucket_and_rel_path(file_path, resolved_root)
        if not bucket:
            continue
        participant = normalize_participant_id(bucket)
        if is_excluded_participant(participant):
            continue
        if not _should_include_transcript(participant, file_path):
            continue

        # Compute file-level metadata once per transcript file.
        suffixes = file_path.suffixes
        if len(suffixes) >= 2:
            original_ext = suffixes[-2]
            file_type = original_ext.lstrip(".").lower()
        else:
            file_type = "json"

        for chat in chats:
            participant_stats = stats_by_participant.get(participant)
            if participant_stats is None:
                participant_stats = ParticipantTranscriptStats(participant=participant)
                stats_by_participant[participant] = participant_stats

            participant_stats.transcript_files.add(file_path)
            participant_stats.transcript_file_types.add(file_type)

            # Collect conversation-level information. Chats produced by
            # iter_loaded_chats/load_chats_for_file follow the same linearized,
            # visible user/assistant message selection used elsewhere in the
            # pipeline (for example, when building per-message tables and
            # plots), so counts and lengths here are directly comparable to
            # those other analyses.
            messages = [
                msg
                for msg in chat.messages
                if isinstance(msg, dict)
                and str(msg.get("role", "")).strip().lower() in ("user", "assistant")
            ]
            conv_length = len(messages)
            participant_stats.num_conversations += 1
            participant_stats.total_messages += conv_length
            participant_stats.conversation_lengths.append(conv_length)

            for msg in messages:
                role = str(msg.get("role", "")).strip().lower()
                content_raw = msg.get("content", "")
                content = (
                    content_raw if isinstance(content_raw, str) else str(content_raw)
                )
                length_words = len(content.split())
                if role == "user":
                    participant_stats.total_user_messages += 1
                    participant_stats.total_user_words += length_words
                elif role == "assistant":
                    participant_stats.total_assistant_words += length_words

                    # Count model slugs for assistant turns when available.
                    model_slug = msg.get("model_slug")
                    if isinstance(model_slug, str):
                        slug = model_slug.strip()
                        if slug:
                            participant_stats.models[slug] = (
                                participant_stats.models.get(slug, 0) + 1
                            )

                ts_label = msg.get("timestamp")
                ts = parse_date_label(str(ts_label)) if ts_label is not None else None
                if ts is not None:
                    if (
                        participant_stats.earliest_timestamp is None
                        or ts < participant_stats.earliest_timestamp
                    ):
                        participant_stats.earliest_timestamp = ts
                    if (
                        participant_stats.latest_timestamp is None
                        or ts > participant_stats.latest_timestamp
                    ):
                        participant_stats.latest_timestamp = ts

            # Fall back to conversation-level date when no message timestamps exist.
            if participant_stats.earliest_timestamp is None or (
                participant_stats.latest_timestamp is None
            ):
                conv_ts = parse_date_label(chat.date_label)
                if conv_ts is not None:
                    if (
                        participant_stats.earliest_timestamp is None
                        or conv_ts < participant_stats.earliest_timestamp
                    ):
                        participant_stats.earliest_timestamp = conv_ts
                    if (
                        participant_stats.latest_timestamp is None
                        or conv_ts > participant_stats.latest_timestamp
                    ):
                        participant_stats.latest_timestamp = conv_ts

    return stats_by_participant


def _compute_ordering_from_transcripts(
    stats_by_participant: Dict[str, ParticipantTranscriptStats],
) -> Dict[str, ParticipantOrderingInfo]:
    """Return participant ordering categories derived from transcript stats.

    This helper inspects per-participant transcript statistics to assign an
    ordering type describing how well each participant's messages can be
    placed on a global timeline. For transcript-derived metadata we apply a
    simplified, file-structure-aware version of the general ordering rules:

    - FULL_DATED: at least one usable timestamp or conversation date is
      available for the participant.
    - GLOBAL_ORDER: no timestamps or dates are available, but all transcripts
      for the participant are contained within a single JSON file so the on-
      disk sequence provides a canonical global ordering.
    - CONVERSATION_ONLY: multiple transcript files exist without any usable
      dates, so ordering is treated as guaranteed only within conversations,
      not across the participant's full history.
    - UNKNOWN: no conversations or messages were observed for the participant.
    """

    results: Dict[str, ParticipantOrderingInfo] = {}

    for participant, stats in stats_by_participant.items():
        num_files = len(stats.transcript_files)
        total_messages = stats.total_messages
        total_conversations = stats.num_conversations

        has_any_dates = stats.earliest_timestamp is not None
        has_indices = total_messages > 0
        effective_has_indices = bool(has_indices and num_files == 1)
        has_any_activity = not (
            total_messages == 0 or total_conversations == 0 or num_files == 0
        )

        ordering_type = resolve_ordering_or_unknown(
            has_any_activity=has_any_activity,
            has_any_dates=has_any_dates,
            total_messages=total_messages,
            total_conversations=total_conversations,
            has_indices=effective_has_indices,
        )

        results[participant] = ParticipantOrderingInfo(
            participant=participant,
            ordering_type=ordering_type,
            # For transcript-derived ordering we only know whether any usable
            # date-like information exists; we expose this symmetrically via
            # both flags for downstream consumers.
            has_timestamps=has_any_dates,
            has_conversation_dates=has_any_dates,
            has_message_indices=has_indices,
        )

    return results


def _write_ordering_json(
    output_path: Path, ordering_info: Dict[str, ParticipantOrderingInfo]
) -> None:
    """Write participant ordering metadata to ``output_path`` as JSON."""

    serializable: Dict[str, Dict[str, object]] = {}
    for participant, info in ordering_info.items():
        serializable[participant] = {
            "ordering_type": str(info.ordering_type.value),
            "has_timestamps": bool(info.has_timestamps),
            "has_conversation_dates": bool(info.has_conversation_dates),
            "has_message_indices": bool(info.has_message_indices),
        }

    resolved = output_path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2, sort_keys=True)


def _write_transcript_stats_csv(
    output_path: Path, stats_by_participant: Dict[str, ParticipantTranscriptStats]
) -> None:
    """Write participant transcript statistics to ``output_path`` as CSV."""

    rows: List[Dict[str, object]] = []

    # Build per-participant rows.
    for participant, stats in stats_by_participant.items():
        num_conversations = stats.num_conversations
        total_messages = stats.total_messages
        total_words = stats.total_user_words + stats.total_assistant_words
        conv_lengths = stats.conversation_lengths

        max_conv_length = max(conv_lengths) if conv_lengths else 0
        if conv_lengths and max_conv_length > 0:
            longest_index = conv_lengths.index(max_conv_length)
            longest_position_fraction = float(longest_index + 1) / float(
                len(conv_lengths)
            )
        else:
            longest_index = -1
            longest_position_fraction = 0.0

        # Per-participant median conversation length.
        if conv_lengths:
            sorted_lengths = sorted(conv_lengths)
            mid = len(sorted_lengths) // 2
            if len(sorted_lengths) % 2 == 1:
                median_conv_length = float(sorted_lengths[mid])
            else:
                median_conv_length = (
                    float(sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2.0
                )
        else:
            median_conv_length = 0.0

        # Round floating-point summaries to a small, readable number of
        # decimal places so that LaTeX tables remain compact.
        median_conv_length = round(median_conv_length, 2)
        longest_position_rate = round(longest_position_fraction, 3)

        file_types = (
            ";".join(sorted(stats.transcript_file_types))
            if stats.transcript_file_types
            else ""
        )

        # Order models by frequency (most to least) and compute the rate of
        # the most frequent model among all model_slug occurrences in the
        # range [0.0, 1.0]. The CSV only records the single most frequent
        # model for each participant.
        if stats.models:
            sorted_models = sorted(
                stats.models.items(), key=lambda item: item[1], reverse=True
            )
            top_model = sorted_models[0][0]
            models_str = top_model
            total_model_events = sum(count for _, count in sorted_models)
            if total_model_events > 0:
                top_model_share = round(
                    float(sorted_models[0][1]) / float(total_model_events),
                    3,
                )
            else:
                top_model_share = 0.0
        else:
            models_str = ""
            top_model_share = ""

        row: Dict[str, object] = {
            "participant": participant,
            "gender": stats.gender or "",
            "age_bin": stats.age_bin or "",
            "num_conversations": num_conversations,
            "total_messages": total_messages,
            "total_words": total_words,
            "max_conversation_length": max_conv_length,
            "median_conversation_length": median_conv_length,
            "longest_conversation_rate": longest_position_rate,
            "files": file_types,
            "top_model": models_str,
            "top_model_rate": top_model_share,
            "span_days": (
                (stats.latest_timestamp - stats.earliest_timestamp).days
                if stats.earliest_timestamp is not None
                and stats.latest_timestamp is not None
                else ""
            ),
        }
        rows.append(row)

    # Sort rows from most to least messages for easier scanning.
    rows.sort(key=lambda item: int(item.get("total_messages", 0)), reverse=True)

    # Append a totals row aggregating counts and global summaries.
    total_num_conversations = sum(
        stats.num_conversations for stats in stats_by_participant.values()
    )
    total_messages_all = sum(
        stats.total_messages for stats in stats_by_participant.values()
    )
    total_words_all = sum(
        stats.total_user_words + stats.total_assistant_words
        for stats in stats_by_participant.values()
    )

    # Global median conversation length across all conversations.
    all_conv_lengths: List[int] = []
    for stats in stats_by_participant.values():
        all_conv_lengths.extend(stats.conversation_lengths)
    if all_conv_lengths:
        sorted_all_lengths = sorted(all_conv_lengths)
        mid_all = len(sorted_all_lengths) // 2
        if len(sorted_all_lengths) % 2 == 1:
            median_conv_length_all = float(sorted_all_lengths[mid_all])
        else:
            median_conv_length_all = (
                float(sorted_all_lengths[mid_all - 1] + sorted_all_lengths[mid_all])
                / 2.0
            )
        median_conv_length_all = round(float(median_conv_length_all), 2)
    else:
        median_conv_length_all = ""
    max_conv_length_all = max(
        (
            max(stats.conversation_lengths)
            for stats in stats_by_participant.values()
            if stats.conversation_lengths
        ),
        default=0,
    )

    # Union of all model slugs across participants for the totals row. As
    # with per-participant rows, record only the single most frequent model
    # and its rate in the range [0.0, 1.0].
    all_model_counts: Dict[str, int] = {}
    for stats in stats_by_participant.values():
        for model_slug, count in stats.models.items():
            all_model_counts[model_slug] = all_model_counts.get(model_slug, 0) + count
    if all_model_counts:
        sorted_all_models = sorted(
            all_model_counts.items(), key=lambda item: item[1], reverse=True
        )
        total_top_model = sorted_all_models[0][0]
        total_models_str = total_top_model
        total_model_events = sum(count for _, count in sorted_all_models)
        if total_model_events > 0:
            top_model_share_all = round(
                float(sorted_all_models[0][1]) / float(total_model_events),
                3,
            )
        else:
            top_model_share_all = 0.0
    else:
        total_models_str = ""
        top_model_share_all = ""

    totals_row: Dict[str, object] = {
        "participant": "TOTAL",
        "gender": "",
        "age_bin": "",
        "num_conversations": total_num_conversations,
        "total_messages": total_messages_all,
        "total_words": total_words_all,
        "max_conversation_length": max_conv_length_all,
        "median_conversation_length": median_conv_length_all,
        "longest_conversation_rate": "",
        "files": "",
        "top_model": total_models_str,
        "top_model_rate": top_model_share_all,
        "span_days": "",
    }
    rows.append(totals_row)

    fieldnames = [
        "participant",
        "gender",
        "age_bin",
        "num_conversations",
        "total_messages",
        "total_words",
        "max_conversation_length",
        "median_conversation_length",
        "longest_conversation_rate",
        "files",
        "top_model",
        "top_model_rate",
        "span_days",
    ]
    write_dicts_to_csv(output_path, fieldnames, rows)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Return parsed command-line arguments for the metadata script.

    Parameters
    ----------
    argv:
        Optional argument vector. When omitted, ``sys.argv`` is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including the transcripts root plus output paths
        for ordering and stats artefacts.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute per-participant ordering categories and transcript-level "
            "descriptive statistics from chat JSON exports."
        ),
    )
    add_transcripts_root_argument(
        parser,
        help_text=(
            "Root directory containing parsed chat JSON transcripts "
            "(default: transcripts_de_ided)."
        ),
    )
    parser.add_argument(
        "--ordering-json",
        type=Path,
        default=Path("analysis") / "participant_ordering.json",
        help=(
            "Output JSON path for participant ordering metadata "
            "(default: analysis/participant_ordering.json)."
        ),
    )
    parser.add_argument(
        "--stats-csv",
        type=Path,
        default=Path("analysis") / "data" / "participant_transcript_stats.csv",
        help=(
            "Output CSV path for participant transcript statistics "
            "(default: analysis/data/participant_transcript_stats.csv)."
        ),
    )
    parser.add_argument(
        "--surveys-dir",
        type=Path,
        default=Path("surveys"),
        help=(
            "Optional directory containing per-participant survey JSON files "
            "named <participant>.json (for example, 105.json). When "
            "provided, the participant transcript statistics CSV will also "
            "include gender, age, and binned age columns."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Script entry point for computing ordering and transcript statistics.

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

    transcripts_root: Path = args.transcripts_root

    transcript_stats = _compute_transcript_stats(transcripts_root)
    if not transcript_stats:
        print(
            "No transcripts found under "
            f"{transcripts_root.expanduser().resolve()}, skipping outputs."
        )
    else:
        _attach_demographics_from_surveys(transcript_stats, args.surveys_dir)

        ordering_info = _compute_ordering_from_transcripts(transcript_stats)
        _write_ordering_json(args.ordering_json, ordering_info)
        print(
            "Wrote participant ordering metadata for "
            f"{len(ordering_info)} participants to {args.ordering_json}"
        )

        # Always write transcript statistics to the requested path.
        _write_transcript_stats_csv(args.stats_csv, transcript_stats)
        print(
            "Wrote transcript statistics for "
            f"{len(transcript_stats)} participants to {args.stats_csv}"
        )

        # Print a lightweight summary to stdout with one row per participant
        # containing their ordering category and the number of original
        # non-JSON source files they have in the transcripts root. For
        # example, *.docx.json transcripts are grouped by their *.docx
        # originals and JSON-only transcripts are ignored.
        print(
            "participant,ordering_type,num_original_non_json_files",
        )
        for participant in sorted(transcript_stats.keys()):
            info = ordering_info.get(participant)
            ordering_type = (
                str(info.ordering_type.value) if info is not None else "UNKNOWN"
            )
            original_non_json_files = set()
            for path in transcript_stats[participant].transcript_files:
                suffixes = path.suffixes
                if len(suffixes) >= 2:
                    original_ext = suffixes[-2].lower()
                elif suffixes:
                    original_ext = suffixes[-1].lower()
                else:
                    original_ext = ""
                # Only count files whose original extension was not JSON,
                # grouping all derived artefacts (for example, *.docx.json)
                # by their non-JSON source stem.
                if original_ext != ".json":
                    original_non_json_files.add(path.with_suffix(""))
            print(
                f"{participant},{ordering_type},{len(original_non_json_files)}",
            )

        # Also mirror the transcript statistics into the canonical
        # analysis/data path so that downstream analyses that rely on that
        # location continue to see up-to-date data, even when the caller
        # overrides --stats-csv.
        default_stats_path = (
            Path("analysis") / "data" / "participant_transcript_stats.csv"
        )
        if (
            args.stats_csv.expanduser().resolve()
            != default_stats_path.expanduser().resolve()
        ):
            _write_transcript_stats_csv(default_stats_path, transcript_stats)
            print("Also wrote transcript statistics to " f"{default_stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
