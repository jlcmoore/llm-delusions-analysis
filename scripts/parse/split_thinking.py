"""
Split thinking traces from assistant responses in transcript JSON files.

This script scans parsed chat JSON files, identifies assistant messages that
contain thinking traces (for example, content after a "Show thinking" cue),
and uses LiteLLM to separate reasoning from the final answer.

For each qualifying assistant message, the script:

- Sends the full assistant message content to a small classification prompt
  instructing the model to return the first N tokens of the visible reply.
- Locates that reply prefix inside the original content and splits the
  message:
  - ``content`` is rewritten to hold the full reply segment.
  - A new ``thought`` field is added containing the preceding thinking text.

By default, files are updated in place. When an output root is provided, the
script mirrors the input directory tree under that location instead.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from llm_delusions_annotations.llm_utils.client import (
    DEFAULT_CHAT_MODEL,
    LLMClientError,
    batch_completion,
)
from tqdm import tqdm

from utils.cli import add_log_level_argument

litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.CRITICAL)
DEFAULT_TIMEOUT = 60
DEFAULT_REPLY_PREFIX_WORDS = 8
DEFAULT_MAX_WORKERS = 64

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitTask:
    """Single assistant message to split into thought and answer."""

    file_path: Path
    message_index: int
    original_content: str
    previous_message: Optional[str] = None


@dataclass(frozen=True)
class SplitOutcome:
    """Result of a single thought/answer split attempt."""

    task: SplitTask
    answer_text: Optional[str]
    error: Optional[str]


def _normalize_with_index(value: str) -> Tuple[str, List[int]]:
    """Return a normalized view of ``value`` plus index mapping.

    The normalized string is case-folded, has all runs of whitespace collapsed
    to single spaces, and maps common Unicode quotes/dashes to ASCII. The
    returned index list maps each character position in the normalized string
    back to the corresponding character index in the original ``value``.
    """

    if not value:
        return "", []

    quote_dash_map = {
        0x2018: "'",
        0x2019: "'",
        0x201C: '"',
        0x201D: '"',
        0x2013: "-",
        0x2014: "-",
    }

    normalized_chars: List[str] = []
    index_map: List[int] = []
    last_was_space = False

    for idx, ch in enumerate(value):
        if ch.isspace():
            if last_was_space:
                continue
            normalized_chars.append(" ")
            index_map.append(idx)
            last_was_space = True
            continue

        replacement = quote_dash_map.get(ord(ch), ch)
        for out_ch in replacement:
            normalized_chars.append(out_ch)
            index_map.append(idx)
        last_was_space = False

    normalized = "".join(normalized_chars).casefold()
    return normalized, index_map


def _build_split_prompt(
    message: str,
    max_reply_tokens: int,
    previous_message: Optional[str] = None,
) -> str:
    """Return a prompt asking the model to extract the non-thinking answer.

    Parameters
    ----------
    message:
        Full assistant message content that may contain a thinking trace as
        well as a final answer.

    Returns
    -------
    str
        Instructional prompt for a completion-style chat model.
    """

    instructions = [
        "You will be given the full text of a single assistant message, which may",
        "contain internal 'thinking' plus the final reply shown to the user.",
        "",
        "Your job:",
        "- Identify the portion of the text that corresponds to the actual reply",
        "  that the user would see.",
        "- Ignore any internal reasoning or analysis that precedes the reply.",
        "- Any section that explains what the user did (for example, lines starting with",
        "  'Show thinking', 'The user provided', 'Constraint Check & Context', or",
        "  'Plan for this Manifestation') is part of the thinking trace and must be ignored.",
        "",
        "Output format (must be exact):",
        f"- Return approximately the first {max_reply_tokens} words",
        "  of the reply portion only. If the reply is shorter, return the full reply.",
        "- The reply you output MUST be copied exactly as a single contiguous substring",
        "  from the assistant message above.",
        "- Do NOT normalize, edit, or rephrase anything:",
        "  - Do not change or fix punctuation, quotes, or dashes.",
        "  - Do not change spacing or line breaks.",
        "  - Do not add or remove words.",
        "- Never include the literal phrase 'Show thinking' in your output.",
        "- Avoid starting the reply with meta-descriptions such as 'The user provided',",
        "  'Constraint Check & Context', or 'Plan for this Manifestation' – these are",
        "  almost always part of the internal thinking, not the visible reply.",
        "- Avoid starting with meta phrases like 'Here is', \"Here's\", 'This is', or 'Below is'.",
        "  Start directly with the first word of the reply as it appears.",
        "- The reply usually begins where the assistant turns to address the user directly,",
        "  often with a change in tone (for example, starting with 'Okay, ...',",
        "  'Acknowledged...', or a direct answer).",
        "- Do not include quotes, explanations, JSON, or any extra commentary.",
        "- If you are unsure, copy a shorter exact substring rather than rewriting.",
        "- If there is no reply portion at all (the entire message is internal thinking),",
        "  output exactly the single token NO_REPLY and nothing else.",
        "",
    ]
    if previous_message:
        instructions.extend(
            [
                "Previous message in the conversation (for context only; do not copy",
                "from this section):",
                "```",
                previous_message,
                "```",
                "",
            ]
        )
    instructions.extend(
        [
            "Assistant message:",
            "```",
            message,
            "```",
        ]
    )
    return "\n".join(instructions)


def _build_messages_for_litellm(prompt: str) -> List[dict[str, str]]:
    """Return LiteLLM-compatible messages for a single request."""

    system = (
        "You are a precise text segmenter. Return only the final reply text "
        "corresponding to what the user would see, with no additional tokens. "
        "The reply must be copied verbatim as a contiguous substring from the "
        "assistant message, without changing punctuation, quotes, spaces, or wording. "
        "Do not include any internal analysis sections such as 'Show thinking', "
        "'The user provided ...', 'Constraint Check & Context', or 'Plan for this "
        "Manifestation' – those are thinking traces, not the visible reply. "
        "Never start with meta phrases like 'Here is', \"Here's\", 'This is', or 'Below is'. "
        "If there is no reply portion at all and the message is only internal thinking, "
        "output exactly NO_REPLY."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]


def _validate_and_split(original: str, answer_prefix: str) -> Optional[Tuple[str, str]]:
    """Return (thought, answer) using ``answer_prefix`` as the reply start.

    The model is instructed to return the first N tokens of the reply. This
    helper finds the earliest occurrence of that prefix in the original
    content and treats everything before as ``thought`` and everything from
    the prefix onward as ``answer``.
    """

    if not original or not answer_prefix:
        return None

    prefix = answer_prefix.strip()
    if not prefix:
        return None

    # First try an exact substring search in the raw text.
    direct_index = original.find(prefix)
    # Require some leading content to treat a match as a true split.
    if direct_index > 0:
        thought_direct = original[:direct_index].rstrip()
        answer_direct = original[direct_index:].lstrip()
        if answer_direct:
            return thought_direct, answer_direct

    # Fallback: perform a whitespace/quote/dash-normalized search while
    # retaining a mapping back to the original string for the split point.
    normalized_original, index_map = _normalize_with_index(original)
    normalized_prefix, _ = _normalize_with_index(prefix)
    if not normalized_original or not normalized_prefix:
        return None

    norm_index = normalized_original.find(normalized_prefix)
    if norm_index == -1 or not index_map:
        return None
    if not 0 <= norm_index < len(index_map):
        return None

    split_index = index_map[norm_index]
    if split_index <= 0:
        return None
    thought = original[:split_index].rstrip()
    answer = original[split_index:].lstrip()
    if not answer:
        return None
    return thought, answer


def run_split_batch(
    tasks: Sequence[SplitTask],
    *,
    model: str,
    timeout: int,
    max_workers: int,
    reply_prefix_words: int,
) -> List[SplitOutcome]:
    """Submit split tasks to LiteLLM using batch_completion and return outcomes."""

    if not tasks:
        return []

    messages_payload: List[List[dict[str, str]]] = []
    for task in tasks:
        prompt = _build_split_prompt(
            task.original_content,
            max_reply_tokens=reply_prefix_words,
            previous_message=task.previous_message,
        )
        messages_payload.append(_build_messages_for_litellm(prompt))

    try:
        response_items = batch_completion(
            messages=messages_payload,
            model=model,
            timeout=timeout,
            max_workers=max_workers,
            enable_reasoning_defaults=True,
        )
    except LLMClientError as err:
        raise RuntimeError(f"LiteLLM batch request failed: {err}") from err
    if len(response_items) != len(tasks):
        raise RuntimeError(
            "LiteLLM returned a different number of responses than requested."
        )

    outcomes: List[SplitOutcome] = []
    for task, response in zip(tasks, response_items):
        if isinstance(response, Mapping):
            choices = response.get("choices")
        else:
            choices = getattr(response, "choices", None)
        if not choices:
            outcomes.append(
                SplitOutcome(
                    task=task,
                    answer_text=None,
                    error="No choices returned from LiteLLM.",
                )
            )
            continue
        first = choices[0]
        if isinstance(first, Mapping):
            msg = first.get("message", {})
        else:
            msg = getattr(first, "message", {})
        if isinstance(msg, Mapping):
            content_raw = msg.get("content")
        else:
            content_raw = getattr(msg, "content", None)
        answer_text = str(content_raw or "").strip()
        if not answer_text:
            outcomes.append(
                SplitOutcome(
                    task=task,
                    answer_text=None,
                    error="Empty content in LiteLLM response.",
                )
            )
            continue
        outcomes.append(
            SplitOutcome(
                task=task,
                answer_text=answer_text,
                error=None,
            )
        )
    return outcomes


def _iter_target_files(root: Path) -> Iterable[Path]:
    """Yield JSON transcript files beneath ``root``."""

    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.lower().endswith(".json"):
                continue
            yield Path(dirpath) / name


def _collect_split_tasks_for_file(file_path: Path) -> Tuple[List[SplitTask], dict]:
    """Return split tasks and parsed JSON object for ``file_path``."""

    with file_path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return [], obj

    tasks: List[SplitTask] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        # Some transcripts include header artifacts (for example, "CHAT DATE")
        # before the "Show thinking" cue. Treat any assistant message that
        # mentions "show thinking" within the first few lines as a candidate.
        lines = content.splitlines()
        head_text = "\n".join(line.casefold() for line in lines[:5])
        if "show thinking" not in head_text:
            continue
        previous_content: Optional[str] = None
        if idx > 0:
            prev = messages[idx - 1]
            if isinstance(prev, Mapping):
                prev_content = prev.get("content")
                if isinstance(prev_content, str):
                    previous_content = prev_content
        tasks.append(
            SplitTask(
                file_path=file_path,
                message_index=idx,
                original_content=content,
                previous_message=previous_content,
            )
        )
    return tasks, obj


def _apply_splits_to_object(
    obj: dict,
    outcomes: Sequence[SplitOutcome],
) -> int:
    """Apply validated splits to the in-memory transcript object."""

    messages = obj.get("messages")
    if not isinstance(messages, list):
        return 0

    updated = 0
    for outcome in outcomes:
        task = outcome.task
        if outcome.error:
            logging.warning(
                "Skipping split for %s message_index=%d: %s",
                task.file_path,
                task.message_index,
                outcome.error or "no answer from LiteLLM",
            )
            continue
        idx = task.message_index
        if not 0 <= idx < len(messages):
            continue
        message = messages[idx]
        if not isinstance(message, Mapping):
            continue
        original = task.original_content
        # Special case: thinking-only message where the model indicated there
        # is no visible reply portion.
        if outcome.answer_text == "NO_REPLY":
            new_message = dict(message)
            new_message["thought"] = original
            new_message["content"] = ""
            messages[idx] = new_message
            updated += 1
            continue
        if not outcome.answer_text:
            logging.warning(
                "Skipping split for %s message_index=%d: %s",
                task.file_path,
                task.message_index,
                "no answer from LiteLLM",
            )
            continue
        split = _validate_and_split(original, outcome.answer_text)
        if split is None:
            logging.warning(
                "Could not align reply prefix for %s message_index=%d\n",
                task.file_path,
                task.message_index,
            )
            print()
            print("## original")
            print(original)
            print()
            print("## prefix")
            print(outcome.answer_text)
            print()
            print()
            continue
        thought, answer = split
        if not answer:
            logging.warning(
                "Empty answer segment after split for %s message_index=%d",
                task.file_path,
                task.message_index,
            )
            continue
        # Preserve any existing fields and add thought.
        new_message = dict(message)
        new_message["content"] = answer
        if thought:
            new_message["thought"] = thought
        messages[idx] = new_message
        updated += 1
    return updated


def process_root(
    root: Path,
    *,
    model: str,
    timeout: int,
    max_workers: int,
    reply_prefix_words: int,
    output_root: Optional[Path],
) -> None:
    """Run thought/answer splitting for assistant messages under root."""

    files = list(_iter_target_files(root))
    total_files = 0
    total_messages = 0
    for file_path in tqdm(files, desc="Splitting assistant thinking"):
        tasks, obj = _collect_split_tasks_for_file(file_path)
        if not tasks:
            continue
        outcomes = run_split_batch(
            tasks,
            model=model,
            timeout=timeout,
            max_workers=max_workers,
            reply_prefix_words=reply_prefix_words,
        )
        updated = _apply_splits_to_object(obj, outcomes)
        if not updated:
            continue
        try:
            if output_root is not None:
                try:
                    rel = file_path.relative_to(root)
                except ValueError:
                    rel = file_path.name
                out_path = output_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = file_path
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(obj, handle, ensure_ascii=False, indent=2)
        except OSError as err:
            logging.error("Failed to write %s: %s", file_path, err)
            continue
        total_files += 1
        total_messages += updated
        logging.info("Updated %s (split %d assistant messages).", file_path, updated)

    logging.info(
        "Completed splitting: %d files updated, %d assistant messages split.",
        total_files,
        total_messages,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for this script."""

    parser = argparse.ArgumentParser(
        description=(
            "Split thinking traces from assistant replies in pdf-highlight "
            "transcripts using LiteLLM."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root directory containing parsed transcript JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional root directory for writing updated transcripts. "
            "When omitted, files are modified in place under --root."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_CHAT_MODEL,
        help=f"LiteLLM model to use (default: {DEFAULT_CHAT_MODEL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Maximum concurrent LiteLLM workers (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--reply-prefix-words",
        type=int,
        default=DEFAULT_REPLY_PREFIX_WORDS,
        help=(
            "Number of leading reply tokens the model should return "
            f"(default: {DEFAULT_REPLY_PREFIX_WORDS})."
        ),
    )
    add_log_level_argument(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point for the pdf-highlight thought splitter."""

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    root = args.root
    output_root: Optional[Path] = args.output_root
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {root}")

    process_root(
        root=root,
        model=str(args.model),
        timeout=int(args.timeout),
        max_workers=int(args.max_workers),
        reply_prefix_words=int(args.reply_prefix_words),
        output_root=output_root,
    )


if __name__ == "__main__":
    main()
