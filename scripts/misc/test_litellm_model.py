"""Generic LiteLLM smoke test for any model.

This script sends a single prompt to a user-specified LiteLLM model and
prints the response. It enables the project's reasoning defaults unless
explicitly disabled.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Mapping, Optional, Sequence

from llm_delusions_annotations.annotation_prompts import disable_litellm_logging
from llm_delusions_annotations.llm_utils.client import LLMClientError, completion


def _build_messages(prompt: str) -> Sequence[Mapping[str, str]]:
    """Return a minimal chat message payload for ``prompt``."""

    return [{"role": "user", "content": prompt}]


def _extract_text(response: object) -> Optional[str]:
    """Return the first available text content from a LiteLLM response."""

    response_dict: Optional[dict[str, object]]
    if isinstance(response, dict):
        response_dict = response
    elif hasattr(response, "model_dump"):
        response_dict = response.model_dump()  # type: ignore[assignment]
    else:
        response_dict = None

    if response_dict is None:
        return None

    choices = response_dict.get("choices")
    if not isinstance(choices, list) or not choices:
        choices = []
    if choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    stripped = content.strip()
                    if stripped:
                        return stripped
                reasoning = message.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
            provider_fields = first.get("provider_specific_fields")
            if isinstance(provider_fields, dict):
                reasoning = provider_fields.get("reasoning")
                if isinstance(reasoning, str) and reasoning.strip():
                    return reasoning.strip()
                reasoning_content = provider_fields.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content.strip():
                    return reasoning_content.strip()

    candidates = response_dict.get("candidates")
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts = [
                        part.get("text")
                        for part in parts
                        if isinstance(part, dict) and part.get("text")
                    ]
                    joined = "\n".join(texts).strip()
                    if joined:
                        return joined
            text = first.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    return None


def _summarize_response(response: object) -> str:
    """Return a short, human-readable summary of a response payload."""

    response_dict: Optional[dict[str, object]]
    if isinstance(response, dict):
        response_dict = response
    elif hasattr(response, "model_dump"):
        response_dict = response.model_dump()  # type: ignore[assignment]
    else:
        response_dict = None

    if response_dict is None:
        return f"Unexpected response type: {type(response)!r}"

    keys = ", ".join(sorted(response_dict.keys()))
    summary_parts = [f"Top-level keys: {keys or '(none)'}"]
    choices = response_dict.get("choices")
    if isinstance(choices, list):
        summary_parts.append(f"choices: {len(choices)}")
    candidates = response_dict.get("candidates")
    if isinstance(candidates, list):
        summary_parts.append(f"candidates: {len(candidates)}")
    return " | ".join(summary_parts)


def main() -> int:
    """Script entry point."""

    parser = argparse.ArgumentParser(
        description="Send a single prompt to a LiteLLM model and print the response."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LiteLLM model identifier to call (required).",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with a short greeting.",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum completion tokens to request.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--no-reasoning-defaults",
        action="store_true",
        help="Disable reasoning defaults when making the request.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the full LiteLLM response payload as JSON.",
    )
    args = parser.parse_args()
    disable_litellm_logging()

    messages = _build_messages(args.prompt)
    try:
        response = completion(
            model=str(args.model),
            messages=messages,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            enable_reasoning_defaults=not args.no_reasoning_defaults,
            reasoning_effort="none",
        )
    except LLMClientError as err:
        print(f"LiteLLM request failed: {err}", file=sys.stderr)
        return 2

    text = _extract_text(response)
    if text:
        print(text)
    else:
        print("No content field found in response.")
        print(_summarize_response(response))
        print("Re-run with --show-raw to inspect the full payload.")

    if args.show_raw:
        print("\nRaw response:")
        if hasattr(response, "model_dump"):
            payload = response.model_dump()
        else:
            payload = response
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
