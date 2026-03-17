"""Cost estimation utilities for LiteLLM chat completions."""

from __future__ import annotations

import itertools
import json
import sys
from typing import Callable, Iterable, List, Optional

import litellm
from llm_delusions_annotations.llm_utils.client import LITELLM_API_ERRORS
from llm_delusions_annotations.utils import AnnotationRequest, to_litellm_messages


class CostEstimationError(RuntimeError):
    """Raised when LiteLLM cost estimation fails.

    Parameters
    ----------
    message:
        High level description of the failure.
    inner:
        Optional underlying LiteLLM exception that triggered the failure.

    Attributes
    ----------
    inner:
        Saved underlying exception instance when available.
    """

    def __init__(self, message: str, *, inner: Optional[BaseException] = None) -> None:
        super().__init__(message)
        self.inner = inner


def estimate_max_request_cost(
    model: str,
    request_payloads: Iterable[AnnotationRequest],
    *,
    max_completion_tokens: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[float, List[dict[str, object]], Optional[int], int]:
    """Estimate the maximum potential cost for the provided request payloads."""

    iterator = iter(request_payloads)
    try:
        first_request = next(iterator)
    except StopIteration:
        return 0.0, [], None, 0

    chained_requests = itertools.chain([first_request], iterator)

    try:
        max_tokens_raw = litellm.get_max_tokens(model)
        max_tokens: Optional[int]
        if isinstance(max_tokens_raw, str):
            max_tokens = int(max_tokens_raw)
        else:
            max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None
    except (ValueError, TypeError, KeyError) as err:
        raise CostEstimationError(f"Failed to parse max token info: {err}") from err
    except LITELLM_API_ERRORS as err:
        raise CostEstimationError(
            f"Failed to fetch max token info: {err}",
            inner=err,
        ) from err

    total_cost = 0.0
    breakdown_map: dict[str, dict[str, object]] = {}
    total_requests = 0

    for annotation_id, annotation_name, messages in chained_requests:
        try:
            prompt_tokens = litellm.token_counter(
                model=model,
                messages=to_litellm_messages(messages),
            )
        except (ValueError, TypeError, KeyError) as err:
            raise CostEstimationError(f"Failed to count tokens: {err}") from err
        except LITELLM_API_ERRORS as err:
            raise CostEstimationError(
                f"Failed to count tokens: {err}",
                inner=err,
            ) from err

        if max_tokens is None:
            available_completion_tokens: Optional[int] = None
        else:
            available_completion_tokens = max(max_tokens - prompt_tokens, 0)

        completion_tokens_for_cost = 0
        assumed_completion_tokens: Optional[int]
        if max_completion_tokens is not None:
            completion_tokens_for_cost = max_completion_tokens
            if available_completion_tokens is not None:
                completion_tokens_for_cost = min(
                    completion_tokens_for_cost,
                    available_completion_tokens,
                )
            completion_tokens_for_cost = max(completion_tokens_for_cost, 0)
            assumed_completion_tokens = completion_tokens_for_cost
        elif available_completion_tokens is not None:
            completion_tokens_for_cost = available_completion_tokens
            assumed_completion_tokens = available_completion_tokens
        else:
            assumed_completion_tokens = None

        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens_for_cost,
            )
        except (ValueError, TypeError, KeyError) as err:
            raise CostEstimationError(f"Failed to compute token cost: {err}") from err
        except LITELLM_API_ERRORS as err:
            raise CostEstimationError(
                f"Failed to compute token cost: {err}",
                inner=err,
            ) from err

        prompt_cost_value = float(prompt_cost or 0.0)
        completion_cost_value = float(completion_cost or 0.0)
        request_cost = prompt_cost_value + completion_cost_value
        total_cost += request_cost
        total_requests += 1
        if progress_callback:
            progress_callback(1)
        elif total_requests % 5000 == 0:
            print(
                f"Cost estimation progress: evaluated {total_requests} requests...",
                file=sys.stderr,
            )

        entry = breakdown_map.setdefault(
            annotation_id,
            {
                "annotation_id": annotation_id,
                "annotation": annotation_name,
                "prompt_tokens": 0,
                "assumed_completion_tokens": (
                    0 if assumed_completion_tokens is not None else None
                ),
                "prompt_cost_usd": 0.0,
                "completion_cost_usd": 0.0,
                "max_request_cost_usd": 0.0,
                "request_count": 0,
            },
        )
        entry["prompt_tokens"] += prompt_tokens
        if entry["assumed_completion_tokens"] is not None and (
            assumed_completion_tokens is not None
        ):
            entry["assumed_completion_tokens"] += assumed_completion_tokens
        entry["prompt_cost_usd"] += prompt_cost_value
        entry["completion_cost_usd"] += completion_cost_value
        entry["max_request_cost_usd"] += request_cost
        entry["request_count"] += 1

        if assumed_completion_tokens is None:
            entry["assumed_completion_tokens"] = None

    breakdown = [
        {
            "annotation_id": item["annotation_id"],
            "annotation": item["annotation"],
            "request_count": item["request_count"],
            "prompt_tokens": item["prompt_tokens"],
            "assumed_completion_tokens": item["assumed_completion_tokens"],
            "prompt_cost_usd": round(item["prompt_cost_usd"], 6),
            "completion_cost_usd": round(item["completion_cost_usd"], 6),
            "max_request_cost_usd": round(item["max_request_cost_usd"], 6),
        }
        for item in breakdown_map.values()
    ]

    return round(total_cost, 6), breakdown, max_tokens, total_requests


def safe_estimate_max_request_cost(
    model: str,
    request_payloads: Iterable[AnnotationRequest],
    *,
    max_completion_tokens: Optional[int] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> tuple[float, List[dict[str, object]], Optional[int], int]:
    """Wrapper around :func:`estimate_max_request_cost` with error handling."""

    try:
        return estimate_max_request_cost(
            model,
            request_payloads,
            max_completion_tokens=max_completion_tokens,
            progress_callback=progress_callback,
        )
    except CostEstimationError as err:
        print(f"Cost estimation failed: {err}", file=sys.stderr)
        return 0.0, [], None, 0


def print_cost_summary(
    model: str,
    max_completion_tokens: int,
    total_cost: float,
    cost_breakdown: List[dict[str, object]],
    max_tokens: Optional[int],
    total_requests: int,
) -> None:
    """Print a JSON summary of cost estimation results."""

    if not cost_breakdown:
        return

    (
        total_prompt_tokens,
        total_assumed_completion_tokens,
    ) = summarize_token_totals(cost_breakdown)

    summary: dict[str, object] = {
        "model": model,
        "max_tokens": max_completion_tokens,
        "model_max_tokens": max_tokens,
        "max_total_cost_usd": total_cost,
        "total_request_count": total_requests,
        "total_prompt_tokens": total_prompt_tokens,
        "total_assumed_completion_tokens": total_assumed_completion_tokens,
    }
    print("\nMax potential cost estimate:")
    print(json.dumps(summary, indent=2))


def summarize_token_totals(
    cost_breakdown: List[dict[str, object]],
) -> tuple[int, Optional[int]]:
    """Return aggregate prompt and completion token counts from a breakdown.

    Parameters
    ----------
    cost_breakdown:
        Per-annotation breakdown records produced by
        :func:`estimate_max_request_cost`.

    Returns
    -------
    tuple[int, Optional[int]]
        Total prompt tokens and the summed assumed completion tokens when
        available. When any entry has ``assumed_completion_tokens`` equal to
        ``None``, the aggregate completion count is reported as ``None``.
    """

    total_prompt_tokens = 0
    total_assumed_completion_tokens: Optional[int] = 0
    for item in cost_breakdown:
        prompt_value = item.get("prompt_tokens")
        if isinstance(prompt_value, int):
            total_prompt_tokens += prompt_value
        elif isinstance(prompt_value, float):
            total_prompt_tokens += int(prompt_value)
        assumed_value = item.get("assumed_completion_tokens")
        if assumed_value is None:
            total_assumed_completion_tokens = None
        elif total_assumed_completion_tokens is not None:
            if isinstance(assumed_value, int):
                total_assumed_completion_tokens += assumed_value
            elif isinstance(assumed_value, float):
                total_assumed_completion_tokens += int(assumed_value)

    return total_prompt_tokens, total_assumed_completion_tokens
