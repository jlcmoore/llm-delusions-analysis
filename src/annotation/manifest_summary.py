"""Pretty-print helpers for manifest token and request statistics.

These utilities are shared between the batch submit pipeline and
ad-hoc analysis scripts such as ``tmp_sum_manifest_tokens.py``.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Set, Tuple

import litellm
from llm_delusions_annotations.classify_messages import MAX_CLASSIFICATION_TOKENS


def _lookup_model_info(model: str) -> Mapping[str, object] | None:
    """Return LiteLLM pricing metadata for ``model`` when available."""

    base_model = model.split("/", 1)[1] if "/" in model else model
    return litellm.model_cost.get(model) or litellm.model_cost.get(base_model)


def print_token_cost_summary(
    *,
    model: str,
    total_tokens: int,
    total_tasks: int,
    max_completion_tokens: int = MAX_CLASSIFICATION_TOKENS,
    batch_discount: float = 0.5,
) -> None:
    """Print token breakdown and approximate on-demand and batch costs."""

    if total_tokens <= 0 or total_tasks <= 0:
        print(
            "\nToken breakdown and cost summary unavailable "
            f"(total_tokens={total_tokens}, total_tasks={total_tasks})."
        )
        return

    completion_tokens = total_tasks * max_completion_tokens
    prompt_tokens = max(total_tokens - completion_tokens, 0)

    print(
        "\nToken breakdown (approximate, based on "
        f"MAX_CLASSIFICATION_TOKENS={max_completion_tokens}):"
    )
    print(f"  Prompt tokens     (estimated): {prompt_tokens}")
    print(f"  Completion tokens (estimated): {completion_tokens}")

    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except (TypeError, ValueError, KeyError):
        print(
            f"\nApproximate cost unavailable for model {model}; "
            "litellm.cost_per_token raised an error."
        )
        return

    try:
        prompt_cost_f = float(prompt_cost)
        completion_cost_f = float(completion_cost)
    except (TypeError, ValueError):
        print(
            f"\nApproximate cost unavailable for model {model}; "
            "could not parse cost values."
        )
        return

    total_cost = prompt_cost_f + completion_cost_f
    discounted_total = total_cost * batch_discount

    model_info = _lookup_model_info(model)
    print("\nPricing details:")
    if model_info is not None:
        input_rate = model_info.get("input_cost_per_token")
        output_rate = model_info.get("output_cost_per_token")
        try:
            input_rate_f = float(input_rate)
            output_rate_f = float(output_rate)
        except (TypeError, ValueError):
            input_rate_f = None
            output_rate_f = None
        if input_rate_f is not None and output_rate_f is not None:
            print(
                f"  model_cost[{model!r}]['input_cost_per_token']  = "
                f"{input_rate_f:.8f}"
            )
            print(
                f"  model_cost[{model!r}]['output_cost_per_token'] = "
                f"{output_rate_f:.8f}"
            )
            print("  Math (on-demand, before batch discount):")
            print(
                f"    prompt_cost     = {prompt_tokens} * {input_rate_f} "
                f"= ${prompt_cost_f:.4f}"
            )
            print(
                f"    completion_cost = {completion_tokens} * {output_rate_f} "
                f"= ${completion_cost_f:.4f}"
            )
            print(
                "    total_cost      = prompt_cost + completion_cost "
                f"= ${total_cost:.4f}"
            )
            per_million_in = input_rate_f * 1_000_000
            per_million_out = output_rate_f * 1_000_000
            print("\n  On-demand rates (per 1M tokens):")
            print(f"    input  : ${per_million_in:.3f} per 1M tokens")
            print(f"    output : ${per_million_out:.3f} per 1M tokens")
            cache_rate = model_info.get("cache_read_input_token_cost")
            try:
                cache_rate_f = float(cache_rate)
            except (TypeError, ValueError):
                cache_rate_f = None
            if cache_rate_f is not None:
                per_million_cache = cache_rate_f * 1_000_000
                print(f"    cached : ${per_million_cache:.3f} per 1M tokens")
        else:
            print(
                "  Pricing metadata for this model is present but could not be "
                "parsed."
            )
    else:
        print(
            "  Pricing metadata for this model is not available in "
            "litellm.model_cost."
        )

    print("\nBatch pricing:")
    print(
        "  Applying a 50% Batch API discount to the on-demand total "
        "(batch_total_cost = total_cost * 0.5)."
    )
    print(f"  batch_total_cost = ${discounted_total:.4f}")
    print(
        "\nApproximate on-demand vs Batch API cost "
        f"for model {model}: on-demand=${total_cost:.4f}, "
        f"batch (50% off)=${discounted_total:.4f}"
    )


def print_annotation_stats(annotation_counts: Counter[str]) -> None:
    """Print per-annotation request counts."""

    if not annotation_counts:
        return

    print("\nAnnotation statistics:")
    print(f"  Unique annotations: {len(annotation_counts)}")
    max_ann_len = max(len(ann_id) for ann_id in annotation_counts.keys())
    max_count_len = max(len(str(count)) for count in annotation_counts.values())
    print("  Requests per annotation (all):")
    for ann_id, count in sorted(annotation_counts.items(), key=lambda item: item[0]):
        ann_col = ann_id.ljust(max_ann_len)
        count_col = str(count).rjust(max_count_len)
        print(f"    {ann_col} : {count_col}")


def print_participant_stats(
    participant_request_counts: Counter[str],
    participant_annotation_counts: Dict[str, Counter[str]] | Mapping[str, Counter[str]],
    participant_message_keys: (
        Dict[str, Set[Tuple[str, int, int]]] | Mapping[str, Set[Tuple[str, int, int]]]
    ),
) -> None:
    """Print per-participant request and message statistics."""

    if not participant_request_counts:
        return

    print("\nParticipant statistics:")
    max_ppt_len = max(len(p) for p in participant_request_counts.keys())
    max_req_len = max(len(str(v)) for v in participant_request_counts.values())
    max_unique_len = max(
        len(str(len(keys))) for keys in participant_message_keys.values()
    )
    max_ann_len = max(
        len(str(len(anns))) for anns in participant_annotation_counts.values()
    )

    per_ann_values: Dict[str, float] = {}
    max_per_ann_len = 0
    for participant, total_req in participant_request_counts.items():
        ann_counts = participant_annotation_counts.get(participant, {})
        num_annotations = len(ann_counts)
        per_annotation = total_req / num_annotations if num_annotations > 0 else 0.0
        per_ann_values[participant] = per_annotation
        text = f"{per_annotation:.2f}"
        max_per_ann_len = max(max_per_ann_len, len(text))

    for participant in sorted(
        participant_request_counts.keys(),
        key=lambda p: participant_request_counts[p],
        reverse=True,
    ):
        total_req = participant_request_counts[participant]
        unique_msgs = len(participant_message_keys.get(participant, set()))
        ann_counts = participant_annotation_counts.get(participant, {})
        num_annotations = len(ann_counts)
        per_annotation = per_ann_values[participant]

        ppt_col = participant.ljust(max_ppt_len)
        req_col = str(total_req).rjust(max_req_len)
        uniq_col = str(unique_msgs).rjust(max_unique_len)
        ann_col = str(num_annotations).rjust(max_ann_len)
        per_ann_col = f"{per_annotation:.2f}".rjust(max_per_ann_len)

        print(
            f"  {ppt_col} : "
            f"requests={req_col}, "
            f"unique_messages={uniq_col}, "
            f"annotations={ann_col}, "
            f"requests_per_annotation={per_ann_col}"
        )


def print_duplicate_warnings(
    duplicate_key_counts: Counter[Tuple[str, str, int, int, str]],
) -> None:
    """Print a warning block when duplicate task rows are detected."""

    duplicate_rows = sum(
        count - 1 for count in duplicate_key_counts.values() if count > 1
    )
    if duplicate_rows <= 0:
        return

    affected_keys = sum(1 for count in duplicate_key_counts.values() if count > 1)
    print("\nWARNING: detected duplicate task rows.")
    print(
        "  Duplicate rows are defined as repeated "
        "(participant, source_path, chat_index, message_index, annotation_id) "
        "combinations."
    )
    print(f"  Total duplicate rows: {duplicate_rows}")
    print(f"  Affected unique (participant, message, annotation) keys: {affected_keys}")
    print("  Top duplicate keys (up to 10):")
    for (
        participant,
        source_path,
        chat_index,
        message_index,
        ann_id,
    ), count in duplicate_key_counts.most_common(10):
        if count <= 1:
            continue
        print(
            "    "
            f"participant={participant}, source_path={source_path}, "
            f"chat_index={chat_index}, message_index={message_index}, "
            f"annotation_id={ann_id}: {count} rows"
        )
