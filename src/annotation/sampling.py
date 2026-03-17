"""Sampling helpers for annotation pipelines."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterator, List, Mapping, Set

from llm_delusions_annotations.chat.chat_utils import MessageContext
from llm_delusions_annotations.classify_messages import ConversationKey


def allocate_per_participant(
    sizes: Mapping[str, int],
    total_sample_size: int,
    *,
    equal: bool,
) -> Dict[str, int]:
    """Return per-participant allocations for a target sample size.

    Parameters
    ----------
    sizes:
        Mapping from participant identifier to the number of available items.
    total_sample_size:
        Target total number of items to allocate across all participants.
    equal:
        When True, aim to allocate the same number of items to each
        participant (as evenly as possible). When False, allocate in
        proportion to each participant's available item count.

    Returns
    -------
    Dict[str, int]
        Mapping from participant identifier to the number of items that
        should be sampled for that participant, never exceeding the
        corresponding entry in ``sizes``.
    """

    participants = sorted(sizes.keys())
    if not participants or total_sample_size <= 0:
        return {name: 0 for name in participants}

    total_messages = sum(max(0, sizes.get(name, 0)) for name in participants)
    if total_messages <= 0:
        return {name: 0 for name in participants}

    if total_sample_size >= total_messages:
        return {name: max(0, sizes.get(name, 0)) for name in participants}

    allocations: Dict[str, int] = {name: 0 for name in participants}

    if equal:
        base = total_sample_size // len(participants)
        for name in participants:
            allocations[name] = min(base, max(0, sizes.get(name, 0)))
        remainder = total_sample_size - sum(allocations.values())
        if remainder > 0:
            remaining_cap = sum(
                max(0, sizes.get(name, 0) - allocations[name]) for name in participants
            )
            remainders: list[tuple[str, float]] = []
            if remaining_cap > 0:
                for name in participants:
                    cap = max(0, sizes.get(name, 0) - allocations[name])
                    if remaining_cap > 0:
                        quota = remainder * (cap / float(remaining_cap))
                    else:
                        quota = 0.0
                    k = int(quota)
                    k = min(k, cap)
                    allocations[name] += k
                    remainder -= k
                    remainders.append((name, quota - k))
            if remainder > 0:
                remainders.sort(key=lambda item: item[1], reverse=True)
                for name, _frac in remainders:
                    if remainder == 0:
                        break
                    if allocations[name] < max(0, sizes.get(name, 0)):
                        allocations[name] += 1
                        remainder -= 1
        if remainder > 0:
            for name in participants:
                if remainder == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remainder -= 1
    else:
        total = float(total_messages)
        remainders: list[tuple[str, float]] = []
        used = 0
        for name in participants:
            size_value = max(0, sizes.get(name, 0))
            quota = total_sample_size * (size_value / total)
            k = int(quota)
            k = min(k, size_value)
            allocations[name] = k
            used += k
            remainders.append((name, quota - k))
        remaining = total_sample_size - used
        if remaining > 0:
            remainders.sort(key=lambda item: item[1], reverse=True)
            for name, _frac in remainders:
                if remaining == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remaining -= 1
        if remaining > 0:
            for name in participants:
                if remaining == 0:
                    break
                if allocations[name] < max(0, sizes.get(name, 0)):
                    allocations[name] += 1
                    remaining -= 1

    return allocations


def limit_conversations_by_participant(
    message_iter: Iterator[MessageContext],
    max_conversations: int,
) -> Iterator[MessageContext]:
    """Yield contexts limited to the first ``max_conversations`` per participant."""

    if max_conversations <= 0:
        yield from message_iter
        return

    participant_counts: defaultdict[str, int] = defaultdict(int)
    allowed_keys: Set[ConversationKey] = set()
    skipped_keys: Set[ConversationKey] = set()

    for context in message_iter:
        conversation_key: ConversationKey = (
            context.participant,
            context.source_path,
            context.chat_key,
            context.chat_index,
        )
        if conversation_key in allowed_keys:
            yield context
            continue
        if conversation_key in skipped_keys:
            continue

        if participant_counts[context.participant] >= max_conversations:
            skipped_keys.add(conversation_key)
            continue

        participant_counts[context.participant] += 1
        allowed_keys.add(conversation_key)
        yield context


def sample_conversations_within_participant(
    message_iter: Iterator[MessageContext],
    sample_size: int,
    rng: random.Random,
) -> List[MessageContext]:
    """Return contexts sampled by selecting random conversations per participant.

    Parameters
    ----------
    message_iter:
        Source iterator yielding message contexts.
    sample_size:
        Maximum number of messages to include across sampled conversations.
    rng:
        Random number generator used for sampling.

    Returns
    -------
    List[MessageContext]
        Sampled contexts with length at most ``sample_size`` preserving the
        original conversation order. When ``sample_size`` is zero or negative,
        all conversations are included in randomized order.
    """

    conversation_messages: dict[ConversationKey, List[MessageContext]] = {}
    conversation_order: dict[ConversationKey, int] = {}
    participant_to_conversations: defaultdict[str, List[ConversationKey]] = defaultdict(
        list
    )

    for index, context in enumerate(message_iter):
        conversation_key: ConversationKey = (
            context.participant,
            context.source_path,
            context.chat_key,
            context.chat_index,
        )
        if conversation_key not in conversation_messages:
            conversation_messages[conversation_key] = []
            conversation_order[conversation_key] = index
            participant_to_conversations[context.participant].append(conversation_key)
        conversation_messages[conversation_key].append(context)

    if not conversation_messages:
        return []

    include_all = sample_size <= 0
    participants = list(participant_to_conversations.keys())
    rng.shuffle(participants)

    selected_conversations: List[ConversationKey] = []
    remaining = sample_size
    for participant in participants:
        conversation_keys = participant_to_conversations[participant][:]
        rng.shuffle(conversation_keys)
        for conversation_key in conversation_keys:
            if include_all:
                selected_conversations.append(conversation_key)
                continue
            conversation_length = len(conversation_messages[conversation_key])
            if conversation_length > remaining and selected_conversations:
                continue
            selected_conversations.append(conversation_key)
            remaining = max(remaining - conversation_length, 0)
            if remaining == 0:
                break
        if not include_all and remaining == 0:
            break

    if not include_all and not selected_conversations:
        fallback_key = min(
            conversation_messages,
            key=lambda candidate: len(conversation_messages[candidate]),
        )
        selected_conversations.append(fallback_key)

    selected_conversations.sort(key=lambda key: conversation_order[key])

    sampled_contexts: List[MessageContext] = []
    for conversation_key in selected_conversations:
        for context in conversation_messages[conversation_key]:
            if not include_all and len(sampled_contexts) >= sample_size:
                return sampled_contexts
            sampled_contexts.append(context)

    return sampled_contexts


def _collect_messages_by_participant(
    message_iter: Iterator[MessageContext],
) -> tuple[
    dict[str, List[tuple[int, MessageContext]]], List[tuple[int, MessageContext]]
]:
    """Return per-participant message buckets and global ordering.

    Messages are grouped by participant while also tracking their original
    global sequence index so that sampled results can later be restored to
    the input order.
    """

    buckets: dict[str, List[tuple[int, MessageContext]]] = {}
    all_collected: List[tuple[int, MessageContext]] = []
    for sequence_index, context in enumerate(message_iter):
        bucket_list = buckets.setdefault(context.participant, [])
        pair = (sequence_index, context)
        bucket_list.append(pair)
        all_collected.append(pair)
    return buckets, all_collected


def _allocate_sampled_pairs(
    buckets: dict[str, List[tuple[int, MessageContext]]],
    sample_size: int,
    rng: random.Random,
    *,
    equal: bool,
) -> List[tuple[int, MessageContext]]:
    """Return sampled (sequence index, context) pairs across participants.

    The allocation per participant is computed using the same allocation
    helper as the manual annotation dataset script. Sampling within each
    participant is performed without replacement.
    """

    participants = sorted(buckets.keys())
    sizes = {participant: len(buckets[participant]) for participant in participants}
    allocations = allocate_per_participant(sizes, sample_size, equal=equal)

    chosen_pairs: List[tuple[int, MessageContext]] = []
    for participant in participants:
        allocation = allocations[participant]
        if allocation <= 0:
            continue
        participant_pairs = buckets[participant]
        if allocation >= len(participant_pairs):
            chosen_pairs.extend(participant_pairs)
            continue
        sampled_indices = rng.sample(range(len(participant_pairs)), allocation)
        for index in sampled_indices:
            chosen_pairs.append(participant_pairs[index])

    return chosen_pairs


def sample_messages_by_participant(
    message_iter: Iterator[MessageContext],
    sample_size: int,
    rng: random.Random,
    *,
    equal: bool = False,
) -> List[MessageContext]:
    """Sample messages with per-participant allocation.

    Parameters
    ----------
    message_iter:
        Source iterator yielding :class:`MessageContext` instances.
    sample_size:
        Maximum total number of messages to sample.
    rng:
        Random number generator used for participant order and index selection.
    equal:
        When True, sample the same number from each participant (as evenly as
        possible). When False, sample in proportion to each participant's total
        message count.

    Returns
    -------
    List[MessageContext]
        Selected contexts preserving the original global iteration order.
    """

    if sample_size <= 0:
        return []

    buckets, all_collected = _collect_messages_by_participant(message_iter)
    total_messages = len(all_collected)
    if total_messages == 0:
        return []
    if sample_size >= total_messages:
        return [ctx for _seq, ctx in all_collected]

    chosen_pairs = _allocate_sampled_pairs(
        buckets,
        sample_size,
        rng,
        equal=equal,
    )

    chosen_pairs.sort(key=lambda pair: pair[0])
    return [ctx for _seq, ctx in chosen_pairs]
