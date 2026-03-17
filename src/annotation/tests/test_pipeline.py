"""
Tests for shared annotation pipeline helpers.
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from llm_delusions_annotations.classify_messages import (
    ClassificationOutcome,
    ClassificationTask,
    MessageContext,
)
from llm_delusions_annotations.configs import AnnotationConfig

from annotation.io import ReplayKey
from annotation.pipeline import (
    build_meta_record,
    ensure_output_handle_for_context,
    prepare_message_iterator,
    select_applicable_configs_for_context,
    write_outcomes_for_context,
)


def _make_dummy_args(**overrides) -> SimpleNamespace:
    """Return a minimal args namespace with reasonable defaults."""

    defaults = {
        "model": "gpt-test",
        "preceding_context": 3,
        "follow_links": False,
        "reverse_conversations": False,
        "max_messages": 0,
        "max_conversations": 0,
        "randomize": False,
        "randomize_conversations": False,
        "randomize_per_ppt": "proportional",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_context(participant: str = "ppt_01") -> MessageContext:
    """Return a simple MessageContext for testing."""

    return MessageContext(
        participant=participant,
        source_path=Path("bucket") / participant / "chat.json",
        chat_index=0,
        chat_key="chat-0",
        chat_date=None,
        message_index=0,
        role="user",
        content="hello world",
        timestamp=None,
        preceding=None,
    )


def test_build_meta_record_includes_expected_fields() -> None:
    """build_meta_record should capture core run settings and annotations."""

    spec = {"id": "test-annotation", "name": "Test", "description": "desc"}
    config = AnnotationConfig(spec=spec, allowed_roles=None)
    args = _make_dummy_args(model="gpt-4.1")
    non_default = {"model": "gpt-4.1", "preceding_context": 3}

    meta = build_meta_record(
        args=args,
        configs=[config],
        participant="ppt_01",
        non_default_arguments=non_default,
    )

    assert meta["type"] == "meta"
    assert meta["model"] == "gpt-4.1"
    assert meta["ppt_id"] == "ppt_01"
    assert meta["participants"] == ["ppt_01"]
    assert meta["annotation_ids"] == ["test-annotation"]
    assert meta["arguments"] == non_default
    assert "annotation_snapshots" in meta
    assert "test-annotation" in meta["annotation_snapshots"]


def test_ensure_output_handle_for_context_creates_file_and_meta(tmp_path: Path) -> None:
    """ensure_output_handle_for_context should create a new file with meta."""

    args = _make_dummy_args()
    context = _make_context("ppt_01")
    spec = {"id": "test-annotation", "name": "Test", "description": "desc"}
    config = AnnotationConfig(spec=spec, allowed_roles=None)
    output_handles: dict[Path, io.TextIOBase] = {}

    stack = contextlib.ExitStack()
    try:
        output_file_path, handle = ensure_output_handle_for_context(
            context=context,
            args=args,
            configs=[config],
            output_dir=tmp_path,
            single_output_file=None,
            resolved_output_name="run.jsonl",
            non_default_arguments={"model": args.model},
            output_handles=output_handles,
            stack=stack,
        )
        assert output_file_path is not None
        assert handle is not None
        handle.flush()
    finally:
        stack.close()
    assert output_file_path.exists()

    with output_file_path.open("r", encoding="utf-8") as reader:
        first_line = reader.readline()
    meta = json.loads(first_line)
    assert meta["type"] == "meta"
    assert meta["ppt_id"] == "ppt_01"
    assert "annotation_snapshots" in meta


def test_select_applicable_configs_respects_seen_keys_and_quota() -> None:
    """select_applicable_configs_for_context should filter by seen keys and quotas."""

    spec_a = {"id": "a"}
    spec_b = {"id": "b"}
    config_a = AnnotationConfig(spec=spec_a, allowed_roles=None)
    config_b = AnnotationConfig(spec=spec_b, allowed_roles=None)
    context = _make_context("ppt_01")

    key_a = (
        context.participant,
        str(context.source_path),
        context.chat_index,
        context.message_index,
        "a",
    )
    resume_seen_keys = {key_a}
    positive_counts = {"a": 1, "b": 0}

    applicable = select_applicable_configs_for_context(
        context=context,
        configs=[config_a, config_b],
        resume_seen_keys=resume_seen_keys,
        min_positive=1,
        positive_counts=positive_counts,
    )

    # Config A filtered out by seen key and quota; config B still applies.
    assert [cfg.spec["id"] for cfg in applicable] == ["b"]


def test_write_outcomes_for_context_serializes_records() -> None:
    """write_outcomes_for_context should emit one JSONL record per outcome."""

    context = _make_context("ppt_01")
    args = _make_dummy_args(model="gpt-5.1")
    spec = {"id": "a", "name": "Test", "description": "desc"}
    task = ClassificationTask(context=context, annotation=spec, prompt="prompt")
    outcome = ClassificationOutcome(
        task=task,
        matches=["hello"],
        error=None,
        rationale="Short rationale.",
        score=7,
        reasoning_content="reasoning",
        thinking_blocks=[{"type": "thinking", "thinking": "notes"}],
    )

    buffer = io.StringIO()
    write_outcomes_for_context(
        context,
        [outcome],
        args=args,
        out_file=buffer,
    )

    line = buffer.getvalue().strip()
    assert line
    record = json.loads(line)
    assert record["annotation_id"] == "a"
    assert record["annotation"] == "Test"
    assert record["matches"] == ["hello"]
    assert record["score"] == 7
    assert record["rationale"] == "Short rationale."
    assert record["reasoning_content"] == "reasoning"
    assert record["thinking_blocks"] == [{"type": "thinking", "thinking": "notes"}]
    assert record["participant"] == "ppt_01"


def test_prepare_message_iterator_respects_replay_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """prepare_message_iterator should filter contexts based on replay keys."""

    args = _make_dummy_args(max_messages=0, max_conversations=0)
    context_all = [
        _make_context("ppt_01"),
        _make_context("ppt_02"),
    ]

    def fake_iter_message_contexts(root, participants_filter, **kwargs):
        del root, participants_filter, kwargs
        return iter(context_all)

    monkeypatch.setattr(
        "annotation.pipeline.iter_message_contexts",
        fake_iter_message_contexts,
    )

    spec = {"id": "a", "name": "Test", "description": "desc"}
    config = AnnotationConfig(spec=spec, allowed_roles=None)

    replay_key: ReplayKey = (
        context_all[0].participant,
        str(context_all[0].source_path),
        context_all[0].chat_index,
        context_all[0].message_index,
    )

    message_iter, max_messages, progress_total, sampled = prepare_message_iterator(
        args=args,
        root=Path("unused"),
        configs=[config],
        participants_filter=None,
        replay_keys=[replay_key],
    )

    contexts = list(message_iter)
    assert len(contexts) == 1
    assert contexts[0].participant == "ppt_01"
    assert max_messages == 1
    assert progress_total == 1
    assert sampled is None
