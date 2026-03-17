"""
Integration tests for classify_chats_batch submit and harvest flows.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from llm_delusions_annotations.classify_messages import MessageContext
from llm_delusions_annotations.configs import AnnotationConfig

from annotation.batch_manifest import (
    decode_custom_id,
    iter_manifests,
    load_manifest_tasks,
)
from annotation.io import iter_jsonl_meta, iter_jsonl_records

BatchCall = tuple[dict[str, object], dict[str, object]]
BatchRecorder = Callable[..., tuple[str, str]]


def _make_context(
    participant: str, source_path: str, chat_index: int, message_index: int
) -> MessageContext:
    """Return a simple MessageContext for submit tests."""

    return MessageContext(
        participant=participant,
        source_path=Path(source_path),
        chat_index=chat_index,
        chat_key="chat-0",
        chat_date=None,
        message_index=message_index,
        role="user",
        content="hello world",
        timestamp=None,
        preceding=None,
    )


def _make_batch_recorder() -> tuple[BatchRecorder, list[BatchCall]]:
    """Return a callable that records create_litellm_batch calls."""

    calls: list[BatchCall] = []

    def recorder(keys_to_messages, **kwargs):
        calls.append((keys_to_messages, kwargs))
        index = len(calls)
        return f"batch-{index}", f"file-{index}"

    return recorder, calls


def _build_batch_results(manifest_dir: Path) -> dict[str, dict[str, str]]:
    """Return fake batch results keyed by batch and custom id."""

    batch_results: dict[str, dict[str, str]] = {}
    for summary in iter_manifests(manifest_dir):
        meta, tasks = load_manifest_tasks(summary.path)
        batch_results[meta["batch_id"]] = {
            task["custom_id"]: json.dumps({"quotes": ["hello"], "score": 7})
            for task in tasks
        }
    return batch_results


def _patch_submit_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    ccb: object,
    fake_configs: list[AnnotationConfig],
) -> list[MessageContext]:
    """Patch config loading and message iteration for submit tests."""

    monkeypatch.setattr(
        ccb,
        "load_annotation_configs",
        lambda annotation_ids: fake_configs,
    )

    contexts = [
        _make_context("ppt_01", "bucket/ppt_01/chat.json", 0, 0),
        _make_context("ppt_01", "bucket/ppt_01/chat.json", 0, 1),
    ]

    monkeypatch.setattr(
        ccb,
        "prepare_message_iterator",
        lambda *args, **_kwargs: (
            iter(contexts),
            len(contexts),
            len(contexts),
            None,
        ),
    )

    return contexts


@pytest.fixture
def fake_configs() -> list[AnnotationConfig]:
    """Return a single dummy annotation configuration."""

    spec = {"id": "a1", "name": "Test A1", "description": "desc"}
    return [AnnotationConfig(spec=spec, allowed_roles=None)]


def test_submit_creates_manifest_and_calls_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_configs: list[AnnotationConfig],
) -> None:
    """submit should create manifests and call create_litellm_batch."""

    ccb = importlib.import_module("scripts.annotation.classify_chats_batch")

    _patch_submit_dependencies(monkeypatch, ccb, fake_configs)

    recorder, calls = _make_batch_recorder()
    monkeypatch.setattr(ccb, "create_litellm_batch", recorder)

    output_root = tmp_path / "outputs"
    input_root = tmp_path / "transcripts"
    input_root.mkdir(parents=True, exist_ok=True)
    assert (
        ccb.run_submit(
            ccb.parse_args(
                [
                    "submit",
                    "--input",
                    str(input_root),
                    "--output",
                    str(output_root),
                    "--annotation",
                    "a1",
                    "--job",
                    "job1",
                    "--batch-size",
                    "2",
                ]
            )
        )
        == 0
    )

    # Ensure a batch was created and manifests written.
    assert calls
    manifest_dir = output_root / "batch_manifests" / "job1"
    assert manifest_dir.exists()

    summary = next(iter_manifests(manifest_dir))
    meta, tasks = load_manifest_tasks(summary.path)
    assert meta["job_name"] == "job1"
    assert meta["batch_id"] == "batch-1"
    assert meta["task_count"] == 2
    assert len(tasks) == 2

    # Check custom_id encoding round-trip for one task.
    values = decode_custom_id(tasks[0]["custom_id"])
    assert values[0] == "ppt_01"
    assert values[1] == "bucket/ppt_01/chat.json"
    assert values[2] == 0
    assert values[3] in {0, 1}
    assert values[4] == "a1"


def test_submit_vertex_provider_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_configs: list[AnnotationConfig],
) -> None:
    """submit should normalize Vertex models and set provider metadata."""

    ccb = importlib.import_module("scripts.annotation.classify_chats_batch")

    _patch_submit_dependencies(monkeypatch, ccb, fake_configs)

    recorder, calls = _make_batch_recorder()
    monkeypatch.setattr(ccb, "create_litellm_batch", recorder)

    output_root = tmp_path / "outputs"
    input_root = tmp_path / "transcripts"
    input_root.mkdir(parents=True, exist_ok=True)

    assert (
        ccb.run_submit(
            ccb.parse_args(
                [
                    "submit",
                    "--input",
                    str(input_root),
                    "--output",
                    str(output_root),
                    "--annotation",
                    "a1",
                    "--job",
                    "job1",
                    "--batch-size",
                    "2",
                    "--model",
                    "vertex_ai/gemini-2.5-flash-lite",
                ]
            )
        )
        == 0
    )

    assert calls
    _messages, kwargs = calls[0]
    assert kwargs["custom_llm_provider"] == "vertex_ai"
    assert kwargs["request_parameters"]["model"] == "gemini-2.5-flash-lite"

    manifest_dir = output_root / "batch_manifests" / "job1"
    summary = next(iter_manifests(manifest_dir))
    meta, _tasks = load_manifest_tasks(summary.path)
    assert meta["provider"] == "vertex_ai"


def test_submit_and_harvest_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_configs: list[AnnotationConfig],
) -> None:
    """submit followed by harvest should write outputs and be idempotent."""

    ccb = importlib.import_module("scripts.annotation.classify_chats_batch")

    _patch_submit_dependencies(monkeypatch, ccb, fake_configs)

    monkeypatch.setattr(ccb, "create_litellm_batch", _make_batch_recorder()[0])

    output_root = tmp_path / "outputs"
    input_root = tmp_path / "transcripts"
    input_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "batch_manifests" / "job1"

    assert (
        ccb.run_submit(
            ccb.parse_args(
                [
                    "submit",
                    "--input",
                    str(input_root),
                    "--output",
                    str(output_root),
                    "--annotation",
                    "a1",
                    "--job",
                    "job1",
                    "--batch-size",
                    "2",
                ]
            )
        )
        == 0
    )

    # Build fake batch results keyed by batch_id and custom_id.
    batch_results = _build_batch_results(manifest_dir)

    monkeypatch.setattr(
        ccb,
        "resume_litellm_batch",
        lambda batch_id, **_kwargs: batch_results.get(batch_id, {}),
    )

    harvest_args = ccb.parse_args(
        [
            "harvest",
            "--output",
            str(output_root),
            "--manifest-dir",
            str(manifest_dir),
        ]
    )

    # First harvest run.
    assert ccb.run_harvest(harvest_args) == 0

    # Collect records after first harvest, ignoring manifest files.
    records_after_first: list[dict] = []
    for path, _meta in iter_jsonl_meta(output_root):
        if "batch_manifests" in path.parts:
            continue
        for record in iter_jsonl_records(path):
            records_after_first.append(record)

    assert len(records_after_first) == 2
    for record in records_after_first:
        assert record["annotation_id"] == "a1"
        assert record["participant"] == "ppt_01"
        assert "content" in record

    # Second harvest run should be idempotent.
    assert ccb.run_harvest(harvest_args) == 0

    records_after_second: list[dict] = []
    for path, _meta in iter_jsonl_meta(output_root):
        if "batch_manifests" in path.parts:
            continue
        for record in iter_jsonl_records(path):
            records_after_second.append(record)

    assert records_after_second == records_after_first
