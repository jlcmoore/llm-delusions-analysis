"""
Tests for LiteLLM batch manifest helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from annotation.batch_manifest import (
    ManifestConfig,
    ManifestSummary,
    create_manifest,
    decode_custom_id,
    encode_custom_id,
    iter_manifests,
    load_manifest_tasks,
    update_manifest_status,
)


def test_create_and_load_manifest_round_trip(tmp_path: Path) -> None:
    """create_manifest followed by load_manifest_tasks should preserve records."""

    manifest_path = tmp_path / "job1" / "batch_0001.jsonl"
    cid_a = encode_custom_id(
        "ppt_01",
        "bucket/ppt_01/chat.json",
        0,
        0,
        "a",
    )
    cid_b = encode_custom_id(
        "ppt_01",
        "bucket/ppt_01/chat.json",
        0,
        1,
        "b",
    )
    tasks = [
        {
            "custom_id": cid_a,
            "participant": "ppt_01",
            "source_path": "bucket/ppt_01/chat.json",
            "chat_index": 0,
            "chat_key": "chat-0",
            "chat_date": None,
            "message_index": 0,
            "role": "user",
            "timestamp": None,
            "annotation_id": "a",
            "annotation_name": "Test A",
            "prompt": "prompt-a",
        },
        {
            "custom_id": cid_b,
            "participant": "ppt_01",
            "source_path": "bucket/ppt_01/chat.json",
            "chat_index": 0,
            "chat_key": "chat-0",
            "chat_date": None,
            "message_index": 1,
            "role": "user",
            "timestamp": None,
            "annotation_id": "b",
            "annotation_name": "Test B",
            "prompt": "prompt-b",
            "written": True,
        },
    ]

    config = ManifestConfig(
        job_name="job1",
        batch_id="batch-1",
        input_file_id="file-1",
        model="gpt-test",
        provider="openai",
        endpoint="/v1/chat/completions",
        arguments={"max_messages": 10},
    )

    create_manifest(
        config,
        tasks=tasks,
        manifest_path=manifest_path,
    )

    meta, loaded_tasks = load_manifest_tasks(manifest_path)

    assert meta["job_name"] == "job1"
    assert meta["batch_id"] == "batch-1"
    assert meta["input_file_id"] == "file-1"
    assert meta["model"] == "gpt-test"
    assert meta["provider"] == "openai"
    assert meta["endpoint"] == "/v1/chat/completions"
    assert meta["arguments"]["max_messages"] == 10
    assert meta["task_count"] == 2

    assert len(loaded_tasks) == 2
    assert loaded_tasks[0]["custom_id"] == cid_a
    assert loaded_tasks[0]["written"] is False
    assert loaded_tasks[1]["custom_id"] == cid_b
    assert loaded_tasks[1]["written"] is True


def test_iter_manifests_yields_valid_meta(tmp_path: Path) -> None:
    """iter_manifests should yield ManifestSummary records for valid manifests."""

    root = tmp_path / "manifests"
    root.mkdir()

    valid_path = root / "batch_valid.jsonl"
    invalid_path = root / "not_a_manifest.jsonl"

    config = ManifestConfig(
        job_name="job-valid",
        batch_id="batch-valid",
        input_file_id="file-valid",
        model="gpt-test",
        provider="openai",
        endpoint="/v1/chat/completions",
        arguments={},
    )

    create_manifest(
        config,
        tasks=[],
        manifest_path=valid_path,
    )

    # Write a file that will not parse as a manifest.
    invalid_path.write_text("not-json\n", encoding="utf-8")

    summaries = list(iter_manifests(root))
    assert len(summaries) == 1
    summary = summaries[0]
    assert isinstance(summary, ManifestSummary)
    assert summary.path == valid_path
    assert summary.meta["batch_id"] == "batch-valid"
    assert summary.meta["job_name"] == "job-valid"


def test_update_manifest_status_overwrites_meta_only(tmp_path: Path) -> None:
    """update_manifest_status should update status and leave tasks intact."""

    manifest_path = tmp_path / "batch.jsonl"
    tasks = [
        {
            "type": "task",
            "custom_id": "id-1",
            "participant": "ppt_01",
        }
    ]
    # Write a simple manifest manually to exercise status update.
    meta = {
        "type": "meta",
        "version": 1,
        "job_name": "job",
        "batch_id": "batch",
        "input_file_id": "file",
        "model": "gpt-test",
        "provider": "openai",
        "endpoint": "/v1/chat/completions",
        "arguments": {},
        "status": "submitted",
        "created_at": "2025-01-01T00:00:00",
        "updated_at": "2025-01-01T00:00:00",
        "task_count": 1,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        for task in tasks:
            handle.write(json.dumps(task) + "\n")

    update_manifest_status(manifest_path, "completed")

    meta_after, tasks_after = load_manifest_tasks(manifest_path)
    assert meta_after["status"] == "completed"
    assert meta_after["task_count"] == 1
    assert len(tasks_after) == 1
    assert tasks_after[0]["custom_id"] == "id-1"


def test_custom_id_round_trip_with_special_characters() -> None:
    """encode_custom_id and decode_custom_id should be reversible."""

    participant = "ppt 01"
    source_path = "bucket/ppt_01/some file|with special.json"
    chat_index = 3
    message_index = 7
    annotation_id = "ann:id/with spaces"

    encoded = encode_custom_id(
        participant,
        source_path,
        chat_index,
        message_index,
        annotation_id,
    )
    decoded = decode_custom_id(encoded)

    assert decoded[0] == participant
    assert decoded[1] == source_path
    assert decoded[2] == chat_index
    assert decoded[3] == message_index
    assert decoded[4] == annotation_id
