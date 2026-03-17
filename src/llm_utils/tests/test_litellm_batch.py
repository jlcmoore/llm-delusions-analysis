"""
Tests for LiteLLM batch utilities.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import litellm
import pytest

from llm_utils.litellm_batch import (
    BatchFailedError,
    BatchTimeoutError,
    create_litellm_batch,
    delete_litellm_file,
    download_litellm_batch_output,
    poll_litellm_batch,
    resume_litellm_batch,
    run_litellm_batch,
)


class FakeLiteLLM:
    """Test double that captures LiteLLM interactions."""

    def __init__(self, *, statuses: list[str], output_lines: list[str]) -> None:
        """Initialise the fake client with statuses and output lines."""

        self.statuses = statuses
        self.output_lines = output_lines
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.retrieve_call_count = 0
        self.batch_file_bytes: bytes | None = None

    def create_file(self, **kwargs):
        """Simulate uploading a batch file and capture the payload."""

        self.calls.append(("create_file", kwargs))
        _, file_obj = kwargs["file"]
        self.batch_file_bytes = file_obj.getvalue()
        return SimpleNamespace(id="file-test")

    def create_batch(self, **kwargs):
        """Simulate scheduling a batch run."""

        self.calls.append(("create_batch", kwargs))
        return SimpleNamespace(id="batch-test")

    def retrieve_batch(self, **kwargs):
        """Return a SimpleNamespace describing the current batch status."""

        self.calls.append(("retrieve_batch", kwargs))
        status_index = min(self.retrieve_call_count, len(self.statuses) - 1)
        status = self.statuses[status_index]
        self.retrieve_call_count += 1

        if status == "completed":
            return SimpleNamespace(
                status="completed", failed_at=None, output_file_id="output-test"
            )

        if status == "failed":
            return SimpleNamespace(
                status="failed",
                failed_at="now",
                errors=["request failed"],
                output_file_id=None,
            )

        return SimpleNamespace(status=status, failed_at=None, output_file_id=None)

    def file_content(self, **kwargs):
        """Return encoded batch output content."""

        self.calls.append(("file_content", kwargs))
        return SimpleNamespace(
            content="\n".join(self.output_lines).encode("utf-8"),
        )

    def file_delete(self, **kwargs):
        """Record a request to delete a batch file."""

        self.calls.append(("file_delete", kwargs))
        return SimpleNamespace(id=kwargs["file_id"])


def test_create_litellm_batch_returns_ids_and_uploads():
    """create_litellm_batch should upload a file and return identifiers."""
    messages = {
        "first": [{"role": "user", "content": "hello there"}],
    }
    fake_client = FakeLiteLLM(statuses=["completed"], output_lines=[])

    batch_id, file_id = create_litellm_batch(
        messages,
        litellm_client=fake_client,
        request_parameters={"model": "gpt-test"},
    )

    assert batch_id == "batch-test"
    assert file_id == "file-test"

    create_file_call = fake_client.calls[0]
    assert create_file_call[0] == "create_file"
    assert fake_client.batch_file_bytes is not None


def test_poll_litellm_batch_completes_and_tracks_status():
    """poll_litellm_batch should eventually return a completed batch."""
    fake_client = FakeLiteLLM(
        statuses=["queued", "running", "completed"],
        output_lines=[],
    )

    retrieve_response = poll_litellm_batch(
        "batch-test",
        litellm_client=fake_client,
        polling_interval=0.0 + 0.01,
        timeout_seconds=1.0,
    )

    assert getattr(retrieve_response, "status") == "completed"
    assert fake_client.retrieve_call_count == 3


def test_poll_litellm_batch_raises_on_failure():
    """poll_litellm_batch should raise BatchFailedError on failure."""
    fake_client = FakeLiteLLM(
        statuses=["failed"],
        output_lines=[],
    )

    with pytest.raises(BatchFailedError):
        poll_litellm_batch(
            "batch-test",
            litellm_client=fake_client,
            polling_interval=0.01,
            timeout_seconds=1.0,
        )


def test_download_litellm_batch_output_reads_content():
    """download_litellm_batch_output should return decoded batch output text."""
    lines = [
        json.dumps(
            {
                "custom_id": "first",
                "response": {"body": {"choices": [{"message": {"content": "hello"}}]}},
            }
        )
    ]
    fake_client = FakeLiteLLM(statuses=["completed"], output_lines=lines)

    content = download_litellm_batch_output(
        "output-test",
        litellm_client=fake_client,
    )

    assert "hello" in content
    file_content_call = fake_client.calls[-1]
    assert file_content_call[0] == "file_content"
    assert file_content_call[1]["file_id"] == "output-test"


def test_delete_litellm_file_invokes_client_delete():
    """delete_litellm_file should invoke the client's delete operation."""
    fake_client = FakeLiteLLM(statuses=["completed"], output_lines=[])

    delete_litellm_file(
        "file-test",
        litellm_client=fake_client,
    )

    assert fake_client.calls[-1][0] == "file_delete"
    assert fake_client.calls[-1][1]["file_id"] == "file-test"

    call_count_after_first = len(fake_client.calls)
    delete_litellm_file(
        "",
        litellm_client=fake_client,
    )
    assert len(fake_client.calls) == call_count_after_first


def test_run_litellm_batch_openai_provider():
    """run_litellm_batch should handle an OpenAI-style provider workflow."""
    messages = {
        "first": [{"role": "user", "content": "hello there"}],
    }
    fake_client = FakeLiteLLM(
        statuses=["in_progress", "completed"],
        output_lines=[
            json.dumps(
                {
                    "custom_id": "first",
                    "response": {
                        "body": {
                            "choices": [
                                {"message": {"content": "hello back"}},
                            ]
                        }
                    },
                }
            ),
        ],
    )

    result = run_litellm_batch(
        messages,
        litellm_client=fake_client,
        request_parameters={"model": "gpt-test"},
    )

    assert result == {"first": "hello back"}

    create_file_call = fake_client.calls[0]
    assert create_file_call[0] == "create_file"
    assert create_file_call[1]["custom_llm_provider"] == "openai"
    assert fake_client.batch_file_bytes is not None

    first_line = fake_client.batch_file_bytes.decode("utf-8").splitlines()[0]
    payload = json.loads(first_line)
    assert payload["body"]["model"] == "gpt-test"

    delete_call = fake_client.calls[-1]
    assert delete_call[0] == "file_delete"
    assert delete_call[1]["file_id"] == "file-test"


def test_run_litellm_batch_anthropic_provider():
    """run_litellm_batch should support Anthropic-style chat completion batches."""
    messages = {
        "first": [{"role": "user", "content": "ping"}],
        "second": [{"role": "user", "content": "pong"}],
    }
    fake_client = FakeLiteLLM(
        statuses=["queued", "running", "completed"],
        output_lines=[
            json.dumps(
                {
                    "custom_id": "first",
                    "response": {
                        "body": {
                            "choices": [
                                {"message": {"content": "first choice"}},
                                {"message": {"content": "alt choice"}},
                            ]
                        }
                    },
                }
            ),
            json.dumps(
                {
                    "custom_id": "second",
                    "response": {
                        "body": {
                            "choices": [
                                {"message": {"content": "second choice"}},
                            ]
                        }
                    },
                }
            ),
        ],
    )

    result = run_litellm_batch(
        messages,
        litellm_client=fake_client,
        custom_llm_provider="anthropic",
        endpoint="/v1/messages",
        request_parameters={"model": "claude-3"},
        return_all_choices=True,
    )

    assert result["first"][0]["message"]["content"] == "first choice"
    assert result["first"][1]["message"]["content"] == "alt choice"
    assert result["second"][0]["message"]["content"] == "second choice"

    for call_name, kwargs in fake_client.calls:
        if call_name in {
            "create_file",
            "create_batch",
            "retrieve_batch",
            "file_content",
            "file_delete",
        }:
            assert kwargs["custom_llm_provider"] == "anthropic"

    first_line = fake_client.batch_file_bytes.decode("utf-8").splitlines()[0]
    payload = json.loads(first_line)
    assert payload["url"] == "/v1/messages"
    assert payload["body"]["model"] == "claude-3"


def test_run_litellm_batch_timeout():
    """run_litellm_batch should raise BatchTimeoutError when polling exceeds the timeout."""
    messages = {"first": [{"role": "user", "content": "ping"}]}
    fake_client = FakeLiteLLM(
        statuses=["running", "running", "running"],
        output_lines=[],
    )

    with pytest.raises(BatchTimeoutError):
        run_litellm_batch(
            messages,
            litellm_client=fake_client,
            polling_interval=0.01,
            timeout_seconds=0.02,
        )


def test_resume_litellm_batch_uses_existing_batch_and_does_not_create_new():
    """resume_litellm_batch should reuse an existing batch without creating new resources."""
    output_lines = [
        json.dumps(
            {
                "custom_id": "first",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"content": "resumed hello"}},
                        ]
                    }
                },
            }
        )
    ]
    fake_client = FakeLiteLLM(
        statuses=["running", "completed"],
        output_lines=output_lines,
    )

    # Ensure helper can be used without creating a new batch or file.
    result = resume_litellm_batch(
        "batch-existing",
        litellm_client=fake_client,
        polling_interval=0.01,
        timeout_seconds=1.0,
    )

    assert result == {"first": "resumed hello"}

    call_names = [name for name, _ in fake_client.calls]
    assert "create_file" not in call_names
    assert "create_batch" not in call_names
    assert "retrieve_batch" in call_names
    assert "file_content" in call_names


def test_process_vertex_batch_without_custom_id():
    """resume_litellm_batch should map results by order when custom_id is absent."""
    output_lines = [
        json.dumps(
            {
                "status": "",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "first response"}]}}]
                },
            }
        ),
        json.dumps(
            {
                "status": "",
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "second response"}]}}
                    ]
                },
            }
        ),
    ]
    fake_client = FakeLiteLLM(
        statuses=["completed"],
        output_lines=output_lines,
    )

    result = resume_litellm_batch(
        "batch-existing",
        litellm_client=fake_client,
        custom_id_order=["first", "second"],
    )

    assert result == {"first": "first response", "second": "second response"}


def test_process_vertex_batch_count_mismatch_raises():
    """process_batch_results should raise on line count mismatches."""
    output_lines = [
        json.dumps(
            {
                "status": "",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "first response"}]}}]
                },
            }
        )
    ]
    fake_client = FakeLiteLLM(
        statuses=["completed"],
        output_lines=output_lines,
    )

    with pytest.raises(ValueError):
        resume_litellm_batch(
            "batch-existing",
            litellm_client=fake_client,
            custom_id_order=["first", "second"],
        )


def _require_env(var_name: str) -> str:
    """Return an environment variable value or skip when unavailable.

    Network-dependent tests are only run when the
    ``LLM_DELUSIONS_RUN_NETWORK_TESTS`` flag is set. This keeps the
    suite reliable in environments without external network access.
    """

    if not os.getenv("LLM_DELUSIONS_RUN_NETWORK_TESTS"):
        pytest.skip("Network tests disabled; set LLM_DELUSIONS_RUN_NETWORK_TESTS=1")

    value = os.getenv(var_name)
    if not value:
        pytest.skip(f"{var_name} is not configured")
    return value


def test_openai_batch_real_call():
    """Run a LiteLLM batch request against OpenAI."""
    _require_env("OPENAI_API_KEY")

    messages = {
        "openai-batch": [{"role": "user", "content": "Reply with the word hello."}],
    }

    start_time = time.monotonic()
    results = run_litellm_batch(
        messages,
        litellm_client=litellm,
        custom_llm_provider="openai",
        request_parameters={
            "model": "gpt-4o-mini",
            "temperature": 0,
            "max_tokens": 32,
        },
        polling_interval=5,
        timeout_seconds=180,
    )
    elapsed = time.monotonic() - start_time

    assert "hello" in results["openai-batch"].lower()
    assert elapsed < 180
