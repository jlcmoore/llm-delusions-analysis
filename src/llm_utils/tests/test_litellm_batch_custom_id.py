"""Tests for batch output custom_id/key handling."""

import pytest

from llm_utils.litellm_batch import process_batch_results


def test_process_batch_results_rejects_missing_custom_id_and_key() -> None:
    """Raise when batch output lacks both custom_id and key."""
    content = (
        '{"request": {"contents": []}, "response": {"candidates": []}}\n'
        '{"request": {"contents": []}, "response": {"candidates": []}}\n'
    )
    with pytest.raises(ValueError, match="custom_id/key"):
        process_batch_results(content)


def test_process_batch_results_accepts_key_field() -> None:
    """Accept and map key to results when custom_id is missing."""
    content = (
        '{"key": "task-1", "response": {"candidates": '
        '[{"content": {"parts": [{"text": "ok"}]}}]}}\n'
    )
    results = process_batch_results(content)
    assert results == {"task-1": "ok"}
