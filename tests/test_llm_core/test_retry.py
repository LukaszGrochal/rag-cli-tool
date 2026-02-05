"""Tests for the retry utility with exponential backoff."""

import pytest

from llm_core.retry import with_retry


def test_retry_succeeds_on_second_attempt():
    """Should retry on failure and return the successful result."""
    call_count = 0

    @with_retry(max_attempts=3, min_wait=0, max_wait=0)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Temporary failure")
        return "success"

    result = flaky_function()
    assert result == "success"
    assert call_count == 2


def test_retry_exhausted_raises():
    """Should raise after exhausting all retries."""

    @with_retry(max_attempts=2, min_wait=0, max_wait=0)
    def always_fails():
        raise ConnectionError("Permanent failure")

    with pytest.raises(ConnectionError):
        always_fails()
