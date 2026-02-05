"""Retry decorator with exponential backoff for API calls."""

from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")

# Exception types that should trigger retries
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def with_retry(
    max_attempts: int = 3,
    min_wait: float = 1,
    max_wait: float = 30,
) -> Callable:
    """Create a retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time in seconds.

    Returns:
        A decorator that adds retry logic to a function.
    """
    return retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        reraise=True,
    )
