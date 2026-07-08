"""
OmniRoute AI — Utility Module
Provides retry logic with exponential backoff (tenacity), logging, and helpers.
"""

import logging
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("omniroute")


# ---------------------------------------------------------------------------
# Retry decorator for all Gemini / LangChain API calls
# ---------------------------------------------------------------------------

def get_retry_decorator(max_attempts: int = 5, min_wait: int = 35, max_wait: int = 120):
    """
    Returns a tenacity retry decorator configured for OpenRouter free tier rate limits.

    Free models return Retry-After: 30s on 429s, so we wait at least 35s to be safe.
    Uses exponential backoff: 35s, 70s, 120s (capped).
    Retries on generic Exceptions (covers RateLimitError, 429, 503, etc.).
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Convenience instance for direct use as @api_retry
api_retry = get_retry_decorator()


# ---------------------------------------------------------------------------
# Async semaphore-based concurrency limiter
# ---------------------------------------------------------------------------

class ConcurrencyLimiter:
    """
    Wraps an asyncio.Semaphore to throttle concurrent API calls.
    Prevents bursting past RPM/TPM limits during the Map phase.
    """

    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        await self._semaphore.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text for logging/display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def estimate_pages(text: str, chars_per_page: int = 3000) -> float:
    """Rough estimate of how many pages a text block spans."""
    return len(text) / chars_per_page
