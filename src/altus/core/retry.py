"""Retry helpers.

Streams are only retried while they have produced no output. Once the model
has emitted a token, replaying the call would duplicate text, so the error is
surfaced in-band as a ``StreamError`` instead.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Callable

from altus.core.errors import AltusError
from altus.core.events import StreamEvent

log = logging.getLogger(__name__)

StreamFactory = Callable[[], AsyncIterator[StreamEvent]]


def backoff_delay(attempt: int, *, base: float = 0.5, cap: float = 30.0) -> float:
    """Exponential backoff with full jitter. ``attempt`` is 0-indexed."""
    return random.uniform(0.0, min(cap, base * (2**attempt)))


async def retry_stream(
    factory: StreamFactory,
    *,
    max_attempts: int = 4,
    base: float = 0.5,
    cap: float = 30.0,
) -> AsyncIterator[StreamEvent]:
    """Yield from ``factory()``, retrying failures that happen before output."""
    for attempt in range(max_attempts):
        emitted = False
        try:
            async for event in factory():
                emitted = True
                yield event
            return
        except asyncio.CancelledError:
            raise
        except AltusError as exc:
            last = attempt == max_attempts - 1
            if emitted or not exc.retryable or last:
                raise
            delay = (
                exc.retry_after
                if exc.retry_after is not None
                else backoff_delay(attempt, base=base, cap=cap)
            )
            log.warning(
                "retrying %s after %.2fs (attempt %d/%d): %s",
                exc.provider or "provider",
                delay,
                attempt + 1,
                max_attempts,
                exc,
            )
            await asyncio.sleep(delay)
