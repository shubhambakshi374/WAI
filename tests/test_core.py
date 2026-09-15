from __future__ import annotations

import time

import pytest

from altus.core.errors import AltusError, RateLimitError, TransientProviderError
from altus.core.events import MessageEnd, StreamError, TextDelta
from altus.core.retry import backoff_delay, retry_stream
from altus.core.session import Session
from altus.core.types import (
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
    new_id,
)
from altus.runner import TurnAccumulator, run_turn
from tests.conftest import FakeProvider, default_events


def test_message_text_excludes_reasoning_and_tools() -> None:
    message = Message(
        role=Role.ASSISTANT,
        content=[
            ReasoningBlock(text="thinking"),
            TextBlock(text="hello "),
            TextBlock(text="world"),
            ToolUseBlock(id="t1", name="bash", input={"cmd": "ls"}),
        ],
    )
    assert message.text == "hello world"
    assert message.reasoning == "thinking"
    assert [t.name for t in message.tool_uses] == ["bash"]


def test_content_blocks_round_trip_through_json() -> None:
    original = Message(
        role=Role.ASSISTANT,
        content=[ReasoningBlock(text="r", signature="sig"), TextBlock(text="t")],
    )
    restored = Message.model_validate(original.model_dump(mode="json"))
    assert restored == original


def test_new_id_is_unique_and_prefixed() -> None:
    ids = [new_id("s_") for _ in range(100)]
    assert all(i.startswith("s_") for i in ids)
    assert len(set(ids)) == 100


def test_new_id_sorts_across_milliseconds() -> None:
    """Ordering holds between milliseconds; within one, the suffix is random."""
    first = new_id()
    time.sleep(0.002)
    second = new_id()
    assert first < second


def test_usage_addition() -> None:
    total = Usage(input_tokens=1, output_tokens=2) + Usage(input_tokens=3, cache_read_tokens=4)
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (4, 2, 4)
    assert total.total_tokens == 6


def test_session_derives_title_from_first_user_message() -> None:
    session = Session()
    session.append(Message.user("  Deploy   the staging   cluster  "))
    assert session.title == "Deploy the staging cluster"
    session.append(Message.assistant("sure"))
    assert session.title == "Deploy the staging cluster"


def test_session_title_is_truncated() -> None:
    session = Session()
    session.append(Message.user("x" * 200))
    assert session.title is not None
    assert len(session.title) == 60
    assert session.title.endswith("…")


def test_accumulator_orders_reasoning_before_text() -> None:
    acc = TurnAccumulator()
    for event in default_events("one two"):
        acc.add(event)
    message = acc.to_message()
    assert message.text == "onetwo"
    assert acc.usage.output_tokens == 3
    assert acc.stop_reason is StopReason.END_TURN


def test_accumulator_records_stream_error() -> None:
    acc = TurnAccumulator()
    acc.add(StreamError(message="boom", retryable=False))
    assert acc.error == "boom"
    assert acc.stop_reason is StopReason.ERROR


async def test_run_turn_folds_events(fake_provider: FakeProvider) -> None:
    acc = await run_turn(fake_provider, Session().to_request())
    assert acc.text == "Hellothere."
    assert len(fake_provider.calls) == 1


def test_backoff_delay_is_bounded() -> None:
    assert all(0 <= backoff_delay(n, base=0.5, cap=10.0) <= 10.0 for n in range(10))


async def test_retry_stream_retries_before_any_output() -> None:
    attempts = 0

    async def factory():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TransientProviderError("503", provider="fake", retry_after=0.0)
        yield TextDelta(text="ok")
        yield MessageEnd()

    events = [e async for e in retry_stream(factory, base=0.0)]
    assert attempts == 3
    assert isinstance(events[0], TextDelta)


async def test_retry_stream_does_not_retry_after_output() -> None:
    attempts = 0

    async def factory():
        nonlocal attempts
        attempts += 1
        yield TextDelta(text="partial")
        raise RateLimitError("429", provider="fake", retry_after=0.0)

    with pytest.raises(RateLimitError):
        async for _ in retry_stream(factory, base=0.0):
            pass
    assert attempts == 1, "replaying after output would duplicate text"


async def test_retry_stream_gives_up_on_non_retryable() -> None:
    async def factory():
        raise AltusError("nope")
        yield  # pragma: no cover

    with pytest.raises(AltusError):
        async for _ in retry_stream(factory, base=0.0):
            pass
