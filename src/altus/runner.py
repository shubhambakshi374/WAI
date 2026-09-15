"""Headless turn execution.

Sits above ``core`` and ``providers`` and below every front end. The CLI, the
TUI and the Phase 2 workflow engine all drive a turn through here, which is
what keeps the provider layer free of UI concerns.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from altus.core.events import (
    MessageEnd,
    ReasoningDelta,
    StreamError,
    StreamEvent,
    TextDelta,
    ToolCallEnd,
    UsageUpdate,
)
from altus.core.retry import retry_stream
from altus.core.types import (
    ChatRequest,
    ContentBlock,
    Message,
    ReasoningBlock,
    Role,
    StopReason,
    TextBlock,
    ToolUseBlock,
    Usage,
)
from altus.providers.base import Provider


@dataclass
class TurnAccumulator:
    """Folds a stream of events back into a single assistant ``Message``."""

    text: str = ""
    reasoning: str = ""
    signature: str | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: StopReason = StopReason.END_TURN
    error: str | None = None
    tool_uses: list[ToolUseBlock] = field(default_factory=list)

    def add(self, event: StreamEvent) -> None:
        match event:
            case TextDelta():
                self.text += event.text
            case ReasoningDelta():
                self.reasoning += event.text
                if event.signature:
                    self.signature = event.signature
            case ToolCallEnd():
                self.tool_uses.append(ToolUseBlock(id=event.id, name=event.name, input=event.input))
            case UsageUpdate():
                self.usage = event.usage
            case MessageEnd():
                self.stop_reason = event.stop_reason
                if event.usage.total_tokens:
                    self.usage = event.usage
            case StreamError():
                self.error = event.message
                self.stop_reason = StopReason.ERROR

    def to_message(self) -> Message:
        """Reasoning first, then text, then tool calls — provider ordering."""
        blocks: list[ContentBlock] = []
        if self.reasoning or self.signature:
            blocks.append(ReasoningBlock(text=self.reasoning, signature=self.signature))
        if self.text:
            blocks.append(TextBlock(text=self.text))
        blocks.extend(self.tool_uses)
        return Message(role=Role.ASSISTANT, content=blocks)


async def stream_turn(
    provider: Provider,
    request: ChatRequest,
    accumulator: TurnAccumulator | None = None,
    *,
    max_attempts: int = 4,
) -> AsyncIterator[StreamEvent]:
    """Stream one turn, folding events into ``accumulator`` as they pass through."""
    acc = accumulator
    async for event in retry_stream(lambda: provider.stream(request), max_attempts=max_attempts):
        if acc is not None:
            acc.add(event)
        yield event


async def run_turn(
    provider: Provider, request: ChatRequest, *, max_attempts: int = 4
) -> TurnAccumulator:
    """Run a turn to completion and return the folded result."""
    acc = TurnAccumulator()
    async for _ in stream_turn(provider, request, acc, max_attempts=max_attempts):
        pass
    return acc
