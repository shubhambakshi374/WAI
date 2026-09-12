"""The streaming event union every provider adapter normalizes onto.

Phase 1 emits only text, usage and lifecycle events. The tool-call and
reasoning variants are defined now on purpose: they cost nothing today and
they are what stops the Phase 2 agent loop from being a breaking change.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from wai.core.types import StopReason, Usage


class MessageStart(BaseModel):
    type: Literal["message_start"] = "message_start"
    model: str
    usage: Usage = Field(default_factory=Usage)


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ReasoningDelta(BaseModel):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    text: str
    signature: str | None = None


class ToolCallStart(BaseModel):
    type: Literal["tool_call_start"] = "tool_call_start"
    index: int
    id: str
    name: str


class ToolCallDelta(BaseModel):
    """A fragment of the tool's JSON arguments. Accumulate by ``index``."""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int
    partial_json: str


class ToolCallEnd(BaseModel):
    type: Literal["tool_call_end"] = "tool_call_end"
    index: int
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class UsageUpdate(BaseModel):
    type: Literal["usage_update"] = "usage_update"
    usage: Usage


class MessageEnd(BaseModel):
    type: Literal["message_end"] = "message_end"
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = Field(default_factory=Usage)


class StreamError(BaseModel):
    """A failure surfaced in-band, after the stream already produced output.

    Failures *before* the first event are raised as ``WaiError`` instead, so
    that ``retry_stream`` can transparently retry them.
    """

    type: Literal["stream_error"] = "stream_error"
    message: str
    retryable: bool = False


StreamEvent = Annotated[
    MessageStart
    | TextDelta
    | ReasoningDelta
    | ToolCallStart
    | ToolCallDelta
    | ToolCallEnd
    | UsageUpdate
    | MessageEnd
    | StreamError,
    Field(discriminator="type"),
]
