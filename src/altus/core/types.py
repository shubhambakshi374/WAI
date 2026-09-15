"""Normalized, provider-agnostic conversation types.

No SDK type ever leaks past this module. Each provider adapter translates its
own wire format into these, and back.
"""

from __future__ import annotations

import secrets
import time
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def new_id(prefix: str = "") -> str:
    """A time-ordered id: millisecond timestamp + random suffix.

    Ids created in different milliseconds sort chronologically. Ids created
    within the same millisecond are unique but unordered relative to each
    other, which is fine for every current caller.
    """
    stamp = format(int(time.time() * 1000), "011x")
    return f"{prefix}{stamp}{secrets.token_hex(5)}"


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class StopReason(StrEnum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    CONTENT_FILTER = "content_filter"
    CANCELLED = "cancelled"
    ERROR = "error"


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Block):
    type: Literal["text"] = "text"
    text: str


class ReasoningBlock(_Block):
    """Extended thinking / reasoning output.

    ``signature`` and ``redacted`` carry provider-opaque data that must be
    echoed back verbatim on the next turn for providers that verify it.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    signature: str | None = None
    redacted: bool = False


class ToolUseBlock(_Block):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(_Block):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str = ""
    is_error: bool = False


class ImageBlock(_Block):
    type: Literal["image"] = "image"
    media_type: str
    data: str
    """Base64-encoded image bytes."""


ContentBlock = Annotated[
    TextBlock | ReasoningBlock | ToolUseBlock | ToolResultBlock | ImageBlock,
    Field(discriminator="type"),
]


class Message(BaseModel):
    role: Role
    content: list[ContentBlock] = Field(default_factory=list)

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role=Role.USER, content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role=Role.ASSISTANT, content=[TextBlock(text=text)])

    @property
    def text(self) -> str:
        """Concatenated text blocks; reasoning and tool blocks are excluded."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def reasoning(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, ReasoningBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class ToolDef(BaseModel):
    """A tool the model may call. Unused in Phase 1; part of the contract."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


class ModelInfo(BaseModel):
    id: str
    provider: str
    display_name: str | None = None
    context_window: int = 0
    max_output_tokens: int = 0
    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False

    @property
    def label(self) -> str:
        return self.display_name or self.id


class ChatRequest(BaseModel):
    """One inference call. Adapters translate this; they never mutate it."""

    model: str
    messages: list[Message]
    system: str | None = None
    max_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolDef] = Field(default_factory=list)
    stop_sequences: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    """Provider-specific passthrough, merged into the request body as-is."""
