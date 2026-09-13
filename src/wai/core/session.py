"""A conversation, independent of how it is displayed or stored."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field

from wai.core.types import ChatRequest, Message, Role, ToolDef, Usage, new_id


def _now() -> datetime:
    return datetime.now(UTC)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: new_id("s_"))
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    title: str | None = None
    provider: str = ""
    model: str = ""
    system: str | None = None
    base_url: str = ""
    """Self-hosted endpoint, persisted so --resume reconnects to the same one."""
    model_supports_tools: bool = True
    """False disables tool declarations entirely --- see agent.build_system_prompt."""
    workspace_root: str = ""
    """Persisted so --resume reconstructs the same workspace."""
    tools_enabled: bool = True
    max_tokens: int = 4096
    temperature: float | None = None
    messages: list[Message] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

    def append(self, message: Message) -> None:
        self.messages.append(message)
        self.updated_at = _now()
        if self.title is None and message.role is Role.USER:
            self.title = self._derive_title(message.text)

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.updated_at = _now()

    def to_request(self, tools: list[ToolDef] | None = None) -> ChatRequest:
        return ChatRequest(
            model=self.model,
            messages=list(self.messages),
            system=self.system,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=list(tools or []),
        )

    @staticmethod
    def _derive_title(text: str, limit: int = 60) -> str:
        line = " ".join(text.strip().split())
        if not line:
            return "Untitled"
        return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
