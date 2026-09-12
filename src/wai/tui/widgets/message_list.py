"""The transcript.

Streamed text is buffered and flushed on a timer rather than written per
token: ``Markdown.update`` re-parses the whole document, so per-token updates
make long replies crawl.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, Label, Markdown, Static

from wai.core.types import Message, Role

_ROLE_LABEL = {Role.USER: "you", Role.ASSISTANT: "assistant", Role.SYSTEM: "system"}


class MessageBubble(Static):
    """One message: an optional collapsed reasoning block, then the body."""

    DEFAULT_CSS = """
    MessageBubble { height: auto; margin: 0 0 1 0; }
    MessageBubble > .role {
        color: $text-muted;
        text-style: bold;
        padding: 0 1;
    }
    MessageBubble.-user > .role { color: $accent; }
    MessageBubble.-assistant > .role { color: $success; }
    MessageBubble > Markdown { height: auto; margin: 0; padding: 0 1; background: transparent; }
    MessageBubble > Collapsible { margin: 0 1; border: none; }
    MessageBubble.-error > Markdown { color: $error; }
    """

    def __init__(self, role: Role, text: str = "", reasoning: str = "") -> None:
        super().__init__()
        self.role = role
        self._text = text
        self._reasoning = reasoning
        self.add_class(f"-{role.value}")

    def compose(self) -> ComposeResult:
        yield Label(_ROLE_LABEL.get(self.role, self.role.value), classes="role")
        if self._reasoning:
            with Collapsible(title="reasoning", collapsed=True, id="reasoning-box"):
                yield Markdown(self._reasoning, id="reasoning-body")
        yield Markdown(self._text, id="body")

    def set_text(self, text: str) -> None:
        self._text = text
        if self.is_mounted:
            self.query_one("#body", Markdown).update(text)

    async def set_reasoning(self, text: str) -> None:
        """Create the reasoning block lazily; providers may never send one."""
        self._reasoning = text
        if not self.is_mounted:
            return
        try:
            self.query_one("#reasoning-body", Markdown).update(text)
        except Exception:
            box = Collapsible(title="reasoning", collapsed=True, id="reasoning-box")
            await self.mount(box, before=self.query_one("#body", Markdown))
            await box.mount(Markdown(text, id="reasoning-body"))

    def mark_error(self) -> None:
        self.add_class("-error")


class MessageList(VerticalScroll):
    DEFAULT_CSS = """
    MessageList { padding: 1 1 0 1; height: 1fr; }
    """

    def clear_messages(self) -> None:
        self.remove_children()

    async def add_message(self, message: Message) -> MessageBubble:
        bubble = MessageBubble(message.role, message.text, message.reasoning)
        await self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble

    async def add_placeholder(self, role: Role) -> MessageBubble:
        bubble = MessageBubble(role)
        await self.mount(bubble)
        self.scroll_end(animate=False)
        return bubble
