"""Multi-line input. Enter sends, Ctrl+J inserts a newline."""

from __future__ import annotations

from dataclasses import dataclass

from textual import events
from textual.message import Message as TextualMessage
from textual.widgets import TextArea


class Composer(TextArea):
    DEFAULT_CSS = """
    Composer {
        height: auto;
        max-height: 12;
        min-height: 3;
        border: round $primary;
        padding: 0 1;
    }
    Composer:focus { border: round $accent; }
    """

    @dataclass
    class Submitted(TextualMessage):
        text: str

    def __init__(self, **kwargs: object) -> None:
        super().__init__(soft_wrap=True, tab_behavior="focus", **kwargs)  # type: ignore[arg-type]

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
