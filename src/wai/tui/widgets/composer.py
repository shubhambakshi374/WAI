"""Multi-line input. Enter sends, Ctrl+J inserts a newline."""

from __future__ import annotations

from dataclasses import dataclass

from textual import events
from textual.message import Message as TextualMessage
from textual.widgets import TextArea

from wai.tui.widgets.command_suggest import CommandSuggestions


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
        #: Set by ChatScreen once both are mounted. Keys reach the focused
        #: TextArea first, so the popup cannot intercept them on its own.
        self.suggestions: CommandSuggestions | None = None

    def _offering(self) -> bool:
        return self.suggestions is not None and bool(self.suggestions.suggestions)

    def _complete(self) -> None:
        """Replace what has been typed with the highlighted command."""
        if self.suggestions is None:
            return
        chosen = self.suggestions.highlighted
        if chosen is None:
            return
        self.text = chosen.completion + " "
        self.move_cursor(self.document.end)
        self.suggestions.hide()

    async def _on_key(self, event: events.Key) -> None:
        offering = self._offering()

        if offering and event.key in ("up", "down"):
            event.prevent_default()
            event.stop()
            assert self.suggestions is not None
            self.suggestions.move(-1 if event.key == "up" else 1)
            return

        if offering and event.key in ("tab", "shift+tab"):
            event.prevent_default()
            event.stop()
            self._complete()
            return

        if event.key == "escape" and self.suggestions is not None and self.suggestions.visible_now:
            event.prevent_default()
            event.stop()
            self.suggestions.hide()
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            # Complete when there is something to complete, otherwise send.
            # Typing `/help` in full and pressing enter should just run it.
            if offering and self.suggestions is not None:
                chosen = self.suggestions.highlighted
                if chosen is not None and chosen.completion != self.text.strip():
                    self._complete()
                    return
                self.suggestions.hide()
            text = self.text.strip()
            if text:
                self.post_message(self.Submitted(text))
                self.clear()
        elif event.key == "ctrl+j":
            event.prevent_default()
            event.stop()
            self.insert("\n")
