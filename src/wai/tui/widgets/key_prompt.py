"""Masked API-key entry.

Stores through ``config.secrets.set_api_key``, so the key lands in the OS
keyring and never touches the config file.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label


class KeyPrompt(ModalScreen[bool]):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    KeyPrompt { align: center middle; }
    KeyPrompt > Vertical {
        width: 66; height: auto;
        border: round $accent; background: $surface; padding: 1 2;
    }
    KeyPrompt .title { text-style: bold; }
    KeyPrompt .hint { color: $text-muted; padding-top: 1; }
    """

    def __init__(self, provider: str) -> None:
        super().__init__()
        self.provider = provider

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"API key for {self.provider}", classes="title", markup=False)
            yield Input(password=True, placeholder="paste and press enter", id="key")
            yield Label("Stored in your OS keyring, never on disk.", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#key", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        from wai.config.secrets import set_api_key

        key = event.value.strip()
        if not key:
            self.dismiss(False)
            return
        try:
            set_api_key(self.provider, key)
        except Exception as exc:
            self.notify(f"Could not store the key: {exc}", severity="error", markup=False)
            self.dismiss(False)
            return
        self.notify(f"Stored the {self.provider} key in your keyring.", markup=False)
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
