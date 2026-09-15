"""Top bar: provider, model, session token totals, streaming state."""

from __future__ import annotations

import contextlib

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.reactive import reactive
from textual.widgets import Label

from wai.core.types import Usage


class StatusBar(Horizontal):
    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        background: $panel;
        color: $text-muted;
        padding: 0 1;
    }
    StatusBar #status-model { width: 1fr; color: $text; text-style: bold; }
    StatusBar #status-usage { width: auto; }
    StatusBar #status-granted { width: auto; padding-left: 2; color: $warning; }
    StatusBar #status-state { width: auto; padding-left: 2; }
    """

    provider: reactive[str] = reactive("")
    model: reactive[str] = reactive("")
    usage: reactive[Usage] = reactive(Usage, always_update=True)
    state: reactive[str] = reactive("ready")
    granted: reactive[str] = reactive("")
    """Tools with a standing session approval. Visible so it is never forgotten."""

    def compose(self) -> ComposeResult:
        # Seed from the current values. Watchers fire before compose runs, so
        # relying on them alone leaves the labels blank until first assignment.
        yield Label(self._model_text(), id="status-model", markup=False)
        yield Label(self._usage_text(), id="status-usage", markup=False)
        yield Label(self._granted_text(), id="status-granted", markup=False)
        yield Label(self.state, id="status-state", markup=False)

    def _model_text(self) -> str:
        return f"{self.provider} · {self.model}"

    def _usage_text(self) -> str:
        return f"{self.usage.input_tokens:,} in / {self.usage.output_tokens:,} out"

    def _set(self, selector: str, text: str) -> None:
        """Update a label, tolerating watchers that fire before compose."""
        with contextlib.suppress(NoMatches):
            self.query_one(selector, Label).update(text)

    def watch_provider(self) -> None:
        self._set("#status-model", self._model_text())

    def watch_model(self) -> None:
        self._set("#status-model", self._model_text())

    def watch_usage(self) -> None:
        self._set("#status-usage", self._usage_text())

    def watch_state(self) -> None:
        self._set("#status-state", self.state)

    def _granted_text(self) -> str:
        return f"⚠ auto: {self.granted}" if self.granted else ""

    def watch_granted(self) -> None:
        self._set("#status-granted", self._granted_text())
