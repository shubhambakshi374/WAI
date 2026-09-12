"""Ctrl+P model switcher, backed by the static catalog so it opens instantly."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from wai.config.secrets import credential_status
from wai.core.types import ModelInfo
from wai.providers.registry import PROVIDER_NAMES, catalog_for


class ModelPicker(ModalScreen[ModelInfo]):
    """Returns the chosen ``ModelInfo``, or ``None`` when dismissed."""

    BINDINGS: ClassVar[list[BindingType]] = [("escape", "dismiss_picker", "Cancel")]

    DEFAULT_CSS = """
    ModelPicker { align: center middle; }
    ModelPicker > Vertical {
        width: 78;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    ModelPicker Label.title { text-style: bold; padding-bottom: 1; }
    ModelPicker Label.hint { color: $text-muted; padding-top: 1; }
    ModelPicker OptionList { height: auto; max-height: 24; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._models: list[ModelInfo] = []

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Select a model", classes="title")
            yield OptionList(*self._build_options(), id="model-options")
            yield Label(
                "Providers without credentials are shown dimmed — wai config set-key <provider>",
                classes="hint",
            )

    def _build_options(self) -> list[Option]:
        options: list[Option] = []
        for provider in PROVIDER_NAMES:
            models = catalog_for(provider)
            if not models:
                continue
            configured = credential_status(provider).available
            for model in models:
                self._models.append(model)
                mark = " " if configured else "·"
                label = f"{mark} {model.label:<30} {provider}"
                options.append(Option(label, id=str(len(self._models) - 1)))
        return options

    def on_mount(self) -> None:
        """Focus the list so arrow keys work without a click."""
        options = self.query_one(OptionList)
        options.focus()
        if options.option_count:
            options.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self.dismiss(self._models[int(event.option.id)])

    def action_dismiss_picker(self) -> None:
        self.dismiss()
