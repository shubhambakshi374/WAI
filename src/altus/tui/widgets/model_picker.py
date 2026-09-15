"""Pick a model, with a filter that falls through to the provider.

The catalog is a starting point, not the truth: it goes stale faster than
providers ship models. So typing something the list does not contain asks the
provider directly, and a fully-typed id is accepted whether or not anything
lists it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList
from textual.widgets.option_list import Option

from altus.core.types import ModelInfo
from altus.providers.registry import catalog_for, known_models

if TYPE_CHECKING:
    from altus.tui.app import AltusApp


class ModelPicker(ModalScreen[ModelInfo | None]):
    """Returns the chosen ``ModelInfo``, or None when dismissed."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Cancel"),
        Binding("down", "focus_list", "Down", show=False),
    ]

    DEFAULT_CSS = """
    ModelPicker { align: center middle; }
    ModelPicker > Vertical {
        width: 84; max-width: 92%; height: auto; max-height: 84%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    ModelPicker .title { text-style: bold; }
    ModelPicker .hint { color: $text-muted; padding-top: 1; }
    ModelPicker OptionList { height: auto; max-height: 16; }
    ModelPicker Input { margin: 1 0 0 0; }
    """

    def __init__(self, provider: str | None = None) -> None:
        super().__init__()
        self.provider = provider
        self.models: list[ModelInfo] = []
        self._searching = False

    @property
    def altus(self) -> AltusApp:
        return cast("AltusApp", self.app)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Select a model", classes="title")
            yield Input(placeholder="filter, or paste any model id", id="model-filter")
            yield OptionList(id="model-options")
            yield Label("", classes="hint", id="model-note")

    def on_mount(self) -> None:
        scope = self.provider or self.altus.session.provider
        # Start from what this provider is known to offer, then everything
        # else, so the likely answer is at the top but nothing is hidden.
        self.models = [*catalog_for(scope), *(m for m in known_models() if m.provider != scope)]
        self._fill(self.models)
        self.query_one("#model-filter", Input).focus()
        self._note(
            f"{scope}: {len(catalog_for(scope))} known · type to filter, "
            "unlisted ids are looked up live"
        )

    def _note(self, text: str) -> None:
        self.query_one("#model-note", Label).update(text)

    def _fill(self, models: list[ModelInfo]) -> None:
        options = self.query_one("#model-options", OptionList)
        options.clear_options()
        options.add_options(
            [
                Option(f"{m.id:<46} {m.provider:<14} {m.display_name or ''}", id=m.id)
                for m in models[:80]
            ]
        )
        if models:
            options.highlighted = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        self.filter_models(event.value)

    @work(exclusive=True, group="model-filter")
    async def filter_models(self, needle: str) -> None:
        from altus.config.secrets import credential_status
        from altus.providers.registry import search_models

        scope = self.provider or self.altus.session.provider
        if not credential_status(scope).available:
            wanted = needle.strip().casefold()
            self._fill(
                [m for m in self.models if wanted in m.id.casefold()] if wanted else self.models
            )
            self._note(f"{scope} has no credentials, so no live lookup. /setup {scope} to add one.")
            return

        result = await search_models(scope, self.altus.config, self.models, needle)
        self.models = result.known
        self._fill(result.matches)
        if result.note:
            self._note(result.note)
        elif not needle.strip():
            self._note("type to filter, unlisted ids are looked up live")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._accept(event.value.strip())

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self._accept(str(event.option.id or ""))

    def _accept(self, wanted: str) -> None:
        options = self.query_one("#model-options", OptionList)
        if options.option_count:
            index = options.highlighted or 0
            listed = str(options.get_option_at_index(index).id or "")
            # A typed id that nothing lists is still valid --- see the module
            # docstring: the catalog is a starting point, not the truth.
            if not wanted or listed.startswith(wanted):
                wanted = listed
        if not wanted:
            return
        found = next((m for m in self.models if m.id == wanted), None)
        scope = self.provider or self.altus.session.provider
        self.dismiss(found or ModelInfo(id=wanted, provider=scope))

    def action_focus_list(self) -> None:
        self.query_one("#model-options", OptionList).focus()

    def action_close(self) -> None:
        self.dismiss(None)
