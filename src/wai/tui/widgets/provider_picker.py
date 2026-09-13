"""Pick a provider, and set one up if it is not ready.

A list you cannot act on is a dead end: the old `/provider` printed eight
names and a hint, leaving you to work out the next command yourself. Choosing
a configured provider switches to it; choosing an unconfigured one opens the
setup wizard for that provider, so the list is the route rather than a signpost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, cast

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, OptionList
from textual.widgets.option_list import Option

from wai.providers import PROVIDER_NAMES

if TYPE_CHECKING:
    from wai.tui.app import WaiApp


class ProviderPicker(ModalScreen[str | None]):
    """Dismisses with the chosen provider name, or None."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "close", "Cancel")]

    DEFAULT_CSS = """
    ProviderPicker { align: center middle; }
    ProviderPicker > Vertical {
        width: 76; max-width: 92%; height: auto; max-height: 80%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    ProviderPicker .title { text-style: bold; padding-bottom: 1; }
    ProviderPicker .hint { color: $text-muted; padding-top: 1; }
    ProviderPicker OptionList { height: auto; max-height: 16; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._names: list[str] = []

    @property
    def wai(self) -> WaiApp:
        return cast("WaiApp", self.app)

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Switch provider", classes="title")
            yield OptionList(*self._options(), id="provider-options")
            yield Label(
                "enter to choose · unconfigured providers open setup · esc to cancel",
                classes="hint",
            )

    def _options(self) -> list[Option]:
        from wai.config.secrets import USES_CREDENTIAL_CHAIN, credential_status
        from wai.providers.local import LocalProvider, configured_endpoints

        endpoints = dict(configured_endpoints(self.wai.config))
        current = self.wai.session.provider
        options: list[Option] = []
        for name in PROVIDER_NAMES:
            status = credential_status(name)
            self._names.append(name)
            if name == LocalProvider.name and endpoints:
                options.append(
                    Option(
                        f"● {name:<15} {'ready':<22} "
                        f"{next(iter(endpoints.values()))}{' ← current' if name == current else ''}",
                        id=name,
                    )
                )
                continue
            if status.available:
                state = "ready"
                detail = status.source
            elif name in USES_CREDENTIAL_CHAIN:
                state = "no cloud credentials"
                detail = ""
            else:
                state = "not set up"
                detail = "enter to configure"
            mark = "●" if status.available else "○"
            here = " ← current" if name == current else ""
            options.append(Option(f"{mark} {name:<15} {state:<22} {detail}{here}", id=name))
        return options

    def on_mount(self) -> None:
        options = self.query_one("#provider-options", OptionList)
        options.focus()
        current = self.wai.session.provider
        options.highlighted = self._names.index(current) if current in self._names else 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id or "") or None)

    def action_close(self) -> None:
        self.dismiss(None)
