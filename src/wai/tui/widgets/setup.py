"""The provider setup wizard.

WAI starts fine with nothing configured; this is how you get from there to a
working session without leaving the app. Four steps: pick a provider, paste a
key, prove it works, choose a default model.

The health check and the model list are **the same call**. Asking the provider
what it serves is an authenticated, read-only request, so a bad key fails here
with the provider's own message rather than on your first real prompt --- and
the reply is exactly the list step four needs.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, ClassVar

from textual import work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, LoadingIndicator, OptionList
from textual.widgets.option_list import Option

from wai.config.secrets import USES_CREDENTIAL_CHAIN, credential_status, set_api_key
from wai.core.types import ModelInfo
from wai.providers import PROVIDER_NAMES

if TYPE_CHECKING:
    from wai.tui.app import WaiApp


class Step(StrEnum):
    PROVIDER = auto()
    KEY = auto()
    CHECKING = auto()
    MODEL = auto()


class SetupWizard(ModalScreen[bool]):
    """Returns True when a provider ends up configured."""

    BINDINGS: ClassVar[list[BindingType]] = [Binding("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    SetupWizard { align: center middle; }
    SetupWizard > Vertical {
        width: 84; max-width: 94%; height: auto; max-height: 90%;
        border: thick $accent; background: $surface; padding: 1 2;
    }
    SetupWizard .title { text-style: bold; }
    SetupWizard .step { color: $text-muted; padding-bottom: 1; }
    SetupWizard .hint { color: $text-muted; padding-top: 1; }
    SetupWizard .problem { color: $error; padding-top: 1; }
    /* A Vertical defaults to 1fr and would stretch the panel to its
       max-height however few options there are. */
    SetupWizard #step-body { height: auto; }
    SetupWizard OptionList { height: auto; max-height: 14; }
    SetupWizard VerticalScroll { height: auto; max-height: 14; }
    SetupWizard Input { margin-top: 1; }
    SetupWizard Horizontal { height: auto; align: right middle; padding-top: 1; }
    SetupWizard Button { margin-left: 1; }
    SetupWizard LoadingIndicator { height: 3; }
    """

    def __init__(self, provider: str | None = None) -> None:
        super().__init__()
        self.provider = provider or ""
        self.step = Step.KEY if provider else Step.PROVIDER
        self.models: list[ModelInfo] = []
        self.problem = ""
        self._searching = False

    @property
    def wai(self) -> WaiApp:
        from typing import cast

        return cast("WaiApp", self.app)

    def compose(self) -> ComposeResult:
        with Vertical(id="body"):
            yield Label("Set up a provider", classes="title")
            yield Label("", id="step-label", classes="step")
            yield Vertical(id="step-body")
            with Horizontal():
                yield Button("Cancel", id="cancel")
                yield Button("Continue", variant="primary", id="next")

    async def on_mount(self) -> None:
        await self._show_step()

    # ------------------------------------------------------------- rendering

    async def _show_step(self) -> None:
        # NOT _render: Widget already owns that name.
        body = self.query_one("#step-body", Vertical)
        await body.remove_children()
        label = self.query_one("#step-label", Label)
        nxt = self.query_one("#next", Button)

        if self.step is Step.PROVIDER:
            label.update("1 of 4 — which provider?")
            options = OptionList(*self._provider_options(), id="providers")
            await body.mount(options)
            options.focus()
            options.highlighted = 0
            nxt.label = "Continue"

        elif self.step is Step.KEY:
            label.update(f"2 of 4 — paste your {self.provider} API key")
            if self.provider in USES_CREDENTIAL_CHAIN:
                await body.mount(
                    Label(
                        f"{self.provider} uses its cloud credential chain "
                        "(AWS_PROFILE, instance roles) rather than an API key.",
                        classes="hint",
                    )
                )
                nxt.label = "Check credentials"
            else:
                await body.mount(Input(password=True, placeholder="sk-…", id="key"))
                await body.mount(Label("Stored in your OS keyring, never on disk.", classes="hint"))
                self.query_one("#key", Input).focus()
                nxt.label = "Check key"
            if self.problem:
                await body.mount(Label(self.problem, classes="problem"))

        elif self.step is Step.CHECKING:
            label.update(f"3 of 4 — asking {self.provider} what it serves…")
            await body.mount(LoadingIndicator())
            nxt.disabled = True

        elif self.step is Step.MODEL:
            label.update(f"4 of 4 — default model  ({len(self.models)} available)")
            await body.mount(Input(placeholder="type to filter, or paste a model id", id="filter"))
            await body.mount(VerticalScroll(OptionList(id="models")))
            await body.mount(
                Label(
                    "Not listed? Keep typing — WAI re-queries the provider.",
                    classes="hint",
                    id="search-note",
                )
            )
            self._fill_models(self.models)
            self.query_one("#filter", Input).focus()
            nxt.disabled = False
            nxt.label = "Finish"

    def _provider_options(self) -> list[Option]:
        options: list[Option] = []
        for name in PROVIDER_NAMES:
            status = credential_status(name)
            mark = "●" if status.available else "○"
            note = "  already configured" if status.available else ""
            options.append(Option(f"{mark} {name}{note}", id=name))
        return options

    def _fill_models(self, models: list[ModelInfo]) -> None:
        options = self.query_one("#models", OptionList)
        options.clear_options()
        options.add_options(
            [Option(f"{m.id}    {m.display_name or ''}", id=m.id) for m in models[:60]]
        )
        if models:
            options.highlighted = 0

    # -------------------------------------------------------------- stepping

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)
        else:
            self.advance()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "key" or event.input.id == "filter":
            self.advance()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.step is Step.PROVIDER:
            self.advance()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter":
            self.filter_models(event.value)

    @work(exclusive=True, group="wizard")
    async def advance(self) -> None:
        if self.step is Step.PROVIDER:
            options = self.query_one("#providers", OptionList)
            index = options.highlighted or 0
            self.provider = str(options.get_option_at_index(index).id or "")
            self.step = Step.KEY
            self.problem = ""
            await self._show_step()
            return

        if self.step is Step.KEY:
            if self.provider not in USES_CREDENTIAL_CHAIN:
                key = self.query_one("#key", Input).value.strip()
                if not key:
                    self.problem = "Paste a key first."
                    await self._show_step()
                    return
                try:
                    set_api_key(self.provider, key)
                except Exception as exc:
                    self.problem = f"Could not store the key in your keyring: {exc}"
                    await self._show_step()
                    return
            self.step = Step.CHECKING
            await self._show_step()
            await self._health_check()
            return

        if self.step is Step.MODEL:
            options = self.query_one("#models", OptionList)
            typed = self.query_one("#filter", Input).value.strip()
            chosen = typed
            if options.option_count:
                index = options.highlighted or 0
                listed = str(options.get_option_at_index(index).id or "")
                # A fully-typed id the list does not know is still valid: the
                # catalog goes stale faster than providers ship models.
                chosen = typed if typed and not listed.startswith(typed) else listed
            if not chosen:
                return
            await self.wai.apply_setup(self.provider, chosen)
            self.dismiss(True)

    async def _health_check(self) -> None:
        # Reached through the registry module, not a re-exported name, so
        # there is a single place to patch or trace.
        from wai.providers import registry
        from wai.providers.registry import catalog_for, merge_models

        try:
            found = await registry.live_models(self.provider, self.wai.config)
        except Exception as exc:
            self.step = Step.KEY
            self.problem = f"{self.provider} rejected that: {_first_line(exc)}"
            self.query_one("#next", Button).disabled = False
            await self._show_step()
            return

        self.models = merge_models(found, catalog_for(self.provider))
        if not self.models:
            # Azure Foundry cannot enumerate deployments; the name is the
            # user's to supply, so let them type it rather than dead-ending.
            self.models = []
        self.step = Step.MODEL
        await self._show_step()

    @work(exclusive=True, group="model-search")
    async def filter_models(self, needle: str) -> None:
        """Same search the model picker uses --- one implementation, one path."""
        from wai.providers.registry import search_models

        note = self.query_one("#search-note", Label)
        if needle.strip() and not self._searching:
            note.update("Searching…")
        self._searching = True
        try:
            result = await search_models(self.provider, self.wai.config, self.models, needle)
        finally:
            self._searching = False
        self.models = result.known
        self._fill_models(result.matches)
        note.update(result.note or "Not listed? Keep typing — WAI re-queries the provider.")

    def action_cancel(self) -> None:
        self.dismiss(False)


def _first_line(exc: BaseException) -> str:
    return str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
