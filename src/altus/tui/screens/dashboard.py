"""Several views of a namespace at once.

The complaint this answers: a DevOps engineer wants a dashboard, and the chat
transcript gives one picture per tool call, buried between messages. Here the
topology, utilisation, live usage and storage sit on one screen and refresh
together.

It populates itself. Waiting for the agent to happen to call the right four
tools would leave the screen blank on open, so the dashboard runs them --- all
four classify ``READ``, so nothing here prompts and nothing here changes
anything.

Refresh is manual. Polling a cluster on a timer is a decision someone should
make deliberately, not a default they discover from their API server's logs.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Vertical
from textual.screen import Screen
from textual.widgets import Input, Label, Static

from altus.core.visuals import Visual

#: Which tools fill each dashboard, and what to call the panels. Every one is a
#: read: the dashboard can never prompt for approval, which is what makes it
#: safe to populate itself on open.
K8S_PANELS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("topology", "k8s_topology", {}),
    ("requests vs limits", "k8s_usage", {}),
    ("live usage", "k8s_top", {}),
    ("storage", "k8s_storage", {}),
)

AWS_PANELS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("network", "aws_topology", {}),
    ("inventory", "aws_inventory", {}),
    ("identity", "aws_whoami", {}),
    # Cost Explorer bills per request, so it is not on a screen that refreshes.
    ("regions", "aws_regions", {}),
)

#: Kept for callers that predate the AWS panels.
PANELS = K8S_PANELS


class Panel(Vertical):
    """One tool's output, with its own heading and failure state."""

    DEFAULT_CSS = """
    Panel { height: 1fr; border: round $panel; padding: 0 1; }
    Panel > .panel-title { text-style: bold; color: $accent; }
    Panel > .panel-note { color: $text-muted; }
    """

    def __init__(self, title: str, *, setting: str) -> None:
        super().__init__()
        self.title_text = title
        self.setting = setting

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, classes="panel-title", markup=False)
        yield Static("loading…", classes="panel-note", markup=False, id="note")

    async def show(self, visual: Visual | None, note: str) -> None:
        from altus.tui.widgets.visuals import build_view

        await self.remove_children()
        await self.mount(Label(self.title_text, classes="panel-title", markup=False))
        if visual is None:
            await self.mount(Static(note, classes="panel-note", markup=False))
            return
        await self.mount(build_view(visual, setting=self.setting))


class DashboardScreen(Screen[None]):
    """Four panels over one namespace."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("n", "focus_namespace", "Namespace"),
    ]

    DEFAULT_CSS = """
    DashboardScreen { background: $surface; }
    DashboardScreen > Vertical { height: 1fr; padding: 1 2; }
    DashboardScreen .head { height: auto; }
    DashboardScreen .title { text-style: bold; }
    DashboardScreen .hint { color: $text-muted; }
    DashboardScreen Input { width: 32; margin: 1 0; }
    DashboardScreen Grid {
        grid-size: 2 2;
        grid-gutter: 1;
        height: 1fr;
    }
    """

    def __init__(
        self, namespace: str = "default", *, setting: str = "auto", cloud: str = "k8s"
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.setting = setting
        self.cloud = cloud
        self.panels = AWS_PANELS if cloud == "aws" else K8S_PANELS

    def compose(self) -> ComposeResult:
        with Vertical():
            with Vertical(classes="head"):
                yield Label(self._heading(), classes="title", id="heading", markup=False)
                yield Input(
                    value=self.namespace,
                    placeholder="namespace" if self.cloud == "k8s" else "region",
                    id="namespace",
                )
                yield Label(
                    "enter to load · r refresh · click a node to open it · escape to go back",
                    classes="hint",
                    markup=False,
                )
            with Grid():
                for title, _tool, _args in self.panels:
                    yield Panel(title, setting=self.setting)

    def _heading(self) -> str:
        context = getattr(getattr(self.app, "config", None), "cloud", None)
        if self.cloud == "aws":
            where = getattr(context, "default_region", None) or "default region"
            return f"AWS · {where}"
        where = getattr(context, "kube_context", None) or "current context"
        return f"{where} · namespace {self.namespace}"

    def on_mount(self) -> None:
        self.action_refresh()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "namespace":
            return
        self.namespace = event.value.strip() or "default"
        self.query_one("#heading", Label).update(self._heading())
        self.action_refresh()

    def action_focus_namespace(self) -> None:
        self.query_one("#namespace", Input).focus()

    def action_refresh(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        registry = getattr(self.app, "registry", None)
        context = getattr(self.app, "tool_ctx", None)
        panels = list(self.query(Panel))
        if registry is None or context is None:
            for panel in panels:
                await panel.show(None, "no tool session")
            return

        async def run(name: str, args: dict[str, Any]) -> Any:
            if name not in registry:
                return None
            # Every panel is a read, so a failure is worth showing rather than
            # raising: one tool being denied should not blank the rest.
            scope = {} if self.cloud == "aws" else {"namespace": self.namespace}
            return await registry.execute(name, {**args, **scope}, context)

        outcomes = await asyncio.gather(
            *(run(tool, args) for _title, tool, args in self.panels), return_exceptions=True
        )
        for panel, (title, tool, _args), outcome in zip(panels, self.panels, outcomes, strict=True):
            if outcome is None:
                await panel.show(None, f"{tool} is not available in this session")
            elif isinstance(outcome, BaseException):
                await panel.show(None, f"{tool} failed: {outcome}")
            elif outcome.is_error:
                await panel.show(None, outcome.content)
            elif outcome.visual is None:
                await panel.show(None, outcome.content or f"{title}: nothing to show")
            else:
                await panel.show(outcome.visual, "")

    def action_close(self) -> None:
        self.app.pop_screen()
