"""Rendering for the ``Visual`` payloads.

The layout decisions --- bar fill, tree connectors, edge grouping --- all live
in ``wai/core/visuals.py`` so the terminal and headless renderings can never
drift. This module only adds colour and Textual widgets on top.
"""

from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import DataTable, Label, Sparkline, Static

from wai.core.visuals import (
    Bars,
    Gauge,
    ResourceGraph,
    Series,
    Table,
    Visual,
    VisualGroup,
)

#: Terminal colours by resource kind, so a type reads at a glance.
KIND_COLOURS: dict[str, str] = {
    "Deployment": "bright_cyan",
    "StatefulSet": "bright_cyan",
    "DaemonSet": "bright_cyan",
    "ReplicaSet": "cyan",
    "Pod": "bright_white",
    "Service": "bright_magenta",
    "Ingress": "magenta",
    "PersistentVolumeClaim": "yellow",
    "PersistentVolume": "yellow",
    "ConfigMap": "bright_black",
    "Secret": "red",
    "Job": "green",
    "CronJob": "green",
    "HorizontalPodAutoscaler": "bright_blue",
    "Node": "blue",
}

UNHEALTHY = (
    "CrashLoopBackOff",
    "Error",
    "Failed",
    "Pending",
    "ImagePullBackOff",
    "ErrImagePull",
    "Evicted",
    "OOMKilled",
    "Terminating",
    "Unknown",
)


def _status_style(status: str) -> str:
    if any(word in status for word in UNHEALTHY):
        return "bold red"
    if "0/" in status:
        return "yellow"
    return "green"


class BarsView(Static):
    """Unicode-block bars. Over-limit bars go red, which is the whole point."""

    DEFAULT_CSS = "BarsView { height: auto; }"

    def __init__(self, model: Bars) -> None:
        super().__init__()
        self.model = model

    def render(self) -> Text:
        text = Text()
        if self.model.title:
            text.append(f"{self.model.title}\n", style="bold")
        for bar in self.model.bars:
            over = bar.limit is not None and bar.limit > 0 and bar.value > bar.limit
            text.append(bar.render() + "\n", style="red" if over else "")
        if not self.model.bars:
            text.append("  (nothing to show)\n", style="dim")
        if self.model.caption:
            text.append(f"  {self.model.caption}\n", style="dim italic")
        return text


class SeriesView(Vertical):
    DEFAULT_CSS = "SeriesView { height: auto; } SeriesView Sparkline { height: 3; }"

    def __init__(self, model: Series) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        if self.model.title:
            yield Label(self.model.title, markup=False)
        if self.model.points:
            yield Sparkline(self.model.points)
        else:
            yield Label("(no data)")
        if self.model.caption:
            yield Label(self.model.caption, markup=False)


class TableView(Vertical):
    DEFAULT_CSS = """
    TableView { height: auto; max-height: 22; }
    TableView DataTable { height: auto; max-height: 18; }
    """

    def __init__(self, model: Table) -> None:
        super().__init__()
        self.model = model

    def compose(self) -> ComposeResult:
        if self.model.title:
            yield Label(self.model.title, markup=False)
        table: DataTable[str] = DataTable(zebra_stripes=True, cursor_type="row")
        yield table
        if self.model.caption:
            yield Label(self.model.caption, markup=False)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(*self.model.columns)
        for row in self.model.rows[:200]:
            # Text(), not str: a cell is cluster output --- a k8s_events message
            # routinely contains brackets, which DataTable would parse as markup.
            table.add_row(*[Text(str(cell)) for cell in row])


class GaugeView(Static):
    DEFAULT_CSS = "GaugeView { height: auto; }"

    def __init__(self, model: Gauge) -> None:
        super().__init__()
        self.model = model

    def render(self) -> Text:
        return Text(self.model.to_text())


class GraphView(Static):
    """The topology tree, coloured by kind and health."""

    DEFAULT_CSS = "GraphView { height: auto; }"

    def __init__(self, model: ResourceGraph) -> None:
        super().__init__()
        self.model = model

    def render(self) -> Text:
        text = Text()
        if self.model.title:
            text.append(f"{self.model.title}\n", style="bold")
        for row in self.model.layout():
            text.append(row.prefix, style="bright_black")
            if row.cross:
                text.append(row.text + "\n", style="dim italic")
                continue
            colour = KIND_COLOURS.get(row.kind, "white")
            if row.status:
                head, _, tail = row.text.partition(row.status)
                text.append(head, style=colour)
                text.append(row.status, style=_status_style(row.status))
                text.append(tail + "\n", style="dim")
            else:
                text.append(row.text + "\n", style=colour)
        if not self.model.nodes:
            text.append("  (no resources found)\n", style="dim")
        if self.model.caption:
            text.append(f"  {self.model.caption}\n", style="dim italic")
        return text


def build_view(model: Visual, *, setting: str = "auto", dark: bool = True) -> Widget:
    """The best rendering this terminal can manage.

    Falls straight through to the text views when graphics are off or the
    variant is not one we draw, so the terminal-only path is unchanged from
    what it has always been.
    """
    from wai.render.capability import Support, resolve

    if resolve(setting) is Support.TEXT:
        return build_text_view(model)
    from wai.tui.widgets.graphics import GraphicsPanel

    if isinstance(model, VisualGroup):
        # A group is several visuals; each one chooses for itself.
        return GroupView(model, setting=setting, dark=dark)
    return GraphicsPanel(model, setting=setting, dark=dark)


def build_text_view(model: Visual) -> Static | Vertical:
    """One renderer per variant. The rendering that needs nothing of the
    terminal beyond colour, and the floor everything else falls back to."""
    match model:
        case Bars():
            return BarsView(model)
        case Series():
            return SeriesView(model)
        case Table():
            return TableView(model)
        case Gauge():
            return GaugeView(model)
        case ResourceGraph():
            return GraphView(model)
        case VisualGroup():
            return GroupView(model)
    return Static(str(model), markup=False)


class GroupView(Vertical):
    DEFAULT_CSS = "GroupView { height: auto; } GroupView > Label { text-style: bold; }"

    def __init__(self, model: VisualGroup, *, setting: str = "auto", dark: bool = True) -> None:
        super().__init__()
        self.model = model
        self.setting = setting
        self.dark = dark

    def compose(self) -> ComposeResult:
        if self.model.title:
            yield Label(self.model.title, markup=False)
        for item in self.model.items:
            yield build_view(item, setting=self.setting, dark=self.dark)


class VisualPanel(Vertical):
    """An inline visual. Enter or click expands it to full screen."""

    DEFAULT_CSS = """
    VisualPanel {
        height: auto;
        margin: 0 1 1 1;
        padding: 0 1;
        border-left: thick $success;
    }
    VisualPanel:focus-within { border-left: thick $accent; }
    VisualPanel > .expand-hint { color: $text-muted; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [Binding("enter", "expand", "Expand")]

    can_focus = True

    def __init__(self, model: Visual, *, setting: str = "auto") -> None:
        super().__init__()
        self.model = model
        self.setting = setting

    def compose(self) -> ComposeResult:
        # The inline preview stays text: it sits in the middle of a scrolling
        # transcript, where a tall drawing pushes the conversation off screen.
        # Expanding is where the room is, so that is where graphics happen.
        yield build_text_view(self.model)
        yield Label("press enter to expand", classes="expand-hint")

    def action_expand(self) -> None:
        setting = getattr(getattr(self.app, "config", None), "ui", None)
        self.app.push_screen(VisualScreen(self.model, setting=getattr(setting, "graphics", "auto")))

    def on_click(self) -> None:
        self.focus()


class VisualScreen(ModalScreen[None]):
    """Full screen. Topology needs the room."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    DEFAULT_CSS = """
    VisualScreen { align: center middle; }
    VisualScreen > Vertical {
        width: 96%; height: 92%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    VisualScreen VerticalScroll { height: 1fr; }
    VisualScreen .hint { color: $text-muted; padding-top: 1; }
    """

    def __init__(self, model: Visual, *, setting: str = "auto") -> None:
        super().__init__()
        self.model = model
        self.setting = setting

    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll():
                yield build_view(self.model, setting=self.setting)
            yield Label(_expand_hint(self.setting), classes="hint", markup=False)

    def action_close(self) -> None:
        self.dismiss()


def _expand_hint(setting: str) -> str:
    """Say how this is being drawn, right where someone is looking at it."""
    from wai.render.capability import explain

    return f"escape to close · {explain(setting)}"
