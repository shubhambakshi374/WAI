"""Drawing ``Visual`` models as images.

Headless: no Textual import anywhere in this package, which
``tests/test_layering.py`` enforces. That is what lets golden-image tests run
with no terminal, and what would let a second front end reuse this unchanged.

``render`` returns ``None`` for anything it would draw *worse* than the
existing text view --- a ``Table`` belongs in a real ``DataTable``, not a
bitmap of one. The caller reads ``None`` as "fall back", which is also the
answer on a terminal with no graphics support.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from wai.core.visuals import (
    Bars,
    Chart,
    Gauge,
    ResourceGraph,
    Series,
    Visual,
    VisualGroup,
)
from wai.render.canvas import Canvas, View
from wai.render.palette import DARK, LIGHT, Palette

DEFAULT_SIZE = (720, 320)


@dataclass(frozen=True)
class Hit:
    """A clickable region, and what it stands for.

    The whole of drill-down rests on this: the widget knows only pixels, and
    ``node_id`` is already ``apiVersion/Kind/namespace/name``, so a click
    turns into a ``k8s_get`` without the front end parsing anything.
    """

    box: tuple[float, float, float, float]
    node_id: str
    label: str = ""

    def contains(self, x: float, y: float) -> bool:
        left, top, right, bottom = self.box
        return left <= x <= right and top <= y <= bottom


@dataclass(frozen=True)
class Rendered:
    png: bytes
    size: tuple[int, int]
    hits: tuple[Hit, ...] = field(default_factory=tuple)

    def hit(self, x: float, y: float) -> Hit | None:
        """Topmost region under a point. Later hits win, matching draw order."""
        for entry in reversed(self.hits):
            if entry.contains(x, y):
                return entry
        return None


def render(
    visual: Visual,
    *,
    size: tuple[int, int] = DEFAULT_SIZE,
    palette: Palette = DARK,
    view: View | None = None,
    font_override: str = "",
) -> Rendered | None:
    """Draw ``visual``, or return None when the text view is the better one."""
    from wai.render import charts, graph

    width, height = size
    if width < 80 or height < 60:
        return None
    resolved = view or View()
    canvas = Canvas(width, height, palette.background)
    hits: tuple[Hit, ...] = ()

    match visual:
        case Bars():
            charts.draw_bars(visual, canvas, palette, resolved)
        case Gauge():
            charts.draw_gauge(visual, canvas, palette, resolved)
        case Series():
            charts.draw_series(visual, canvas, palette, resolved)
        case Chart():
            charts.draw_chart(visual, canvas, palette, resolved)
        case ResourceGraph():
            hits = graph.draw_graph(visual, canvas, palette, resolved)
        case _:
            # Table and VisualGroup. A table drawn as pixels loses selection,
            # sorting and copy-paste and gains nothing; a group is several
            # visuals and is the caller's to lay out.
            return None

    return Rendered(png=canvas.to_png(), size=size, hits=hits)


def renderable(visual: Visual) -> bool:
    """Whether ``render`` would draw this, without drawing it."""
    return isinstance(visual, Bars | Gauge | Series | Chart | ResourceGraph)


__all__ = [
    "DARK",
    "DEFAULT_SIZE",
    "LIGHT",
    "Hit",
    "Palette",
    "Rendered",
    "View",
    "VisualGroup",
    "render",
    "renderable",
]
