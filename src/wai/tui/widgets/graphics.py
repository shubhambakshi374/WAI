"""Showing a ``Visual`` as well as this terminal allows.

Three renderings, one decision:

* an **image**, where the terminal speaks Kitty's protocol or Sixel;
* a **cell map** of box-drawing characters, which works everywhere and keeps
  labels crisp because they are never rasterised;
* the **text view** that has always been here, for a dumb terminal or when
  graphics are switched off.

The fallback is the load-bearing part. A terminal that cannot draw must get a
working screen, not an error and not a faceful of escape codes --- so every
step down happens quietly, including when the image library itself fails at
draw time.
"""

from __future__ import annotations

from typing import Any

from rich.style import Style
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.events import Click
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static

from wai.core.visuals import ResourceGraph, Visual
from wai.render import DARK, LIGHT, Palette, render, renderable
from wai.render.capability import Support, resolve

#: A cell is roughly twice as tall as it is wide, so an image asked for in
#: character units has to be scaled by this to come out square.
CELL_ASPECT = 2.1


class NodeSelected(Message):
    """A node was clicked. Carries what drill-down needs and nothing else."""

    def __init__(self, node_id: str, label: str) -> None:
        super().__init__()
        self.node_id = node_id
        self.label = label


class CellMap(Static):
    """A ``CellGrid`` as styled terminal text.

    ``render`` returns a Rich ``Text`` built by appending, so nothing here is
    ever parsed as markup --- a resource name containing a bracket is drawn,
    not interpreted. See tests/test_markup_safety.py.
    """

    DEFAULT_CSS = "CellMap { height: auto; }"

    def __init__(self, model: ResourceGraph, palette: Palette) -> None:
        super().__init__()
        self.model = model
        self.palette = palette
        self._grid: Any = None

    def render(self) -> Text:
        from wai.render.cells import draw_graph

        width = max(20, self.size.width or 80)
        height = max(8, self.size.height or 24)
        self._grid = draw_graph(self.model, width=width, height=height, palette=self.palette)
        text = Text()
        for row in self._grid.rows:
            for cell in row:
                text.append(
                    cell.char,
                    style=Style(color=cell.fg or None, bold=cell.bold) if cell.fg else None,
                )
            text.append("\n")
        return text

    def get_content_height(self, container: Any, viewport: Any, width: int) -> int:
        # Tall enough for the whole map; the panel scrolls rather than clipping.
        from wai.render.cells import CELLS

        depth = len(self.model.nodes) or 1
        return min(60, int(CELLS.node_height) * depth + 8)

    def on_click(self, event: Click) -> None:
        if self._grid is None:
            return
        hit = self._grid.hit(event.x, event.y)
        if hit is not None:
            self.post_message(NodeSelected(hit.node_id, hit.label))


class ImageMap(Widget):
    """An image, with the hit map that makes it clickable."""

    DEFAULT_CSS = "ImageMap { height: auto; }"

    def __init__(self, model: Visual, palette: Palette) -> None:
        super().__init__()
        self.model = model
        self.palette = palette
        self._rendered: Any = None

    def compose(self) -> ComposeResult:
        from textual_image.widget import AutoImage

        width = max(240, (self.size.width or 80) * 8)
        height = max(120, int((self.size.height or 20) * 8 * CELL_ASPECT / 2))
        self._rendered = render(self.model, size=(width, height), palette=self.palette)
        if self._rendered is None:
            yield from _text_view(self.model)
            return

        import io

        # on_error is why this is safe to attempt: if the protocol fails at
        # draw time --- a terminal that claimed support and lied --- the widget
        # swaps itself for the cell map rather than showing a broken screen.
        yield AutoImage(
            io.BytesIO(self._rendered.png),
            on_error=lambda _exc: _fallback(self.model, self.palette),
        )

    def on_click(self, event: Click) -> None:
        if self._rendered is None:
            return
        scale_x = self._rendered.size[0] / max(1, self.size.width)
        scale_y = self._rendered.size[1] / max(1, self.size.height)
        hit = self._rendered.hit(event.x * scale_x, event.y * scale_y)
        if hit is not None:
            self.post_message(NodeSelected(hit.node_id, hit.label))


def _fallback(model: Visual, palette: Palette) -> Widget:
    if isinstance(model, ResourceGraph):
        return CellMap(model, palette)
    from wai.tui.widgets.visuals import build_text_view

    return build_text_view(model)


def _text_view(model: Visual) -> ComposeResult:
    from wai.tui.widgets.visuals import build_text_view

    yield build_text_view(model)


class GraphicsPanel(Vertical):
    """Picks a rendering and stands out of the way."""

    DEFAULT_CSS = "GraphicsPanel { height: auto; }"

    def __init__(self, model: Visual, *, setting: str = "auto", dark: bool = True) -> None:
        super().__init__()
        self.model = model
        self.setting = setting
        self.palette = DARK if dark else LIGHT
        self.support = resolve(setting)

    def compose(self) -> ComposeResult:
        if self.support is Support.TEXT or not renderable(self.model):
            yield from _text_view(self.model)
            return
        if self.support is Support.IMAGE:
            yield ImageMap(self.model, self.palette)
            return
        # Cells draw a topology properly. Everything else --- bars, gauges,
        # series --- is better served by the text views, which were written for
        # exactly this width and already say the numbers out loud.
        if isinstance(self.model, ResourceGraph):
            yield CellMap(self.model, self.palette)
        else:
            yield from _text_view(self.model)
