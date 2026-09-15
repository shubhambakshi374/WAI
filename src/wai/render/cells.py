"""The topology map drawn in character cells.

The image renderer needs a terminal that speaks Kitty's graphics protocol or
Sixel. Apple's Terminal.app speaks neither, and neither does a plain ssh
session or a CI log --- and ``textual-image``'s halfcell fallback, while it
preserves the *shape* of a graph, turns every label into mush at two pixels per
row.

So this back end draws the same layout with box-drawing characters and leaves
the text as text. Labels stay crisp at any terminal size, because they are
never rasterised. That makes it better than the image for a label-dense
topology even on a terminal that could show one.

What it cannot do is arcs. Two nodes on the same row with a `selects` between
them have nowhere to route a curve through a character grid, so cross-relations
are listed beneath the map instead --- the same compromise the original text
view makes, kept deliberately.

Output is plain data: a grid of ``Cell``. No Textual, no Rich --- the front end
turns cells into whatever its framework wants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from wai.core.visuals import ResourceGraph
from wai.render.layout import Metrics, Placed, cross_edges, forest, layout, walk
from wai.render.palette import Palette

#: A node is four rows: border, kind, name, border. Three would fit only one
#: content line, and dropping either the kind or the name loses the thing that
#: makes the map readable at a glance.
CELLS = Metrics(node_width=22.0, node_height=4.0, gap_x=2.0, gap_y=2.0, pad=0.0)

#: Rounded, because square corners read as a table and these are objects.
TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = "╭", "╮", "╰", "╯"
HORIZONTAL, VERTICAL = "─", "│"
TEE_DOWN, TEE_UP, CROSS = "┬", "┴", "┼"


class Cell(NamedTuple):
    char: str = " "
    fg: str = ""
    bold: bool = False


@dataclass(frozen=True)
class CellHit:
    """A node's rectangle, in columns and rows."""

    box: tuple[int, int, int, int]
    node_id: str
    label: str = ""

    def contains(self, column: int, row: int) -> bool:
        left, top, right, bottom = self.box
        return left <= column <= right and top <= row <= bottom


@dataclass
class CellGrid:
    """Rows of styled cells, plus the hit map."""

    width: int
    height: int
    rows: list[list[Cell]] = field(default_factory=list)
    hits: tuple[CellHit, ...] = ()

    def __post_init__(self) -> None:
        if not self.rows:
            self.rows = [[Cell() for _ in range(self.width)] for _ in range(self.height)]

    def put(self, column: int, row: int, cell: Cell) -> None:
        if 0 <= row < self.height and 0 <= column < self.width:
            self.rows[row][column] = cell

    def write(self, column: int, row: int, text: str, *, fg: str = "", bold: bool = False) -> None:
        for offset, char in enumerate(text):
            self.put(column + offset, row, Cell(char, fg, bold))

    def hit(self, column: int, row: int) -> CellHit | None:
        for entry in reversed(self.hits):
            if entry.contains(column, row):
                return entry
        return None

    def to_text(self) -> str:
        """Unstyled, for tests and for anywhere that cannot carry colour."""
        return "\n".join("".join(cell.char for cell in row).rstrip() for row in self.rows).rstrip()


def _elide(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def draw_graph(
    model: ResourceGraph,
    *,
    width: int,
    height: int,
    palette: Palette,
) -> CellGrid:
    """The same layout the image back end uses, in cells."""
    title_rows = 1 if model.title else 0
    relations = cross_edges(model)
    # Reserve room for the relation list; without it the map would be drawn
    # over the very thing that makes it a graph.
    relation_rows = min(len(relations), 6) + (1 if relations else 0)
    caption_rows = 1 if model.caption else 0

    grid = CellGrid(width=width, height=height)
    if model.title:
        grid.write(0, 0, _elide(model.title, width), fg=palette.ink, bold=True)
    if not model.nodes:
        grid.write(0, title_rows, "(no resources found)", fg=palette.muted)
        return grid

    map_top = title_rows
    map_height = max(4, height - title_rows - relation_rows - caption_rows)

    roots = forest(model)
    layout(roots, CELLS, max(0.2, width / max(1.0, map_height)))
    placed = walk(roots)

    hits: list[CellHit] = []
    for entry in placed:
        left = round(entry.x)
        top = map_top + round(entry.y)
        right = left + int(CELLS.node_width) - 1
        bottom = top + int(CELLS.node_height) - 1
        if left >= width or top >= map_top + map_height:
            continue  # off the visible map; scrolling is the widget's business
        _draw_node(grid, entry, left, top, right, bottom, palette, width)
        hits.append(
            CellHit(
                box=(left, top, min(right, width - 1), bottom),
                node_id=entry.node.id,
                label=f"{entry.node.kind}/{entry.node.name}",
            )
        )
        _draw_connectors(grid, entry, left, bottom, palette, map_top, width)

    row = map_top + map_height
    if relations:
        grid.write(0, row, "relationships", fg=palette.muted)
        names = {node.id: node for node in model.nodes}
        for offset, (source, relation, target) in enumerate(relations[:6]):
            colour, _dashed = palette.relation(relation)
            head = names.get(source)
            tail = names.get(target)
            if head is None or tail is None:
                continue
            line = f"  {head.kind}/{head.name} ─{relation}→ {tail.kind}/{tail.name}"
            grid.write(0, row + 1 + offset, _elide(line, width), fg=colour)
        row += 1 + min(len(relations), 6)
    if model.caption:
        grid.write(0, min(row, height - 1), _elide(model.caption, width), fg=palette.muted)

    grid.hits = tuple(hits)
    return grid


def _draw_node(
    grid: CellGrid,
    entry: Placed,
    left: int,
    top: int,
    right: int,
    bottom: int,
    palette: Palette,
    width: int,
) -> None:
    node = entry.node
    kind_colour = palette.kind(node.kind)
    inner = right - left - 1

    grid.put(left, top, Cell(TOP_LEFT, kind_colour))
    grid.put(right, top, Cell(TOP_RIGHT, kind_colour))
    grid.put(left, bottom, Cell(BOTTOM_LEFT, kind_colour))
    grid.put(right, bottom, Cell(BOTTOM_RIGHT, kind_colour))
    for column in range(left + 1, right):
        grid.put(column, top, Cell(HORIZONTAL, kind_colour))
        grid.put(column, bottom, Cell(HORIZONTAL, kind_colour))
    for row in range(top + 1, bottom):
        grid.put(left, row, Cell(VERTICAL, kind_colour))
        grid.put(right, row, Cell(VERTICAL, kind_colour))

    grid.write(left + 1, top + 1, _elide(node.kind, inner), fg=kind_colour)

    # Status shares the name's line and is right-aligned, so its width comes
    # out of the name's budget first --- the same collision the image back end
    # had to fix, and the same fix.
    status = _elide(node.status, max(0, inner // 2)) if node.status else ""
    grid.write(left + 1, top + 2, _elide(node.name, inner - len(status) - 1), fg=palette.ink)
    if status:
        grid.write(right - len(status), top + 2, status, fg=palette.status(node.status))


def _draw_connectors(
    grid: CellGrid,
    entry: Placed,
    left: int,
    bottom: int,
    palette: Palette,
    map_top: int,
    width: int,
) -> None:
    """The ownership tree's own lines: a stem down, a bus across, a drop to
    each child. Drawn after the boxes so a bus never cuts through one."""
    if not entry.children:
        return
    colour, _dashed = palette.relation("owns")
    centre = left + int(CELLS.node_width) // 2
    bus_row = bottom + 1
    grid.put(centre, bus_row, Cell(VERTICAL, colour))

    child_centres = [round(child.x) + int(CELLS.node_width) // 2 for child in entry.children]
    child_top = map_top + round(entry.children[0].y)
    if len(child_centres) == 1 and child_centres[0] == centre:
        for row in range(bus_row, child_top):
            grid.put(centre, row, Cell(VERTICAL, colour))
        return

    for column in range(min(child_centres), max(child_centres) + 1):
        grid.put(column, bus_row, Cell(HORIZONTAL, colour))
    grid.put(centre, bus_row, Cell(TEE_UP, colour))
    for column in child_centres:
        # A child directly under the parent's stem needs a crossing, not a
        # tee --- writing the tee would erase the up-tick and the bus would
        # look as though it connects to nothing above it.
        junction = CROSS if column == centre else TEE_DOWN
        grid.put(column, bus_row, Cell(junction, colour))
        for row in range(bus_row + 1, child_top):
            grid.put(column, row, Cell(VERTICAL, colour))
