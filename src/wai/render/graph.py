"""The topology map: an ownership tree with relationships drawn over it.

Why a tidy tree rather than a force-directed layout. Kubernetes ownership *is*
hierarchical --- ``ownerReferences`` make a forest --- so a spring layout
would spend its effort rediscovering a structure we already know, and would
rearrange itself every time you looked at it. A Reingold-Tilford layout is
deterministic: the same namespace draws identically twice, and adding an
unrelated Deployment does not move anything that was already on screen. That
stability is what makes the picture usable for watching something change.

The non-ownership edges --- selects, routes-to, mounts, uses, scales --- are
arcs laid over the tree. They are what keep this a graph rather than merely a
tree, which is exactly the distinction ``ResourceGraph`` makes in text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from wai.core.visuals import GraphNode, ResourceGraph
from wai.render.canvas import Canvas, View
from wai.render.palette import Palette

if TYPE_CHECKING:
    from wai.render import Hit

NODE_WIDTH = 132.0
NODE_HEIGHT = 34.0
GAP_X = 18.0
GAP_Y = 30.0
PAD = 16.0


@dataclass
class _Placed:
    node: GraphNode
    depth: int
    x: float = 0.0
    y: float = 0.0
    children: list[_Placed] = field(default_factory=list)


def _forest(model: ResourceGraph) -> list[_Placed]:
    """Ownership edges into a forest, preserving the model's node order.

    Anything with no owner inside the graph is a root --- which is correct for
    a namespace view, where a Deployment's owner is simply not present.
    """
    by_id = {node.id: _Placed(node=node, depth=0) for node in model.nodes}
    owned: set[str] = set()
    for edge in model.edges:
        if edge.relation != "owns":
            continue
        parent, child = by_id.get(edge.source), by_id.get(edge.target)
        if parent is None or child is None or child.node.id in owned:
            continue
        parent.children.append(child)
        owned.add(child.node.id)

    # Siblings sort by identity, not by the order edges arrived in. The API
    # does not promise a stable order between polls, and a topology that
    # reshuffles itself every refresh is exactly what choosing a tidy tree over
    # a force-directed layout was meant to avoid.
    for placed in by_id.values():
        placed.children.sort(key=lambda child: (_root_rank(child.node.kind), child.node.name))

    roots = sorted(
        (placed for node_id, placed in by_id.items() if node_id not in owned),
        key=lambda placed: (_root_rank(placed.node.kind), placed.node.name),
    )

    def set_depth(entry: _Placed, depth: int, seen: set[str]) -> None:
        # A cycle in ownerReferences should not be possible, but a malformed
        # cluster should still draw rather than recurse forever.
        if entry.node.id in seen:
            entry.children = []
            return
        entry.depth = depth
        for child in entry.children:
            set_depth(child, depth + 1, seen | {entry.node.id})

    for root in roots:
        set_depth(root, 0, set())
    return roots


def _extent(entry: _Placed) -> tuple[float, float]:
    right, bottom = entry.x + NODE_WIDTH, entry.y + NODE_HEIGHT
    for child in entry.children:
        child_right, child_bottom = _extent(child)
        right, bottom = max(right, child_right), max(bottom, child_bottom)
    return right, bottom


def _shift(entry: _Placed, dx: float, dy: float) -> None:
    entry.x += dx
    entry.y += dy
    for child in entry.children:
        _shift(child, dx, dy)


def _place_tree(root: _Placed, cursor: float) -> float:
    """One subtree, packed from ``cursor``. Leaves fill left to right and
    parents centre over their children --- Reingold-Tilford at this scale."""
    position = cursor

    def place(entry: _Placed) -> float:
        nonlocal position
        entry.y = entry.depth * (NODE_HEIGHT + GAP_Y)
        if not entry.children:
            entry.x = position
            position += NODE_WIDTH + GAP_X
            return entry.x
        spans = [place(child) for child in entry.children]
        entry.x = (spans[0] + spans[-1]) / 2
        return entry.x

    place(root)
    return position


def _layout(roots: list[_Placed], target_ratio: float) -> tuple[float, float]:
    """Place every root subtree, wrapping into rows.

    Packing every root onto one line is what leaves a wide graph as a thin
    strip across the top of a large canvas. Wrapping to roughly the canvas
    aspect uses the space instead, and keeps each subtree intact --- a tree
    split across a row boundary would be unreadable.

    Roots that own something come first: the workloads are the structure, and
    the loose Services and ConfigMaps read better gathered after them.
    """
    # Stable sort, so the identity ordering from _forest survives underneath
    # the grouping of owners before loose resources.
    ordered = sorted(roots, key=lambda r: not r.children)
    if not ordered:
        return (PAD * 2, PAD * 2)

    widths: list[float] = []
    for root in ordered:
        end = _place_tree(root, 0.0)
        widths.append(max(end - GAP_X, NODE_WIDTH))

    total = sum(widths) + GAP_X * (len(widths) - 1)
    tallest = max(_extent(root)[1] for root in ordered)
    # Choose a row width that lands near the canvas aspect rather than the
    # widest possible line.
    row_limit = max(
        NODE_WIDTH, (total * (tallest + GAP_Y) * target_ratio) ** 0.5 if tallest else total
    )

    x = y = 0.0
    row_height = 0.0
    for root, width in zip(ordered, widths, strict=True):
        if x > 0 and x + width > row_limit:
            x, y = 0.0, y + row_height + GAP_Y * 1.6
            row_height = 0.0
        _shift(root, x - root.x + (width - NODE_WIDTH) / 2 if not root.children else x, y)
        x += width + GAP_X
        row_height = max(row_height, _extent(root)[1] - y)

    width = height = 0.0
    for root in ordered:
        right, bottom = _extent(root)
        width, height = max(width, right), max(height, bottom)
    return width + PAD, height + PAD


def _root_rank(kind: str) -> int:
    """Workload kinds first, then routing, then the rest."""
    order = ("Deployment", "StatefulSet", "DaemonSet", "CronJob", "Job", "Ingress", "Service")
    return order.index(kind) if kind in order else len(order)


def _walk(roots: list[_Placed]) -> list[_Placed]:
    out: list[_Placed] = []

    def visit(entry: _Placed) -> None:
        out.append(entry)
        for child in entry.children:
            visit(child)

    for root in roots:
        visit(root)
    return out


def draw_graph(
    model: ResourceGraph, canvas: Canvas, palette: Palette, view: View
) -> tuple[Hit, ...]:
    """Draw, and return the hit regions drill-down needs."""
    from wai.render import Hit

    if not model.nodes:
        canvas.text((PAD, PAD), model.title or "topology", fill=palette.ink, size=12)
        canvas.text((PAD, PAD + 22), "(no resources found)", fill=palette.muted, size=10)
        return ()

    top = PAD
    if model.title:
        canvas.text((PAD, PAD), model.title, fill=palette.ink, size=12)
        top = PAD + 22

    roots = _forest(model)
    content_width, content_height = _layout(
        roots, (canvas.width - 2 * PAD) / max(1.0, canvas.height - top - PAD)
    )
    placed = _walk(roots)
    positions = {entry.node.id: entry for entry in placed}

    # Fit to the canvas, then apply the view's own zoom on top, so a fresh
    # render always shows everything and zooming is a deliberate act.
    usable_width = canvas.width - 2 * PAD
    usable_height = canvas.height - top - PAD
    fit = min(usable_width / content_width, usable_height / content_height, 1.0)
    scale = fit * view.scale
    offset_x = PAD + view.offset[0]
    offset_y = top + view.offset[1]

    def screen(entry: _Placed) -> tuple[float, float, float, float]:
        left = offset_x + entry.x * scale
        node_top = offset_y + entry.y * scale
        return (left, node_top, left + NODE_WIDTH * scale, node_top + NODE_HEIGHT * scale)

    # Ownership first, so relation arcs sit on top of the backbone.
    for entry in placed:
        parent_box = screen(entry)
        for child in entry.children:
            child_box = screen(child)
            start = ((parent_box[0] + parent_box[2]) / 2, parent_box[3])
            end = ((child_box[0] + child_box[2]) / 2, child_box[1])
            colour, _dashed = palette.relation("owns")
            elbow = (start[1] + end[1]) / 2
            canvas.line(
                [start, (start[0], elbow), (end[0], elbow), end],
                fill=colour,
                width=1.0,
            )

    for edge in model.edges:
        if edge.relation == "owns":
            continue
        source, target = positions.get(edge.source), positions.get(edge.target)
        if source is None or target is None:
            continue
        source_box, target_box = screen(source), screen(target)
        colour, dashed = palette.relation(edge.relation)
        start = ((source_box[0] + source_box[2]) / 2, source_box[1])
        finish = ((target_box[0] + target_box[2]) / 2, target_box[1])
        # Clamp the arc so it cannot sail off the top of the canvas, which is
        # what happened to every edge between two nodes on the first row.
        headroom = max(4.0, min(start[1], finish[1]) - top)
        canvas.curve(
            start,
            finish,
            fill=colour,
            lift=min(26 * scale, headroom * 0.8),
            width=1.0,
            dashed=dashed,
        )

    hits: list[Hit] = []
    for entry in placed:
        box = screen(entry)
        if box[2] < 0 or box[0] > canvas.width or box[3] < 0 or box[1] > canvas.height:
            continue  # panned out of sight; no point drawing or hit-testing it
        node = entry.node
        kind_colour = palette.kind(node.kind)
        canvas.rect(box, fill=palette.surface, outline=kind_colour, radius=4 * scale, width=1)
        # A status stripe down the leading edge: health readable at a glance,
        # without relying on reading the status text at small scales.
        if node.status:
            canvas.rect(
                (box[0], box[1], box[0] + 3 * scale, box[3]), fill=palette.status(node.status)
            )

        # Type sizes scale with the boxes. Leaving them fixed is how the
        # labels came to sit on top of their own borders at small scales.
        inner = box[2] - box[0] - 14 * scale
        kind_size = max(6, round(9 * scale))
        name_size = max(7, round(10 * scale))
        status_size = max(6, round(8 * scale))

        # The status sits on the same line as the name, so its width has to be
        # taken out of the name's budget first --- otherwise a long status like
        # CrashLoopBackOff is drawn straight over the name it describes.
        status = ""
        status_width = 0.0
        if node.status and scale > 0.75:
            status = canvas.elide(node.status, inner * 0.7, size=status_size)
            status_width = canvas.measure(status, size=status_size) + 8 * scale

        if scale > 0.45:
            canvas.text(
                (box[0] + 7 * scale, box[1] + 6 * scale),
                canvas.elide(node.kind, inner, size=kind_size),
                fill=kind_colour,
                size=kind_size,
            )
            canvas.text(
                (box[0] + 7 * scale, box[1] + 6 * scale + kind_size + 3 * scale),
                canvas.elide(node.name, max(8.0, inner - status_width), size=name_size),
                fill=palette.ink,
                size=name_size,
            )
        if status:
            canvas.text(
                (box[2] - 6 * scale, box[3] - 5 * scale),
                status,
                fill=palette.status(node.status),
                size=status_size,
                anchor="rs",
            )
        hits.append(Hit(box=box, node_id=node.id, label=f"{node.kind}/{node.name}"))

    if model.caption:
        canvas.text((PAD, canvas.height - PAD - 10), model.caption, fill=palette.muted, size=9)
    return tuple(hits)
