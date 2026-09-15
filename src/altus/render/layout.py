"""Where the nodes go. Backend-agnostic.

This is the part worth having once. It computes positions in whatever unit the
caller's ``Metrics`` are expressed in --- pixels for the image renderer, cells
for the terminal one --- so the two back ends are genuinely the same engine
with different rulers, rather than one reimplementing the other and drifting.

The layout is Reingold-Tilford: leaves pack left to right, parents centre over
their children. Deterministic by construction, which is the whole reason it
was chosen over a force-directed layout. Kubernetes ownership already *is* a
tree; a spring layout would spend its effort rediscovering that and would
rearrange itself every time you looked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from altus.core.visuals import GraphNode, ResourceGraph

#: Workload kinds first, then routing, then the rest. Used for sibling order
#: and for which root subtrees are drawn first.
ROOT_ORDER = (
    "Deployment",
    "StatefulSet",
    "DaemonSet",
    "CronJob",
    "Job",
    "Ingress",
    "Service",
)


def root_rank(kind: str) -> int:
    return ROOT_ORDER.index(kind) if kind in ROOT_ORDER else len(ROOT_ORDER)


@dataclass(frozen=True)
class Metrics:
    """One node's footprint, and the space between them, in the caller's unit."""

    node_width: float
    node_height: float
    gap_x: float
    gap_y: float
    pad: float = 0.0


@dataclass
class Placed:
    node: GraphNode
    depth: int
    x: float = 0.0
    y: float = 0.0
    children: list[Placed] = field(default_factory=list)


def forest(model: ResourceGraph) -> list[Placed]:
    """Ownership edges into a forest.

    Anything with no owner *inside the graph* is a root, which is correct for a
    namespace view where a Deployment's owner is simply not present.
    """
    by_id = {node.id: Placed(node=node, depth=0) for node in model.nodes}
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
    # reshuffles itself every refresh is exactly what a tidy tree avoids.
    for placed in by_id.values():
        placed.children.sort(key=lambda child: (root_rank(child.node.kind), child.node.name))

    roots = sorted(
        (placed for node_id, placed in by_id.items() if node_id not in owned),
        key=lambda placed: (root_rank(placed.node.kind), placed.node.name),
    )

    def set_depth(entry: Placed, depth: int, seen: set[str]) -> None:
        # ownerReferences should never cycle, but a malformed cluster must
        # still draw rather than recurse until the stack runs out.
        if entry.node.id in seen:
            entry.children = []
            return
        entry.depth = depth
        for child in entry.children:
            set_depth(child, depth + 1, seen | {entry.node.id})

    for root in roots:
        set_depth(root, 0, set())
    return roots


def left_edge(entry: Placed) -> float:
    """The leftmost x in a subtree.

    Not the root's own x: a parent is centred *over* its children, so its x is
    the middle of their span. Shifting a subtree by its root's x therefore
    moves it left by half its own width and drops whatever falls below zero.
    """
    return min([entry.x, *(left_edge(child) for child in entry.children)])


def extent(entry: Placed, metrics: Metrics) -> tuple[float, float]:
    right, bottom = entry.x + metrics.node_width, entry.y + metrics.node_height
    for child in entry.children:
        child_right, child_bottom = extent(child, metrics)
        right, bottom = max(right, child_right), max(bottom, child_bottom)
    return right, bottom


def shift(entry: Placed, dx: float, dy: float) -> None:
    entry.x += dx
    entry.y += dy
    for child in entry.children:
        shift(child, dx, dy)


def walk(roots: list[Placed]) -> list[Placed]:
    out: list[Placed] = []

    def visit(entry: Placed) -> None:
        out.append(entry)
        for child in entry.children:
            visit(child)

    for root in roots:
        visit(root)
    return out


def _place_tree(root: Placed, cursor: float, metrics: Metrics) -> float:
    position = cursor

    def place(entry: Placed) -> float:
        nonlocal position
        entry.y = entry.depth * (metrics.node_height + metrics.gap_y)
        if not entry.children:
            entry.x = position
            position += metrics.node_width + metrics.gap_x
            return entry.x
        spans = [place(child) for child in entry.children]
        entry.x = (spans[0] + spans[-1]) / 2
        return entry.x

    place(root)
    return position


def layout(roots: list[Placed], metrics: Metrics, target_ratio: float) -> tuple[float, float]:
    """Place every root subtree, wrapping into rows. Returns the content size.

    Packing every root onto one line is what leaves a wide graph as a thin
    strip across the top of a large surface. Wrapping to roughly the target
    aspect uses the space, and keeps each subtree intact --- a tree split
    across a row boundary would be unreadable.

    Roots that own something come first: the workloads are the structure, and
    the loose Services and ConfigMaps read better gathered after them. The
    sort is stable, so the identity ordering from ``forest`` survives beneath
    that grouping.
    """
    ordered = sorted(roots, key=lambda r: not r.children)
    if not ordered:
        return (metrics.pad * 2, metrics.pad * 2)

    widths: list[float] = []
    for root in ordered:
        end = _place_tree(root, 0.0, metrics)
        widths.append(max(end - metrics.gap_x, metrics.node_width))

    total = sum(widths) + metrics.gap_x * (len(widths) - 1)
    tallest = max(extent(root, metrics)[1] for root in ordered)
    row_limit = max(
        metrics.node_width,
        (total * (tallest + metrics.gap_y) * target_ratio) ** 0.5 if tallest else total,
    )

    x = y = 0.0
    row_height = 0.0
    for root, width in zip(ordered, widths, strict=True):
        if x > 0 and x + width > row_limit:
            x, y = 0.0, y + row_height + metrics.gap_y * 1.6
            row_height = 0.0
        shift(root, x - left_edge(root), y)
        x += width + metrics.gap_x
        row_height = max(row_height, extent(root, metrics)[1] - y)

    width = height = 0.0
    for root in ordered:
        right, bottom = extent(root, metrics)
        width, height = max(width, right), max(height, bottom)
    return width + metrics.pad, height + metrics.pad


def cross_edges(model: ResourceGraph) -> list[tuple[str, str, str]]:
    """Everything that is not ownership, as (source, relation, target).

    These are what keep a topology a graph rather than merely a tree, and each
    back end shows them differently --- arcs where there are pixels, an
    annotated list where there are only cells.
    """
    return [
        (edge.source, edge.relation, edge.target) for edge in model.edges if edge.relation != "owns"
    ]


def caption_with_omissions(caption: str, omitted: int) -> str:
    """Fold "and there is more" into a caption.

    Shared because both back ends drop nodes that will not fit, and a map that
    quietly leaves them out claims a completeness it does not have --- the
    caption goes on counting them, so the reader has no way to know the picture
    is partial.
    """
    if omitted <= 0:
        return caption
    # Leads rather than trails. Captions are elided to the panel width, and a
    # note appended to an already-full one is cut off --- which is how this
    # landed in the output looking exactly like the bug it fixes.
    note = f"{omitted} more not shown — the map is taller than the space"
    return f"{note} · {caption}" if caption else note
