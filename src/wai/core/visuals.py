"""Inert descriptions of things worth drawing.

The point of this module: **the model gets a text summary, the human gets the
chart.** A tool returns compact prose in ``ToolOutcome.content`` and a
``Visual`` beside it. Only the prose reaches the LLM, so a full cluster
topology costs almost no context, while the TUI still renders something
readable.

Every variant implements ``to_text()`` so headless callers --- ``wai chat
--once``, and the Phase 3 workflow engine --- get a usable rendering with no
terminal at all. The graph layout lives here rather than in the TUI for the
same reason: it is the only correct place if both front ends need it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

FULL = "█"
EMPTY = "░"
OVER = "▓"
DEFAULT_WIDTH = 28


def _bar(fraction: float, width: int = DEFAULT_WIDTH, *, over: bool = False) -> str:
    fraction = max(0.0, min(1.0, fraction))
    filled = round(fraction * width)
    return (OVER if over else FULL) * filled + EMPTY * (width - filled)


def _pct(value: float, total: float) -> str:
    return f"{value / total * 100:.0f}%" if total else "—"


class Bar(BaseModel):
    label: str
    value: float
    limit: float | None = None
    """A real ceiling --- a resource limit or quota. Shown as `value / limit`
    with a percentage, because against a limit a percentage is meaningful."""
    scale: float | None = None
    """Bar length only, never displayed. For comparing sizes against each
    other when there is no ceiling: a percentage there would be read as a fill
    level and would be a lie."""
    request: float | None = None
    """A second marker, shown alongside: what was requested versus used."""
    unit: str = ""
    note: str = ""

    def render(self, width: int = DEFAULT_WIDTH, label_width: int = 28) -> str:
        limit = self.limit
        ceiling = limit or self.scale or max(self.value, self.request or 0.0)
        over = limit is not None and limit > 0 and self.value > limit
        bar = _bar(self.value / ceiling if ceiling else 0.0, width, over=over)
        amount = f"{_fmt(self.value)}{self.unit}"
        if self.limit:
            amount += f" / {_fmt(self.limit)}{self.unit}  {_pct(self.value, self.limit)}"
        extra = f"  req {_fmt(self.request)}{self.unit}" if self.request is not None else ""
        warn = "  ⚠ over limit" if over else ""
        note = f"  {self.note}" if self.note else ""
        label = _shorten(self.label, label_width)
        return f"  {label:<{label_width}} {bar} {amount}{extra}{warn}{note}"


def _shorten(text: str, width: int) -> str:
    """Truncate in the middle. `elasticsearch-data-esm-cluster-0` and `-1`
    share a prefix, so cutting the tail makes them indistinguishable."""
    if len(text) <= width:
        return text
    head = (width - 1) // 2
    tail = width - 1 - head
    return f"{text[:head]}…{text[-tail:]}"


def _fmt(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000 and value == int(value):
        return f"{int(value):,}"
    return f"{value:g}"


class Bars(BaseModel):
    """The workhorse: requests vs limits vs actual, storage, quota headroom."""

    type: Literal["bars"] = "bars"
    title: str = ""
    bars: list[Bar] = Field(default_factory=list)
    caption: str = ""
    label_width: int = 28

    def to_text(self) -> str:
        lines = [self.title] if self.title else []
        lines += [bar.render(label_width=self.label_width) for bar in self.bars] or [
            "  (nothing to show)"
        ]
        if self.caption:
            lines.append(f"  {self.caption}")
        return "\n".join(lines)


class Series(BaseModel):
    type: Literal["series"] = "series"
    title: str = ""
    points: list[float] = Field(default_factory=list)
    at: list[float] = Field(default_factory=list)
    """Unix timestamps, one per point. Optional and additive.

    Without them there is no time axis, only a shape --- a sparkline can say
    "it went up" and cannot say when or how fast. Length is not enforced by the
    schema; ``timed`` is the check, so a mismatched pair degrades to an
    unlabelled shape rather than raising at render time.
    """
    unit: str = ""
    caption: str = ""
    label: str = ""
    """Names this line when several share a Chart's axes."""

    @property
    def timed(self) -> bool:
        return len(self.at) == len(self.points) and len(self.points) > 1

    def to_text(self) -> str:
        if not self.points:
            return f"{self.title}\n  (no data)".strip()
        blocks = "▁▂▃▄▅▆▇█"
        low, high = min(self.points), max(self.points)
        span = (high - low) or 1.0
        spark = "".join(blocks[min(7, int((p - low) / span * 7))] for p in self.points)
        head = f"{self.title}\n" if self.title else ""
        return f"{head}  {spark}  {_fmt(low)} to {_fmt(high)}{self.unit}" + (
            f"\n  {self.caption}" if self.caption else ""
        )


class Table(BaseModel):
    type: Literal["table"] = "table"
    title: str = ""
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: str = ""

    def to_text(self, max_rows: int = 40) -> str:
        if not self.columns:
            return self.title or "(empty table)"
        shown = self.rows[:max_rows]
        widths = [
            max(len(self.columns[i]), *(len(str(r[i])) for r in shown))
            if shown
            else len(self.columns[i])
            for i in range(len(self.columns))
        ]
        lines = [self.title] if self.title else []
        lines.append(
            "  " + "  ".join(c.ljust(w) for c, w in zip(self.columns, widths, strict=True))
        )
        lines.append("  " + "  ".join("─" * w for w in widths))
        lines += [
            "  " + "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths, strict=True))
            for row in shown
        ]
        if len(self.rows) > max_rows:
            lines.append(f"  … {len(self.rows) - max_rows} more rows")
        if self.caption:
            lines.append(f"  {self.caption}")
        return "\n".join(lines)


class Gauge(BaseModel):
    type: Literal["gauge"] = "gauge"
    title: str = ""
    value: float = 0.0
    total: float = 0.0
    unit: str = ""
    caption: str = ""

    def to_text(self) -> str:
        bar = _bar(self.value / self.total if self.total else 0.0)
        head = f"{self.title}\n" if self.title else ""
        return (
            f"{head}  {bar} {_fmt(self.value)}{self.unit} / {_fmt(self.total)}{self.unit}"
            f"  {_pct(self.value, self.total)}" + (f"\n  {self.caption}" if self.caption else "")
        )


# ------------------------------------------------------------------ the graph

OWNS = "owns"

#: Marker per kind, so a node's type is readable at a glance.
KIND_MARKS: dict[str, str] = {
    "Namespace": "▣",
    "Deployment": "◆",
    "StatefulSet": "◆",
    "DaemonSet": "◆",
    "ReplicaSet": "◇",
    "Pod": "●",
    "Service": "◈",
    "Ingress": "⬡",
    "PersistentVolumeClaim": "▤",
    "PersistentVolume": "▥",
    "ConfigMap": "▢",
    "Secret": "▨",
    "Job": "◐",
    "CronJob": "◑",
    "HorizontalPodAutoscaler": "⇅",
    "Node": "▪",
}


class GraphNode(BaseModel):
    id: str
    """Stable identity: `Kind/namespace/name`, e.g. `Deployment/shop/web`.

    Built by ``cloud.k8s.node_id``. Deliberately not apiVersion-qualified:
    an apiVersion contains a slash of its own, and the id has to stay
    splittable by a front end turning a click into a lookup."""
    kind: str
    name: str
    namespace: str = ""
    status: str = ""
    detail: str = ""

    @property
    def mark(self) -> str:
        return KIND_MARKS.get(self.kind, "○")

    @property
    def label(self) -> str:
        base = f"{self.mark} {self.kind}/{self.name}"
        if self.status:
            base += f"  {self.status}"
        return f"{base}  {self.detail}" if self.detail else base


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    """owns · selects · routes-to · mounts · uses · scales."""


class ResourceGraph(BaseModel):
    """The birds-eye view.

    Rendered as an ownership forest with non-ownership links annotated in
    place. Kubernetes topology really is hierarchical --- ownerReferences make
    a tree --- and a node-link diagram stops being readable past about twenty
    nodes in eighty columns. The cross-edges are what keep it a graph rather
    than merely a tree.
    """

    type: Literal["graph"] = "graph"
    title: str = ""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    caption: str = ""

    def by_id(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}

    def roots(self) -> list[GraphNode]:
        """Nodes nothing owns. Orphans included, so nothing is ever hidden."""
        owned = {e.target for e in self.edges if e.relation == OWNS}
        known = self.by_id()
        return [n for n in self.nodes if n.id not in owned or n.id not in known]

    def children_of(self, node_id: str) -> list[str]:
        return [e.target for e in self.edges if e.relation == OWNS and e.source == node_id]

    def cross_edges(self, node_id: str) -> list[str]:
        """Non-ownership links touching this node, grouped so they stay short.

        Grouped by relation and direction: a Service fronting twelve pods is
        one line, not twelve. Both directions are kept --- from a Pod you want
        "selected by", from a Service you want "selects".
        """
        lookup = self.by_id()
        outbound: dict[str, list[GraphNode]] = {}
        inbound: dict[str, list[GraphNode]] = {}
        for edge in self.edges:
            if edge.relation == OWNS:
                continue
            if edge.source == node_id and (other := lookup.get(edge.target)):
                outbound.setdefault(edge.relation, []).append(other)
            elif edge.target == node_id and (other := lookup.get(edge.source)):
                inbound.setdefault(edge.relation, []).append(other)
        return [
            *(f"→ {rel} {_describe(nodes)}" for rel, nodes in sorted(outbound.items())),
            *(f"← {rel} by {_describe(nodes)}" for rel, nodes in sorted(inbound.items())),
        ]

    def layout(self) -> list[GraphRow]:
        """Rows with their tree connectors already drawn.

        Shared between the plain-text and the Textual renderer so the two can
        never drift. Guards against cycles in ownerReferences: illegal, but
        they do occur after a botched restore, and unbounded recursion would
        take the whole session down.
        """
        lookup = self.by_id()
        rows: list[GraphRow] = []
        visited: set[str] = set()

        def walk(node_id: str, ancestors: tuple[bool, ...]) -> None:
            node = lookup.get(node_id)
            if node is None:
                return
            prefix = _prefix(ancestors)
            if node_id in visited:
                rows.append(
                    GraphRow(
                        prefix=prefix, text=f"↺ {node.kind}/{node.name} (shown above)", cross=True
                    )
                )
                return
            visited.add(node_id)
            rows.append(
                GraphRow(prefix=prefix, text=node.label, kind=node.kind, status=node.status)
            )

            links = self.cross_edges(node_id)
            children = sorted(
                self.children_of(node_id),
                key=lambda c: (lookup[c].kind, lookup[c].name) if c in lookup else ("", c),
            )
            total = len(links) + len(children)
            for index, link in enumerate(links):
                last = index == total - 1
                rows.append(
                    GraphRow(prefix=_prefix((*ancestors, last), leaf=True), text=link, cross=True)
                )
            for offset, child in enumerate(children):
                walk(child, (*ancestors, len(links) + offset == total - 1))

        for root in sorted(self.roots(), key=lambda n: (_root_order(n.kind), n.kind, n.name)):
            walk(root.id, ())
        for node in self.nodes:  # unreachable nodes (a cycle with no root)
            if node.id not in visited:
                walk(node.id, ())
        return rows

    def to_text(self) -> str:
        lines = [self.title] if self.title else []
        lines += [f"  {row.prefix}{row.text}" for row in self.layout()]
        if not self.nodes:
            lines.append("  (no resources found)")
        if self.caption:
            lines.append(f"  {self.caption}")
        return "\n".join(lines)


class GraphRow(BaseModel):
    """One rendered line: connectors already drawn, text still colourable."""

    prefix: str = ""
    text: str = ""
    kind: str = ""
    status: str = ""
    cross: bool = False


#: Workloads first, then what fronts them, then what they consume.
_ROOT_ORDER = {
    "Namespace": 0,
    "Deployment": 1,
    "StatefulSet": 1,
    "DaemonSet": 1,
    "CronJob": 2,
    "Job": 2,
    "Pod": 3,
    "Service": 4,
    "Ingress": 5,
}


def _root_order(kind: str) -> int:
    return _ROOT_ORDER.get(kind, 9)


def _describe(nodes: list[GraphNode]) -> str:
    if len(nodes) == 1:
        return f"{nodes[0].kind}/{nodes[0].name}"
    kinds = {n.kind for n in nodes}
    kind = next(iter(kinds)) if len(kinds) == 1 else "resources"
    return f"{len(nodes)} {kind}{'s' if len(nodes) != 1 and kind != 'resources' else ''}"


def _prefix(ancestors: tuple[bool, ...], *, leaf: bool = False) -> str:
    if not ancestors:
        return ""
    stem = "".join("   " if last else "│  " for last in ancestors[:-1])
    return stem + ("└─ " if ancestors[-1] else "├─ ") + ("· " if leaf else "")


class VisualGroup(BaseModel):
    type: Literal["group"] = "group"
    title: str = ""
    items: list[Visual] = Field(default_factory=list)

    def to_text(self) -> str:
        parts = [self.title] if self.title else []
        parts += [item.to_text() for item in self.items]
        return "\n\n".join(p for p in parts if p)


class Chart(BaseModel):
    """Several series on one pair of axes.

    A separate variant rather than a list on ``Series`` because the point is
    comparison: these lines share a scale, and two charts side by side with
    different scales invite exactly the wrong reading.
    """

    type: Literal["chart"] = "chart"
    title: str = ""
    series: list[Series] = Field(default_factory=list)
    unit: str = ""
    caption: str = ""

    @property
    def timed(self) -> bool:
        return bool(self.series) and all(line.timed for line in self.series)

    def to_text(self) -> str:
        if not self.series:
            return f"{self.title}\n  (no data)".strip()
        parts = [self.title] if self.title else []
        for line in self.series:
            rendered = line.to_text()
            if line.label and not line.title:
                rendered = f"{line.label}\n{rendered}"
            parts.append(rendered)
        if self.caption:
            parts.append(f"  {self.caption}")
        return "\n".join(p for p in parts if p)


Visual = Annotated[
    Bars | Series | Chart | Table | Gauge | ResourceGraph | VisualGroup,
    Field(discriminator="type"),
]

VisualGroup.model_rebuild()
