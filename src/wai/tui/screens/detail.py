"""One object, in full, reached by clicking it on a map.

Read-only by construction. The only tool this can reach is ``k8s_get``, which
classifies ``READ``, so no approval is involved and the gate in
``tools/k8s/base.py`` is untouched. That is the reason drill-down is safe to
put behind a single click: there is nothing here that could change anything.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Label, Static


class NodeDetail(ModalScreen[None]):
    """The object behind a node, fetched when the screen opens."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Back"),
        Binding("q", "close", "Back"),
    ]

    DEFAULT_CSS = """
    NodeDetail { align: center middle; }
    NodeDetail > Vertical {
        width: 90%; height: 86%;
        border: round $accent; background: $surface; padding: 1 2;
    }
    NodeDetail .subject { text-style: bold; }
    NodeDetail .hint { color: $text-muted; padding-top: 1; }
    NodeDetail VerticalScroll { height: 1fr; }
    """

    def __init__(self, node_id: str, label: str) -> None:
        super().__init__()
        self.node_id = node_id
        self.label = label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(self.label or self.node_id, classes="subject", markup=False)
            with VerticalScroll():
                yield Static("loading…", id="body", markup=False)
            yield Label("escape to go back", classes="hint", markup=False)

    def on_mount(self) -> None:
        self.run_worker(self._load(), exclusive=True)

    async def _load(self) -> None:
        from wai.cloud.k8s import split_node_id

        body = self.query_one("#body", Static)
        parts = split_node_id(self.node_id)
        if parts is None:
            body.update(f"cannot read {self.node_id!r}: not a Kind/namespace/name identity")
            return

        kind, namespace, name = parts
        registry = getattr(self.app, "registry", None)
        context = getattr(self.app, "tool_ctx", None)
        if registry is None or context is None or "k8s_get" not in registry:
            body.update("Kubernetes tools are not available in this session.")
            return

        args = {"kind": kind, "name": name}
        if namespace:
            args["namespace"] = namespace
        outcome = await registry.execute("k8s_get", args, context)
        body.update(outcome.content or "(empty)")

    def action_close(self) -> None:
        self.dismiss()
