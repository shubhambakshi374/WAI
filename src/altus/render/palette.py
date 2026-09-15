"""Colours for drawn output, as plain hex.

Deliberately not read from a Textual theme. ``altus.render`` has to stay
importable without a terminal --- the layering guard enforces it, and golden
image tests depend on it --- so the front end hands a palette in rather than
the renderer reaching out for one.

The vocabulary matches ``core/visuals.py`` so a drawing and its text rendering
say the same thing about the same object: a CrashLoopBackOff pod is the error
colour in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Kubernetes kinds, coloured by what they are rather than how they are doing.
#: Mirrors KIND_MARKS in core/visuals.py --- if a kind gains a mark there it
#: should gain a colour here.
KIND_COLOURS: dict[str, str] = {
    "Deployment": "#7aa2f7",
    "StatefulSet": "#7aa2f7",
    "DaemonSet": "#7aa2f7",
    "ReplicaSet": "#5c7ec2",
    "Pod": "#9ece6a",
    "Service": "#bb9af7",
    "Ingress": "#f7768e",
    "PersistentVolumeClaim": "#e0af68",
    "ConfigMap": "#7dcfff",
    "Secret": "#e0af68",
    "Job": "#9ece6a",
    "CronJob": "#9ece6a",
    "HorizontalPodAutoscaler": "#bb9af7",
    "Node": "#c0caf5",
}

#: How an edge is drawn. Ownership is the backbone and gets a solid line; the
#: rest are relationships laid over it and are dashed, so the tree stays
#: readable underneath.
RELATION_STYLES: dict[str, tuple[str, bool]] = {
    "owns": ("#4a5268", False),
    "selects": ("#bb9af7", True),
    "routes-to": ("#f7768e", True),
    "mounts": ("#e0af68", True),
    "uses": ("#7dcfff", True),
    "scales": ("#9d7cd8", True),
    "secures": ("#f7768e", True),
}


@dataclass(frozen=True)
class Palette:
    """Everything a drawing needs, with no reference to any UI framework."""

    background: str = "#16131a"
    surface: str = "#1d1923"
    ink: str = "#ece8ef"
    muted: str = "#9d95a8"
    rule: str = "#302b39"
    accent: str = "#ca9cc6"

    ok: str = "#9ece6a"
    warn: str = "#e0af68"
    error: str = "#f7768e"

    kinds: dict[str, str] = field(default_factory=lambda: dict(KIND_COLOURS))
    relations: dict[str, tuple[str, bool]] = field(default_factory=lambda: dict(RELATION_STYLES))

    def kind(self, name: str) -> str:
        return self.kinds.get(name, self.muted)

    def relation(self, name: str) -> tuple[str, bool]:
        return self.relations.get(name, (self.rule, True))

    def status(self, text: str) -> str:
        """Health from a status string, using the same words the text views do."""
        lowered = text.casefold()
        if not lowered:
            return self.muted
        if any(word in lowered for word in ("crashloop", "error", "failed", "evicted")):
            return self.error
        if any(word in lowered for word in ("pending", "terminating", "notready", "unknown")):
            return self.warn
        if any(word in lowered for word in ("running", "ready", "bound", "active", "complete")):
            return self.ok
        return self.muted


DARK = Palette()

LIGHT = Palette(
    background="#f6f4f8",
    surface="#ffffff",
    ink="#191520",
    muted="#6f6878",
    rule="#e2dfe6",
    accent="#6d3d6b",
    ok="#2f7367",
    warn="#9a5c08",
    error="#b3324a",
)
