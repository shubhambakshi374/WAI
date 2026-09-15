"""Pillow primitives: fonts, axes, boxes, arrows.

Pillow rather than matplotlib, and the reason is worth keeping written down.
``textual-image`` already depends on Pillow, so drawing with it costs nothing
extra; matplotlib would add numpy and a ~60MB install, and a second of import
time on a tool whose startup is already noticeable. Bars and gauges were
hand-rolled in ``core/visuals.py`` for the terminal; this is the same exercise
with pixels.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

#: Drawn at this multiple and handed to the terminal at its natural size, so
#: text stays sharp on a HiDPI display. Everything in this module works in
#: logical units and multiplies at the edges.
SUPERSAMPLE = 2

#: Tried in order. All are ordinary system faces --- nothing is vendored,
#: because a font is a licence and a few hundred kilobytes we do not need:
#: Pillow's built-in Aileron scales properly since 10.1 and is a fine fallback.
FONT_CANDIDATES: tuple[str, ...] = (
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
)


@functools.lru_cache(maxsize=32)
def font(size: int, *, override: str = "") -> Any:
    """A face at ``size``, from the first candidate that loads.

    Cached because a drawing asks for the same two or three sizes repeatedly
    and opening a TTF per call is the slowest thing here.
    """
    candidates = (override, *FONT_CANDIDATES) if override else FONT_CANDIDATES
    for path in candidates:
        if not path or not Path(path).exists():
            continue
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class View:
    """Zoom and pan. The widget re-renders at a new View rather than scaling a
    bitmap, so text stays crisp at every zoom level."""

    scale: float = 1.0
    offset: tuple[float, float] = (0.0, 0.0)
    focus: str = ""
    """A node id to keep centred, if the caller has one."""

    def zoomed(self, factor: float) -> View:
        return View(max(0.25, min(6.0, self.scale * factor)), self.offset, self.focus)

    def panned(self, dx: float, dy: float) -> View:
        x, y = self.offset
        return View(self.scale, (x + dx, y + dy), self.focus)


class Canvas:
    """A drawing surface in logical units.

    Supersampling is handled here so callers never think about it: give
    logical coordinates, get a sharp image out of ``to_png``.
    """

    def __init__(self, width: int, height: int, background: str) -> None:
        self.width = width
        self.height = height
        self._scale = SUPERSAMPLE
        self._image = Image.new("RGB", (width * self._scale, height * self._scale), background)
        self._draw = ImageDraw.Draw(self._image)

    # ------------------------------------------------------------- geometry

    def _p(self, value: float) -> int:
        return round(value * self._scale)

    def _box(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        return (self._p(box[0]), self._p(box[1]), self._p(box[2]), self._p(box[3]))

    # -------------------------------------------------------------- drawing

    def rect(
        self,
        box: tuple[float, float, float, float],
        *,
        fill: str | None = None,
        outline: str | None = None,
        radius: float = 0,
        width: float = 1,
    ) -> None:
        if radius > 0:
            self._draw.rounded_rectangle(
                self._box(box),
                radius=self._p(radius),
                fill=fill,
                outline=outline,
                width=self._p(width),
            )
        else:
            self._draw.rectangle(self._box(box), fill=fill, outline=outline, width=self._p(width))

    def line(
        self,
        points: list[tuple[float, float]],
        *,
        fill: str,
        width: float = 1,
        dashed: bool = False,
    ) -> None:
        scaled = [(self._p(x), self._p(y)) for x, y in points]
        if not dashed:
            self._draw.line(scaled, fill=fill, width=self._p(width), joint="curve")
            return
        # Dash by walking each segment. Pillow has no dash support, and a
        # dashed relation edge is what keeps the ownership tree readable
        # underneath the overlaid ones.
        on, off = self._p(5), self._p(4)
        for (x1, y1), (x2, y2) in itertools.pairwise(scaled):
            dx, dy = x2 - x1, y2 - y1
            length = (dx * dx + dy * dy) ** 0.5 or 1.0
            ux, uy = dx / length, dy / length
            position = 0.0
            while position < length:
                end = min(position + on, length)
                self._draw.line(
                    [
                        (x1 + ux * position, y1 + uy * position),
                        (x1 + ux * end, y1 + uy * end),
                    ],
                    fill=fill,
                    width=self._p(width),
                )
                position = end + off

    def curve(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        fill: str,
        lift: float = 24,
        width: float = 1,
        dashed: bool = True,
    ) -> None:
        """A quadratic arc between two points.

        Relation edges arc rather than cut straight across, so they read as
        laid *over* the tree instead of being mistaken for part of it.
        """
        (x1, y1), (x2, y2) = start, end
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 - lift
        points = []
        steps = 24
        for step in range(steps + 1):
            t = step / steps
            inv = 1 - t
            points.append(
                (
                    inv * inv * x1 + 2 * inv * t * cx + t * t * x2,
                    inv * inv * y1 + 2 * inv * t * cy + t * t * y2,
                )
            )
        self.line(points, fill=fill, width=width, dashed=dashed)

    def dot(self, centre: tuple[float, float], radius: float, *, fill: str) -> None:
        x, y = centre
        self._draw.ellipse(self._box((x - radius, y - radius, x + radius, y + radius)), fill=fill)

    def text(
        self,
        at: tuple[float, float],
        content: str,
        *,
        fill: str,
        size: int = 11,
        anchor: str = "la",
        font_override: str = "",
    ) -> None:
        self._draw.text(
            (self._p(at[0]), self._p(at[1])),
            content,
            font=font(size * self._scale, override=font_override),
            fill=fill,
            anchor=anchor,
        )

    def measure(self, content: str, *, size: int = 11, font_override: str = "") -> float:
        """Logical width of ``content``. Used for centring and for deciding
        when a label has to be elided."""
        face = font(size * self._scale, override=font_override)
        return self._draw.textlength(content, font=face) / self._scale

    def elide(self, content: str, limit: float, *, size: int = 11, font_override: str = "") -> str:
        if self.measure(content, size=size, font_override=font_override) <= limit:
            return content
        cut = content
        while cut and self.measure(cut + "…", size=size, font_override=font_override) > limit:
            cut = cut[:-1]
        return (cut + "…") if cut else ""

    # ---------------------------------------------------------------- output

    def to_png(self) -> bytes:
        import io

        buffer = io.BytesIO()
        self._image.resize((self.width, self.height), Image.Resampling.LANCZOS).save(
            buffer, format="PNG", optimize=True
        )
        return buffer.getvalue()
