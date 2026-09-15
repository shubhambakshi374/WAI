"""Bars, gauges and series, drawn.

These say the same thing the text renderings in ``core/visuals.py`` say. Where
that module writes `████░░░░ 3.2 / 4 cores 80%`, this one draws it --- the
numbers, the limit and the over-limit distinction all survive, because a
picture that drops the limit is prettier and less useful.
"""

from __future__ import annotations

from wai.core.visuals import Bars, Gauge, Series
from wai.render.canvas import Canvas, View
from wai.render.palette import Palette

PAD = 12
TITLE_SIZE = 12
LABEL_SIZE = 10


def _fmt(value: float) -> str:
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _header(canvas: Canvas, palette: Palette, title: str, y: float) -> float:
    if not title:
        return y
    canvas.text((PAD, y), title, fill=palette.ink, size=TITLE_SIZE)
    return y + TITLE_SIZE + 8


def _caption(canvas: Canvas, palette: Palette, caption: str) -> None:
    if caption:
        canvas.text(
            (PAD, canvas.height - PAD - LABEL_SIZE),
            caption,
            fill=palette.muted,
            size=LABEL_SIZE,
        )


def draw_bars(model: Bars, canvas: Canvas, palette: Palette, view: View) -> None:
    """Horizontal bars with the request marker and the over-limit case kept.

    ``Bar`` distinguishes a limit (a real ceiling, so a percentage means
    something) from a scale (relative sizing only). Drawing both as the same
    bar would turn one of them into a lie, so only a limit gets a percentage.
    """
    y = _header(canvas, palette, model.title, PAD)
    bars = model.bars
    if not bars:
        canvas.text((PAD, y), "(no data)", fill=palette.muted, size=LABEL_SIZE)
        return

    label_width = min(
        max((canvas.measure(b.label, size=LABEL_SIZE) for b in bars), default=0) + 8,
        canvas.width * 0.32,
    )
    bottom = canvas.height - PAD - (LABEL_SIZE + 6 if model.caption else 0)
    available = bottom - y
    row = min(26.0, available / len(bars)) if bars else 0
    thickness = max(6.0, row * 0.52)
    track_left = PAD + label_width
    track_right = canvas.width - PAD - 96

    for index, bar in enumerate(bars):
        top = y + index * row
        if top + thickness > bottom:
            break
        canvas.text(
            (PAD, top + thickness / 2),
            canvas.elide(bar.label, label_width - 8, size=LABEL_SIZE),
            fill=palette.muted,
            size=LABEL_SIZE,
            anchor="lm",
        )
        canvas.rect(
            (track_left, top, track_right, top + thickness),
            fill=palette.rule,
            radius=thickness / 2,
        )

        ceiling = bar.limit or bar.scale or max(bar.value, bar.request or 0.0) or 1.0
        over = bar.limit is not None and bar.limit > 0 and bar.value > bar.limit
        fraction = max(0.0, min(1.0, bar.value / ceiling))
        end = track_left + (track_right - track_left) * fraction
        if end > track_left:
            canvas.rect(
                (track_left, top, end, top + thickness),
                fill=palette.error if over else palette.accent,
                radius=thickness / 2,
            )

        if bar.request is not None and ceiling:
            marker = track_left + (track_right - track_left) * min(1.0, bar.request / ceiling)
            canvas.line(
                [(marker, top - 2), (marker, top + thickness + 2)],
                fill=palette.ink,
                width=1,
            )

        amount = f"{_fmt(bar.value)}{bar.unit}"
        if bar.limit:
            amount += f" / {_fmt(bar.limit)}{bar.unit}  {bar.value / bar.limit * 100:.0f}%"
        canvas.text(
            (canvas.width - PAD, top + thickness / 2),
            amount,
            fill=palette.error if over else palette.muted,
            size=LABEL_SIZE,
            anchor="rm",
        )

    _caption(canvas, palette, model.caption)


def draw_gauge(model: Gauge, canvas: Canvas, palette: Palette, view: View) -> None:
    """One number, large, with its context beneath it."""
    y = _header(canvas, palette, model.title, PAD)
    centre_y = y + (canvas.height - y - PAD) / 2

    # Gauge names its ceiling `total`, not `limit` --- unlike Bar, which
    # distinguishes the two. Reading the model rather than assuming.
    ceiling = model.total or model.value or 1.0
    fraction = max(0.0, min(1.0, model.value / ceiling))
    track_top = centre_y + 14
    thickness = 10.0
    canvas.rect(
        (PAD, track_top, canvas.width - PAD, track_top + thickness),
        fill=palette.rule,
        radius=thickness / 2,
    )
    end = PAD + (canvas.width - 2 * PAD) * fraction
    colour = palette.error if fraction >= 0.9 else palette.warn if fraction >= 0.75 else palette.ok
    if end > PAD:
        canvas.rect((PAD, track_top, end, track_top + thickness), fill=colour, radius=thickness / 2)

    canvas.text(
        (PAD, centre_y),
        f"{_fmt(model.value)}{model.unit}",
        fill=palette.ink,
        size=24,
        anchor="lm",
    )
    if model.total:
        canvas.text(
            (canvas.width - PAD, centre_y),
            f"of {_fmt(model.total)}{model.unit}  ({fraction * 100:.0f}%)",
            fill=palette.muted,
            size=LABEL_SIZE,
            anchor="rm",
        )
    _caption(canvas, palette, model.caption)


def draw_series(model: Series, canvas: Canvas, palette: Palette, view: View) -> None:
    """A line with real axes.

    This is the variant the terminal served worst: a sparkline has no axis, so
    it shows a shape and hides every number. Here the extremes are labelled.
    """
    y = _header(canvas, palette, model.title, PAD)
    points = model.points
    bottom = canvas.height - PAD - (LABEL_SIZE + 6 if model.caption else 0) - LABEL_SIZE - 4
    left = PAD + 44
    right = canvas.width - PAD

    if len(points) < 2:
        canvas.text((PAD, y), "(not enough data)", fill=palette.muted, size=LABEL_SIZE)
        _caption(canvas, palette, model.caption)
        return

    low, high = min(points), max(points)
    span = (high - low) or 1.0
    # Pad the range so a flat line does not sit exactly on the axis.
    low, high = low - span * 0.08, high + span * 0.08
    span = high - low

    for step in range(4):
        grid_y = y + (bottom - y) * step / 3
        canvas.line([(left, grid_y), (right, grid_y)], fill=palette.rule, width=0.5)
        canvas.text(
            (left - 6, grid_y),
            _fmt(high - span * step / 3),
            fill=palette.muted,
            size=9,
            anchor="rm",
        )

    span_x = right - left
    plotted = [
        (left + span_x * index / (len(points) - 1), bottom - (value - low) / span * (bottom - y))
        for index, value in enumerate(points)
    ]
    canvas.line(plotted, fill=palette.accent, width=1.6)
    canvas.dot(plotted[-1], 2.5, fill=palette.accent)
    canvas.text(
        (right, bottom + 4),
        f"{_fmt(points[-1])}{model.unit}",
        fill=palette.ink,
        size=LABEL_SIZE,
        anchor="rt",
    )
    _caption(canvas, palette, model.caption)
