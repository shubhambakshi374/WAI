"""Bars, gauges and series, drawn.

These say the same thing the text renderings in ``core/visuals.py`` say. Where
that module writes `████░░░░ 3.2 / 4 cores 80%`, this one draws it --- the
numbers, the limit and the over-limit distinction all survive, because a
picture that drops the limit is prettier and less useful.
"""

from __future__ import annotations

from wai.core.visuals import Bars, Chart, Gauge, Series
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


def _clock(stamp: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(stamp, UTC).strftime("%H:%M")


def draw_series(model: Series, canvas: Canvas, palette: Palette, view: View) -> None:
    """One line. Kept as its own entry point because a lone Series is the
    common case and should not have to be wrapped in a Chart to be drawn."""
    draw_chart(
        Chart(title=model.title, series=[model], unit=model.unit, caption=model.caption),
        canvas,
        palette,
        view,
    )


def draw_chart(model: Chart, canvas: Canvas, palette: Palette, view: View) -> None:
    """Several lines on shared axes.

    Shared is the point: comparison is why a chart holds more than one series,
    and giving each its own scale would invite exactly the wrong reading.
    """
    y = _header(canvas, palette, model.title, PAD)
    lines = [line for line in model.series if len(line.points) >= 2]
    # The legend needs a row of its own. Drawn at the top of the plot area it
    # lands on the title's descenders, which is where it was.
    legend = [line for line in lines if line.label] if len(lines) > 1 else []
    if legend:
        y += 14
    bottom = canvas.height - PAD - (LABEL_SIZE + 6 if model.caption else 0) - LABEL_SIZE - 6
    left = PAD + 46
    right = canvas.width - PAD

    if not lines:
        canvas.text((PAD, y), "(not enough data)", fill=palette.muted, size=LABEL_SIZE)
        _caption(canvas, palette, model.caption)
        return

    low = min(min(line.points) for line in lines)
    high = max(max(line.points) for line in lines)
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

    # A time axis where the timestamps are there, and nothing where they are
    # not --- inventing evenly spaced ticks for irregular samples would be a
    # graph that lies about when things happened.
    timed = [line for line in lines if line.timed]
    if timed:
        stamps = [stamp for line in timed for stamp in line.at]
        first, last = min(stamps), max(stamps)
        if last > first:
            for fraction in (0.0, 0.5, 1.0):
                at_x = left + (right - left) * fraction
                canvas.text(
                    (at_x, bottom + 5),
                    _clock(first + (last - first) * fraction),
                    fill=palette.muted,
                    size=9,
                    anchor="mt" if fraction == 0.5 else ("lt" if fraction == 0.0 else "rt"),
                )

    palettes = [palette.accent, palette.ok, palette.warn, palette.error, palette.muted]
    legend_x = float(left)
    for index, line in enumerate(lines):
        colour = palettes[index % len(palettes)]
        if line.timed:
            stamps = [stamp for entry in timed for stamp in entry.at] or line.at
            first, last = min(stamps), max(stamps)
            width = (last - first) or 1.0
            positions = [left + (right - left) * ((stamp - first) / width) for stamp in line.at]
        else:
            positions = [
                left + (right - left) * step / (len(line.points) - 1)
                for step in range(len(line.points))
            ]
        plotted = [
            (at_x, bottom - (value - low) / span * (bottom - y))
            for at_x, value in zip(positions, line.points, strict=True)
        ]
        canvas.line(plotted, fill=colour, width=1.6)
        canvas.dot(plotted[-1], 2.5, fill=colour)

        if line in legend:
            canvas.dot((legend_x + 3, y - 8), 3, fill=colour)
            canvas.text((legend_x + 10, y - 8), line.label, fill=palette.muted, size=9, anchor="lm")
            legend_x += 16 + canvas.measure(line.label, size=9)

    if not timed:
        # Where there is no clock, the last value is the most useful label the
        # right-hand edge can carry. With a clock, that space is the end time.
        latest = lines[-1]
        canvas.text(
            (right, bottom + 5),
            f"{_fmt(latest.points[-1])}{latest.unit or model.unit}",
            fill=palette.ink,
            size=LABEL_SIZE,
            anchor="rt",
        )
    _caption(canvas, palette, model.caption)
