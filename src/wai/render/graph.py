"""The topology map as an image.

The layout is in ``wai.render.layout`` --- shared with the terminal-cell back
end, so the two are the same engine measured in different units rather than
two implementations that drift.

What is specific to pixels is here: node boxes, and the non-ownership edges
drawn as arcs over the tree. Arcs are what keep the picture a graph rather than
merely a tree, and they are the one thing character cells cannot do, which is
why the cell back end lists them instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wai.core.visuals import ResourceGraph
from wai.render.canvas import Canvas, View
from wai.render.layout import Metrics, Placed, caption_with_omissions, forest, layout, walk
from wai.render.palette import Palette

if TYPE_CHECKING:
    from wai.render import Hit

PIXELS = Metrics(node_width=132.0, node_height=34.0, gap_x=18.0, gap_y=30.0, pad=16.0)
PAD = PIXELS.pad


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

    roots = forest(model)
    content_width, content_height = layout(
        roots, PIXELS, (canvas.width - 2 * PAD) / max(1.0, canvas.height - top - PAD)
    )
    placed = walk(roots)
    positions = {entry.node.id: entry for entry in placed}

    # Fit to the canvas, then apply the view's own zoom on top, so a fresh
    # render always shows everything and zooming is a deliberate act.
    usable_width = canvas.width - 2 * PAD
    usable_height = canvas.height - top - PAD
    fit = min(usable_width / content_width, usable_height / content_height, 1.0)
    scale = fit * view.scale
    offset_x = PAD + view.offset[0]
    offset_y = top + view.offset[1]

    def screen(entry: Placed) -> tuple[float, float, float, float]:
        left = offset_x + entry.x * scale
        node_top = offset_y + entry.y * scale
        return (
            left,
            node_top,
            left + PIXELS.node_width * scale,
            node_top + PIXELS.node_height * scale,
        )

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
    omitted = 0
    for entry in placed:
        box = screen(entry)
        if box[2] < 0 or box[0] > canvas.width or box[3] < 0 or box[1] > canvas.height:
            # Panned out of sight; no point drawing or hit-testing it. Counted
            # rather than forgotten, because a caption that keeps counting a
            # node nobody can see makes the picture look complete.
            omitted += 1
            continue
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
        hits.append(
            Hit(
                box=box,
                node_id=node.id,
                label=f"{node.kind}/{node.name}",
                reader=node.reader,
            )
        )

    caption = caption_with_omissions(model.caption, omitted)
    if caption:
        canvas.text((PAD, canvas.height - PAD - 10), caption, fill=palette.muted, size=9)
    return tuple(hits)
