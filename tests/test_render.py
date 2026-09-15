"""Drawing ``Visual`` models. No terminal involved.

**A note on golden images.** The plan for this called for byte-comparing
rendered PNGs. That is wrong here and the reason is worth recording: font
discovery walks system paths, so macOS draws with SF Mono and a Linux CI
runner with DejaVu, and identical inputs produce different bytes on different
machines. Golden bytes would fail on CI for a reason that has nothing to do
with a regression.

So these assert the things that are actually invariant --- geometry, hit
regions, determinism on one machine, and the fallback contract --- and leave
pixel fidelity to the one-off visual check a human does.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from wai.core.visuals import (
    Bar,
    Bars,
    Gauge,
    GraphEdge,
    GraphNode,
    ResourceGraph,
    Series,
    Table,
    VisualGroup,
)
from wai.render import DARK, LIGHT, Hit, View, render, renderable


def node(kind: str, name: str, status: str = "", namespace: str = "shop") -> GraphNode:
    # Kind/namespace/name --- what cloud.k8s.node_id actually builds. An earlier
    # version of these tests invented an apiVersion-qualified id the producer
    # never emits, which would have hidden a drill-down that could not parse.
    return GraphNode(
        id=f"{kind}/{namespace}/{name}",
        kind=kind,
        name=name,
        namespace=namespace,
        status=status,
    )


def owns(parent: GraphNode, child: GraphNode) -> GraphEdge:
    return GraphEdge(source=parent.id, target=child.id, relation="owns")


@pytest.fixture
def graph() -> ResourceGraph:
    deployment = node("Deployment", "web", "2/2")
    replicaset = node("ReplicaSet", "web-7d9")
    good = node("Pod", "web-7d9-aaa", "Running")
    bad = node("Pod", "web-7d9-bbb", "CrashLoopBackOff")
    service = node("Service", "web")
    return ResourceGraph(
        title="shop",
        nodes=[deployment, replicaset, good, bad, service],
        edges=[
            owns(deployment, replicaset),
            owns(replicaset, good),
            owns(replicaset, bad),
            GraphEdge(source=service.id, target=good.id, relation="selects"),
        ],
    )


BARS = Bars(
    title="CPU",
    bars=[
        Bar(label="web-a", value=340, limit=500, request=250, unit="m"),
        Bar(label="web-b", value=612, limit=500, request=250, unit="m"),
    ],
)
GAUGE = Gauge(title="memory", value=6.4, total=8.0, unit="Gi")
SERIES = Series(title="restarts", points=[0, 1, 3, 3, 7, 12])


# ------------------------------------------------------------------ contract


@pytest.mark.parametrize("model", [BARS, GAUGE, SERIES])
def test_every_drawable_variant_produces_a_valid_png(model: object) -> None:
    out = render(model, size=(480, 200))  # type: ignore[arg-type]
    assert out is not None
    image = Image.open(io.BytesIO(out.png))
    assert image.format == "PNG"
    assert image.size == (480, 200), "the caller's size is what comes back"


def test_a_table_falls_back_rather_than_being_drawn_badly() -> None:
    """A table as pixels loses selection, sorting and copy-paste, and gains
    nothing. None is how the caller is told to use the real DataTable."""
    model = Table(columns=["name"], rows=[["web"]])
    assert render(model) is None
    assert not renderable(model)


def test_a_group_is_the_callers_to_lay_out() -> None:
    assert render(VisualGroup(items=[BARS, GAUGE])) is None


def test_a_canvas_too_small_to_read_falls_back() -> None:
    assert render(BARS, size=(40, 20)) is None


@pytest.mark.parametrize("model", [BARS, GAUGE, SERIES])
def test_renderable_agrees_with_render(model: object) -> None:
    assert renderable(model) is (render(model, size=(320, 160)) is not None)  # type: ignore[arg-type]


# -------------------------------------------------------------- degenerate


def test_empty_models_draw_something_rather_than_crashing() -> None:
    """A namespace with nothing in it is an ordinary answer, not an error."""
    for model in (
        Bars(title="CPU", bars=[]),
        Series(title="restarts", points=[]),
        ResourceGraph(title="empty", nodes=[], edges=[]),
        Gauge(title="memory", value=0, total=0),
    ):
        out = render(model, size=(320, 160))
        assert out is not None, f"{type(model).__name__} should still draw"


def test_a_series_of_one_point_does_not_divide_by_zero() -> None:
    assert render(Series(points=[4.0]), size=(320, 160)) is not None


def test_a_flat_series_does_not_divide_by_zero() -> None:
    assert render(Series(points=[4.0, 4.0, 4.0]), size=(320, 160)) is not None


def test_a_bar_over_its_limit_still_draws() -> None:
    over = Bars(bars=[Bar(label="x", value=900, limit=100)])
    assert render(over, size=(320, 160)) is not None


# ------------------------------------------------------------------- hits


def test_every_node_gets_exactly_one_hit_region(graph: ResourceGraph) -> None:
    out = render(graph, size=(820, 420))
    assert out is not None
    assert len(out.hits) == len(graph.nodes)
    assert {hit.node_id for hit in out.hits} == {n.id for n in graph.nodes}


def test_hit_regions_do_not_overlap(graph: ResourceGraph) -> None:
    """Overlapping regions mean a click is ambiguous, and drill-down opens
    whichever node happened to be drawn last."""
    out = render(graph, size=(820, 420))
    assert out is not None
    boxes = [hit.box for hit in out.hits]
    for index, (ax1, ay1, ax2, ay2) in enumerate(boxes):
        for bx1, by1, bx2, by2 in boxes[index + 1 :]:
            overlaps = ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2
            assert not overlaps, f"{(ax1, ay1, ax2, ay2)} overlaps {(bx1, by1, bx2, by2)}"


def test_a_click_resolves_to_the_node_under_it(graph: ResourceGraph) -> None:
    out = render(graph, size=(820, 420))
    assert out is not None
    for hit in out.hits:
        left, top, right, bottom = hit.box
        found = out.hit((left + right) / 2, (top + bottom) / 2)
        assert found is not None and found.node_id == hit.node_id


def test_a_click_on_empty_space_resolves_to_nothing(graph: ResourceGraph) -> None:
    out = render(graph, size=(820, 420))
    assert out is not None
    assert out.hit(-5, -5) is None


def test_hit_node_ids_are_what_a_drill_down_needs(graph: ResourceGraph) -> None:
    """The id already encodes apiVersion/Kind/namespace/name, which is why the
    front end can build a k8s_get from a click without parsing pixels."""
    out = render(graph, size=(820, 420))
    assert out is not None
    from wai.cloud.k8s import split_node_id

    for hit in out.hits:
        assert split_node_id(hit.node_id) is not None, hit.node_id
        assert hit.label


# ------------------------------------------------------------ determinism


def test_the_same_graph_renders_identically_twice(graph: ResourceGraph) -> None:
    first = render(graph, size=(820, 420))
    second = render(graph, size=(820, 420))
    assert first is not None and second is not None
    assert first.png == second.png
    assert first.hits == second.hits


def test_node_order_does_not_change_the_layout(graph: ResourceGraph) -> None:
    """A tidy tree is chosen over a force-directed layout precisely so the
    picture is stable. If input order moved things, that claim is false."""
    shuffled = ResourceGraph(
        title=graph.title,
        nodes=list(reversed(graph.nodes)),
        edges=list(reversed(graph.edges)),
    )
    original = render(graph, size=(820, 420))
    other = render(shuffled, size=(820, 420))
    assert original is not None and other is not None
    assert {(h.node_id, h.box) for h in original.hits} == {(h.node_id, h.box) for h in other.hits}


# ------------------------------------------------------------------- view


def test_zooming_in_makes_nodes_bigger(graph: ResourceGraph) -> None:
    normal = render(graph, size=(820, 420), view=View())
    zoomed = render(graph, size=(820, 420), view=View(scale=2.0))
    assert normal is not None and zoomed is not None

    def area(out: object) -> float:
        hit = next(h for h in out.hits if h.node_id.endswith("/web"))  # type: ignore[attr-defined]
        left, top, right, bottom = hit.box
        return (right - left) * (bottom - top)

    assert area(zoomed) > area(normal) * 2


def test_panning_moves_every_node_by_the_same_amount(graph: ResourceGraph) -> None:
    still = render(graph, size=(820, 420), view=View())
    panned = render(graph, size=(820, 420), view=View(offset=(40.0, 25.0)))
    assert still is not None and panned is not None
    by_id = {hit.node_id: hit.box for hit in still.hits}
    for hit in panned.hits:
        before = by_id[hit.node_id]
        assert hit.box[0] == pytest.approx(before[0] + 40)
        assert hit.box[1] == pytest.approx(before[1] + 25)


def test_zoom_is_clamped_so_it_cannot_be_driven_to_nothing() -> None:
    tiny = View()
    for _ in range(20):
        tiny = tiny.zoomed(0.5)
    assert tiny.scale >= 0.25
    huge = View()
    for _ in range(20):
        huge = huge.zoomed(2.0)
    assert huge.scale <= 6.0


def test_nodes_panned_off_canvas_get_no_hit_region(graph: ResourceGraph) -> None:
    """Hit-testing something invisible would make a click land on a node the
    user cannot see."""
    out = render(graph, size=(820, 420), view=View(offset=(-5000.0, 0.0)))
    assert out is not None
    assert out.hits == ()


# ---------------------------------------------------------------- palette


def test_status_colours_follow_the_words_the_text_views_use() -> None:
    assert DARK.status("Running") == DARK.ok
    assert DARK.status("CrashLoopBackOff") == DARK.error
    assert DARK.status("Pending") == DARK.warn
    assert DARK.status("") == DARK.muted
    assert DARK.status("something we have never seen") == DARK.muted


def test_both_palettes_cover_every_relation_the_graph_model_uses() -> None:
    for relation in ("owns", "selects", "routes-to", "mounts", "uses", "scales"):
        for palette in (DARK, LIGHT):
            colour, _dashed = palette.relation(relation)
            assert colour.startswith("#")


def test_ownership_is_the_only_solid_edge() -> None:
    """The tree is the backbone; everything laid over it is dashed, which is
    what keeps the backbone readable underneath."""
    _colour, dashed = DARK.relation("owns")
    assert not dashed
    for relation in ("selects", "routes-to", "mounts", "uses", "scales"):
        assert DARK.relation(relation)[1], f"{relation} should be dashed"


def test_an_unknown_relation_still_draws(graph: ResourceGraph) -> None:
    graph.edges.append(
        GraphEdge(source=graph.nodes[0].id, target=graph.nodes[-1].id, relation="invented")
    )
    assert render(graph, size=(820, 420)) is not None


def test_palettes_differ_so_the_light_one_is_not_a_copy() -> None:
    assert LIGHT.background != DARK.background
    assert LIGHT.ink != DARK.ink


# ------------------------------------------------------------------ layout


def test_a_cycle_in_ownership_does_not_hang() -> None:
    """ownerReferences should never cycle, but a malformed cluster must still
    draw rather than recurse until the stack runs out."""
    first, second = node("Pod", "a"), node("Pod", "b")
    looped = ResourceGraph(
        nodes=[first, second],
        edges=[owns(first, second), owns(second, first)],
    )
    assert render(looped, size=(480, 240)) is not None


def test_an_edge_naming_a_missing_node_is_ignored() -> None:
    present = node("Pod", "a")
    orphan = ResourceGraph(
        nodes=[present],
        edges=[GraphEdge(source=present.id, target="Pod/shop/gone", relation="selects")],
    )
    out = render(orphan, size=(480, 240))
    assert out is not None
    assert len(out.hits) == 1


def test_a_large_graph_still_fits_the_canvas() -> None:
    """The whole point: past twenty nodes the text view gives up, so this one
    must not."""
    nodes = [node("Pod", f"pod-{index:02d}", "Running") for index in range(40)]
    big = ResourceGraph(title="busy", nodes=nodes, edges=[])
    out = render(big, size=(900, 500))
    assert out is not None
    assert len(out.hits) == 40
    for hit in out.hits:
        assert hit.box[0] >= -1 and hit.box[1] >= -1
        assert hit.box[2] <= 901 and hit.box[3] <= 501


def test_hit_regions_line_up_with_the_hit_helper(graph: ResourceGraph) -> None:
    out = render(graph, size=(820, 420))
    assert out is not None
    assert isinstance(out.hits[0], Hit)
    assert out.size == (820, 420)


# ------------------------------------------------------- the cell back end

# Terminal.app speaks neither Kitty's protocol nor Sixel, and the halfcell
# fallback preserves a graph's shape while turning every label into mush. So
# the same layout is drawn with box-drawing characters, where text stays text.


def test_the_cell_map_draws_every_node(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    grid = draw_graph(graph, width=120, height=30, palette=DARK)
    assert len(grid.hits) == len(graph.nodes)
    text = grid.to_text()
    for node in graph.nodes:
        assert node.name[:8] in text, f"{node.name} missing from the map"


def test_cell_hits_are_in_columns_and_rows_not_pixels(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    grid = draw_graph(graph, width=120, height=30, palette=DARK)
    for hit in grid.hits:
        left, top, right, bottom = hit.box
        assert 0 <= left <= right < 120
        assert 0 <= top <= bottom < 30
        assert bottom - top == 3, "a node is four rows: border, kind, name, border"


def test_a_cell_click_resolves_to_the_node_under_it(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    grid = draw_graph(graph, width=120, height=30, palette=DARK)
    for hit in grid.hits:
        left, top, right, bottom = hit.box
        found = grid.hit((left + right) // 2, (top + bottom) // 2)
        assert found is not None and found.node_id == hit.node_id


def test_cell_hit_regions_do_not_overlap(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    grid = draw_graph(graph, width=120, height=30, palette=DARK)
    boxes = [hit.box for hit in grid.hits]
    for index, (ax1, ay1, ax2, ay2) in enumerate(boxes):
        for bx1, by1, bx2, by2 in boxes[index + 1 :]:
            assert not (ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2)


def test_cross_relations_are_listed_because_cells_cannot_draw_arcs(
    graph: ResourceGraph,
) -> None:
    """The image back end arcs them over the tree. A character grid has nowhere
    to route a curve, so they are named instead --- losing them entirely would
    reduce the graph to a tree."""
    from wai.render.cells import draw_graph

    text = draw_graph(graph, width=120, height=30, palette=DARK).to_text()
    assert "relationships" in text
    assert "selects" in text


def test_the_cell_map_is_deterministic(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    first = draw_graph(graph, width=120, height=30, palette=DARK)
    second = draw_graph(graph, width=120, height=30, palette=DARK)
    assert first.to_text() == second.to_text()
    assert first.hits == second.hits


def test_a_narrow_terminal_does_not_crash(graph: ResourceGraph) -> None:
    from wai.render.cells import draw_graph

    for width in (20, 40, 60):
        grid = draw_graph(graph, width=width, height=24, palette=DARK)
        for row in grid.rows:
            assert len(row) == width, "nothing may be written past the edge"


def test_an_empty_graph_says_so_rather_than_drawing_nothing() -> None:
    from wai.render.cells import draw_graph

    grid = draw_graph(
        ResourceGraph(title="empty", nodes=[], edges=[]), width=60, height=10, palette=DARK
    )
    assert "no resources found" in grid.to_text()
    assert grid.hits == ()


def test_status_is_kept_and_the_name_gives_way_to_it() -> None:
    """The same collision the image back end had: both share a line, so the
    status takes its width out of the name's budget first."""
    from wai.render.cells import draw_graph

    long_name = node("Pod", "a-very-long-pod-name-indeed", "CrashLoopBackOff")
    grid = draw_graph(ResourceGraph(nodes=[long_name], edges=[]), width=60, height=12, palette=DARK)
    text = grid.to_text()
    assert "CrashLoop" in text, "the status survives"
    assert "a-very-long-pod-name-indeed" not in text, "the name gave way"


def test_both_back_ends_are_the_same_engine(graph: ResourceGraph) -> None:
    """The architectural claim, asserted rather than assumed.

    Not that the two produce identical positions --- they wrap to their own
    surface's aspect, and adapting is the point. What must hold in both is the
    structure the shared engine computes: a parent sits above its children, and
    siblings run left to right in the same order.
    """
    from wai.render.cells import draw_graph as draw_cells

    image = render(graph, size=(900, 460))
    cells = draw_cells(graph, width=120, height=30, palette=DARK)
    assert image is not None

    owns = {edge.target: edge.source for edge in graph.edges if edge.relation == "owns"}
    for boxes in (
        {hit.node_id: hit.box for hit in image.hits},
        {hit.node_id: hit.box for hit in cells.hits},
    ):
        for child, parent in owns.items():
            assert boxes[parent][1] < boxes[child][1], f"{parent} must sit above {child}"

        siblings = sorted(
            (node for node in graph.nodes if owns.get(node.id) == "ReplicaSet/shop/web-7d9"),
            key=lambda node: node.name,
        )
        positions = [boxes[node.id][0] for node in siblings]
        assert positions == sorted(positions), "siblings run left to right by name"


def test_the_layout_engine_is_shared_not_copied() -> None:
    """Both back ends import the same module. If one ever grows its own
    forest-building or placement, this is what notices."""
    import ast
    from pathlib import Path

    import wai.render

    root = Path(wai.render.__file__).parent
    for name in ("graph.py", "cells.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert "wai.render.layout" in imported, f"{name} should use the shared layout"
        source = (root / name).read_text(encoding="utf-8")
        assert "def forest(" not in source, f"{name} reimplements forest()"


# ------------------------------------------------------ what the terminal can do

# Detection is a pure function over an environment mapping, not a probe. A probe
# needs a real TTY --- so it cannot run here, cannot run in CI, and answers
# differently depending on who is watching.


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"TERM": "xterm-kitty"}, "image"),
        ({"TERM": "xterm-256color", "KITTY_WINDOW_ID": "1"}, "image"),
        ({"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}, "image"),
        ({"TERM": "xterm-256color", "WEZTERM_PANE": "0"}, "image"),
        ({"TERM": "xterm-256color", "TERM_PROGRAM": "iTerm.app"}, "image"),
        # The one the author actually runs.
        ({"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}, "cells"),
        ({"TERM": "xterm-256color"}, "cells"),
        ({"TERM": "xterm-256color", "TERM_PROGRAM": "vscode"}, "cells"),
        ({"TERM": "dumb"}, "text"),
        ({}, "text"),
    ],
)
def test_terminals_are_recognised_or_assumed_modest(env: dict[str, str], expected: str) -> None:
    from wai.render.capability import detect

    assert detect(env).value == expected


def test_warp_is_denied_despite_claiming_support() -> None:
    """Warp answers the graphics query affirmatively but does not implement the
    unicode placeholders textual-image uses. Believing it produces a broken
    screen, so it is denied by name rather than trusted."""
    from wai.render.capability import detect

    assert detect({"TERM": "xterm-256color", "TERM_PROGRAM": "WarpTerminal"}).value == "cells"


@pytest.mark.parametrize("wrapper", [{"TMUX": "/tmp/x,1,0"}, {"TERM": "screen-256color"}])
def test_multiplexers_do_not_get_images(wrapper: dict[str, str]) -> None:
    """Images inside tmux need passthrough that is off by default and mangles
    output when missing. Not worth the gamble."""
    from wai.render.capability import detect

    env = {"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1", **wrapper}
    assert detect(env).value == "cells"


def test_an_unknown_terminal_gets_cells_not_images() -> None:
    """Guessing low costs a nicer picture. Guessing high sprays escape codes."""
    from wai.render.capability import detect

    assert detect({"TERM": "something-nobody-has-heard-of"}).value == "cells"


# -------------------------------------------------------------- the setting


@pytest.mark.parametrize(
    ("setting", "expected"),
    [("off", "text"), ("text", "text"), ("cells", "cells"), ("image", "image"), ("auto", "image")],
)
def test_the_setting_is_applied_on_top_of_detection(setting: str, expected: str) -> None:
    from wai.render.capability import resolve

    assert resolve(setting, {"TERM": "xterm-kitty"}).value == expected


def test_asking_for_images_on_a_terminal_that_cannot_show_them_yields_cells() -> None:
    """The explicit values are a ceiling, not a floor. The alternative to
    refusing is a screen full of escape codes."""
    from wai.render.capability import resolve

    env = {"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"}
    assert resolve("image", env).value == "cells"
    assert resolve("on", env).value == "cells"


def test_an_unknown_setting_falls_back_to_auto() -> None:
    from wai.render.capability import available, resolve

    env = {"TERM": "xterm-kitty"}
    assert resolve("nonsense", env) == available(env)


def test_off_beats_a_capable_terminal() -> None:
    from wai.render.capability import resolve

    assert resolve("off", {"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1"}).value == "text"


def test_support_levels_are_ordered() -> None:
    from wai.render.capability import Support

    assert Support.IMAGE.at_least(Support.CELLS)
    assert Support.CELLS.at_least(Support.TEXT)
    assert not Support.TEXT.at_least(Support.CELLS)


# ----------------------------------------------------------------- explaining


def test_every_reason_for_no_pictures_is_explainable() -> None:
    """ "Why are there no pictures" should be answerable inside the app."""
    from wai.render.capability import explain

    cases = {
        "off": {"TERM": "xterm-kitty"},
        "auto": {"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"},
        "cells": {"TERM": "xterm-kitty"},
    }
    for setting, env in cases.items():
        message = explain(setting, env)
        assert message and not message.endswith(("  ", ":"))
        assert message[0].islower(), "reads as a continuation of a label"


def test_the_explanation_names_the_terminals_that_would_work() -> None:
    from wai.render.capability import explain

    message = explain("auto", {"TERM": "xterm-256color", "TERM_PROGRAM": "Apple_Terminal"})
    assert "Kitty" in message and "Ghostty" in message


def test_tmux_gets_its_own_explanation_rather_than_the_generic_one() -> None:
    from wai.render.capability import explain

    message = explain("auto", {"TERM": "xterm-kitty", "KITTY_WINDOW_ID": "1", "TMUX": "/tmp/x"})
    assert "tmux" in message
