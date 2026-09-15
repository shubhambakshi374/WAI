"""Text we did not write must never be parsed as Textual markup.

Issue #1: a tool-call title built from `repr()` of a list argument contained
`[`, Textual parsed it as a style tag, and the widget raised MarkupError
mid-compose. The same mistake was repeated across the TUI --- including the
approval modal, where the text comes from the model itself.

`textual.markup.escape()` does not solve this: it escapes well-formed closing
tags and leaves an unterminated `[` alone. The only reliable answer is not to
parse, so this module guards both halves of that:

* the AST scan below fails when a widget is handed derived text with markup
  still enabled, which is what stops the *next* one being written wrong;
* the render tests fail when one of the inputs verified to crash in #1 crashes
  again.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from textual.content import Content

import wai

TUI = Path(wai.__file__).parent / "tui"

#: Widgets whose first positional argument is parsed as markup unless told
#: otherwise. `notify` takes the same keyword.
MARKUP_CALLS = {"Static", "Label", "notify"}

#: Reported as `file:line  call  hint`, the way the layering guard reports.
HINT = (
    "pass markup=False, or wrap the value in Content(...) --- "
    "a bracket in derived text is parsed as a style tag and can raise"
)


def _modules() -> list[Path]:
    return sorted(TUI.rglob("*.py"))


def _is_literal_text(node: ast.expr) -> bool:
    """Is this argument text we wrote, rather than text we were handed?

    A plain string literal is safe --- we control it. An f-string is not: the
    interpolated parts are exactly the derived text this rule is about.
    """
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _is_neutralised(node: ast.expr) -> bool:
    """Wrapped in something that cannot be read as markup.

    ``Content``/``Text`` carry no markup by construction; ``literal`` escapes
    the one character that starts a tag. See ``wai/tui/markup.py``.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"Content", "Text", "literal"}
    )


def _disables_markup(call: ast.Call) -> bool:
    for keyword in call.keywords:
        if keyword.arg == "markup" and isinstance(keyword.value, ast.Constant):
            return keyword.value.value is False
    return False


def _offenders(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else node.func.attr
            if isinstance(node.func, ast.Attribute)
            else ""
        )

        if name in MARKUP_CALLS:
            if not node.args:
                continue
            first = node.args[0]
            if _is_literal_text(first) or _is_neutralised(first) or _disables_markup(node):
                continue
            found.append((node.lineno, name))

        elif name == "Collapsible":
            title = next((k.value for k in node.keywords if k.arg == "title"), None)
            if title is None or _is_literal_text(title) or _is_neutralised(title):
                continue
            found.append((node.lineno, "Collapsible(title=…)"))
    return found


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_derived_text_is_never_handed_to_a_markup_parser(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = _offenders(tree)
    assert not offenders, "\n".join(
        f"{path.relative_to(TUI.parent)}:{line}  {call}  — {HINT}" for line, call in offenders
    )


# --------------------------------------------------------------- the inputs


#: Every one of these raised MarkupError somewhere in the TUI before the fix.
#: Keep them verbatim --- they are evidence, not examples.
HOSTILE = [
    pytest.param(
        "⋯ k8s_kubectl(args=['get', 'pods', '-A', '--field-selector=st…)",
        id="issue-1-truncated-list-repr",
    ),
    pytest.param("Readiness probe failed: statuscode: [/]", id="event-message-auto-close"),
    pytest.param('failed to start container "web": [/bold]', id="event-message-closing-tag"),
    pytest.param("kubectl get pods -l app=[/bold", id="model-supplied-argv"),
]


@pytest.mark.parametrize("text", HOSTILE)
def test_the_markup_parser_really_does_reject_these(text: str) -> None:
    """The guard above is only worth having because this is true.

    If Textual ever stops raising on these, the rule is still right --- markup
    in derived text is silently wrong even when it parses --- but this test
    should be the thing that tells us the failure mode changed.
    """
    from textual.markup import MarkupError

    with pytest.raises(MarkupError):
        Content.from_markup(text)


@pytest.mark.parametrize("text", HOSTILE)
def test_content_renders_them_as_the_literal_text_they_are(text: str) -> None:
    assert Content(text).plain == text


# ------------------------------------------------------ through real widgets


async def test_the_tool_call_from_issue_1_renders() -> None:
    """The exact shape that crashed: a list argument, repr()d and truncated
    mid-bracket by summarize_args."""
    from textual.app import App, ComposeResult

    from wai.tui.widgets.tool_call import ToolCallWidget

    args = {"args": ["get", "pods", "-A", "--field-selector=status.phase=Running"]}

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield ToolCallWidget("call-1", "k8s_kubectl", args)

    app = Harness()
    async with app.run_test() as pilot:
        widget = app.query_one(ToolCallWidget)
        widget.finish(summary="exit 0", is_error=False)
        await pilot.pause()


async def test_an_approval_prompt_carrying_model_supplied_brackets_renders() -> None:
    """`path` for the CLI tools is the model's own argv. A crash here takes out
    the gate itself, so this one matters more than the reported one."""
    from wai.tools.approval import ApprovalRequest
    from wai.tui.widgets.approval import ApprovalModal

    request = ApprovalRequest(
        tool="k8s_kubectl",
        action="run",
        path="kubectl get pods -l app=[/bold -o jsonpath=[/]",
        target="cluster AKS_QAM · namespace [/]",
        diff="$ kubectl get pods -l app=[/bold",
        recoverability="unknown — [/] cannot be undone",
        dry_run="not dry-run [/]",
    )
    from textual.app import App, ComposeResult
    from textual.widgets import Label

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield Label("harness")

    app = Harness()
    async with app.run_test() as pilot:
        app.push_screen(ApprovalModal(request))
        await pilot.pause()
        assert isinstance(app.screen, ApprovalModal)


async def test_an_events_table_with_bracketed_messages_renders() -> None:
    """Real cluster text. `k8s_events` puts event messages in a cell, and those
    routinely contain brackets."""
    from textual.app import App, ComposeResult

    from wai.core.visuals import Table
    from wai.tui.widgets.visuals import build_view

    model = Table(
        title="Events [/]",
        columns=["type", "message"],
        rows=[
            ["Warning", "Readiness probe failed: statuscode: [/]"],
            ["Warning", 'failed to start container "web": [/bold]'],
        ],
        caption="2 warnings [/]",
    )

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield build_view(model)

    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
