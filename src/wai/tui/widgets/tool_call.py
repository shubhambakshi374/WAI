"""One line per tool call, expandable to the arguments.

Tool activity is high-volume, so the collapsed line is what you normally read;
the detail is one keypress away, matching how the reasoning block behaves in
``message_list.py``.
"""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.widgets import Collapsible, Static

from wai.tui.markup import literal


def summarize_args(args: dict[str, Any], limit: int = 48) -> str:
    if not args:
        return ""
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return rendered if len(rendered) <= limit else rendered[: limit - 1] + "…"


class ToolCallWidget(Static):
    """Pending while the tool runs, then resolved to success or failure."""

    DEFAULT_CSS = """
    ToolCallWidget { height: auto; margin: 0 0 1 0; }
    ToolCallWidget > Collapsible {
        margin: 0 1;
        border: none;
        background: transparent;
    }
    ToolCallWidget.-running > Collapsible { color: $text-muted; }
    ToolCallWidget.-failed   > Collapsible { color: $error; }
    ToolCallWidget.-done     > Collapsible { color: $success; }
    """

    def __init__(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        super().__init__()
        self.call_id = call_id
        self.tool_name = name
        self.args = args
        self.add_class("-running")

    def compose(self) -> ComposeResult:
        # literal(), not a bare string: Collapsible runs a str title through
        # Content.from_markup, and these titles carry repr()d tool arguments.
        # A list argument puts brackets in the title, summarize_args truncates
        # mid-bracket, and Textual raises trying to read a style tag. See #1.
        with Collapsible(title=literal(self._title("⋯")), collapsed=True):
            yield Static(self._detail(), id="tool-detail", markup=False)

    def _title(self, mark: str, summary: str = "") -> str:
        head = f"{mark} {self.tool_name}({summarize_args(self.args)})"
        return f"{head} — {summary}" if summary else head

    def _detail(self) -> str:
        try:
            return json.dumps(self.args, indent=2, default=str)
        except TypeError, ValueError:
            return str(self.args)

    def finish(self, *, summary: str, is_error: bool) -> None:
        self.remove_class("-running")
        self.add_class("-failed" if is_error else "-done")
        if self.is_mounted:
            title = self._title("✗" if is_error else "✓", summary)
            self.query_one(Collapsible).title = literal(title)
