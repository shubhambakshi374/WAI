"""The approval modal, and the policy that drives it from the agent loop.

The modal shows the actual diff, not a description of one. Approving a change
you cannot see is not consent, and a prompt that only says "write_file wants
to modify main.py" trains people to hit yes.

The escalation button is deliberately the least prominent of the three, and
says which tool it applies to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from wai.tools.approval import ApprovalRequest, Decision

if TYPE_CHECKING:
    from textual.app import App


class ApprovalModal(ModalScreen[Decision]):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "reject", "Reject"),
        Binding("y", "approve", "Approve"),
        Binding("n", "reject", "Reject"),
        Binding("a", "approve_always", "Always"),
    ]

    DEFAULT_CSS = """
    ApprovalModal { align: center middle; }
    ApprovalModal > Vertical {
        width: 100;
        max-width: 92%;
        height: auto;
        max-height: 85%;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    ApprovalModal .headline { text-style: bold; color: $warning; }
    ApprovalModal .path { text-style: bold; padding-bottom: 1; }
    ApprovalModal .recover { color: $text-muted; padding-bottom: 1; }
    ApprovalModal .recover.-danger { color: $error; }
    ApprovalModal VerticalScroll {
        height: auto;
        max-height: 24;
        border: round $panel;
        padding: 0 1;
        margin-bottom: 1;
    }
    ApprovalModal .diff-add { color: $success; }
    ApprovalModal .diff-del { color: $error; }
    ApprovalModal .diff-meta { color: $text-muted; }
    ApprovalModal Horizontal { height: auto; align: right middle; }
    ApprovalModal Button { margin-left: 1; }
    """

    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__()
        self.request = request

    def compose(self) -> ComposeResult:
        req = self.request
        with Vertical():
            yield Label(f"{req.action.upper()} — approval required", classes="headline")
            yield Label(req.path, classes="path")
            if req.recoverability:
                danger = "cannot be undone" in req.recoverability
                yield Label(req.recoverability, classes=f"recover{' -danger' if danger else ''}")
            with VerticalScroll():
                yield Static(self._render_diff(), markup=False, id="diff")
            with Horizontal():
                yield Button("Reject  (n)", variant="error", id="reject")
                yield Button(f"Always allow {req.tool}  (a)", id="always")
                yield Button("Approve  (y)", variant="success", id="approve")

    def _render_diff(self) -> str:
        return self.request.diff or "(no preview available)"

    def on_mount(self) -> None:
        # Focus Reject: the safe option should be what Enter hits.
        self.query_one("#reject", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(
            {
                "approve": Decision.ALLOW,
                "always": Decision.ALLOW_ALWAYS,
                "reject": Decision.DENY,
            }[event.button.id or "reject"]
        )

    def action_approve(self) -> None:
        self.dismiss(Decision.ALLOW)

    def action_approve_always(self) -> None:
        self.dismiss(Decision.ALLOW_ALWAYS)

    def action_reject(self) -> None:
        self.dismiss(Decision.DENY)


class InteractiveApproval:
    """Asks the user, through the TUI.

    ``SessionApprovals`` wraps this and owns both the ``always`` bookkeeping
    and the lock that stops two modals racing to the screen.
    """

    def __init__(self, app: App[None]) -> None:
        self._app = app

    async def request(self, req: ApprovalRequest) -> Decision:
        decision = await self._app.push_screen_wait(ApprovalModal(req))
        # Dismissed without a choice (escape through the stack) means no.
        return decision if isinstance(decision, Decision) else Decision.DENY
