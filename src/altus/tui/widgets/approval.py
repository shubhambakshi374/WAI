"""The approval modal, and the policy that drives it from the agent loop.

The modal shows the actual diff, not a description of one. Approving a change
you cannot see is not consent, and a prompt that only says "write_file wants
to modify main.py" trains people to hit yes.

The escalation button is deliberately the least prominent of the three, and
says which tool it applies to. It disappears entirely for a protected target or
a privileged operation --- a standing grant on ``k8s_exec`` would be
indistinguishable from having no gate at all, so it is never on offer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from altus.tools.approval import ApprovalRequest, Decision

if TYPE_CHECKING:
    from textual.app import App

#: Why a given tool is privileged, in the user's terms rather than the API's.
#: "k8s_exec wants approval" is not a decision anyone can make; "runs a command
#: inside a running container" is.
PRIVILEGED_REASONS = {
    "k8s_exec": "runs a command inside a running container",
    "k8s_attach": "joins a process already running in a container",
    "k8s_cp": "copies files between your machine and a container",
    "k8s_port_forward": "opens a tunnel from this machine into the cluster network",
    "k8s_drain": "evicts every pod from a node and takes it out of service",
    "k8s_node": "changes whether a node accepts work",
    "k8s_patch": "changes who may do what, or takes capacity out of service",
    "k8s_create": "mints a credential or grants rights",
    "k8s_delete": "removes an unbounded set of objects in one call",
    "k8s_kubectl": "runs a cluster command outside Altus's structured tools",
    "helm": "installs or removes a release, many objects at once",
}


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
    ApprovalModal .path { text-style: bold; }
    ApprovalModal .target { color: $warning; text-style: bold; padding-bottom: 1; }
    ApprovalModal .dry-run { color: $success; padding-bottom: 1; }
    ApprovalModal .protected {
        background: $error 20%;
        color: $error;
        text-style: bold;
        padding: 0 1;
        margin-bottom: 1;
    }
    ApprovalModal Input { margin-bottom: 1; }
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
        #: The token a protected or privileged operation demands. Typing the
        #: name is the whole point: it forces you to read which environment you
        #: are changing.
        self.challenge = _challenge(request.target) if request.needs_challenge else ""

    def compose(self) -> ComposeResult:
        req = self.request
        with Vertical():
            yield Label(
                f"{req.action.upper()} — approval required", classes="headline", markup=False
            )
            yield Label(req.path, classes="path", markup=False)
            if req.target:
                yield Label(req.target, classes="target", markup=False)
            if self.challenge:
                yield Label(self._challenge_banner(), classes="protected", markup=False)
                yield Input(placeholder=self.challenge, id="challenge")
            if req.dry_run:
                yield Label(f"✓ {req.dry_run}", classes="dry-run", markup=False)
            if req.recoverability:
                danger = (
                    "cannot be undone" in req.recoverability or "permanent" in req.recoverability
                )
                yield Label(
                    req.recoverability,
                    classes=f"recover{' -danger' if danger else ''}",
                    markup=False,
                )
            with VerticalScroll():
                yield Static(self._render_diff(), markup=False, id="diff")
            with Horizontal():
                yield Button("Reject  (n)", variant="error", id="reject")
                if not self.challenge:
                    # No standing grant for a protected target: the whole point
                    # is that each change is looked at.
                    yield Button(f"Always allow {req.tool}  (a)", id="always")
                yield Button("Approve  (y)", variant="success", id="approve")

    def _challenge_banner(self) -> str:
        """Say *why* a name has to be typed. The two reasons are not the same:
        one is where the change lands, the other is what the change can do."""
        req = self.request
        if req.sensitivity.needs_challenge:
            reason = PRIVILEGED_REASONS.get(req.tool, "runs with elevated privilege")
            where = " IN A PROTECTED ENVIRONMENT" if req.protected else ""
            return f"⚠ PRIVILEGED{where} — {reason}\n  type  {self.challenge}  to confirm"
        return f"⚠ PROTECTED ENVIRONMENT — type  {self.challenge}  to confirm"

    def _render_diff(self) -> str:
        return self.request.diff or "(no preview available)"

    def on_mount(self) -> None:
        # Focus Reject: the safe option should be what Enter hits.
        self.query_one("#reject", Button).focus()

    def _challenge_met(self) -> bool:
        if not self.challenge:
            return True
        try:
            typed = self.query_one("#challenge", Input).value.strip()
        except Exception:
            return False
        return typed == self.challenge

    def _refuse_unconfirmed(self) -> None:
        what = (
            "a privileged operation"
            if self.request.sensitivity.needs_challenge
            else "a change to a protected environment"
        )
        self.notify(f"Type {self.challenge} to confirm {what}.", severity="warning", markup=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        choice = {
            "approve": Decision.ALLOW,
            "always": Decision.ALLOW_ALWAYS,
            "reject": Decision.DENY,
        }[event.button.id or "reject"]
        if choice is not Decision.DENY and not self._challenge_met():
            self._refuse_unconfirmed()
            return
        self.dismiss(choice)

    def action_approve(self) -> None:
        if not self._challenge_met():
            self._refuse_unconfirmed()
            return
        self.dismiss(Decision.ALLOW)

    def action_approve_always(self) -> None:
        if self.challenge:
            self._refuse_unconfirmed()
            return
        self.dismiss(Decision.ALLOW_ALWAYS)

    def action_reject(self) -> None:
        self.dismiss(Decision.DENY)


def _challenge(target: str) -> str:
    """The word that must be typed. The context name, not a generic yes ---
    typing it is what makes you read which environment this is."""
    if not target:
        return "confirm"
    head = target.split("·")[0].strip()
    return head.replace("cluster ", "").strip() or "confirm"


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
