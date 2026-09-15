"""TUI tests. Driven by a scripted fake provider — no network."""

from __future__ import annotations

import asyncio
import contextlib
from typing import ClassVar

from textual.widgets import Label
from textual.worker import WorkerCancelled

from tests.conftest import FakeProvider, default_events
from wai.agent import unanswered_tool_uses
from wai.config.models import Config
from wai.core.errors import RateLimitError
from wai.core.events import MessageEnd, MessageStart, ReasoningDelta, TextDelta, Usage
from wai.core.types import ModelInfo, Role
from wai.tui.app import WaiApp
from wai.tui.widgets.composer import Composer
from wai.tui.widgets.message_list import MessageBubble
from wai.tui.widgets.status_bar import StatusBar


def make_app(provider: FakeProvider | None = None) -> WaiApp:
    config = Config()
    config.ui.stream_flush_ms = 10
    return WaiApp(config=config, provider=provider or FakeProvider())


async def _send(pilot, text: str) -> None:
    composer = pilot.app.screen.query_one(Composer)
    composer.text = text
    await pilot.press("enter")


async def _settle(pilot, screen=None) -> None:
    """Wait for the streaming worker and its final flush."""
    await pilot.app.workers.wait_for_complete()
    await pilot.pause()
    await pilot.pause()


async def test_app_boots_and_shows_model() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        status = pilot.app.screen.query_one(StatusBar)
        assert status.model == "claude-sonnet-5"
        assert status.state == "ready"


async def test_send_streams_into_transcript() -> None:
    provider = FakeProvider()
    app = make_app(provider)
    async with app.run_test() as pilot:
        await _send(pilot, "hello")
        await _settle(pilot)

        bubbles = pilot.app.screen.query(MessageBubble)
        assert len(bubbles) == 2
        assert bubbles.first().role is Role.USER
        assert app.session.messages[-1].text == "Hellothere."
        assert len(provider.calls) == 1
        assert provider.calls[0].messages[0].text == "hello"


async def test_usage_reaches_the_status_bar() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "hello")
        await _settle(pilot)
        status = pilot.app.screen.query_one(StatusBar)
        assert status.usage.output_tokens == 3
        assert status.state == "ready"


async def test_turn_is_persisted_and_resumable() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "remember me")
        await _settle(pilot)
        session_id = app.session.id

    reloaded = app.store.load(session_id)
    assert [m.text for m in reloaded.messages] == ["remember me", "Hellothere."]
    assert reloaded.title == "remember me"
    assert reloaded.usage.output_tokens == 3


async def test_reasoning_is_rendered_in_a_collapsible() -> None:
    events = [
        MessageStart(model="fake-1"),
        ReasoningDelta(text="thinking hard"),
        TextDelta(text="answer"),
        MessageEnd(usage=Usage(output_tokens=1)),
    ]
    app = make_app(FakeProvider(events))
    async with app.run_test() as pilot:
        await _send(pilot, "why?")
        await _settle(pilot)
        assert app.session.messages[-1].reasoning == "thinking hard"
        bubble = pilot.app.screen.query(MessageBubble).last()
        assert bubble.query("#reasoning-body")


async def test_provider_error_is_surfaced_not_raised() -> None:
    app = make_app(FakeProvider(error=RateLimitError("slow down", provider="fake")))
    async with app.run_test() as pilot:
        await _send(pilot, "hi")
        await _settle(pilot)
        assert "slow down" in app.session.messages[-1].text
        assert pilot.app.screen.query_one(StatusBar).state == "ready"


async def test_empty_input_is_ignored() -> None:
    provider = FakeProvider()
    app = make_app(provider)
    async with app.run_test() as pilot:
        await _send(pilot, "   ")
        await pilot.pause()
        assert provider.calls == []


async def test_new_session_clears_the_transcript() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "first")
        await _settle(pilot)
        first_id = app.session.id

        await pilot.press("ctrl+n")
        await pilot.pause()

        assert app.session.id != first_id
        assert app.session.messages == []
        assert len(pilot.app.screen.query(MessageBubble)) == 0
        # The first session is still on disk.
        assert app.store.load(first_id).messages[0].text == "first"


async def test_switch_model_updates_session_and_caps_max_tokens() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        app.session.max_tokens = 100_000
        await app.switch_model(
            ModelInfo(id="deepseek-chat", provider="deepseek", max_output_tokens=8192)
        )
        await pilot.pause()
        assert app.session.model == "deepseek-chat"
        assert app.session.provider == "deepseek"
        assert app.session.max_tokens == 8192
        assert app.store.load(app.session.id).model == "deepseek-chat"


async def test_resume_loads_a_prior_session() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "earlier work")
        await _settle(pilot)
        session_id = app.session.id

    resumed = WaiApp(config=app.config, provider=FakeProvider(), resume=session_id)
    async with resumed.run_test() as pilot:
        assert resumed.session.id == session_id
        assert len(pilot.app.screen.query(MessageBubble)) == 2


async def test_resume_last_falls_back_when_missing() -> None:
    app = WaiApp(config=Config(), provider=FakeProvider(), resume="does-not-exist")
    async with app.run_test():
        assert app.session.messages == []


# --------------------------------------------------------------------- bindings


async def test_key_bindings_reach_the_screen() -> None:
    """Regression guard.

    Without priority=True the focused TextArea swallows ctrl+c (copy) and the
    App swallows ctrl+p (command palette) and ctrl+q, which silently makes
    cancel and the model picker unreachable.
    """
    app = make_app()
    async with app.run_test() as pilot:
        resolved = {
            key: binding.binding.action for key, binding in pilot.app.screen.active_bindings.items()
        }
        assert resolved["ctrl+c"] == "cancel_stream"
        assert resolved["ctrl+p"] == "pick_model"
        assert resolved["ctrl+n"] == "new_session"
        assert resolved["ctrl+q"] == "quit_app"


async def test_ctrl_c_cancels_an_in_flight_stream() -> None:
    """The provider hangs; ctrl+c must stop it and leave the app usable."""
    stalled = asyncio.Event()

    class HangingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            self.calls.append(request)
            yield TextDelta(text="partial")
            stalled.set()
            await asyncio.Event().wait()  # never completes

    app = make_app(HangingProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "hang please")
        await asyncio.wait_for(stalled.wait(), timeout=2)
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert pilot.app.screen.query_one(StatusBar).state == "cancelled"
        # The partial text is kept as content; "cancelled" is a display marker
        # only, so the stored conversation stays free of UI annotations.
        assert "partial" in app.session.messages[-1].text
        assert "partial" in app.store.load(app.session.id).messages[-1].text
        assert unanswered_tool_uses(app.session) == []


async def test_ctrl_c_with_no_stream_exits() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


async def test_ctrl_p_opens_the_model_picker() -> None:
    from wai.tui.widgets.model_picker import ModelPicker

    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert isinstance(pilot.app.screen, ModelPicker)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(pilot.app.screen, ModelPicker)


async def test_model_picker_switches_the_session_model() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.model != "claude-sonnet-5"
        assert pilot.app.screen.query_one(StatusBar).model == app.session.model


async def test_second_send_is_rejected_while_streaming() -> None:
    stalled = asyncio.Event()

    class HangingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            self.calls.append(request)
            stalled.set()
            await asyncio.Event().wait()
            yield TextDelta(text="never")

    provider = HangingProvider()
    app = make_app(provider)
    async with app.run_test() as pilot:
        await _send(pilot, "first")
        await asyncio.wait_for(stalled.wait(), timeout=2)
        await _send(pilot, "second")
        await pilot.pause()
        assert len(provider.calls) == 1
        pilot.app.screen.workers.cancel_group(pilot.app.screen, "turn")
        with contextlib.suppress(WorkerCancelled):
            await pilot.app.workers.wait_for_complete()


async def test_status_bar_renders_its_labels_not_just_reactives() -> None:
    """Regression guard.

    ``is_mounted`` is False during on_mount, so guarding label updates on it
    silently skipped every initial render. Assert the rendered text, because
    asserting the reactive value passes even when the bar is visibly blank.
    """
    app = make_app()
    async with app.run_test() as pilot:
        bar = pilot.app.screen.query_one(StatusBar)
        rendered = {label.id: str(label.render()) for label in bar.query(Label)}
        assert rendered["status-model"] == "anthropic · claude-sonnet-5"
        assert rendered["status-usage"] == "0 in / 0 out"
        assert rendered["status-state"] == "ready"


async def test_status_bar_state_renders_while_streaming() -> None:
    stalled = asyncio.Event()

    class HangingProvider(FakeProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            self.calls.append(request)
            stalled.set()
            await asyncio.Event().wait()
            yield TextDelta(text="never")

    app = make_app(HangingProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "hi")
        await asyncio.wait_for(stalled.wait(), timeout=2)
        await pilot.pause()
        state = pilot.app.screen.query_one("#status-state", Label)
        assert "working" in str(state.render())

        await pilot.press("ctrl+c")
        with contextlib.suppress(WorkerCancelled):
            await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert str(state.render()) == "cancelled"


# ------------------------------------------------------------------ tool calls


def _tool_events(call_id: str = "t1", name: str = "list_dir"):  # type: ignore[no-untyped-def]
    from wai.core.events import ToolCallEnd, ToolCallStart
    from wai.core.types import StopReason

    return [
        MessageStart(model="fake-1"),
        ToolCallStart(index=0, id=call_id, name=name),
        ToolCallEnd(index=0, id=call_id, name=name, input={}),
        MessageEnd(stop_reason=StopReason.TOOL_USE, usage=Usage(output_tokens=1)),
    ]


class ToolThenTextProvider(FakeProvider):
    """One tool-calling turn, then a plain answer."""

    def __init__(self) -> None:
        super().__init__([])
        self.turn = 0

    async def stream(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        self.turn += 1
        events = _tool_events() if self.turn == 1 else default_events("All done.")
        for event in events:
            yield event


async def test_tool_calls_render_and_resolve() -> None:
    from wai.tui.widgets.tool_call import ToolCallWidget

    app = make_app(ToolThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "what is here?")
        await _settle(pilot)

        widgets = pilot.app.screen.query(ToolCallWidget)
        assert len(widgets) == 1
        widget = widgets.first()
        assert widget.tool_name == "list_dir"
        assert widget.has_class("-done"), "should resolve out of the running state"
        assert not widget.has_class("-running")


async def test_tool_turn_persists_without_orphans() -> None:
    app = make_app(ToolThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "what is here?")
        await _settle(pilot)

    reloaded = app.store.load(app.session.id)
    assert unanswered_tool_uses(reloaded) == []
    assert reloaded.messages[-1].text == "Alldone."  # fixture splits on spaces
    assert reloaded.workspace_root == str(app.workspace.root)
    assert reloaded.tools_enabled is True


async def test_failing_tool_marks_the_widget() -> None:
    from wai.tui.widgets.tool_call import ToolCallWidget

    class BadTool(ToolThenTextProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            self.calls.append(request)
            self.turn += 1
            events = _tool_events(name="nope") if self.turn == 1 else default_events("Sorry.")
            for event in events:
                yield event

    app = make_app(BadTool())
    async with app.run_test() as pilot:
        await _send(pilot, "go")
        await _settle(pilot)
        widget = pilot.app.screen.query(ToolCallWidget).first()
        assert widget.has_class("-failed")


async def test_provider_error_is_recorded_in_the_session() -> None:
    """A resumed session should show that the turn failed."""
    app = make_app(FakeProvider(error=RateLimitError("slow down", provider="fake")))
    async with app.run_test() as pilot:
        await _send(pilot, "hi")
        await _settle(pilot)
        assert "slow down" in app.session.messages[-1].text
        assert app.store.load(app.session.id).messages[-1].text.startswith("**error:**")


async def test_no_tools_flag_disables_them() -> None:
    provider = FakeProvider()
    app = WaiApp(config=Config(), provider=provider, no_tools=True)
    async with app.run_test() as pilot:
        await _send(pilot, "hi")
        await _settle(pilot)
        assert app.session.tools_enabled is False
        assert provider.calls[0].tools == []


# -------------------------------------------------------------------- approval


def _write_events(call_id: str = "w1", **args: object):  # type: ignore[no-untyped-def]
    from wai.core.events import ToolCallEnd, ToolCallStart
    from wai.core.types import StopReason

    payload = {"path": "note.txt", "content": "hello\n", **args}
    return [
        MessageStart(model="fake-1"),
        ToolCallStart(index=0, id=call_id, name="write_file"),
        ToolCallEnd(index=0, id=call_id, name="write_file", input=payload),
        MessageEnd(stop_reason=StopReason.TOOL_USE, usage=Usage(output_tokens=1)),
    ]


class WriteThenTextProvider(FakeProvider):
    def __init__(self, **args: object) -> None:
        super().__init__([])
        self.turn = 0
        self.args = args

    async def stream(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        self.turn += 1
        events = _write_events(**self.args) if self.turn == 1 else default_events("Saved.")
        for event in events:
            yield event


def _app_in(tmp_path, provider, **kw):  # type: ignore[no-untyped-def]
    config = Config()
    config.ui.stream_flush_ms = 10
    return WaiApp(config=config, provider=provider, workspace_root=str(tmp_path), **kw)


async def test_write_prompts_and_approval_applies_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        # Wait for the modal rather than the worker: the worker is blocked on it.
        for _ in range(80):
            await pilot.pause()
            if isinstance(pilot.app.screen, ApprovalModal):
                break
        assert isinstance(pilot.app.screen, ApprovalModal)
        assert "note.txt" in pilot.app.screen.request.path

        await pilot.press("y")
        await _settle(pilot)

    assert (tmp_path / "note.txt").read_text() == "hello\n"


async def test_rejecting_the_prompt_writes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        for _ in range(80):
            await pilot.pause()
            if isinstance(pilot.app.screen, ApprovalModal):
                break
        await pilot.press("n")
        await _settle(pilot)

    assert not (tmp_path / "note.txt").exists()
    assert unanswered_tool_uses(app.session) == []


async def test_escape_rejects(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        for _ in range(80):
            await pilot.pause()
            if isinstance(pilot.app.screen, ApprovalModal):
                break
        await pilot.press("escape")
        await _settle(pilot)

    assert not (tmp_path / "note.txt").exists()


async def test_auto_approve_skips_the_prompt(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider(), auto_approve=True)
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        await _settle(pilot)
        assert not isinstance(pilot.app.screen, ApprovalModal)

    assert (tmp_path / "note.txt").read_text() == "hello\n"


async def test_always_allow_is_shown_in_the_status_bar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A standing grant must never be invisible."""
    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        for _ in range(80):
            await pilot.pause()
            if isinstance(pilot.app.screen, ApprovalModal):
                break
        await pilot.press("a")
        await _settle(pilot)

        assert app.approvals.always_allowed == frozenset({"write_file"})
        granted = pilot.app.screen.query_one("#status-granted", Label)
        assert "write_file" in str(granted.render())


async def test_modal_focuses_reject_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Enter should hit the safe option, not the destructive one."""
    from textual.widgets import Button

    from wai.tui.widgets.approval import ApprovalModal

    app = _app_in(tmp_path, WriteThenTextProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "make a note")
        for _ in range(80):
            await pilot.pause()
            if isinstance(pilot.app.screen, ApprovalModal):
                break
        focused = pilot.app.focused
        assert isinstance(focused, Button) and focused.id == "reject"

        await pilot.press("escape")
        await _settle(pilot)


# --------------------------------------------------------------- slash commands


def test_command_detection_ignores_paths() -> None:
    """`/etc/hosts` in a prompt is a path, not a command."""
    from wai.tui.commands import is_command

    assert is_command("/help")
    assert is_command("/kube use AKS_QAM")
    assert not is_command("/etc/hosts")
    assert not is_command("/")
    assert not is_command("what does / mean")
    assert not is_command("read /var/log/syslog for me")


def test_command_parsing_handles_quotes() -> None:
    from wai.tui.commands import parse

    assert parse("/kube add ~/my config") == ("kube", ["add", "~/my", "config"])
    assert parse('/kube add "~/my config"') == ("kube", ["add", "~/my config"])
    assert parse("/HELP") == ("help", [])


async def test_unknown_command_is_reported_not_sent_to_the_model() -> None:
    provider = FakeProvider()
    app = make_app(provider)
    async with app.run_test() as pilot:
        await _send(pilot, "/depoly now")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert provider.calls == [], "a mistyped command must not become a prompt"
        assert app.session.messages == []
        rendered = _notices(pilot)
        assert "Unknown command" in rendered or "depoly" in rendered


async def test_help_lists_commands() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/help")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _notices(pilot)
        for expected in ("/provider", "/kube", "/login", "/model"):
            assert expected in rendered


async def test_provider_command_opens_an_interactive_picker() -> None:
    """A list you cannot act on is a dead end --- the old one just printed text."""
    from wai.tui.widgets.provider_picker import ProviderPicker

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/provider")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ProviderPicker):
                break
        assert isinstance(pilot.app.screen, ProviderPicker)
        await pilot.press("escape")
        await pilot.pause()


async def test_providers_is_an_alias() -> None:
    """`/providers` was an unknown-command error, which is just annoying."""
    from wai.tui.widgets.provider_picker import ProviderPicker

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/providers")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ProviderPicker):
                break
        assert isinstance(pilot.app.screen, ProviderPicker)
        assert "Unknown command" not in _notices(pilot)
        await pilot.press("escape")
        await pilot.pause()


async def test_choosing_a_configured_provider_switches(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from wai.config.secrets import CredentialStatus
    from wai.tui.widgets.provider_picker import ProviderPicker

    monkeypatch.setattr(
        "wai.config.secrets.credential_status",
        lambda name: CredentialStatus(name, True, "env"),
    )
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/provider")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ProviderPicker):
                break
        picker = pilot.app.screen
        picker.query_one("#provider-options").highlighted = 3  # deepseek
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if not isinstance(pilot.app.screen, ProviderPicker):
                break
        assert app.session.provider == "deepseek"


async def test_choosing_an_unconfigured_provider_opens_setup(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The missing link: the list is the route to setting one up."""
    from wai.tui.widgets.provider_picker import ProviderPicker
    from wai.tui.widgets.setup import SetupWizard

    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):  # first-run wizard opens; dismiss it
            await pilot.pause()
            if isinstance(pilot.app.screen, SetupWizard):
                break
        await pilot.press("escape")
        await pilot.pause()

        await _send(pilot, "/provider")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ProviderPicker):
                break
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, SetupWizard):
                break
        assert isinstance(pilot.app.screen, SetupWizard)
        assert pilot.app.screen.provider == "anthropic"


async def test_kube_command_never_writes_the_kubeconfig(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    kubeconfig = tmp_path / "config"
    kubeconfig.write_text(
        "apiVersion: v1\nkind: Config\ncurrent-context: staging\n"
        "clusters:\n- name: c\n  cluster: {server: https://x}\n"
        "contexts:\n- name: AKS_EU_PROD\n  context: {cluster: c, user: u}\n"
        "- name: staging\n  context: {cluster: c, user: u}\n"
        "users:\n- name: u\n  user: {}\n"
    )
    monkeypatch.setenv("KUBECONFIG", str(kubeconfig))
    before = kubeconfig.read_bytes()

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/kube")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _notices(pilot)
        assert "AKS_EU_PROD" in rendered
        assert "protected" in rendered, "the prod context must be flagged"

        await _send(pilot, "/kube use staging")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert app.config.cloud.kube_context == "staging"

    assert kubeconfig.read_bytes() == before, "~/.kube/config must never be written"


async def test_login_status_lists_every_cloud() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/login")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _notices(pilot)
        for cloud in ("k8s", "aws", "azure", "gcp"):
            assert cloud in rendered


async def test_login_rejects_an_unknown_cloud() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/login oracle")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _notices(pilot)
        assert "Unknown cloud" in rendered


async def test_model_command_switches_directly() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/model claude-opus-5")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert app.session.model == "claude-opus-5"


async def test_key_command_opens_a_masked_prompt() -> None:
    from wai.tui.widgets.key_prompt import KeyPrompt

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/key openai")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, KeyPrompt):
                break
        assert isinstance(pilot.app.screen, KeyPrompt)
        from textual.widgets import Input

        assert pilot.app.screen.query_one(Input).password is True, "the key must be masked"
        await pilot.press("escape")
        await pilot.pause()


async def test_key_command_refuses_bedrock() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/key bedrock")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _notices(pilot)
        assert "credential chain" in rendered


def _notices(pilot) -> str:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.message_list import NoticeBubble

    return "\n".join(n.text for n in pilot.app.screen.query(NoticeBubble))


# ------------------------------------------------------------------- visuals


def _visual_events(call_id: str = "v1"):  # type: ignore[no-untyped-def]
    from wai.core.events import ToolCallEnd, ToolCallStart
    from wai.core.types import StopReason

    return [
        MessageStart(model="fake-1"),
        ToolCallStart(index=0, id=call_id, name="k8s_topology"),
        ToolCallEnd(index=0, id=call_id, name="k8s_topology", input={}),
        MessageEnd(stop_reason=StopReason.TOOL_USE, usage=Usage(output_tokens=1)),
    ]


class ChartingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.turn = 0

    async def stream(self, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        self.turn += 1
        events = _visual_events() if self.turn == 1 else default_events("Done.")
        for event in events:
            yield event


class ChartTool:
    """A stand-in k8s_topology that returns a graph without a cluster."""

    name = "k8s_topology"
    description = "topology"
    input_schema: ClassVar[dict] = {}
    read_only = True

    async def run(self, args, ctx):  # type: ignore[no-untyped-def]
        from wai.core.visuals import GraphEdge, GraphNode, ResourceGraph
        from wai.tools.base import ToolOutcome

        graph = ResourceGraph(
            title="shop",
            nodes=[
                GraphNode(id="d", kind="Deployment", name="web", status="2/2 ready"),
                GraphNode(id="p", kind="Pod", name="web-aaa", status="Running"),
            ],
            edges=[GraphEdge(source="d", target="p", relation="owns")],
        )
        return ToolOutcome(content=graph.to_text(), summary="2 resources", visual=graph)


def _charting_app(provider):  # type: ignore[no-untyped-def]
    from wai.tools.registry import ToolRegistry

    config = Config()
    config.ui.stream_flush_ms = 10
    app = WaiApp(config=config, provider=provider)
    app.registry = ToolRegistry([ChartTool()])  # type: ignore[list-item]
    return app


async def test_a_tool_visual_renders_inline() -> None:
    from wai.tui.widgets.visuals import VisualPanel

    app = _charting_app(ChartingProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "show me the cluster")
        await _settle(pilot)
        panels = pilot.app.screen.query(VisualPanel)
        assert len(panels) == 1


async def test_the_chart_is_not_sent_to_the_model() -> None:
    """The economy of the whole design: charts cost no context."""
    provider = ChartingProvider()
    app = _charting_app(provider)
    async with app.run_test() as pilot:
        await _send(pilot, "show me the cluster")
        await _settle(pilot)

    # The second request replays the conversation; no visual may appear in it.
    replayed = provider.calls[-1]
    serialised = str([m.model_dump() for m in replayed.messages])
    assert "visual" not in serialised
    assert "ResourceGraph" not in serialised


async def test_enter_expands_a_visual_full_screen() -> None:
    from wai.tui.widgets.visuals import VisualPanel, VisualScreen

    app = _charting_app(ChartingProvider())
    async with app.run_test() as pilot:
        await _send(pilot, "show me the cluster")
        await _settle(pilot)

        pilot.app.screen.query(VisualPanel).first().focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(pilot.app.screen, VisualScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(pilot.app.screen, VisualScreen)


async def _open_modal(pilot, request):  # type: ignore[no-untyped-def]
    """Push an ApprovalModal and capture its decision via callback.

    push_screen_wait needs a worker; a callback is the equivalent that a test
    can drive directly.
    """
    from wai.tui.widgets.approval import ApprovalModal

    captured: list[object] = []
    pilot.app.push_screen(ApprovalModal(request), callback=captured.append)
    for _ in range(40):
        await pilot.pause()
        if isinstance(pilot.app.screen, ApprovalModal):
            break
    return pilot.app.screen, captured


async def test_protected_target_demands_a_typed_confirmation() -> None:
    """A keypress is not enough for production; you must type the cluster name."""
    from textual.widgets import Button, Input

    from wai.tools.approval import ApprovalRequest, Decision

    request = ApprovalRequest(
        tool="k8s_delete",
        action="delete",
        path="Deployment/web",
        target="cluster AKS_EU_PROD · namespace payments",
        dry_run="server-side dry run succeeded",
        protected=True,
    )
    app = make_app()
    async with app.run_test() as pilot:
        screen, captured = await _open_modal(pilot, request)
        assert screen.challenge == "AKS_EU_PROD"
        assert not screen.query("#always"), "no standing grant on a protected target"

        screen.query_one("#approve", Button).press()
        await pilot.pause()
        assert captured == [], "approval must not go through unconfirmed"

        await pilot.press("y")
        await pilot.pause()
        assert captured == [], "the shortcut must not bypass the challenge either"

        screen.query_one("#challenge", Input).value = "AKS_EU_PROD"
        await pilot.pause()
        screen.query_one("#approve", Button).press()
        await pilot.pause()
        assert captured == [Decision.ALLOW]


async def test_protected_target_can_still_be_rejected_immediately() -> None:
    from textual.widgets import Button

    from wai.tools.approval import ApprovalRequest, Decision

    request = ApprovalRequest(
        tool="k8s_delete",
        action="delete",
        path="Deployment/web",
        target="cluster AKS_EU_PROD",
        protected=True,
    )
    app = make_app()
    async with app.run_test() as pilot:
        screen, captured = await _open_modal(pilot, request)
        screen.query_one("#reject", Button).press()
        await pilot.pause()
        assert captured == [Decision.DENY], "rejecting never needs a challenge"


async def test_unprotected_target_still_approves_on_a_keypress() -> None:
    from wai.tools.approval import ApprovalRequest, Decision

    request = ApprovalRequest(
        tool="k8s_scale",
        action="scale",
        path="Deployment/web",
        target="cluster AKS_QAM · namespace shop",
    )
    app = make_app()
    async with app.run_test() as pilot:
        screen, captured = await _open_modal(pilot, request)
        assert screen.challenge == ""
        await pilot.press("y")
        await pilot.pause()
        assert captured == [Decision.ALLOW]


# ------------------------------------------------------- command suggestions


def _suggest(pilot):  # type: ignore[no-untyped-def]
    from wai.tui.widgets.command_suggest import CommandSuggestions

    return pilot.app.screen.query_one(CommandSuggestions)


async def _type(pilot, text: str):  # type: ignore[no-untyped-def]
    from wai.tui.widgets.composer import Composer

    composer = pilot.app.screen.query_one(Composer)
    composer.text = text
    await pilot.pause()
    await pilot.pause()
    return composer


async def test_a_bare_slash_offers_every_command() -> None:
    """The point of the popup: you should not need to know the names already."""
    app = make_app()
    async with app.run_test() as pilot:
        await _type(pilot, "/")
        panel = _suggest(pilot)
        assert panel.visible_now
        names = {s.command.name for s in panel.suggestions}
        assert {"help", "kube", "login", "provider", "model", "tools"} <= names


async def test_suggestions_filter_as_you_type() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _type(pilot, "/k")
        assert {s.command.name for s in _suggest(pilot).suggestions} == {"key", "kube"}
        await _type(pilot, "/ku")
        assert [s.command.name for s in _suggest(pilot).suggestions] == ["kube"]


async def test_one_character_does_not_match_on_summaries() -> None:
    """`/k` must not offer `model` because its summary contains "Pick"."""
    app = make_app()
    async with app.run_test() as pilot:
        await _type(pilot, "/k")
        assert "model" not in {s.command.name for s in _suggest(pilot).suggestions}


async def test_no_popup_for_a_path_or_ordinary_prompt() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        for text in ("explain /etc/hosts", "what does / mean", "/xyzzy"):
            await _type(pilot, text)
            assert not _suggest(pilot).visible_now, text


async def test_past_the_command_name_it_becomes_a_usage_hint() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _type(pilot, "/kube ")
        panel = _suggest(pilot)
        assert panel.suggestions == [], "arguments are being typed, not a command name"
        assert panel.visible_now
        hint = str(pilot.app.screen.query_one("#suggestion-hint").render())
        assert "use <ctx>" in hint


async def test_tab_completes_the_highlighted_command() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        composer = await _type(pilot, "/ku")
        await pilot.press("tab")
        await pilot.pause()
        assert composer.text == "/kube "


async def test_arrows_move_the_highlight_and_enter_completes() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        composer = await _type(pilot, "/")
        await pilot.press("down")
        await pilot.pause()
        chosen = _suggest(pilot).highlighted
        assert chosen is not None
        await pilot.press("enter")
        await pilot.pause()
        assert composer.text == f"/{chosen.command.name} "


async def test_a_fully_typed_command_runs_on_enter() -> None:
    """Completing an already-complete command would make every command
    take two Enters, which is the wrong trade."""
    app = make_app()
    async with app.run_test() as pilot:
        composer = await _type(pilot, "/help")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert composer.text == "", "it was sent, not completed"
        assert "Commands:" in _notices(pilot)


async def test_escape_dismisses_but_keeps_what_was_typed() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        composer = await _type(pilot, "/k")
        assert _suggest(pilot).visible_now
        await pilot.press("escape")
        await pilot.pause()
        assert not _suggest(pilot).visible_now
        assert composer.text == "/k"


async def test_arrows_still_move_the_cursor_when_no_popup_is_open() -> None:
    """The popup must not steal navigation from ordinary editing."""
    app = make_app()
    async with app.run_test() as pilot:
        composer = await _type(pilot, "line one")
        composer.insert("\n")
        composer.insert("line two")
        await pilot.pause()
        before = composer.cursor_location
        await pilot.press("up")
        await pilot.pause()
        assert composer.cursor_location != before, "up should move the cursor"


async def test_sending_a_prompt_hides_any_popup() -> None:
    provider = FakeProvider()
    app = make_app(provider)
    async with app.run_test() as pilot:
        await _type(pilot, "/")
        assert _suggest(pilot).visible_now
        await _type(pilot, "hello there")
        await pilot.press("enter")
        await _settle(pilot)
        assert not _suggest(pilot).visible_now
        assert len(provider.calls) == 1


# ------------------------------------------------------------- setup wizard

FAKE_MODELS = [
    ModelInfo(id="claude-opus-5", provider="anthropic", display_name="Opus 5"),
    ModelInfo(id="claude-haiku-4-5", provider="anthropic", display_name="Haiku 4.5"),
]


def _no_credentials(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from wai.config.secrets import CredentialStatus

    monkeypatch.setattr(
        "wai.config.secrets.credential_status",
        lambda name: CredentialStatus(name, False, False),
    )


async def _reach_model_step(pilot, key: str = "sk-test"):  # type: ignore[no-untyped-def]
    from textual.widgets import Input

    from wai.tui.widgets.setup import Step

    wizard = pilot.app.screen
    await pilot.press("enter")  # accept the highlighted provider
    await pilot.pause()
    await pilot.pause()
    wizard.query_one("#key", Input).value = key
    await pilot.press("enter")
    for _ in range(80):
        await pilot.pause()
        if (wizard.step in (Step.MODEL, Step.KEY) and wizard.problem) or wizard.step is Step.MODEL:
            break
    return wizard


async def test_wai_starts_with_nothing_configured(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Loading must not depend on having a provider --- that is the whole point."""
    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.is_running
        assert app.configured_providers == []


async def test_first_run_opens_the_wizard_and_says_why(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from wai.tui.widgets.setup import SetupWizard

    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, SetupWizard):
                break
        assert isinstance(pilot.app.screen, SetupWizard)
        await pilot.press("escape")
        await pilot.pause()
        assert "No provider is configured" in _notices(pilot)


async def test_escape_leaves_a_usable_app(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Skipping setup must not trap you in a modal."""
    from wai.tui.widgets.setup import SetupWizard

    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, SetupWizard):
                break
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(pilot.app.screen, SetupWizard)
        await _type(pilot, "/help")
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "Commands:" in _notices(pilot)


async def test_wizard_stores_the_key_checks_it_and_lists_models(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import keyring

    from wai.config.secrets import KEYRING_SERVICE
    from wai.tui.widgets.setup import Step

    _no_credentials(monkeypatch)
    monkeypatch.setattr("wai.providers.registry.live_models", _returning(FAKE_MODELS))
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ == "SetupWizard":
                break
        wizard = await _reach_model_step(pilot)
        assert wizard.step is Step.MODEL
        assert wizard.provider == "anthropic"
        # The keyring here is the in-memory stand-in from conftest.
        assert keyring.get_password(KEYRING_SERVICE, "anthropic") == "sk-test"
        ids = {m.id for m in wizard.models}
        assert {"claude-opus-5", "claude-haiku-4-5"} <= ids


async def test_a_bad_key_fails_at_the_health_check_not_the_first_prompt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Finding out on your first message is the failure mode this replaces."""
    from wai.core.errors import AuthenticationError
    from wai.tui.widgets.setup import Step

    _no_credentials(monkeypatch)

    async def rejecting(name, config, **kw):  # type: ignore[no-untyped-def]
        raise AuthenticationError("invalid x-api-key", provider=name)

    monkeypatch.setattr("wai.providers.registry.live_models", rejecting)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ == "SetupWizard":
                break
        wizard = await _reach_model_step(pilot, "sk-wrong")
        assert wizard.step is Step.KEY, "it goes back so you can retype"
        assert "invalid x-api-key" in wizard.problem


async def test_finishing_sets_the_session_and_the_default_profile(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _no_credentials(monkeypatch)
    monkeypatch.setattr("wai.providers.registry.live_models", _returning(FAKE_MODELS))
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ == "SetupWizard":
                break
        await _reach_model_step(pilot)
        await pilot.press("enter")  # accept the highlighted model
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ != "SetupWizard":
                break
        assert app.session.provider == "anthropic"
        assert app.session.model in {m.id for m in FAKE_MODELS} | {
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-fable-5-1",
            "claude-haiku-4-5-20251001",
        }
        # Persisted, so the next session starts configured.
        assert app.config.profiles[app.config.default_profile].provider == "anthropic"


async def test_unmatched_filter_requeries_the_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The live search: if it is not in the first pull, ask again."""
    from textual.widgets import Input

    calls: list[str] = []
    later = [*FAKE_MODELS, ModelInfo(id="claude-brand-new-6", provider="anthropic")]

    async def growing(name, config, **kw):  # type: ignore[no-untyped-def]
        calls.append(name)
        return FAKE_MODELS if len(calls) == 1 else later

    _no_credentials(monkeypatch)
    monkeypatch.setattr("wai.providers.registry.live_models", growing)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ == "SetupWizard":
                break
        wizard = await _reach_model_step(pilot)
        assert len(calls) == 1, "the health check is the first pull"

        wizard.query_one("#filter", Input).value = "brand-new"
        for _ in range(80):
            await pilot.pause()
            if len(calls) > 1:
                break
        assert len(calls) == 2, "an unmatched filter re-queries the provider"
        assert any(m.id == "claude-brand-new-6" for m in wizard.models)


def _returning(models):  # type: ignore[no-untyped-def]
    async def _fetch(name, config, **kw):  # type: ignore[no-untyped-def]
        return list(models)

    return _fetch


async def test_model_picker_filters_and_falls_through_to_the_provider(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Not in the catalog must not mean not available."""
    from textual.widgets import Input

    from wai.config.secrets import CredentialStatus
    from wai.tui.widgets.model_picker import ModelPicker

    monkeypatch.setattr(
        "wai.config.secrets.credential_status", lambda name: CredentialStatus(name, True, "env")
    )
    calls: list[str] = []

    async def live(name, config, **kw):  # type: ignore[no-untyped-def]
        calls.append(name)
        return [ModelInfo(id="claude-unreleased-9", provider="anthropic")]

    monkeypatch.setattr("wai.providers.registry.live_models", live)

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/model")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ModelPicker):
                break
        picker = pilot.app.screen

        picker.query_one("#model-filter", Input).value = "sonnet"
        await pilot.pause()
        await pilot.pause()
        assert calls == [], "a local match must not cost a network call"

        picker.query_one("#model-filter", Input).value = "unreleased"
        for _ in range(60):
            await pilot.pause()
            if calls:
                break
        assert calls == ["anthropic"], "an unmatched filter asks the provider"
        assert any(m.id == "claude-unreleased-9" for m in picker.models)


async def test_model_picker_accepts_an_id_nothing_lists(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from textual.widgets import Input

    from wai.config.secrets import CredentialStatus
    from wai.tui.widgets.model_picker import ModelPicker

    monkeypatch.setattr(
        "wai.config.secrets.credential_status", lambda name: CredentialStatus(name, True, "env")
    )

    async def nothing(name, config, **kw):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr("wai.providers.registry.live_models", nothing)

    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/model")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ModelPicker):
                break
        picker = pilot.app.screen
        picker.query_one("#model-filter", Input).value = "some-brand-new-model"
        for _ in range(60):
            await pilot.pause()
            if picker.query_one("#model-options").option_count == 0:
                break
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause()
            if not isinstance(pilot.app.screen, ModelPicker):
                break
        assert app.session.model == "some-brand-new-model"


async def test_model_picker_says_so_when_there_is_no_key(monkeypatch) -> None:
    from textual.widgets import Input, Label

    from wai.tui.widgets.model_picker import ModelPicker

    _no_credentials(monkeypatch)
    app = make_app()
    async with app.run_test() as pilot:
        for _ in range(40):
            await pilot.pause()
            if pilot.app.screen.__class__.__name__ == "SetupWizard":
                break
        await pilot.press("escape")
        await pilot.pause()

        await _send(pilot, "/model")
        for _ in range(40):
            await pilot.pause()
            if isinstance(pilot.app.screen, ModelPicker):
                break
        picker = pilot.app.screen
        picker.query_one("#model-filter", Input).value = "nothing-like-this"
        await pilot.pause()
        await pilot.pause()
        note = str(picker.query_one("#model-note", Label).render())
        assert "no credentials" in note and "/setup" in note


def test_every_command_written_is_a_command_reachable() -> None:
    """`/profile` shipped defined but unregistered, so it existed in the source
    and not in the app. Nothing caught it because nothing asserted the two
    lists agree --- so assert it for every command, not just that one.
    """
    import inspect

    from wai.tui.commands import builtin

    written = {
        name.removeprefix("cmd_")
        for name, value in vars(builtin).items()
        if name.startswith("cmd_") and inspect.isfunction(value)
    }
    registered = {command.name for command in builtin.build_registry().unique}
    assert written - registered == set(), "defined but unreachable"


async def test_profile_command_is_reachable_from_the_app() -> None:
    app = make_app()
    async with app.run_test() as pilot:
        await _send(pilot, "/profile")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "Profiles:" in _notices(pilot)

        await _send(pilot, "/help")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert "/profile" in _notices(pilot), "and it is advertised"
