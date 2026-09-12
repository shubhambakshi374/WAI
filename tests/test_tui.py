"""TUI tests. Driven by a scripted fake provider — no network."""

from __future__ import annotations

import asyncio
import contextlib

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
