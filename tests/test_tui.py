"""TUI tests. Driven by a scripted fake provider — no network."""

from __future__ import annotations

import asyncio
import contextlib

from textual.widgets import Label
from textual.worker import WorkerCancelled

from tests.conftest import FakeProvider
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
        assert "partial" in app.session.messages[-1].text
        assert "cancelled" in app.session.messages[-1].text
        # The partial turn is still persisted.
        assert "partial" in app.store.load(app.session.id).messages[-1].text


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
        assert "streaming" in str(state.render())

        await pilot.press("ctrl+c")
        with contextlib.suppress(WorkerCancelled):
            await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert str(state.render()) == "cancelled"
