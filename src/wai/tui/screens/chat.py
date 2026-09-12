"""The chat screen: transcript, composer, and the streaming worker."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, ClassVar, cast

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer

from wai.core.errors import WaiError
from wai.core.events import MessageStart, ReasoningDelta, StreamError, TextDelta, UsageUpdate
from wai.core.types import Message, Role, Usage
from wai.runner import TurnAccumulator, stream_turn
from wai.tui.widgets.composer import Composer
from wai.tui.widgets.message_list import MessageBubble, MessageList
from wai.tui.widgets.model_picker import ModelPicker
from wai.tui.widgets.status_bar import StatusBar

if TYPE_CHECKING:
    from wai.tui.app import WaiApp


class ChatScreen(Screen[None]):
    # priority=True is required, not cosmetic: without it the focused
    # TextArea swallows ctrl+c (its copy binding) and the App swallows
    # ctrl+p and ctrl+q, leaving cancel and the model picker unreachable.
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+p", "pick_model", "Model", priority=True),
        Binding("ctrl+n", "new_session", "New", priority=True),
        Binding("ctrl+c", "cancel_stream", "Cancel", priority=True),
        Binding("ctrl+q", "quit_app", "Quit", priority=True),
    ]

    @property
    def wai(self) -> WaiApp:
        """The app, typed. ``Screen.app`` is only ever ``App[Any]``."""
        return cast("WaiApp", self.app)

    def __init__(self) -> None:
        super().__init__()
        self._bubble: MessageBubble | None = None
        self._acc: TurnAccumulator | None = None
        self._flush_timer: Timer | None = None
        self._dirty = False

    def compose(self) -> ComposeResult:
        yield StatusBar(id="status")
        yield MessageList(id="transcript")
        yield Composer(id="composer")
        yield Footer()

    async def on_mount(self) -> None:
        app = self.wai
        status = self.query_one(StatusBar)
        status.provider = app.session.provider
        status.model = app.session.model
        status.usage = app.session.usage
        transcript = self.query_one(MessageList)
        for message in app.session.messages:
            await transcript.add_message(message)
        self.query_one(Composer).focus()

    # ------------------------------------------------------------------ sending

    @on(Composer.Submitted)
    async def _on_submit(self, event: Composer.Submitted) -> None:
        if self._acc is not None:
            self.notify("Still streaming — press Ctrl+C to cancel.", severity="warning")
            return
        app = self.wai
        message = Message.user(event.text)
        app.session.append(message)
        app.store.append_message(app.session, message)
        await self.query_one(MessageList).add_message(message)
        self._bubble = await self.query_one(MessageList).add_placeholder(Role.ASSISTANT)
        self.run_turn()

    @work(exclusive=True, group="turn")
    async def run_turn(self) -> None:
        app = self.wai
        status = self.query_one(StatusBar)
        status.state = "streaming…"
        acc = TurnAccumulator()
        self._acc = acc
        # Buffered flush: Markdown.update re-parses the document, so writing
        # per token would make long replies crawl.
        interval = max(app.config.ui.stream_flush_ms, 10) / 1000
        self._flush_timer = self.set_interval(interval, self._flush)
        cancelled = False
        try:
            async for event in stream_turn(app.provider, app.session.to_request(), acc):
                match event:
                    case TextDelta() | ReasoningDelta():
                        self._dirty = True
                    case MessageStart() | UsageUpdate():
                        status.usage = app.session.usage + acc.usage
                    case StreamError():
                        self.notify(event.message, severity="error", timeout=10)
        except asyncio.CancelledError:
            cancelled = True
            acc.text += "\n\n_(cancelled)_"
            raise
        except WaiError as exc:
            acc.error = str(exc)
            acc.text += f"\n\n**error:** {exc}"
            self.notify(str(exc), severity="error", timeout=10)
        finally:
            await self._finish_turn(acc, cancelled=cancelled)

    async def _finish_turn(self, acc: TurnAccumulator, *, cancelled: bool) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._dirty = True
        await self._flush()
        app = self.wai
        message = acc.to_message()
        if message.content:
            app.session.append(message)
            app.session.record_usage(acc.usage)
            app.store.append_message(app.session, message)
            app.store.update_header(app.session)
        if acc.error and self._bubble is not None:
            self._bubble.mark_error()
        status = self.query_one(StatusBar)
        status.usage = app.session.usage
        status.state = "cancelled" if cancelled else "ready"
        self._acc = None
        self._bubble = None
        self.query_one(Composer).focus()

    async def _flush(self) -> None:
        """Push buffered deltas into the transcript. Cheap when nothing changed."""
        if not self._dirty or self._bubble is None or self._acc is None:
            return
        self._dirty = False
        self._bubble.set_text(self._acc.text)
        if self._acc.reasoning and self.wai.config.ui.show_reasoning:
            await self._bubble.set_reasoning(self._acc.reasoning)
        self.query_one(MessageList).scroll_end(animate=False)

    # ------------------------------------------------------------------ actions

    def action_cancel_stream(self) -> None:
        if self._acc is None:
            self.app.exit()
            return
        self.workers.cancel_group(self, "turn")

    async def action_new_session(self) -> None:
        if self._acc is not None:
            self.notify("Cancel the stream first.", severity="warning")
            return
        app = self.wai
        app.start_new_session()
        self.query_one(MessageList).clear_messages()
        status = self.query_one(StatusBar)
        status.usage = Usage()
        status.state = "ready"
        self.notify("Started a new session.")

    @work
    async def action_pick_model(self) -> None:
        if self._acc is not None:
            self.notify("Cancel the stream first.", severity="warning")
            return
        chosen = await self.app.push_screen_wait(ModelPicker())
        if chosen is None:
            return
        app = self.wai
        await app.switch_model(chosen)
        status = self.query_one(StatusBar)
        status.provider = app.session.provider
        status.model = app.session.model
        self.notify(f"Switched to {chosen.label} ({chosen.provider}).")

    def action_quit_app(self) -> None:
        self.app.exit()
