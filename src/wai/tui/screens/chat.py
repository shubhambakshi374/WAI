"""The chat screen: transcript, composer, and the agent worker."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, ClassVar, cast

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.timer import Timer
from textual.widgets import Footer

from wai.agent import run_agent
from wai.core.errors import WaiError
from wai.core.events import (
    IterationEnd,
    MessageStart,
    ReasoningDelta,
    StreamError,
    TextDelta,
    ToolDenied,
    ToolFinished,
    ToolStarted,
    UsageUpdate,
)
from wai.core.types import Message, Role, Usage
from wai.tui.widgets.composer import Composer
from wai.tui.widgets.message_list import MessageBubble, MessageList
from wai.tui.widgets.model_picker import ModelPicker
from wai.tui.widgets.status_bar import StatusBar
from wai.tui.widgets.tool_call import ToolCallWidget

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
        self._tools: dict[str, ToolCallWidget] = {}
        self._text = ""
        self._reasoning = ""
        # NOT _running: MessagePump owns that name and sets it True once the
        # pump starts, which silently made every submit look like a busy turn.
        self._turn_active = False
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
            if message.text or message.reasoning:
                await transcript.add_message(message)
        self.query_one(Composer).focus()

    # ------------------------------------------------------------------ sending

    @on(Composer.Submitted)
    async def _on_submit(self, event: Composer.Submitted) -> None:
        if self._turn_active:
            self.notify("Still working — press Ctrl+C to cancel.", severity="warning")
            return
        app = self.wai
        message = Message.user(event.text)
        app.session.append(message)
        await self.query_one(MessageList).add_message(message)
        self.run_turn()

    @work(exclusive=True, group="turn")
    async def run_turn(self) -> None:
        app = self.wai
        status = self.query_one(StatusBar)
        status.state = "working…"
        self._turn_active = True
        self._tools.clear()
        # run_agent appends to the session itself; remember where we started so
        # only the new messages get persisted.
        start = len(app.session.messages) - 1
        # Buffered flush: Markdown.update re-parses the document, so writing
        # per token would make long replies crawl.
        interval = max(app.config.ui.stream_flush_ms, 10) / 1000
        self._flush_timer = self.set_interval(interval, self._flush)
        cancelled = False
        error: str | None = None

        try:
            async for event in run_agent(
                app.provider,
                app.session,
                app.registry,
                app.tool_ctx,
                max_iterations=app.config.tools.max_iterations,
                output_budget=app.config.tools.max_output_bytes,
            ):
                match event:
                    case MessageStart():
                        await self._begin_assistant()
                    case TextDelta():
                        self._text += event.text
                        self._dirty = True
                    case ReasoningDelta():
                        self._reasoning += event.text
                        self._dirty = True
                    case ToolStarted():
                        await self._flush()
                        await self._add_tool(event)
                    case ToolFinished():
                        self._finish_tool(event)
                    case ToolDenied():
                        self.notify(f"Rejected {event.name}.", severity="warning")
                    case UsageUpdate():
                        status.usage = app.session.usage
                    case StreamError():
                        self.notify(event.message, severity="error", timeout=10)
                    case IterationEnd():
                        await self._flush()
                        self._close_bubble()
        except asyncio.CancelledError:
            cancelled = True
            self._text += "\n\n_(cancelled)_"
            raise
        except WaiError as exc:
            error = str(exc)
            self.notify(error, severity="error", timeout=10)
        finally:
            await self._finish_turn(start, cancelled=cancelled, error=error)

    async def _finish_turn(self, start: int, *, cancelled: bool, error: str | None = None) -> None:
        if self._flush_timer is not None:
            self._flush_timer.stop()
            self._flush_timer = None
        self._dirty = True
        await self._flush()

        app = self.wai
        if error:
            # Persist the failure: a resumed session should show what happened.
            app.session.append(Message.assistant(f"**error:** {error}"))
            bubble = self._bubble or await self.query_one(MessageList).add_placeholder(
                Role.ASSISTANT
            )
            bubble.set_text(f"**error:** {error}")
            bubble.mark_error()
        for message in app.session.messages[start:]:
            app.store.append_message(app.session, message)
        app.store.update_header(app.session)

        for widget in self._tools.values():
            if widget.has_class("-running"):
                widget.finish(summary="cancelled", is_error=True)

        status = self._status()
        if status is not None:
            status.usage = app.session.usage
            status.state = "cancelled" if cancelled else "ready"
            status.granted = ", ".join(sorted(app.approvals.always_allowed))
        self._turn_active = False
        self._close_bubble()
        with contextlib.suppress(NoMatches):
            self.query_one(Composer).focus()

    def _status(self) -> StatusBar | None:
        """None once the screen is being torn down mid-turn."""
        try:
            return self.query_one(StatusBar)
        except NoMatches:
            return None

    # ------------------------------------------------------------------ helpers

    async def _begin_assistant(self) -> None:
        """One bubble per assistant message; the loop produces several."""
        await self._flush()
        self._close_bubble()
        self._bubble = await self.query_one(MessageList).add_placeholder(Role.ASSISTANT)

    def _close_bubble(self) -> None:
        self._bubble = None
        self._text = ""
        self._reasoning = ""

    async def _add_tool(self, event: ToolStarted) -> None:
        widget = ToolCallWidget(event.id, event.name, event.args)
        self._tools[event.id] = widget
        await self.query_one(MessageList).mount(widget)
        self.query_one(MessageList).scroll_end(animate=False)

    def _finish_tool(self, event: ToolFinished) -> None:
        widget = self._tools.get(event.id)
        if widget is not None:
            widget.finish(summary=event.summary, is_error=event.is_error)

    async def _flush(self) -> None:
        """Push buffered deltas into the transcript. Cheap when nothing changed."""
        if not self._dirty or self._bubble is None:
            return
        self._dirty = False
        self._bubble.set_text(self._text)
        if self._reasoning and self.wai.config.ui.show_reasoning:
            await self._bubble.set_reasoning(self._reasoning)
        self.query_one(MessageList).scroll_end(animate=False)

    # ------------------------------------------------------------------ actions

    def action_cancel_stream(self) -> None:
        if not self._turn_active:
            self.app.exit()
            return
        self.workers.cancel_group(self, "turn")

    async def action_new_session(self) -> None:
        if self._turn_active:
            self.notify("Cancel the current turn first.", severity="warning")
            return
        app = self.wai
        app.start_new_session()
        self.query_one(MessageList).clear_messages()
        self._tools.clear()
        status = self.query_one(StatusBar)
        status.usage = Usage()
        status.state = "ready"
        self.notify("Started a new session.")

    @work
    async def action_pick_model(self) -> None:
        if self._turn_active:
            self.notify("Cancel the current turn first.", severity="warning")
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
