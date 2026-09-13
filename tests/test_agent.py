"""Agent loop tests. Driven by a scripted fake provider --- no network."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest

from wai.agent import (
    build_system_prompt,
    run_agent,
    unanswered_tool_uses,
)
from wai.config.models import ProviderSettings
from wai.core.events import (
    IterationEnd,
    MessageEnd,
    MessageStart,
    StreamEvent,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from wai.core.session import Session
from wai.core.types import (
    ChatRequest,
    Message,
    Role,
    StopReason,
    ToolResultBlock,
    Usage,
)
from wai.providers.base import BaseProvider, ProviderCapabilities
from wai.tools import ToolContext, default_registry
from wai.workspace import Workspace


class ScriptedProvider(BaseProvider):
    """Replays one scripted event list per iteration, in order."""

    name = "scripted"
    capabilities = ProviderCapabilities()

    def __init__(self, turns: Sequence[Sequence[StreamEvent]]) -> None:
        super().__init__(api_key="k", settings=ProviderSettings())
        self.turns = [list(t) for t in turns]
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.turns) - 1)
        for event in self.turns[index]:
            yield event


def tool_turn(*calls: tuple[str, str, dict[str, Any]]) -> list[StreamEvent]:
    """A turn whose stop reason is TOOL_USE, requesting the given calls."""
    from wai.core.events import ToolCallEnd, ToolCallStart

    events: list[StreamEvent] = [MessageStart(model="scripted")]
    for i, (call_id, name, args) in enumerate(calls):
        events.append(ToolCallStart(index=i, id=call_id, name=name))
        events.append(ToolCallEnd(index=i, id=call_id, name=name, input=args))
    events.append(
        MessageEnd(stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=5, output_tokens=2))
    )
    return events


def text_turn(text: str = "Done.") -> list[StreamEvent]:
    return [
        MessageStart(model="scripted"),
        TextDelta(text=text),
        MessageEnd(stop_reason=StopReason.END_TURN, usage=Usage(input_tokens=5, output_tokens=3)),
    ]


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "main.py").write_text("def hello():\n    return 1\n")
    (root / "README.md").write_text("# Repo\n")
    return root


@pytest.fixture
def ctx(tree: Path) -> ToolContext:
    return ToolContext(workspace=Workspace(root=tree))


@pytest.fixture
def session(tree: Path) -> Session:
    s = Session(provider="scripted", model="m", workspace_root=str(tree))
    s.append(Message.user("what is in this repo?"))
    return s


@pytest.fixture
def registry():  # type: ignore[no-untyped-def]
    return default_registry(kubernetes=False)


async def drive(provider, session, registry, ctx, **kw):  # type: ignore[no-untyped-def]
    return [e async for e in run_agent(provider, session, registry, ctx, **kw)]


# ------------------------------------------------------------------- happy path


async def test_no_tool_call_ends_in_one_iteration(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([text_turn("Hello.")])
    events = await drive(provider, session, registry, ctx)

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 1 and ends[0].final and ends[0].tool_calls == 0
    assert session.messages[-1].text == "Hello."
    assert len(provider.requests) == 1


async def test_single_tool_call_round_trip(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider(
        [tool_turn(("t1", "read_file", {"path": "src/main.py"})), text_turn("It defines hello.")]
    )
    events = await drive(provider, session, registry, ctx)

    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert [e.name for e in started] == ["read_file"]
    assert finished[0].is_error is False
    assert "read 2 lines" in finished[0].summary

    # user, assistant(tool_use), user(tool_result), assistant(text)
    assert [m.role for m in session.messages] == [
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]
    result = session.messages[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert "def hello():" in result.content
    assert unanswered_tool_uses(session) == []


async def test_tools_are_declared_on_the_request(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([text_turn()])
    await drive(provider, session, registry, ctx)
    names = [t.name for t in provider.requests[0].tools]
    assert names == registry.names
    assert "write_file" in names and "read_file" in names


async def test_parallel_tool_calls_all_answered(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider(
        [
            tool_turn(
                ("a", "read_file", {"path": "src/main.py"}),
                ("b", "list_dir", {}),
                ("c", "glob", {"pattern": "**/*.py"}),
            ),
            text_turn(),
        ]
    )
    events = await drive(provider, session, registry, ctx)

    assert len([e for e in events if isinstance(e, ToolStarted)]) == 3
    assert len([e for e in events if isinstance(e, ToolFinished)]) == 3
    results = session.messages[2].content
    assert len(results) == 3
    assert {b.tool_use_id for b in results if isinstance(b, ToolResultBlock)} == {"a", "b", "c"}
    assert unanswered_tool_uses(session) == []


async def test_tool_error_is_fed_back_not_raised(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider(
        [tool_turn(("t1", "read_file", {"path": "/etc/passwd"})), text_turn("I cannot.")]
    )
    events = await drive(provider, session, registry, ctx)

    finished = next(e for e in events if isinstance(e, ToolFinished))
    assert finished.is_error and finished.summary == "denied"
    result = session.messages[2].content[0]
    assert isinstance(result, ToolResultBlock) and result.is_error
    assert "outside the workspace" in result.content
    assert session.messages[-1].text == "I cannot."


async def test_unknown_tool_is_fed_back(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([tool_turn(("t1", "rm_rf", {})), text_turn()])
    await drive(provider, session, registry, ctx)
    result = session.messages[2].content[0]
    assert isinstance(result, ToolResultBlock) and "unknown tool" in result.content


async def test_multi_iteration_loop(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider(
        [
            tool_turn(("t1", "list_dir", {})),
            tool_turn(("t2", "read_file", {"path": "src/main.py"})),
            text_turn("Two files."),
        ]
    )
    events = await drive(provider, session, registry, ctx)

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert [e.tool_calls for e in ends] == [1, 1, 0]
    assert ends[-1].final
    assert unanswered_tool_uses(session) == []


# ------------------------------------------------------------------------ caps


async def test_iteration_cap_terminates_visibly(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    """The model must be told it was stopped, not left appearing to stall."""
    provider = ScriptedProvider([tool_turn(("t1", "list_dir", {}))])  # loops forever
    events = await drive(provider, session, registry, ctx, max_iterations=3)

    ends = [e for e in events if isinstance(e, IterationEnd)]
    assert len(ends) == 3 and ends[-1].final
    last_result = session.messages[-1].content[-1]
    assert isinstance(last_result, ToolResultBlock)
    assert "reached the 3-iteration limit" in last_result.content
    assert unanswered_tool_uses(session) == [], "the cap notice must not dangle"


async def test_output_budget_truncates(session, registry, ctx, tree) -> None:  # type: ignore[no-untyped-def]
    (tree / "big.txt").write_text("y" * 20_000 + "\n")
    provider = ScriptedProvider([tool_turn(("t1", "read_file", {"path": "big.txt"})), text_turn()])
    await drive(provider, session, registry, ctx, output_budget=500)
    result = session.messages[2].content[0]
    assert isinstance(result, ToolResultBlock)
    assert len(result.content) < 800
    assert "budget is exhausted" in result.content


async def test_budget_is_shared_across_parallel_calls(session, registry, ctx, tree) -> None:  # type: ignore[no-untyped-def]
    (tree / "a.txt").write_text("a" * 5000)
    (tree / "b.txt").write_text("b" * 5000)
    provider = ScriptedProvider(
        [
            tool_turn(("a", "read_file", {"path": "a.txt"}), ("b", "read_file", {"path": "b.txt"})),
            text_turn(),
        ]
    )
    await drive(provider, session, registry, ctx, output_budget=600)
    blocks = [b for b in session.messages[2].content if isinstance(b, ToolResultBlock)]
    assert sum(len(b.content) for b in blocks) < 1200
    assert any("budget is exhausted" in b.content for b in blocks)


# ---------------------------------------------------------------- cancellation


async def test_cancelling_during_tool_execution_leaves_no_orphan(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    """The invariant that keeps a cancelled session resumable.

    Anthropic and Bedrock reject a conversation with an unanswered tool_use,
    so a Ctrl+C mid-execution must still write stand-in results.
    """
    provider = ScriptedProvider([tool_turn(("t1", "read_file", {"path": "src/main.py"}))])

    class Slow:
        name = "read_file"
        description = ""
        input_schema: ClassVar[dict[str, Any]] = {}
        read_only = True

        async def run(self, args, c):  # type: ignore[no-untyped-def]
            await asyncio.Event().wait()

    from wai.tools.registry import ToolRegistry

    slow_registry = ToolRegistry([Slow()])  # type: ignore[list-item]
    task = asyncio.create_task(drive(provider, session, slow_registry, ctx))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert unanswered_tool_uses(session) == []
    last = session.messages[-1].content[0]
    assert isinstance(last, ToolResultBlock) and last.is_error
    assert "Cancelled" in last.content


async def test_cancelling_during_streaming_leaves_no_orphan(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    """Same invariant when the cancel lands mid-stream, after a tool_use arrived."""
    from wai.core.events import ToolCallEnd, ToolCallStart

    class Hanging(ScriptedProvider):
        async def stream(self, request):  # type: ignore[no-untyped-def]
            self.requests.append(request)
            yield MessageStart(model="scripted")
            yield ToolCallStart(index=0, id="t1", name="read_file")
            yield ToolCallEnd(index=0, id="t1", name="read_file", input={"path": "src/main.py"})
            await asyncio.Event().wait()

    task = asyncio.create_task(drive(Hanging([]), session, registry, ctx))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert unanswered_tool_uses(session) == []


async def test_cancelled_session_survives_anthropic_translation(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    """End-to-end proof: the cancelled session is still a valid conversation."""
    from wai.providers.anthropic import _to_anthropic_message

    provider = ScriptedProvider([tool_turn(("t1", "list_dir", {})), text_turn()])
    await drive(provider, session, registry, ctx)

    wire = [_to_anthropic_message(m) for m in session.messages]
    requested = {b["id"] for m in wire for b in m["content"] if b.get("type") == "tool_use"}
    answered = {
        b["tool_use_id"] for m in wire for b in m["content"] if b.get("type") == "tool_result"
    }
    assert requested and requested == answered


# --------------------------------------------------------------- system prompt


def test_system_prompt_names_the_workspace_and_tools(session, registry) -> None:  # type: ignore[no-untyped-def]
    prompt = build_system_prompt(session, registry)
    assert prompt is not None
    assert session.workspace_root in prompt
    assert "read_file" in prompt and "grep" in prompt


def test_system_prompt_keeps_the_profile_prompt_first(session, registry) -> None:  # type: ignore[no-untyped-def]
    session.system = "You are a careful SRE."
    prompt = build_system_prompt(session, registry)
    assert prompt is not None and prompt.startswith("You are a careful SRE.")


def test_system_prompt_omits_tools_when_disabled(session, registry) -> None:  # type: ignore[no-untyped-def]
    session.tools_enabled = False
    session.system = "Be terse."
    assert build_system_prompt(session, registry) == "Be terse."


async def test_disabling_tools_declares_none(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    session.tools_enabled = False
    provider = ScriptedProvider([text_turn()])
    await drive(provider, session, registry, ctx)
    assert provider.requests[0].tools == []


# -------------------------------------------------------------------- fidelity


async def test_reasoning_blocks_survive_between_iterations(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    """Providers that verify thinking signatures need them echoed back verbatim."""
    from wai.core.events import ReasoningDelta
    from wai.core.types import ReasoningBlock

    turn = tool_turn(("t1", "list_dir", {}))
    turn.insert(1, ReasoningDelta(text="let me look"))
    turn.insert(2, ReasoningDelta(text="", signature="sig-1"))
    provider = ScriptedProvider([turn, text_turn()])
    await drive(provider, session, registry, ctx)

    assistant = session.messages[1]
    reasoning = [b for b in assistant.content if isinstance(b, ReasoningBlock)]
    assert reasoning and reasoning[0].signature == "sig-1"
    # And it is still there on the next request the provider received.
    replayed = provider.requests[1].messages[1]
    assert any(isinstance(b, ReasoningBlock) for b in replayed.content)


async def test_tool_use_precedes_tool_result_in_history(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([tool_turn(("t1", "list_dir", {})), text_turn()])
    await drive(provider, session, registry, ctx)
    kinds = [type(b).__name__ for m in session.messages for b in m.content]
    assert kinds.index("ToolUseBlock") < kinds.index("ToolResultBlock")


async def test_usage_accumulates_across_iterations(session, registry, ctx) -> None:  # type: ignore[no-untyped-def]
    provider = ScriptedProvider([tool_turn(("t1", "list_dir", {})), text_turn()])
    await drive(provider, session, registry, ctx)
    assert session.usage.output_tokens == 5  # 2 from the tool turn, 3 from the text turn
