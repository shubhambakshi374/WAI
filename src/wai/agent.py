"""The agent loop: inference, tool execution, repeat.

Composes ``runner.stream_turn`` (one inference call) with tool execution. The
CLI, the TUI and the Phase 2 workflow engine all drive it, so it stays free of
any UI import.

Three invariants this module exists to hold:

* **Every ``tool_use`` gets a matching ``tool_result``** --- including when the
  turn is cancelled and when the iteration cap is hit. Anthropic and Bedrock
  reject a conversation containing an unanswered ``tool_use``, so skipping the
  results on Ctrl+C would leave a session that can never be resumed.
* **Reasoning blocks survive between iterations**, signature included, because
  the providers that verify them need them echoed back verbatim.
* **The loop terminates visibly.** Hitting the cap tells the model so, rather
  than stalling silently.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from wai.core.events import AgentEvent, IterationEnd, ToolFinished, ToolStarted
from wai.core.session import Session
from wai.core.types import Message, Role, StopReason, ToolResultBlock, ToolUseBlock
from wai.providers.base import Provider
from wai.runner import TurnAccumulator, stream_turn
from wai.tools.base import ToolContext
from wai.tools.registry import ToolRegistry
from wai.workspace import Workspace

if TYPE_CHECKING:
    from wai.config.models import Config

DEFAULT_MAX_ITERATIONS = 25
DEFAULT_OUTPUT_BUDGET = 100 * 1024
"""Total tool-result bytes per iteration. One unbounded read can otherwise
exhaust the context window and end the session."""

CANCELLED_RESULT = "Cancelled by the user before this tool ran."
BUDGET_NOTE = "\n\n[truncated: this turn's tool output budget is exhausted]"


def build_system_prompt(session: Session, registry: ToolRegistry) -> str | None:
    """Compose the profile's system prompt with a workspace preamble."""
    parts: list[str] = []
    if session.system:
        parts.append(session.system)
    if session.tools_enabled and len(registry):
        parts.append(
            "You have read-only access to a workspace on the user's machine, "
            f"rooted at {session.workspace_root}.\n"
            f"Tools available: {', '.join(registry.names)}.\n"
            "Paths are relative to the workspace root. You cannot read outside it, "
            "and credential files (.env, private keys, and similar) are blocked by "
            "design. Prefer glob and grep to locate code before reading whole files."
        )
    return "\n\n".join(parts) if parts else None


@dataclass
class _Batch:
    blocks: list[ToolResultBlock] = field(default_factory=list)
    finished: list[ToolFinished] = field(default_factory=list)


async def run_agent(
    provider: Provider,
    session: Session,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    output_budget: int = DEFAULT_OUTPUT_BUDGET,
    max_attempts: int = 4,
) -> AsyncIterator[AgentEvent]:
    """Drive a full turn to completion, executing tools as the model asks.

    Appends every message it produces to ``session`` and yields the wider
    ``AgentEvent`` union so callers can render tool activity.
    """
    tool_defs = registry.to_tool_defs() if session.tools_enabled else []
    system = build_system_prompt(session, registry)

    for iteration in range(max_iterations):
        request = session.to_request(tools=tool_defs)
        request.system = system
        acc = TurnAccumulator()

        try:
            async for event in stream_turn(provider, request, acc, max_attempts=max_attempts):
                yield event
        except asyncio.CancelledError:
            # Persist whatever streamed, and answer any tool_use it already
            # contains -- an unanswered one makes the session unresumable.
            _append_turn(session, acc, cancelled=True)
            raise

        _append_turn(session, acc, cancelled=False)
        pending = list(acc.tool_uses)

        if acc.stop_reason is not StopReason.TOOL_USE or not pending:
            yield IterationEnd(index=iteration, tool_calls=0, final=True)
            return

        for call in pending:
            yield ToolStarted(id=call.id, name=call.name, args=call.input)

        try:
            batch = await _execute_all(pending, registry, ctx, output_budget)
        except asyncio.CancelledError:
            session.append(_results_message(_cancelled_results(pending)))
            raise

        for finished in batch.finished:
            yield finished

        final = iteration == max_iterations - 1
        if final:
            batch.blocks[-1] = _with_cap_notice(batch.blocks[-1], max_iterations)

        session.append(_results_message(batch.blocks))
        yield IterationEnd(index=iteration, tool_calls=len(pending), final=final)
        if final:
            return


def _append_turn(session: Session, acc: TurnAccumulator, *, cancelled: bool) -> None:
    """Append the assistant message, plus stand-in results if we are unwinding."""
    message = acc.to_message()
    if not message.content:
        return
    session.append(message)
    session.record_usage(acc.usage)
    if cancelled and acc.tool_uses:
        session.append(_results_message(_cancelled_results(acc.tool_uses)))


async def _execute_all(
    calls: list[ToolUseBlock],
    registry: ToolRegistry,
    ctx: ToolContext,
    output_budget: int,
) -> _Batch:
    """Run every requested call concurrently. Read-only, so order is free."""
    began = time.monotonic()
    outcomes = await asyncio.gather(
        *(registry.execute(call.name, call.input, ctx) for call in calls)
    )
    elapsed = int((time.monotonic() - began) * 1000)

    batch = _Batch()
    remaining = output_budget
    for call, outcome in zip(calls, outcomes, strict=True):
        content, remaining = _clip(outcome.content, remaining)
        batch.blocks.append(
            ToolResultBlock(tool_use_id=call.id, content=content, is_error=outcome.is_error)
        )
        batch.finished.append(
            ToolFinished(
                id=call.id,
                name=call.name,
                summary=outcome.summary,
                is_error=outcome.is_error,
                duration_ms=elapsed,
            )
        )
    return batch


def _clip(content: str, remaining: int) -> tuple[str, int]:
    if remaining <= 0:
        return BUDGET_NOTE.strip(), 0
    if len(content) <= remaining:
        return content, remaining - len(content)
    return content[:remaining] + BUDGET_NOTE, 0


def _with_cap_notice(block: ToolResultBlock, cap: int) -> ToolResultBlock:
    """Carry the stop notice on the last real result, never as a dangling block."""
    return ToolResultBlock(
        tool_use_id=block.tool_use_id,
        content=(
            f"{block.content}\n\n[reached the {cap}-iteration limit; "
            "summarize what you found and what still needs doing]"
        ),
        is_error=block.is_error,
    )


def _cancelled_results(calls: list[ToolUseBlock]) -> list[ToolResultBlock]:
    return [
        ToolResultBlock(tool_use_id=c.id, content=CANCELLED_RESULT, is_error=True) for c in calls
    ]


def _results_message(blocks: list[ToolResultBlock]) -> Message:
    """Tool results always travel as a user message; adapters reshape from there."""
    return Message(role=Role.USER, content=list(blocks))


def unanswered_tool_uses(session: Session) -> list[str]:
    """Tool-use ids with no matching tool_result. Should always be empty.

    A non-empty result means the session cannot be resumed against Anthropic
    or Bedrock, so this is what the cancellation tests assert on.
    """
    answered: set[str] = set()
    requested: list[str] = []
    for message in session.messages:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                requested.append(block.id)
            elif isinstance(block, ToolResultBlock):
                answered.add(block.tool_use_id)
    return [i for i in requested if i not in answered]


def build_workspace(
    config: Config,
    *,
    root: str | Path | None = None,
    extra_roots: Sequence[str] = (),
) -> Workspace:
    """The workspace for a session: cwd plus any opt-in roots."""
    return Workspace(
        root=Path(root) if root is not None else Path.cwd(),
        extra_roots=tuple(Path(p) for p in (*config.workspace.extra_roots, *extra_roots)),
        deny_secrets=config.workspace.deny_secrets,
    )


def build_tool_context(config: Config, workspace: Workspace) -> ToolContext:
    return ToolContext(workspace=workspace, max_file_bytes=config.tools.max_file_bytes)
