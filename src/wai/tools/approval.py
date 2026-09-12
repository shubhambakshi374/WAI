"""The approval gate for tools that change things.

Read-only tools never reach this. Every mutating tool must get a ``Decision``
before it touches disk, and the default policy is ``DenyAll`` --- failing
closed matters more than convenience when the caller forgot to wire a policy.

The escalation path (``ALLOW_ALWAYS``) is where safety actually erodes, so it
is deliberately narrow: scoped to one tool, held in memory for one session,
never written to disk, and visible in the UI while it is active.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class Decision(StrEnum):
    ALLOW = "allow"
    ALLOW_ALWAYS = "allow_always"
    """Allow this, and every later call to the same tool this session."""
    DENY = "deny"


@dataclass(frozen=True)
class ApprovalRequest:
    """Everything a human needs to decide, assembled before anything happens."""

    tool: str
    action: str
    """Short verb for the prompt: create, overwrite, edit, delete."""
    path: str
    """Workspace-relative, for display."""
    diff: str = ""
    """Unified diff, or a summary when a diff makes no sense."""
    recoverability: str = ""
    """Whether git could get this back. The key fact for a delete."""
    destructive: bool = False

    @property
    def summary(self) -> str:
        return f"{self.action} {self.path}"


@runtime_checkable
class ApprovalPolicy(Protocol):
    async def request(self, req: ApprovalRequest) -> Decision: ...


class DenyAll:
    """The default. A tool with no policy wired must not be able to write."""

    reason = "no approval policy is configured, so changes are refused"

    async def request(self, req: ApprovalRequest) -> Decision:
        return Decision.DENY


class AllowAll:
    """Non-interactive consent: `--yes`, and tests."""

    async def request(self, req: ApprovalRequest) -> Decision:
        return Decision.ALLOW


@dataclass
class RecordingPolicy:
    """AllowAll that remembers what it was asked. For tests."""

    decision: Decision = Decision.ALLOW
    seen: list[ApprovalRequest] = field(default_factory=list)

    async def request(self, req: ApprovalRequest) -> Decision:
        self.seen.append(req)
        return self.decision


class SessionApprovals:
    """Wraps a policy with the session-scoped ``always allow`` bookkeeping.

    Also serializes prompts. Without the lock, two writes in one turn would
    race to put a modal on screen; the agent loop already runs mutating tools
    sequentially, and this is the second belt.
    """

    def __init__(self, delegate: ApprovalPolicy) -> None:
        self._delegate = delegate
        self._always: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def always_allowed(self) -> frozenset[str]:
        return frozenset(self._always)

    def revoke_all(self) -> None:
        self._always.clear()

    async def request(self, req: ApprovalRequest) -> Decision:
        if req.tool in self._always:
            return Decision.ALLOW
        async with self._lock:
            # Re-check: an earlier queued prompt may have granted it.
            if req.tool in self._always:
                return Decision.ALLOW
            decision = await self._delegate.request(req)
        if decision is Decision.ALLOW_ALWAYS:
            self._always.add(req.tool)
            return Decision.ALLOW
        return decision
