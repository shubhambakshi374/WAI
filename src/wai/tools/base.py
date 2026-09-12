"""The tool contract.

Tool failures are *data*, not exceptions: a tool that cannot do its job
returns an error ``ToolOutcome`` which the agent loop hands back to the model
as an error ``tool_result``. Raising into the loop would abort a turn that the
model could have recovered from by trying a different path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from wai.workspace import Workspace

DEFAULT_MAX_FILE_BYTES = 256 * 1024
DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_ENTRIES = 500
DEFAULT_MAX_MATCHES = 200


@dataclass
class ToolContext:
    """What a tool is allowed to touch.

    Phase 3 seam: an approval policy and a cancellation token belong here.
    Neither is built yet --- every tool in this increment is read-only.
    """

    workspace: Workspace
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    max_lines: int = DEFAULT_MAX_LINES
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_matches: int = DEFAULT_MAX_MATCHES


@dataclass
class ToolOutcome:
    """A tool's result: what the model sees, plus a label for the UI."""

    content: str
    is_error: bool = False
    summary: str = ""

    @classmethod
    def error(cls, message: str, *, summary: str = "") -> ToolOutcome:
        return cls(content=message, is_error=True, summary=summary or "failed")


@runtime_checkable
class Tool(Protocol):
    """What the registry, the agent loop and the Phase 2 engine may assume."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, Any]: ...

    @property
    def read_only(self) -> bool: ...

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome: ...


class BaseTool(ABC):
    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    read_only: ClassVar[bool] = True
    input_schema: ClassVar[dict[str, Any]] = {}

    @abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        """Do the work. Return an error outcome rather than raising."""


def truncated_note(shown: int, total: int, unit: str) -> str:
    return f"\n\n[truncated: showing {shown} of {total} {unit}]"
