"""Tool lookup and dispatch.

``execute`` never raises. Every failure --- unknown tool, bad arguments, a
denied path, an OS error --- comes back as an error ``ToolOutcome`` so the
agent loop can hand it to the model as an error ``tool_result``. A raised
exception would abort a turn the model could have recovered from.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

from wai.core.errors import PathNotAllowed, ToolError
from wai.core.types import ToolDef
from wai.tools.base import Tool, ToolContext, ToolOutcome
from wai.tools.edit import DeletePathTool, EditFileTool, WriteFileTool
from wai.tools.fs import GlobTool, GrepTool, ListDirTool, ReadFileTool

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def is_read_only(self, name: str) -> bool:
        """Unknown tools count as mutating, so they take the cautious path."""
        tool = self._tools.get(name)
        return bool(tool and tool.read_only)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def to_tool_defs(self) -> list[ToolDef]:
        """The provider-facing declarations, in a stable order."""
        return [
            ToolDef(name=tool.name, description=tool.description, input_schema=tool.input_schema)
            for tool in sorted(self._tools.values(), key=lambda t: t.name)
        ]

    async def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome.error(
                f"unknown tool {name!r}. Available: {', '.join(self.names) or 'none'}",
                summary="unknown tool",
            )
        if not isinstance(args, dict):
            return ToolOutcome.error("tool arguments must be an object", summary="bad arguments")
        try:
            return await tool.run(args, ctx)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        except ToolError as exc:
            return ToolOutcome.error(str(exc), summary="failed")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.warning("tool %s failed: %s", name, exc, exc_info=True)
            return ToolOutcome.error(f"{name} failed: {exc}", summary="failed")


def default_registry(*, writes: bool = True, kubernetes: bool | None = None) -> ToolRegistry:
    """The tool set for a session.

    Kubernetes tools register only when the SDK is installed, so a missing
    extra is a visible absence rather than an import error at call time.
    """
    tools: list[Tool] = [ReadFileTool(), ListDirTool(), GlobTool(), GrepTool()]
    if writes:
        tools += [WriteFileTool(), EditFileTool(), DeletePathTool()]

    if kubernetes is None:
        from wai.cloud.base import integration

        entry = integration("k8s")
        kubernetes = bool(entry and entry.available)
    if kubernetes:
        from wai.tools.k8s import k8s_tools

        tools += list(k8s_tools())
    return ToolRegistry(tools)
