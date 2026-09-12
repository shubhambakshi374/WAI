"""The mutating tools: write_file, edit_file, delete_path.

Each one follows the same shape, and the order matters:

1. resolve through the workspace (containment + secret denylist),
2. work out exactly what would change,
3. ask the approval policy, with a diff and a recoverability note,
4. only then touch the disk, atomically.

Nothing writes before step 3. The approval prompt shows the real diff, not a
description of one, because approving a change you cannot see is not consent.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from wai.core.errors import PathNotAllowed
from wai.tools._git import recoverability
from wai.tools.approval import ApprovalRequest, Decision
from wai.tools.base import BaseTool, ToolContext, ToolOutcome
from wai.tools.fs import _looks_binary

DENY_WRITE_COMPONENTS: frozenset[str] = frozenset({".git"})
"""Never writable, whatever the workspace says. Corrupting the object store is
not a recoverable mistake, and it is exactly where a confused model reaches."""

MAX_DIFF_LINES = 200
DENIED = "The user rejected this change."


def _check_writable(target: Path, ctx: ToolContext) -> str | None:
    if any(part in DENY_WRITE_COMPONENTS for part in target.parts):
        return f"{ctx.workspace.relative(target)} is inside a .git directory and is never writable"
    return None


def _diff(before: str, after: str, label: str) -> str:
    lines = list(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            n=3,
        )
    )
    if not lines:
        return "(no textual change)"
    if len(lines) > MAX_DIFF_LINES:
        shown = lines[:MAX_DIFF_LINES]
        return "".join(shown) + f"\n… {len(lines) - MAX_DIFF_LINES} more diff lines"
    return "".join(lines)


def _read_text(path: Path) -> str | None:
    """Existing content, or None when the file is absent or unreadable as text."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if _looks_binary(raw):
        return None
    return raw.decode("utf-8", errors="replace")


def _atomic_write(target: Path, content: str) -> None:
    """Write via a temp file in the same directory, then rename.

    A crash or a full disk mid-write leaves the original intact rather than a
    half-written file.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        if target.exists():
            shutil.copymode(target, tmp)
        os.replace(tmp, target)
    except BaseException:
        # Cleanup must never mask the real error.
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


async def _ask(ctx: ToolContext, request: ApprovalRequest) -> tuple[bool, ToolOutcome | None]:
    decision = await ctx.approvals.request(request)
    if decision is Decision.DENY:
        return False, ToolOutcome.rejected(DENIED)
    return True, None


class WriteFileTool(BaseTool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create a new file, or replace the entire contents of an existing one. "
        "Requires user approval. For changing part of an existing file prefer "
        "edit_file, which is cheaper and easier for the user to review."
    )
    read_only: ClassVar[bool] = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."},
            "content": {"type": "string", "description": "The complete file contents."},
        },
        "required": ["path", "content"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return ToolOutcome.error("path is required")
        if "content" not in args:
            return ToolOutcome.error("content is required")
        content = str(args["content"])

        try:
            target = ctx.workspace.resolve(raw_path)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        if (blocked := _check_writable(target, ctx)) is not None:
            return ToolOutcome.error(blocked, summary="denied")
        if target.is_dir():
            return ToolOutcome.error(
                f"{ctx.workspace.relative(target)} is a directory", summary="is a directory"
            )

        label = ctx.workspace.relative(target)
        existed = target.exists()
        before = _read_text(target) if existed else ""
        if existed and before is None:
            return ToolOutcome.error(
                f"{label} exists and is not a text file; refusing to overwrite it",
                summary="binary",
            )
        if existed and before == content:
            return ToolOutcome(content=f"{label} already has these contents.", summary="no change")

        ok, refusal = await _ask(
            ctx,
            ApprovalRequest(
                tool=self.name,
                action="create" if not existed else "overwrite",
                path=label,
                diff=_diff(before or "", content, label),
                recoverability=(await recoverability(target)) if existed else "",
                destructive=existed,
            ),
        )
        if not ok:
            return refusal or ToolOutcome.rejected(DENIED)

        try:
            await asyncio.to_thread(_atomic_write, target, content)
        except OSError as exc:
            return ToolOutcome.error(f"could not write {label}: {exc}", summary="failed")

        lines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        verb = "wrote" if existed else "created"
        return ToolOutcome(
            content=f"{verb} {label} ({lines} lines)",
            summary=f"{verb} {label}",
        )


class EditFileTool(BaseTool):
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace an exact string in a file. old_string must appear exactly once "
        "unless replace_all is true, so include enough surrounding context to be "
        "unambiguous. Requires user approval."
    )
    read_only: ClassVar[bool] = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return ToolOutcome.error("path is required")
        if "old_string" not in args or "new_string" not in args:
            return ToolOutcome.error("old_string and new_string are required")
        old = str(args["old_string"])
        new = str(args["new_string"])
        replace_all = bool(args.get("replace_all"))
        if not old:
            return ToolOutcome.error("old_string must not be empty; use write_file to create")
        if old == new:
            return ToolOutcome.error("old_string and new_string are identical")

        try:
            target = ctx.workspace.resolve(raw_path)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        if (blocked := _check_writable(target, ctx)) is not None:
            return ToolOutcome.error(blocked, summary="denied")

        label = ctx.workspace.relative(target)
        if not target.exists():
            return ToolOutcome.error(f"no such file: {label}", summary="not found")
        before = _read_text(target)
        if before is None:
            return ToolOutcome.error(f"{label} is not a text file", summary="binary")

        occurrences = before.count(old)
        if occurrences == 0:
            return ToolOutcome.error(
                f"old_string was not found in {label}. It must match exactly, "
                "including whitespace and indentation.",
                summary="no match",
            )
        if occurrences > 1 and not replace_all:
            return ToolOutcome.error(
                f"old_string appears {occurrences} times in {label}. Include more "
                "surrounding context to make it unique, or pass replace_all.",
                summary=f"{occurrences} matches",
            )

        after = before.replace(old, new) if replace_all else before.replace(old, new, 1)

        ok, refusal = await _ask(
            ctx,
            ApprovalRequest(
                tool=self.name,
                action="edit",
                path=label,
                diff=_diff(before, after, label),
                recoverability=await recoverability(target),
                destructive=True,
            ),
        )
        if not ok:
            return refusal or ToolOutcome.rejected(DENIED)

        try:
            await asyncio.to_thread(_atomic_write, target, after)
        except OSError as exc:
            return ToolOutcome.error(f"could not write {label}: {exc}", summary="failed")

        count = occurrences if replace_all else 1
        return ToolOutcome(
            content=f"edited {label} ({count} replacement{'s' if count != 1 else ''})",
            summary=f"edited {label}",
        )


class DeletePathTool(BaseTool):
    name: ClassVar[str] = "delete_path"
    description: ClassVar[str] = (
        "Delete a file, or an empty directory. Pass recursive=true to delete a "
        "directory and everything under it. Requires user approval and cannot be "
        "undone unless git has the file."
    )
    read_only: ClassVar[bool] = False
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "recursive": {
                "type": "boolean",
                "description": "Required to delete a non-empty directory.",
            },
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return ToolOutcome.error("path is required")
        recursive = bool(args.get("recursive"))

        try:
            target = ctx.workspace.resolve(raw_path)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        if (blocked := _check_writable(target, ctx)) is not None:
            return ToolOutcome.error(blocked, summary="denied")

        label = ctx.workspace.relative(target)
        if target == ctx.workspace.root:
            return ToolOutcome.error("refusing to delete the workspace root", summary="denied")
        if not target.exists() and not target.is_symlink():
            return ToolOutcome.error(f"no such path: {label}", summary="not found")

        is_dir = target.is_dir() and not target.is_symlink()
        if is_dir:
            contents = list(target.iterdir())
            if contents and not recursive:
                return ToolOutcome.error(
                    f"{label} is not empty ({len(contents)} entries). "
                    "Pass recursive=true to delete it and everything inside.",
                    summary="not empty",
                )
            count = sum(1 for _ in target.rglob("*")) if contents else 0
            preview = f"delete directory {label} and {count} item(s) inside"
        else:
            preview = f"delete file {label}"

        ok, refusal = await _ask(
            ctx,
            ApprovalRequest(
                tool=self.name,
                action="delete",
                path=label,
                diff=preview,
                recoverability=await recoverability(target),
                destructive=True,
            ),
        )
        if not ok:
            return refusal or ToolOutcome.rejected(DENIED)

        def _remove() -> None:
            if is_dir:
                shutil.rmtree(target)
            else:
                target.unlink()

        try:
            await asyncio.to_thread(_remove)
        except OSError as exc:
            return ToolOutcome.error(f"could not delete {label}: {exc}", summary="failed")
        return ToolOutcome(content=f"deleted {label}", summary=f"deleted {label}")
