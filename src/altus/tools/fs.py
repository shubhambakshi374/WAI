"""The read-only filesystem tools: read_file, list_dir, glob, grep.

Every one of them goes through ``Workspace.resolve`` before touching the disk,
and every one caps its output. An uncapped read of a lockfile is enough to
blow the context window and end the session.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path
from typing import Any, ClassVar

from altus.core.errors import PathNotAllowed
from altus.tools._ignore import IgnoreFilter
from altus.tools.base import BaseTool, ToolContext, ToolOutcome, truncated_note

_BINARY_SNIFF_BYTES = 8192


def _human_size(num: int) -> str:
    value = float(num)
    for unit in ("B", "K", "M", "G"):
        if value < 1024 or unit == "G":
            return f"{value:.0f}{unit}"
        value /= 1024
    return f"{value:.0f}G"


def _looks_binary(raw: bytes) -> bool:
    return b"\x00" in raw[:_BINARY_SNIFF_BYTES]


class ReadFileTool(BaseTool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a text file from the workspace. Output is line-numbered so you can "
        "request a specific range with offset and limit. Long files are truncated; "
        "read the next chunk by passing offset."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workspace root."},
            "offset": {"type": "integer", "description": "1-based first line to read."},
            "limit": {"type": "integer", "description": "Maximum number of lines."},
        },
        "required": ["path"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            return ToolOutcome.error("path is required")
        try:
            target = ctx.workspace.resolve(raw_path)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")

        label = ctx.workspace.relative(target)
        if not target.exists():
            return ToolOutcome.error(f"no such file: {label}", summary="not found")
        if target.is_dir():
            return ToolOutcome.error(
                f"{label} is a directory --- use list_dir", summary="is a directory"
            )

        try:
            data = await asyncio.to_thread(target.read_bytes)
        except OSError as exc:
            return ToolOutcome.error(f"could not read {label}: {exc}", summary="unreadable")

        if _looks_binary(data):
            return ToolOutcome.error(
                f"{label} is a binary file ({_human_size(len(data))}); not shown.",
                summary="binary",
            )

        clipped_bytes = len(data) > ctx.max_file_bytes
        if clipped_bytes:
            data = data[: ctx.max_file_bytes]
        text = data.decode("utf-8", errors="replace")
        if not text:
            return ToolOutcome(content=f"{label} is empty.", summary="empty")

        lines = text.splitlines()
        total = len(lines)
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or ctx.max_lines)
        limit = max(1, min(limit, ctx.max_lines))

        window = lines[offset - 1 : offset - 1 + limit]
        if not window:
            return ToolOutcome.error(
                f"offset {offset} is past the end of {label} ({total} lines)",
                summary="bad offset",
            )

        width = len(str(offset + len(window) - 1))
        body = "\n".join(f"{offset + i:>{width}}\t{line}" for i, line in enumerate(window))
        header = f"{label} ({total} lines)\n"
        note = ""
        last = offset + len(window) - 1
        if last < total or clipped_bytes:
            note = f"\n\n[truncated at line {last} of {total}; continue with offset={last + 1}]"
            if clipped_bytes:
                note = (
                    f"\n\n[file exceeds {_human_size(ctx.max_file_bytes)}; "
                    f"showing lines {offset}-{last}, continue with offset={last + 1}]"
                )
        return ToolOutcome(
            content=header + body + note,
            summary=f"read {len(window)} line{'s' if len(window) != 1 else ''} of {label}",
        )


class ListDirTool(BaseTool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = (
        "List the contents of a directory in the workspace. Directories first, "
        "then files with sizes. Gitignored entries and .git are hidden unless "
        "all=true."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory, default the root."},
            "all": {"type": "boolean", "description": "Include gitignored entries."},
        },
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        raw_path = str(args.get("path") or ".").strip() or "."
        try:
            target = ctx.workspace.resolve(raw_path)
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")

        label = ctx.workspace.relative(target)
        if not target.exists():
            return ToolOutcome.error(f"no such directory: {label}", summary="not found")
        if not target.is_dir():
            return ToolOutcome.error(
                f"{label} is a file --- use read_file", summary="not a directory"
            )

        show_all = bool(args.get("all"))
        ignores = IgnoreFilter(ctx.workspace, enabled=not show_all)

        def _scan() -> tuple[list[str], int]:
            dirs: list[str] = []
            files: list[str] = []
            skipped = 0
            with os.scandir(target) as it:
                for entry in sorted(it, key=lambda e: e.name):
                    path = Path(entry.path)
                    if ignores.is_ignored(path):
                        skipped += 1
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        dirs.append(f"{entry.name}/")
                    else:
                        try:
                            size = _human_size(entry.stat(follow_symlinks=False).st_size)
                        except OSError:
                            size = "?"
                        files.append(f"{entry.name}  ({size})")
            return dirs + files, skipped

        try:
            entries, skipped = await asyncio.to_thread(_scan)
        except OSError as exc:
            return ToolOutcome.error(f"could not list {label}: {exc}", summary="unreadable")

        if not entries:
            hint = f" ({skipped} hidden by .gitignore)" if skipped else ""
            return ToolOutcome(content=f"{label} is empty{hint}.", summary="empty")

        total = len(entries)
        shown = entries[: ctx.max_entries]
        body = f"{label}/\n" + "\n".join(f"  {e}" for e in shown)
        if total > len(shown):
            body += truncated_note(len(shown), total, "entries")
        if skipped:
            body += f"\n[{skipped} entr{'y' if skipped == 1 else 'ies'} hidden by .gitignore]"
        return ToolOutcome(content=body, summary=f"{total} entries in {label}")


class GlobTool(BaseTool):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        "Find files by glob pattern, e.g. '**/*.py' or 'src/**/test_*.py'. "
        "Returns paths newest first. Gitignored files and .git are excluded."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern."},
            "path": {"type": "string", "description": "Directory to search from."},
        },
        "required": ["pattern"],
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            return ToolOutcome.error("pattern is required")
        try:
            base = ctx.workspace.resolve(str(args.get("path") or "."))
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")
        if not base.is_dir():
            return ToolOutcome.error(
                f"{ctx.workspace.relative(base)} is not a directory", summary="not a directory"
            )

        ignores = IgnoreFilter(ctx.workspace)

        def _search() -> list[Path]:
            found: list[Path] = []
            for candidate in base.glob(pattern):
                if ignores.is_ignored(candidate) or not candidate.is_file():
                    continue
                if not ctx.workspace.is_allowed(candidate):
                    continue  # symlink out, or a denylisted secret
                found.append(candidate)
            found.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            return found

        try:
            matches = await asyncio.to_thread(_search)
        except (OSError, ValueError) as exc:
            return ToolOutcome.error(f"glob failed: {exc}", summary="failed")

        if not matches:
            return ToolOutcome(
                content=f"no files match {pattern!r} under {ctx.workspace.relative(base)}",
                summary="0 matches",
            )
        shown = matches[: ctx.max_entries]
        body = "\n".join(ctx.workspace.relative(p) for p in shown)
        if len(matches) > len(shown):
            body += truncated_note(len(shown), len(matches), "files")
        return ToolOutcome(content=body, summary=f"{len(matches)} files match {pattern}")


class GrepTool(BaseTool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents with a regular expression. Returns matching lines "
        "prefixed with path:line. Gitignored files and .git are excluded."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regular expression."},
            "path": {"type": "string", "description": "Directory or file to search."},
            "glob": {"type": "string", "description": "Restrict to files matching this glob."},
            "ignore_case": {"type": "boolean"},
        },
        "required": ["pattern"],
    }

    #: Overridable so tests can force the pure-Python path.
    use_ripgrep: ClassVar[bool] = True

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutcome:
        pattern = str(args.get("pattern", ""))
        if not pattern:
            return ToolOutcome.error("pattern is required")
        try:
            re.compile(pattern)
        except re.error as exc:
            return ToolOutcome.error(f"invalid regular expression: {exc}", summary="bad pattern")
        try:
            base = ctx.workspace.resolve(str(args.get("path") or "."))
        except PathNotAllowed as exc:
            return ToolOutcome.error(str(exc), summary="denied")

        glob = args.get("glob")
        ignore_case = bool(args.get("ignore_case"))
        rg = shutil.which("rg") if self.use_ripgrep else None
        if rg:
            lines = await self._ripgrep(rg, pattern, base, glob, ignore_case, ctx)
        else:
            lines = await asyncio.to_thread(
                self._python_grep, pattern, base, glob, ignore_case, ctx
            )
        if lines is None:
            return ToolOutcome.error("search failed", summary="failed")
        if not lines:
            return ToolOutcome(content=f"no matches for {pattern!r}", summary="0 matches")

        total = len(lines)
        shown = lines[: ctx.max_matches]
        body = "\n".join(shown)
        if total > len(shown):
            body += truncated_note(len(shown), total, "matches")
        return ToolOutcome(content=body, summary=f"{total} matches for {pattern}")

    async def _ripgrep(
        self,
        rg: str,
        pattern: str,
        base: Path,
        glob: str | None,
        ignore_case: bool,
        ctx: ToolContext,
    ) -> list[str] | None:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never", "--max-count", "50"]
        if ignore_case:
            cmd.append("--ignore-case")
        if glob:
            cmd += ["--glob", glob]
        cmd += ["--regexp", pattern, str(base)]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, _ = await proc.communicate()
        except OSError:
            return None
        if proc.returncode not in (0, 1):
            return None
        # rg prints absolute paths; shorten them against the workspace root.
        prefix = str(ctx.workspace.root) + os.sep
        return [
            line[len(prefix) :] if line.startswith(prefix) else line
            for line in out.decode("utf-8", errors="replace").splitlines()
        ]

    def _python_grep(
        self,
        pattern: str,
        base: Path,
        glob: str | None,
        ignore_case: bool,
        ctx: ToolContext,
    ) -> list[str]:
        regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
        ignores = IgnoreFilter(ctx.workspace)
        results: list[str] = []
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if len(results) >= ctx.max_matches * 2:
                break
            if not path.is_file() or ignores.is_ignored(path):
                continue
            if glob and not fnmatch.fnmatch(path.name, glob):
                continue
            if not ctx.workspace.is_allowed(path):
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if _looks_binary(raw):
                continue
            label = ctx.workspace.relative(path)
            for number, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    results.append(f"{label}:{number}:{line}")
                    if len(results) >= ctx.max_matches * 2:
                        break
        return results
