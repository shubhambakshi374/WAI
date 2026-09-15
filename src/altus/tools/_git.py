"""Is this file recoverable if we destroy it?

The single most useful thing an approval prompt can tell you about a delete or
an overwrite. Committed and clean means `git checkout` gets it back; untracked
means it is gone for good.

Everything here is read-only git plumbing, and every failure degrades to "I
don't know" rather than blocking the operation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

TIMEOUT = 5.0

RECOVERABLE = "tracked by git and unmodified — recoverable with git checkout"
MODIFIED = "tracked by git but has uncommitted changes — those changes would be lost"
UNTRACKED = "not tracked by git — this cannot be undone"
NO_REPO = "not inside a git repository — this cannot be undone"
UNKNOWN = "git status unavailable — assume this cannot be undone"


async def _git(*args: str, cwd: Path) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
    except OSError, TimeoutError:
        return 1, ""
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


async def recoverability(path: Path) -> str:
    """One human sentence about whether ``path`` could be restored."""
    directory = path.parent if path.parent.exists() else path
    if not directory.exists():
        return UNKNOWN

    code, _ = await _git("rev-parse", "--is-inside-work-tree", cwd=directory)
    if code != 0:
        return NO_REPO

    code, _ = await _git("ls-files", "--error-unmatch", "--", str(path), cwd=directory)
    if code != 0:
        return UNTRACKED

    code, out = await _git("status", "--porcelain", "--", str(path), cwd=directory)
    if code != 0:
        return UNKNOWN
    return MODIFIED if out.strip() else RECOVERABLE
