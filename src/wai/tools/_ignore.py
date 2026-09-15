"""Gitignore filtering for the traversal tools.

Not a nicety: in a real repo an unfiltered ``glob **/*.py`` returns `.venv`
and `node_modules` and buries the actual source, and every one of those paths
is billed as context.

Git applies a directory's ``.gitignore`` to everything beneath it, so patterns
are matched relative to the directory that declared them. ``.git`` itself is
always skipped regardless of any ignore file.
"""

from __future__ import annotations

from pathlib import Path

from wai.workspace import Workspace

ALWAYS_SKIP: frozenset[str] = frozenset({".git"})


class IgnoreFilter:
    """Caches one compiled pattern set per directory that declares a .gitignore."""

    def __init__(self, workspace: Workspace, *, enabled: bool = True) -> None:
        self.workspace = workspace
        self.enabled = enabled
        self._specs: dict[Path, object | None] = {}

    def is_ignored(self, path: Path) -> bool:
        if any(part in ALWAYS_SKIP for part in path.parts):
            return True
        if not self.enabled:
            return False
        root = self.workspace.root
        try:
            relative = path.relative_to(root)
        except ValueError:
            return False  # outside the root; containment is resolve()'s job

        # Walk root -> containing directory, applying each .gitignore to the
        # part of the path below it.
        current = root
        for part in (*relative.parts[:-1], None):
            spec = self._spec_for(current)
            if spec is not None:
                below = path.relative_to(current).as_posix()
                if path.is_dir():
                    below += "/"
                if spec.match_file(below):  # type: ignore[attr-defined]
                    return True
            if part is None:
                break
            current = current / part
        return False

    def _spec_for(self, directory: Path) -> object | None:
        if directory in self._specs:
            return self._specs[directory]
        spec: object | None = None
        gitignore = directory / ".gitignore"
        if gitignore.is_file():
            try:
                import pathspec

                spec = pathspec.PathSpec.from_lines(
                    "gitignore", gitignore.read_text(encoding="utf-8").splitlines()
                )
            except OSError, UnicodeDecodeError, ImportError:
                spec = None
        self._specs[directory] = spec
        return spec
