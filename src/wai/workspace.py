"""The workspace: a rooted, shared filesystem context.

This is a first-class primitive, not a path check bolted onto the tools. A
Phase 2 flow constructs one Workspace and hands the same instance to every
step, so nothing here assumes a single session and every workspace carries an
``id`` a flow can refer to.

Its other job is containment. ``resolve`` is the single choke point every tool
goes through, and it enforces two separate rules:

* **Containment** --- the resolved path must sit under the root or one of the
  opt-in extra roots. Symlinks are resolved *before* the check, because a link
  inside the root pointing at ``~/.ssh`` would otherwise walk straight out.

* **The secret denylist** --- applied even *inside* an allowed root. WAI ships
  context to eight external LLM APIs by design, so "model reads .env, model
  quotes it back, key lands in a provider's log" is the most plausible way this
  tool leaks a credential. The denylist targets secrets that turn up
  *incidentally* in a working tree; a path you had to opt into deliberately via
  ``extra_roots`` is already a deliberate choice, with ``.ssh`` and ``.gnupg``
  the exceptions that stay denied everywhere.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from wai.core.errors import PathNotAllowed
from wai.core.types import new_id

DENY_BASENAMES: frozenset[str] = frozenset(
    {
        ".netrc",
        "_netrc",
        ".pgpass",
        ".npmrc",
        ".pypirc",
        "credentials.json",
    }
)

DENY_GLOBS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
)

DENY_DIR_COMPONENTS: frozenset[str] = frozenset({".ssh", ".gnupg"})
"""Denied wherever they appear --- opting in via extra_roots does not unlock them."""

DENY_SUFFIX_PATHS: tuple[tuple[str, ...], ...] = (
    (".aws", "credentials"),
    (".docker", "config.json"),
)


def _canonical(path: str | Path) -> Path:
    """Expand ``~``, make absolute, and resolve every symlink.

    ``strict=False`` so a path whose final component does not exist still has
    its existing parents resolved --- which is what containment needs.
    """
    return Path(os.path.expanduser(str(path))).resolve()


@dataclass(frozen=True)
class Workspace:
    """A rooted filesystem context shared by everything operating inside it."""

    root: Path
    extra_roots: tuple[Path, ...] = ()
    deny_secrets: bool = True
    id: str = field(default_factory=lambda: new_id("ws_"))

    def __post_init__(self) -> None:
        # Canonicalize once, at construction, so every later comparison is
        # between fully-resolved paths.
        object.__setattr__(self, "root", _canonical(self.root))
        object.__setattr__(self, "extra_roots", tuple(_canonical(p) for p in self.extra_roots))

    @property
    def roots(self) -> tuple[Path, ...]:
        return (self.root, *self.extra_roots)

    # ------------------------------------------------------------------ checks

    def resolve(self, candidate: str | Path) -> Path:
        """Canonicalize and validate a path. The only way tools touch the disk.

        Raises ``PathNotAllowed`` when the path escapes every root or matches
        the secret denylist.
        """
        raw = str(candidate)
        target = _canonical(self.root / raw if not Path(raw).is_absolute() else raw)

        if not self._contained(target):
            allowed = ", ".join(str(r) for r in self.roots)
            raise PathNotAllowed(
                f"{raw!r} resolves to {target}, which is outside the workspace. "
                f"Allowed roots: {allowed}",
                path=str(target),
                reason="outside_workspace",
            )

        if self.deny_secrets and (hit := self._denied(target)) is not None:
            raise PathNotAllowed(
                f"{self.relative(target)} is blocked as a likely secret ({hit}). "
                "WAI sends file contents to external model providers, so credential "
                "files are never readable.",
                path=str(target),
                reason="denylisted",
            )

        return target

    def is_allowed(self, path: str | Path) -> bool:
        try:
            self.resolve(path)
        except PathNotAllowed:
            return False
        return True

    def _contained(self, resolved: Path) -> bool:
        # is_relative_to, never a string prefix: "/workspace-other" must not
        # match a root of "/work".
        return any(resolved == r or resolved.is_relative_to(r) for r in self.roots)

    def _denied(self, resolved: Path) -> str | None:
        """Return the matching denylist rule, or None."""
        name = resolved.name
        if name in DENY_BASENAMES:
            return name
        for pattern in DENY_GLOBS:
            if fnmatch.fnmatch(name, pattern):
                return pattern
        parts = resolved.parts
        for component in DENY_DIR_COMPONENTS:
            if component in parts:
                return f"{component}/"
        for suffix in DENY_SUFFIX_PATHS:
            if len(parts) >= len(suffix) and parts[-len(suffix) :] == suffix:
                return "/".join(suffix)
        return None

    # ----------------------------------------------------------------- display

    def relative(self, path: str | Path) -> str:
        """A short label for output: relative to the root when it sits under it."""
        target = Path(path)
        try:
            return str(target.relative_to(self.root)) or "."
        except ValueError:
            return str(target)

    def __str__(self) -> str:
        return str(self.root)


def default_workspace(
    root: str | Path | None = None,
    extra_roots: tuple[str, ...] | list[str] = (),
    *,
    deny_secrets: bool = True,
) -> Workspace:
    """The workspace for a plain interactive session: the current directory."""
    return Workspace(
        root=Path(root) if root is not None else Path.cwd(),
        extra_roots=tuple(Path(p) for p in extra_roots),
        deny_secrets=deny_secrets,
    )
