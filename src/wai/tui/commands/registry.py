"""Slash-command dispatch.

Anything the composer submits starting with ``/`` is handled here instead of
being sent to the model. An unrecognised command is reported as unknown ---
silently forwarding ``/depoly`` to the LLM as a prompt would be baffling.

Commands return a ``CommandResult`` rather than touching the screen, so they
are testable without a running app.
"""

from __future__ import annotations

import shlex
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wai.tui.app import WaiApp


@dataclass
class CommandResult:
    """What a command wants shown. ``None`` body means it handled its own UI."""

    body: str = ""
    severity: str = "information"
    """information | warning | error"""
    title: str = ""
    handled: bool = True

    @classmethod
    def error(cls, body: str) -> CommandResult:
        return cls(body=body, severity="error")

    @classmethod
    def warn(cls, body: str) -> CommandResult:
        return cls(body=body, severity="warning")

    @classmethod
    def silent(cls) -> CommandResult:
        """The command drove the UI itself (opened a modal, and so on)."""
        return cls(body="")


Handler = Callable[["WaiApp", list[str]], Awaitable[CommandResult]]


@dataclass
class Command:
    name: str
    summary: str
    usage: str = ""
    handler: Handler | None = None
    aliases: tuple[str, ...] = ()


@dataclass
class CommandRegistry:
    commands: dict[str, Command] = field(default_factory=dict)

    def register(self, command: Command) -> None:
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command

    def get(self, name: str) -> Command | None:
        return self.commands.get(name)

    @property
    def unique(self) -> list[Command]:
        seen: dict[str, Command] = {}
        for command in self.commands.values():
            seen.setdefault(command.name, command)
        return sorted(seen.values(), key=lambda c: c.name)


def is_command(text: str) -> bool:
    """A leading slash, but not a path like ``/etc/hosts`` or a bare ``/``."""
    stripped = text.strip()
    if not stripped.startswith("/") or len(stripped) < 2:
        return False
    head = stripped[1:].split(maxsplit=1)[0]
    return bool(head) and "/" not in head and not head[0].isspace()


def parse(text: str) -> tuple[str, list[str]]:
    stripped = text.strip()[1:]
    try:
        parts = shlex.split(stripped)
    except ValueError:
        parts = stripped.split()
    return (parts[0].casefold(), parts[1:]) if parts else ("", [])


async def dispatch(app: WaiApp, registry: CommandRegistry, text: str) -> CommandResult:
    name, args = parse(text)
    command = registry.get(name)
    if command is None or command.handler is None:
        known = ", ".join(f"/{c.name}" for c in registry.unique)
        return CommandResult.error(f"Unknown command /{name}. Try: {known}")
    return await command.handler(app, args)
