"""Slash commands. The only UI-aware part of the cloud/provider surface."""

from wai.tui.commands.builtin import build_registry
from wai.tui.commands.registry import (
    Command,
    CommandRegistry,
    CommandResult,
    dispatch,
    is_command,
    parse,
)

__all__ = [
    "Command",
    "CommandRegistry",
    "CommandResult",
    "build_registry",
    "dispatch",
    "is_command",
    "parse",
]
