"""What this terminal can actually show.

Deliberately a pure function over an environment mapping rather than a probe.
A probe needs a real TTY, which means it cannot run in a test, cannot run in
CI, and answers differently depending on who is watching --- and the one thing
this has to get right is the *negative* case, because a terminal that cannot
draw images must degrade silently rather than emit escape codes at someone.

Detection is therefore a denylist and an allowlist of terminals we know, and
the default when we recognise nothing is ``CELLS``: box-drawing characters
work everywhere Textual works, so guessing low costs almost nothing while
guessing high sprays garbage across the screen.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum


class Support(StrEnum):
    """Ordered worst to best; comparison is by explicit rank, not by value."""

    TEXT = "text"
    """No box drawing. A dumb terminal, or a pipe."""
    CELLS = "cells"
    """Box-drawing characters and colour. Effectively everywhere."""
    IMAGE = "image"
    """Kitty's graphics protocol or Sixel."""

    @property
    def rank(self) -> int:
        return {"text": 0, "cells": 1, "image": 2}[self.value]

    def at_least(self, other: Support) -> bool:
        return self.rank >= other.rank


#: Terminals that render images. ``TERM_PROGRAM`` values, matched exactly.
IMAGE_PROGRAMS = frozenset({"iTerm.app", "WezTerm", "ghostty", "Ghostty", "rio"})

#: Environment variables whose mere presence identifies an image-capable host.
IMAGE_MARKERS = ("KITTY_WINDOW_ID", "GHOSTTY_RESOURCES_DIR", "WEZTERM_PANE", "KONSOLE_VERSION")

#: Warp answers the graphics-protocol query affirmatively, but does not
#: implement the unicode placeholders textual-image relies on, so the query is
#: worse than useless here: believing it produces a broken screen. Denied by
#: name rather than trusted.
DENIED_PROGRAMS = frozenset({"WarpTerminal", "Warp"})


def detect(env: Mapping[str, str] | None = None) -> Support:
    """The best rendering this terminal is known to support."""
    values = os.environ if env is None else env

    term = values.get("TERM", "")
    if not term or term in {"dumb", "unknown"}:
        return Support.TEXT

    program = values.get("TERM_PROGRAM", "")
    if program in DENIED_PROGRAMS:
        return Support.CELLS

    # Inside tmux or screen, graphics need passthrough that is off by default
    # and silently mangles output when it is missing. Not worth the gamble.
    if values.get("TMUX") or term.startswith("screen"):
        return Support.CELLS

    if program in IMAGE_PROGRAMS:
        return Support.IMAGE
    if any(values.get(marker) for marker in IMAGE_MARKERS):
        return Support.IMAGE
    if "kitty" in term or "ghostty" in term:
        return Support.IMAGE

    # Apple_Terminal, vscode, plain xterm and everything unrecognised. xterm
    # *can* do sixel, but only when built and started for it, and there is no
    # way to know from here.
    return Support.CELLS


def available(env: Mapping[str, str] | None = None) -> Support:
    """What is usable right now: detection, capped by what is installed.

    The graphics extra is optional, so a terminal that could show images still
    gets cells when ``textual-image`` is absent. That is a visible, fixable
    state rather than an import error at draw time.
    """
    found = detect(env)
    if found is Support.IMAGE and not images_installed():
        return Support.CELLS
    return found


def images_installed() -> bool:
    import importlib.util

    return importlib.util.find_spec("textual_image") is not None


def resolve(setting: str, env: Mapping[str, str] | None = None) -> Support:
    """Apply the ``[ui] graphics`` setting to what the terminal offers.

    ``auto`` follows the terminal. The explicit values are a ceiling, not a
    floor: asking for images on a terminal that cannot show them still yields
    cells, because the alternative is a broken screen.
    """
    match setting:
        case "off" | "text":
            return Support.TEXT
        case "cells":
            return min(available(env), Support.CELLS, key=lambda s: s.rank)
        case "image" | "on":
            return available(env)
        case _:
            return available(env)


def explain(setting: str, env: Mapping[str, str] | None = None) -> str:
    """One line saying what is being used and why.

    "Why are there no pictures" should have an answer inside the app rather
    than requiring someone to read this module.
    """
    values = os.environ if env is None else env
    program = values.get("TERM_PROGRAM") or values.get("TERM") or "unknown terminal"
    chosen = resolve(setting, env)

    if setting in {"off", "text"}:
        return f"graphics off by configuration; using text views ({program})"
    if chosen is Support.IMAGE:
        return f"drawing images ({program})"
    if not images_installed():
        return (
            f"drawing with box characters: the graphics extra is not installed "
            f"({program}). Add it with: uv sync --extra graphics"
        )
    if setting == "cells":
        return f"drawing with box characters by configuration ({program})"
    if values.get("TMUX"):
        return "drawing with box characters: tmux needs passthrough for images"
    if program in DENIED_PROGRAMS:
        return f"drawing with box characters: {program} reports image support it does not have"
    return (
        f"drawing with box characters: {program} supports no image protocol. "
        "Kitty, Ghostty, WezTerm and iTerm2 do."
    )
