"""Rendering text we did not write.

Textual parses `[...]` in a string as a style tag. Almost everything the TUI
displays is text we were handed rather than text we wrote --- tool arguments,
cluster output, provider errors, the model's own words --- so almost everything
has to say "render this literally".

Where a widget offers ``markup=False``, use that; it is plainer at the call
site. ``literal()`` is for the places that offer no such switch, of which
``Collapsible(title=...)`` is the one that matters.
"""

from __future__ import annotations


def literal(text: str) -> str:
    """Escape ``text`` so Textual markup renders it exactly as given.

    Only ``[`` is escaped, and deliberately not ``\\``. Textual treats a
    backslash as an escape *only* in front of a bracket, so doubling backslashes
    would turn a Windows path or a regex into a corrupted one.

    ``textual.markup.escape`` is not a substitute: it escapes well-formed tags
    and leaves an unterminated ``[`` alone, which is precisely the shape that
    raised MarkupError in issue #1 --- a truncated ``repr()`` of a list argument.
    """
    return text.replace("[", r"\[")
