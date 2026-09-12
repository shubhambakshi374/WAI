"""Layout regression snapshots.

These compare rendered SVG, so they are expected to churn when Textual is
upgraded or the styling changes deliberately. Refresh with:

    uv run pytest tests/test_snapshots.py --snapshot-update
"""

from __future__ import annotations

from typing import Any

from tests.conftest import FakeProvider
from tests.test_tui import make_app


def test_empty_chat_layout(snap_compare: Any) -> None:
    assert snap_compare(make_app(), terminal_size=(90, 26))


def test_chat_with_a_completed_turn(snap_compare: Any) -> None:
    async def run_before(pilot: Any) -> None:
        pilot.app.screen.query_one("Composer").text = "how do I drain a node?"
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.pause()

    assert snap_compare(make_app(FakeProvider()), terminal_size=(90, 26), run_before=run_before)
