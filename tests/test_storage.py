from __future__ import annotations

from pathlib import Path

import pytest

from altus.core.session import Session
from altus.core.types import Message, ReasoningBlock, Role, Usage
from altus.storage.sessions import SessionStore


@pytest.fixture
def store(tmp_path: Path) -> SessionStore:
    return SessionStore(tmp_path / "sessions")


def test_round_trip_preserves_messages_and_blocks(store: SessionStore) -> None:
    session = Session(provider="anthropic", model="claude-sonnet-5")
    store.create(session)
    user = Message.user("hello")
    assistant = Message(
        role=Role.ASSISTANT,
        content=[ReasoningBlock(text="hmm", signature="sig"), *Message.assistant("hi").content],
    )
    for message in (user, assistant):
        session.append(message)
        store.append_message(session, message)
    store.update_header(session)

    loaded = store.load(session.id)
    assert loaded.id == session.id
    assert loaded.title == "hello"
    assert loaded.model == "claude-sonnet-5"
    assert loaded.messages == [user, assistant]
    assert loaded.messages[1].reasoning == "hmm"


def test_update_header_preserves_messages(store: SessionStore) -> None:
    session = Session(model="a")
    store.create(session)
    message = Message.user("first")
    session.append(message)
    store.append_message(session, message)

    session.model = "b"
    session.record_usage(Usage(input_tokens=5, output_tokens=7))
    store.update_header(session)

    loaded = store.load(session.id)
    assert loaded.model == "b"
    assert loaded.usage.output_tokens == 7
    assert len(loaded.messages) == 1


def test_append_survives_missing_header(store: SessionStore) -> None:
    """update_header rebuilds the file when it was never created."""
    session = Session(model="m")
    session.append(Message.user("hi"))
    store.update_header(session)
    assert store.load(session.id).messages[0].text == "hi"


def test_corrupt_line_is_skipped(store: SessionStore) -> None:
    session = Session(model="m")
    store.create(session)
    message = Message.user("good")
    session.append(message)
    store.append_message(session, message)
    with store.path_for(session.id).open("a") as fh:
        fh.write("{not json\n")
    assert len(store.load(session.id).messages) == 1


def test_list_is_newest_first_and_latest_id(store: SessionStore) -> None:
    import os
    import time

    ids = []
    for name in ("one", "two", "three"):
        session = Session(model=name)
        store.create(session)
        ids.append(session.id)
        time.sleep(0.01)
    # mtime resolution varies by filesystem; make the ordering unambiguous.
    for offset, session_id in enumerate(ids):
        path = store.path_for(session_id)
        os.utime(path, (offset, offset))

    listed = store.list_sessions()
    assert [s.model for s in listed] == ["three", "two", "one"]
    assert store.latest_id() == ids[-1]


def test_delete(store: SessionStore) -> None:
    session = Session()
    store.create(session)
    assert store.delete(session.id) is True
    assert store.delete(session.id) is False


def test_load_missing_raises(store: SessionStore) -> None:
    with pytest.raises(FileNotFoundError):
        store.load("nope")
