"""JSONL session persistence.

One record per line, appended as each message completes, so an interrupted run
never loses history. The first line is always the session header.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from wai.core.session import Session
from wai.core.types import Message

log = logging.getLogger(__name__)


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def create(self, session: Session) -> None:
        """Write the header line. Overwrites any existing file for this id."""
        header = session.model_dump(mode="json", exclude={"messages"})
        with self.path_for(session.id).open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "header", **header}) + "\n")

    def append_message(self, session: Session, message: Message) -> None:
        record = {"kind": "message", "message": message.model_dump(mode="json")}
        with self.path_for(session.id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def update_header(self, session: Session) -> None:
        """Rewrite the header in place, preserving every message line.

        Called when the title, model or cumulative usage changes.
        """
        path = self.path_for(session.id)
        if not path.exists():
            self.create(session)
            for message in session.messages:
                self.append_message(session, message)
            return
        lines = path.read_text(encoding="utf-8").splitlines()
        header = session.model_dump(mode="json", exclude={"messages"})
        lines[0:1] = [json.dumps({"kind": "header", **header})]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def load(self, session_id: str) -> Session:
        path = self.path_for(session_id)
        if not path.exists():
            raise FileNotFoundError(f"no such session: {session_id}")
        session: Session | None = None
        messages: list[Message] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping corrupt line %d in %s", lineno, path)
                continue
            kind = record.pop("kind", None)
            if kind == "header":
                session = Session.model_validate(record)
            elif kind == "message":
                messages.append(Message.model_validate(record["message"]))
        if session is None:
            raise ValueError(f"session {session_id} has no header record")
        session.messages = messages
        return session

    def list_sessions(self, limit: int = 50) -> list[Session]:
        """Headers only, newest first. Messages are not loaded."""
        out: list[Session] = []
        paths = sorted(self.root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in paths[:limit]:
            try:
                with path.open(encoding="utf-8") as fh:
                    record = json.loads(fh.readline())
                record.pop("kind", None)
                out.append(Session.model_validate(record))
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                log.warning("skipping unreadable session %s: %s", path.name, exc)
        return out

    def latest_id(self) -> str | None:
        sessions = self.list_sessions(limit=1)
        return sessions[0].id if sessions else None

    def delete(self, session_id: str) -> bool:
        path = self.path_for(session_id)
        if path.exists():
            path.unlink()
            return True
        return False
