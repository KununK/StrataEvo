"""Transparent JSON persistence: a session is simply a list of messages."""

from __future__ import annotations

import json
from pathlib import Path

from .types import Message


class SessionStore:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Message.from_dict(item) for item in data["messages"]]

    def save(self, session_id: str, messages: list[Message]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(session_id)
        temporary = path.with_suffix(".tmp")
        data = {"messages": [item.to_dict() for item in messages]}
        temporary.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, session_id: str) -> Path:
        if not session_id or not all(char.isalnum() or char in "-_" for char in session_id):
            raise ValueError("session_id may contain only letters, numbers, '-' and '_'")
        return self.directory / f"{session_id}.json"
