"""Observable lifecycle events without coupling the agent to a UI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class Event:
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    time: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


EventHandler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> Callable[[], None]:
        self._handlers.append(handler)
        return lambda: self._handlers.remove(handler)

    def emit(self, event_type: str, **data: Any) -> None:
        event = Event(event_type, data)
        for handler in list(self._handlers):
            handler(event)
