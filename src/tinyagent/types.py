"""The small set of data types shared by every Tinyagent component."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value not in (None, [], "")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        calls = [ToolCall(**call) for call in data.get("tool_calls", [])]
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=calls,
            tool_call_id=data.get("tool_call_id"),
            name=data.get("name"),
        )


@dataclass(slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(slots=True)
class ModelResponse:
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


@dataclass(slots=True)
class AgentResult:
    output: str
    messages: list[Message]
    usage: Usage
    steps: int
    stop_reason: str
