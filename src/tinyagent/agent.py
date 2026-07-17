"""The complete agent loop. This is the best file to read first."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .events import EventBus
from .model import Model
from .session import SessionStore
from .tool import Approval, ToolRegistry
from .types import AgentResult, Message, Usage

DEFAULT_SYSTEM_PROMPT = """You are a capable coding and research agent.
Use tools when they provide evidence or are needed to change the environment.
Inspect before editing, keep changes scoped, and report concrete results.
When the task is complete, answer directly without calling more tools."""


@dataclass(slots=True)
class Agent:
    model: Model
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_steps: int = 20
    approval: Approval | None = None
    events: EventBus = field(default_factory=EventBus)
    session_store: SessionStore | None = None
    context_limit_chars: int = 100_000
    _cancelled: threading.Event = field(default_factory=threading.Event, init=False)

    def __post_init__(self) -> None:
        if self.max_steps <= 0 or self.context_limit_chars <= 0:
            raise ValueError("max_steps and context_limit_chars must be positive")

    def run(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        history: list[Message] | None = None,
    ) -> AgentResult:
        """Run model -> tools -> model until the model returns plain text."""
        self._cancelled.clear()
        messages = self._initial_messages(session_id, history)
        if not messages or messages[0].role != "system":
            messages.insert(0, Message("system", self.system_prompt))
        messages.append(Message("user", prompt))
        total_usage = Usage()
        self.events.emit("run_start", prompt=prompt, session_id=session_id)

        for step in range(1, self.max_steps + 1):
            if self._cancelled.is_set():
                return self._finish("", messages, total_usage, step - 1, "cancelled", session_id)
            messages = self._compact(messages)
            self.events.emit("model_start", step=step)
            response = self.model.complete(messages, self.tools.schemas)
            total_usage = total_usage + response.usage
            assistant = response.message
            messages.append(assistant)
            self.events.emit("model_end", step=step, message=assistant)

            if not assistant.tool_calls:
                return self._finish(
                    assistant.content,
                    messages,
                    total_usage,
                    step,
                    "completed",
                    session_id,
                )

            for call in assistant.tool_calls:
                if self._cancelled.is_set():
                    return self._finish("", messages, total_usage, step, "cancelled", session_id)
                result = self._execute_tool(call.name, call.arguments)
                messages.append(Message("tool", result, tool_call_id=call.id, name=call.name))

        return self._finish("", messages, total_usage, self.max_steps, "max_steps", session_id)

    def cancel(self) -> None:
        """Cooperatively stop before the next model or tool call."""
        self._cancelled.set()

    def _execute_tool(self, name: str, arguments: dict) -> str:
        self.events.emit("tool_start", name=name, arguments=arguments)
        item = self.tools.get(name)
        if item is None:
            result = f"Error: unknown tool '{name}'"
        elif item.requires_approval and (
            self.approval is None or not self.approval(name, arguments)
        ):
            result = "Error: tool call denied by approval policy"
        else:
            try:
                result = item.run(arguments)
            except Exception as error:
                result = f"Error: {type(error).__name__}: {error}"
        self.events.emit("tool_end", name=name, result=result)
        return result

    def _initial_messages(
        self, session_id: str | None, history: list[Message] | None
    ) -> list[Message]:
        if history is not None:
            return list(history)
        if session_id and self.session_store:
            return self.session_store.load(session_id)
        return []

    def _finish(
        self,
        output: str,
        messages: list[Message],
        usage: Usage,
        steps: int,
        reason: str,
        session_id: str | None,
    ) -> AgentResult:
        if session_id and self.session_store:
            self.session_store.save(session_id, messages)
        result = AgentResult(output, messages, usage, steps, reason)
        self.events.emit("run_end", result=result)
        return result

    def _compact(self, messages: list[Message]) -> list[Message]:
        """Drop complete old turns without breaking model/tool message groups."""
        size = sum(len(message.content) for message in messages)
        if size <= self.context_limit_chars or len(messages) <= 4:
            return messages
        # A user message starts a turn. Keeping the latest complete turn ensures
        # every assistant tool call stays adjacent to all of its tool results.
        user_positions = [index for index, message in enumerate(messages) if message.role == "user"]
        if len(user_positions) < 2:
            return messages
        keep_from = user_positions[-1]
        system, tail = messages[0], messages[keep_from:]
        removed = messages[1:keep_from]
        summary = "Earlier context omitted:\n" + "\n".join(
            f"{message.role}: {message.content[:300]}" for message in removed if message.content
        )
        return [system, Message("system", summary[-self.context_limit_chars // 2 :]), *tail]
