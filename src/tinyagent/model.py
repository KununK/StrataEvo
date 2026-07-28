"""Model boundary plus a dependency-free OpenAI-compatible implementation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .types import Message, ModelResponse, ToolCall, Usage

DEFAULT_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


class Model(Protocol):
    """Anything that turns conversation history into one assistant message."""

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse: ...


@dataclass(slots=True)
class OpenAICompatibleModel:
    """Calls OpenAI or any server implementing `/v1/chat/completions`."""

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    timeout: float = 120.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        self.base_url = base_url.rstrip("/")

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openai(message) for message in messages],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = list(tools)
            body["tool_choice"] = "auto"
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"model request failed ({error.code}): {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"model request failed: {error.reason}") from error
        except TimeoutError as error:
            raise RuntimeError(f"model request timed out after {self.timeout:g}s") from error

        choice = payload["choices"][0]
        raw = choice["message"]
        calls = [
            ToolCall(
                id=call["id"],
                name=call["function"]["name"],
                arguments=_parse_arguments(call["function"].get("arguments", "{}")),
            )
            for call in raw.get("tool_calls", [])
        ]
        usage = payload.get("usage", {})
        return ModelResponse(
            Message("assistant", raw.get("content") or "", calls),
            Usage(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
            choice.get("finish_reason"),
        )


class ScriptedModel:
    """Deterministic model useful in tests and agent research."""

    def __init__(self, responses: Sequence[Message | ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[list[Message]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        self.requests.append(list(messages))
        if not self.responses:
            raise RuntimeError("ScriptedModel has no response left")
        response = self.responses.pop(0)
        return response if isinstance(response, ModelResponse) else ModelResponse(response)


def _parse_arguments(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"model returned invalid tool arguments: {raw}") from error
    if not isinstance(value, dict):
        raise RuntimeError("tool arguments must be a JSON object")
    return value


def _message_to_openai(message: Message) -> dict[str, Any]:
    content: str | None = message.content
    if message.role == "assistant" and message.tool_calls and not content:
        content = None
    data: dict[str, Any] = {"role": message.role, "content": content}
    if message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.name:
        data["name"] = message.name
    return data
