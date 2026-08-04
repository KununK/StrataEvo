"""Validated JSON model calls used by evolution controllers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tinyagent import Message, Model

STRUCTURED_MAX_TOKENS = 2048


@dataclass(slots=True)
class StructuredResponse[T]:
    value: T
    input_tokens: int
    output_tokens: int
    raw_output: str
    attempts: list[str]


def request_json[T](
    model: Model,
    messages: list[Message],
    parser: Callable[[dict[str, Any]], T],
    *,
    label: str,
    repair_retries: int,
) -> StructuredResponse[T]:
    attempts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    last_error: ValueError | None = None
    for attempt_number in range(repair_retries + 1):
        request = list(messages)
        if last_error is not None:
            request.append(
                Message(
                    "user",
                    f"The previous {label} response was invalid: {last_error}. "
                    "Return a new valid JSON object only.",
                )
            )
        response = model.complete(
            request,
            [],
            max_tokens=STRUCTURED_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw_output = response.message.content
        attempts.append(raw_output)
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        try:
            value = parser(extract_json_object(raw_output))
        except (KeyError, TypeError, ValueError) as error:
            last_error = ValueError(str(error))
            if attempt_number == repair_retries:
                break
            continue
        return StructuredResponse(value, input_tokens, output_tokens, raw_output, attempts)
    raise ValueError(
        f"{label} remained invalid after {len(attempts)} attempt(s): {last_error}"
    ) from last_error


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = lines[1:] if lines and lines[0].startswith("```") else lines
        lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
        content = "\n".join(lines).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    try:
        data = json.loads(content[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"response contains invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data
