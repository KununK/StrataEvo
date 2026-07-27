"""Validated JSON model calls shared by diagnosis and planning."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from tinyagent import Message, Model


@dataclass(slots=True)
class StructuredResponse[T]:
    value: T
    input_tokens: int
    output_tokens: int
    raw_output: str
    attempts: list[str]
    errors: list[str]


class StructuredOutputError(ValueError):
    """A structured model call that remained invalid after repair attempts."""

    def __init__(
        self,
        label: str,
        attempts: list[str],
        errors: list[str],
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.label = label
        self.attempts = attempts
        self.errors = errors
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        super().__init__(
            f"{label} remained invalid after {len(attempts)} attempt(s): {errors[-1]}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.label,
            "attempts": self.attempts,
            "errors": self.errors,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


def request_json[T](
    model: Model,
    messages: list[Message],
    parser: Callable[[dict[str, Any]], T],
    *,
    label: str,
    repair_retries: int,
) -> StructuredResponse[T]:
    attempts: list[str] = []
    errors: list[str] = []
    input_tokens = 0
    output_tokens = 0
    current_messages = list(messages)
    for attempt_number in range(repair_retries + 1):
        response = model.complete(current_messages, [])
        raw_output = response.message.content
        attempts.append(raw_output)
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        try:
            value = parser(extract_json_object(raw_output))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
            if attempt_number == repair_retries:
                break
            if attempt_number == 0:
                current_messages = [
                    *messages,
                    response.message,
                    Message(
                        "user",
                        f"Your {label} JSON was invalid: {error}. Regenerate it and return only "
                        "the requested JSON object.",
                    ),
                ]
            else:
                current_messages = [
                    Message(
                        "system",
                        "Repair structured output. Return only one valid JSON object, preserve "
                        "the intended content, and correct the stated syntax or schema error.",
                    ),
                    Message(
                        "user",
                        f"Output type: {label}\nValidation error: {error}\n\n"
                        f"Invalid output:\n{raw_output}",
                    ),
                ]
            continue
        return StructuredResponse(
            value,
            input_tokens,
            output_tokens,
            raw_output,
            attempts,
            errors,
        )
    raise StructuredOutputError(label, attempts, errors, input_tokens, output_tokens)


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
