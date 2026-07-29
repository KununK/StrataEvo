"""Turn ordinary typed Python functions into tools an LLM can call."""

from __future__ import annotations

import inspect
import json
import shlex
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import (
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

Approval = Callable[[str, dict[str, Any]], bool]


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]
    requires_approval: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def run(self, arguments: dict[str, Any]) -> str:
        try:
            inspect.signature(self.function).bind(**arguments)
        except TypeError as error:
            raise ValueError(f"invalid arguments for {self.name}: {error}") from error
        result = self.function(**arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools or []:
            self.register(item)

    def register(self, item: Tool) -> Tool:
        if item.name in self._tools:
            raise ValueError(f"duplicate tool: {item.name}")
        self._tools[item.name] = item
        return item

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [item.schema for item in self._tools.values()]

    def with_description_addenda(self, addenda: dict[str, str]) -> ToolRegistry:
        """Return a registry with additional model-facing guidance."""
        unknown = set(addenda) - self._tools.keys()
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
        return ToolRegistry(
            [
                replace(item, description=f"{item.description}\n\n{addenda[item.name]}")
                if addenda.get(item.name)
                else item
                for item in self._tools.values()
            ]
        )

    def __iter__(self) -> Iterator[Tool]:
        return iter(self._tools.values())


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
):
    """Decorator that derives JSON Schema from a function's type hints."""

    def wrap(fn: Callable[..., Any]) -> Tool:
        signature = inspect.signature(fn)
        hints = get_type_hints(fn)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter_name, parameter in signature.parameters.items():
            properties[parameter_name] = _json_schema(hints.get(parameter_name, str))
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
            else:
                properties[parameter_name]["default"] = parameter.default
        doc = inspect.getdoc(fn) or ""
        return Tool(
            name or fn.__name__,
            description or doc.split("\n", 1)[0] or fn.__name__,
            fn,
            {"type": "object", "properties": properties, "required": required},
            requires_approval,
        )

    return wrap(function) if function is not None else wrap


def _json_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return {"enum": list(args)}
    if origin in (list, tuple):
        return {"type": "array", "items": _json_schema(args[0] if args else str)}
    if origin is dict:
        return {"type": "object"}
    if origin in (Union, types.UnionType):
        return {"anyOf": [_json_schema(arg) for arg in args]}
    types_by_annotation = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        type(None): "null",
    }
    return {"type": types_by_annotation.get(annotation, "string")}


def allow_all(_name: str, _arguments: dict[str, Any]) -> bool:
    """Approve every side-effecting tool call."""
    return True


def terminal_approval(name: str, arguments: dict[str, Any]) -> bool:
    """Ask the terminal user before a side-effecting tool call."""
    preview = arguments.get("command") or arguments.get("path") or str(arguments)
    answer = input(f"Approve {name}({shlex.quote(str(preview))})? [y/N] ")
    return answer.strip().lower() in {"y", "yes"}
