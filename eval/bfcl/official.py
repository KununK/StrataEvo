"""Small bridge to a pinned checkout of the official BFCL implementation."""

from __future__ import annotations

import copy
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from tinyagent import Tool

CLASS_MODULES = {
    "GorillaFileSystem": "gorilla_file_system",
    "MathAPI": "math_api",
    "MessageAPI": "message_api",
    "TwitterAPI": "posting_api",
    "TicketAPI": "ticket_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}
FUNCTION_DOCS = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}
STATELESS_CLASSES = {"MathAPI"}


class OfficialBFCL:
    """Load BFCL data, tools, and the official multi-turn checker."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = locate_bfcl_root(root)
        source_root = str(self.root.parent)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

    def load_tasks(self, category: str) -> list[dict[str, Any]]:
        tasks = _read_jsonl(self.root / "data" / f"BFCL_v4_{category}.json")
        answers = {
            row["id"]: row["ground_truth"]
            for row in _read_jsonl(
                self.root / "data" / "possible_answer" / f"BFCL_v4_{category}.json"
            )
        }
        for task in tasks:
            task_id = str(task["id"])
            if task_id not in answers:
                raise ValueError(f"BFCL ground truth is missing task {task_id}")
            task["task_id"] = task_id
            task["prompt"] = "\n".join(
                str(message.get("content", ""))
                for turn in task.get("question", [])
                for message in turn
            )
            task["ground_truth"] = answers[task_id]
            task["function"] = self._load_function_docs(task)
        return tasks

    def create_tools(
        self,
        task: dict[str, Any],
        addenda: dict[str, str] | None = None,
    ) -> list[Tool]:
        instances = self._create_instances(task)
        descriptions = {item["name"]: item for item in task["function"]}
        unknown = set(addenda or {}) - {"*", *descriptions}
        if unknown:
            raise ValueError(f"unknown BFCL tools: {', '.join(sorted(unknown))}")

        tools = []
        for name, document in descriptions.items():
            method = next(
                (
                    getattr(instance, name)
                    for instance in instances.values()
                    if hasattr(instance, name)
                ),
                None,
            )
            if method is not None:
                description = str(document.get("description", ""))
                additions = [
                    text.strip()
                    for key in ("*", name)
                    if (text := (addenda or {}).get(key, "")).strip()
                ]
                if additions:
                    description += "\n\n" + "\n".join(additions)
                tools.append(
                    Tool(
                        name=name,
                        description=description,
                        function=_tool_function(method),
                        parameters=_normalize_schema(document.get("parameters", {})),
                    )
                )
        if not tools:
            raise ValueError(f"BFCL task {task['task_id']} exposed no executable tools")
        return tools

    def check(self, task: dict[str, Any], calls: list[list[list[str]]]) -> dict[str, Any]:
        checker = importlib.import_module(
            "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker"
        ).multi_turn_checker
        return checker(
            calls,
            task["ground_truth"],
            task,
            "multi_turn_base",
            f"strataevo_{os.getpid()}_{id(calls)}",
        )

    def _create_instances(self, task: dict[str, Any]) -> dict[str, Any]:
        instances = {}
        for class_name in task["involved_classes"]:
            module_name = CLASS_MODULES.get(class_name)
            if module_name is None:
                raise ValueError(f"unsupported BFCL backend class: {class_name}")
            module = importlib.import_module(
                "bfcl_eval.eval_checker.multi_turn_eval.func_source_code." + module_name
            )
            instance = getattr(module, class_name)()
            if class_name not in STATELESS_CLASSES:
                scenario = copy.deepcopy(task.get("initial_config", {}).get(class_name, {}))
                instance._load_scenario(scenario)
            instances[class_name] = instance
        return instances

    def _load_function_docs(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        excluded = set(task.get("excluded_function", []))
        documents = []
        for class_name in task["involved_classes"]:
            filename = FUNCTION_DOCS.get(class_name)
            if filename is None:
                raise ValueError(f"unsupported BFCL function documentation: {class_name}")
            documents.extend(
                row
                for row in _read_jsonl(self.root / "data" / "multi_turn_func_doc" / filename)
                if row["name"] not in excluded
            )
        return documents


def locate_bfcl_root(root: str | Path | None = None) -> Path:
    requested = Path(root or os.getenv("BFCL_ROOT", "eval/vendor/bfcl")).expanduser().resolve()
    if (requested / "data" / "BFCL_v4_multi_turn_base.json").is_file():
        return requested
    candidates = [requested, requested / "berkeley-function-call-leaderboard"]
    for candidate in candidates:
        package = candidate / "bfcl_eval"
        if (package / "data" / "BFCL_v4_multi_turn_base.json").is_file():
            return package
    raise FileNotFoundError(
        f"BFCL V4 was not found under {requested}; run ./eval/setup_bfcl.sh or set BFCL_ROOT"
    )


def format_call(name: str, arguments: dict[str, Any]) -> str:
    values = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({values})"


def _tool_function(method):
    def invoke(**arguments):
        return method(**arguments)

    return invoke


def _normalize_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    schema = {
        key: _normalize_schema(item) if isinstance(item, dict) else item
        for key, item in value.items()
    }
    if schema.get("type") == "dict":
        schema["type"] = "object"
    return schema


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing official BFCL file: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
