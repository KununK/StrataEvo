"""Evaluate Tinyagent on MBPP through an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.coding_agent import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    add_common_arguments,
    load_local_rows,
    run_benchmark,
    validate_common_arguments,
)
from eval.coding_agent import run_agent_task as run_coding_task
from tinyagent import Model

from .execution import evaluate_source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinyagent on MBPP")
    add_common_arguments(
        parser,
        dataset="google-research-datasets/mbpp",
        output_dir="eval/outputs/mbpp/qwen3-coder-agent",
    )
    parser.add_argument("--dataset-config", default="sanitized")
    return parser.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_file:
        rows = load_local_rows(args.dataset_file)
    else:
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
        rows = [dict(row) for row in dataset]
    tasks = [_normalize_task(row, index) for index, row in enumerate(rows)]
    end = None if args.limit is None else args.offset + args.limit
    return tasks[args.offset : end]


def run_agent_task(
    task: dict[str, Any],
    model: Model,
    candidate_path: Path,
    *,
    session_dir: Path | None = None,
    max_steps: int = 12,
    test_timeout: float = 10.0,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = USER_PROMPT,
    tool_description_addenda: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return run_coding_task(
        task,
        model,
        candidate_path,
        task_source=render_task_file(task),
        evaluate=lambda source, timeout: evaluate_source(task, source, timeout=timeout),
        benchmark="mbpp",
        session_dir=session_dir,
        max_steps=max_steps,
        test_timeout=test_timeout,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_description_addenda=tool_description_addenda,
    )


def render_task_file(task: dict[str, Any]) -> str:
    """Expose the official MBPP description and public tests as valid Python data."""
    return (
        f"TASK = {task['prompt']!r}\n\n"
        f"ENTRY_POINT = {task['entry_point']!r}\n"
        f"TEST_IMPORTS = {task['test_imports']!r}\n"
        f"TEST_SETUP = {task['test_setup']!r}\n"
        f"TESTS = {task['test_list']!r}\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_common_arguments(args)
    tasks = load_tasks(args)
    dataset = args.dataset_file or (
        f"{args.dataset}:{args.dataset_config}" if args.dataset_config else args.dataset
    )
    return run_benchmark(
        args,
        tasks,
        run_agent_task,
        benchmark="mbpp",
        description="MBPP Agent",
        dataset=dataset,
        split=args.split,
    )


def _normalize_task(row: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = _first_text(row, ("prompt", "text"))
    tests = _string_list(row.get("test_list"))
    if not tests:
        raise ValueError(f"MBPP task {index} has no test_list")
    task_id = row.get("task_id")
    if task_id is None:
        raise ValueError(f"MBPP task {index} has no task_id")
    return {
        "task_id": str(task_id),
        "prompt": prompt,
        "entry_point": _infer_entry_point(tests),
        "test_list": tests,
        "test_setup": "\n".join(_string_list(row.get("test_setup_code"))),
        "test_imports": "\n".join(_string_list(row.get("test_imports"))),
    }


def _first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"MBPP row has none of the prompt columns: {list(names)}")


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def _infer_entry_point(tests: list[str]) -> str:
    for test in tests:
        match = re.search(r"\bassert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match:
            return match.group(1)
    for test in tests:
        match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", test)
        if match:
            return match.group(1)
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
