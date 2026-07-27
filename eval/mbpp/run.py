"""Evaluate Tinyagent on MBPP through an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.coding_agent import read_jsonl, run_benchmark
from eval.coding_agent import run_agent_task as run_coding_task
from tinyagent import Model

from .execution import evaluate_source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinyagent on MBPP")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--dataset", default="google-research-datasets/mbpp")
    parser.add_argument("--dataset-config", default="sanitized")
    parser.add_argument("--dataset-file", help="local JSON or JSONL tasks instead of Hugging Face")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="eval/outputs/mbpp/qwen3-coder-agent")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--model-timeout", type=float, default=120.0)
    parser.add_argument("--test-timeout", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_file:
        rows = _load_local_rows(Path(args.dataset_file))
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
    _validate_args(args)
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


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("test", data.get("tasks"))
    if not isinstance(data, list):
        raise ValueError("local dataset must be a JSON list or JSONL file")
    return data


def _validate_args(args: argparse.Namespace) -> None:
    if args.offset < 0 or args.limit is not None and args.limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if args.workers <= 0:
        raise ValueError("workers must be positive")


if __name__ == "__main__":
    raise SystemExit(main())
