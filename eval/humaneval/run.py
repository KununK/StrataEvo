"""Evaluate Tinyagent on HumanEval through an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.coding_agent import read_jsonl, run_benchmark
from eval.coding_agent import run_agent_task as run_coding_task
from tinyagent import Model

from .execution import evaluate_source

REQUIRED_COLUMNS = {"task_id", "prompt", "test", "entry_point"}
SYSTEM_PROMPT = """You are a coding agent being evaluated on one Python task.
Work only in the provided workspace. Read task.py, then create solution.py containing
the complete valid Python source for the task, including the requested function.
You may inspect files and run local checks. Hidden evaluation tests are unavailable.
Do not only print the code in chat: the final solution must exist in solution.py.
Keep the process concise. After one relevant local check passes, stop using tools and
return a final response."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinyagent on HumanEval")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--dataset", default="openai/openai_humaneval")
    parser.add_argument("--dataset-file", help="local JSON or JSONL tasks instead of Hugging Face")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="eval/outputs/humaneval/qwen3-coder-agent")
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
        tasks = _load_local_tasks(Path(args.dataset_file))
    else:
        from datasets import load_dataset

        tasks = [dict(row) for row in load_dataset(args.dataset, split=args.split)]
    for index, task in enumerate(tasks):
        missing = REQUIRED_COLUMNS.difference(task)
        if missing:
            raise ValueError(f"task {index} is missing columns: {sorted(missing)}")
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
        task_source=str(task["prompt"]),
        evaluate=lambda source, timeout: evaluate_source(task, source, timeout=timeout),
        benchmark="humaneval",
        system_prompt=SYSTEM_PROMPT,
        session_dir=session_dir,
        max_steps=max_steps,
        test_timeout=test_timeout,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    tasks = load_tasks(args)
    return run_benchmark(
        args,
        tasks,
        run_agent_task,
        benchmark="humaneval",
        description="HumanEval Agent",
        dataset=args.dataset_file or args.dataset,
        split=args.split,
    )


def _load_local_tasks(path: Path) -> list[dict[str, Any]]:
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
