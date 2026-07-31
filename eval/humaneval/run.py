"""Evaluate Tinyagent on HumanEval through an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
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

REQUIRED_COLUMNS = {"task_id", "prompt", "test", "entry_point"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinyagent on HumanEval")
    add_common_arguments(
        parser,
        dataset="openai/openai_humaneval",
        output_dir="eval/outputs/humaneval/qwen3-coder-agent",
    )
    return parser.parse_args(argv)


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dataset_file:
        tasks = load_local_rows(args.dataset_file)
    else:
        from datasets import load_dataset

        tasks = [dict(row) for row in load_dataset(args.dataset, split=args.split)]
    for index, task in enumerate(tasks):
        missing = REQUIRED_COLUMNS.difference(task)
        if missing:
            raise ValueError(f"task {index} is missing columns: {sorted(missing)}")
    end = None if args.limit is None else args.offset + args.limit
    return tasks[args.offset : end]


def render_task_file(task: dict[str, Any]) -> str:
    return str(task["prompt"])


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
        benchmark="humaneval",
        session_dir=session_dir,
        max_steps=max_steps,
        test_timeout=test_timeout,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tool_description_addenda=tool_description_addenda,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_common_arguments(args)
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


if __name__ == "__main__":
    raise SystemExit(main())
