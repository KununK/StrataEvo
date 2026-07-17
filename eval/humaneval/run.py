"""Evaluate Tinyagent on HumanEval through an OpenAI-compatible model server."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from tqdm import tqdm

from tinyagent import (
    Agent,
    Model,
    OpenAICompatibleModel,
    SessionStore,
    ToolRegistry,
    Workspace,
    allow_all,
)

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
    """Run one fresh agent workspace, preserve its candidate, then evaluate it."""
    started = time.perf_counter()
    with TemporaryDirectory(prefix="strataevo_humaneval_agent_") as directory:
        workspace = Path(directory)
        task_path = workspace / "task.py"
        task_path.write_text(task["prompt"], encoding="utf-8")
        task_path.chmod(0o444)
        session_id = _safe_name(task["task_id"])
        session_store = SessionStore(session_dir) if session_dir else None
        if session_dir:
            (session_dir / f"{session_id}.json").unlink(missing_ok=True)
        agent = Agent(
            model=model,
            tools=ToolRegistry(Workspace(workspace).tools()),
            system_prompt=SYSTEM_PROMPT,
            max_steps=max_steps,
            approval=allow_all,
            session_store=session_store,
        )
        try:
            agent_result = agent.run(
                "Read task.py, create solution.py, and run at most one concise local check. "
                "Then return a final response without further tool calls.",
                session_id=session_id if session_store else None,
            )
            agent_error = ""
        except Exception as error:
            agent_result = None
            agent_error = f"{type(error).__name__}: {error}"

        solution_path = workspace / "solution.py"
        source = solution_path.read_text(encoding="utf-8") if solution_path.is_file() else ""

    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if source:
        candidate_path.write_text(source, encoding="utf-8")

    generation = {
        "task_id": task["task_id"],
        "entry_point": task["entry_point"],
        "prompt": task["prompt"],
        "session_id": session_id,
        "session_path": str(session_dir / f"{session_id}.json") if session_dir else None,
        "candidate_path": str(candidate_path) if source else None,
        "agent_output": agent_result.output if agent_result else "",
        "agent_error": agent_error,
        "stop_reason": agent_result.stop_reason if agent_result else "agent_error",
        "steps": agent_result.steps if agent_result else 0,
        "usage": {
            "input_tokens": agent_result.usage.input_tokens if agent_result else 0,
            "output_tokens": agent_result.usage.output_tokens if agent_result else 0,
        },
        "messages": [message.to_dict() for message in agent_result.messages]
        if agent_result
        else [],
        "generation_seconds": time.perf_counter() - started,
    }

    if agent_error:
        test_result = _failed_result("agent_error", agent_error)
    elif not source:
        test_result = _failed_result("missing_candidate", "solution.py was not created")
    else:
        test_result = evaluate_source(task, source, timeout=test_timeout)

    result = {
        "task_id": task["task_id"],
        "entry_point": task["entry_point"],
        "candidate_path": str(candidate_path) if source else None,
        "agent_stop_reason": generation["stop_reason"],
        "agent_steps": generation["steps"],
        "input_tokens": generation["usage"]["input_tokens"],
        "output_tokens": generation["usage"]["output_tokens"],
        "generation_seconds": generation["generation_seconds"],
        **test_result,
    }
    return generation, result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.offset < 0 or args.limit is not None and args.limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    output_dir = Path(args.output_dir)
    if args.no_resume and output_dir.exists():
        shutil.rmtree(output_dir)
    candidates_dir = output_dir / "candidates"
    sessions_dir = output_dir / "sessions"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generations_path = output_dir / "generations.jsonl"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    tasks = load_tasks(args)
    selected_task_ids = {task["task_id"] for task in tasks}
    existing = [row for row in _read_jsonl(results_path) if row.get("task_id") in selected_task_ids]
    completed = {row["task_id"] for row in existing}
    rows = list(existing)
    model = OpenAICompatibleModel(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.model_timeout,
    )

    pending_tasks = [task for task in tasks if task["task_id"] not in completed]
    with tqdm(
        total=len(tasks),
        initial=len(tasks) - len(pending_tasks),
        desc="HumanEval Agent",
        unit="task",
        dynamic_ncols=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[Future[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]] = {}
            for task in pending_tasks:
                task_id = task["task_id"]
                candidate = candidates_dir / f"{_safe_name(task_id)}.py"
                future = executor.submit(
                    run_agent_task,
                    task,
                    model,
                    candidate,
                    session_dir=sessions_dir,
                    max_steps=args.max_steps,
                    test_timeout=args.test_timeout,
                )
                futures[future] = task

            for future in as_completed(futures):
                task = futures[future]
                task_id = task["task_id"]
                generation, result = future.result()
                _append_jsonl(generations_path, generation)
                _append_jsonl(results_path, result)
                rows.append(result)
                completed.add(task_id)
                summary = _write_summary(summary_path, rows, args, len(tasks))
                progress.set_postfix(
                    status=result["status"],
                    pass_at_1=f"{summary['pass_at_1']:.3f}",
                )
                progress.update()
                tqdm.write(
                    f"{task_id}: {result['status']} | "
                    f"pass@1={summary['pass_at_1']:.3f} "
                    f"({summary['passed']}/{summary['evaluated']})"
                )

    summary = _write_summary(summary_path, rows, args, len(tasks))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def _load_local_tasks(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("test", data.get("tasks"))
    if not isinstance(data, list):
        raise ValueError("local dataset must be a JSON list or JSONL file")
    return data


def _failed_result(status: str, error: str) -> dict[str, Any]:
    return {
        "passed": False,
        "compile_passed": False,
        "status": status,
        "returncode": None,
        "stdout": "",
        "stderr": error,
        "test_seconds": 0.0,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    selected_tasks: int,
) -> dict[str, Any]:
    passed = sum(bool(row.get("passed")) for row in rows)
    statuses: dict[str, int] = {}
    for row in rows:
        status = row.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    evaluated = len(rows)
    summary = {
        "model": args.model,
        "base_url": args.base_url,
        "dataset": args.dataset_file or args.dataset,
        "split": args.split,
        "selected_tasks": selected_tasks,
        "evaluated": evaluated,
        "passed": passed,
        "pass_at_1": passed / evaluated if evaluated else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "average_agent_steps": (
            sum(row.get("agent_steps", 0) for row in rows) / evaluated if evaluated else 0.0
        ),
        "average_generation_seconds": (
            sum(row.get("generation_seconds", 0.0) for row in rows) / evaluated
            if evaluated
            else 0.0
        ),
        "average_test_seconds": (
            sum(row.get("test_seconds", 0.0) for row in rows) / evaluated if evaluated else 0.0
        ),
        "total_input_tokens": sum(row.get("input_tokens", 0) for row in rows),
        "total_output_tokens": sum(row.get("output_tokens", 0) for row in rows),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _safe_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)


if __name__ == "__main__":
    raise SystemExit(main())
