"""Shared runtime for file-based coding-agent benchmarks."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol

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

SYSTEM_PROMPT = """You are a coding agent being evaluated on one Python task.
Work only in the provided workspace. Read task.py, then create solution.py containing
the complete valid Python source for the task, including the requested function.
You may inspect files and run local checks. Do not only print the code in chat:
the final solution must exist in solution.py. Keep the process concise. After one
relevant local check passes, stop using tools and return a final response."""

USER_PROMPT = (
    "Read task.py, create solution.py, and run at most one concise local check. "
    "Then return a final response without further tool calls."
)


class TaskRunner(Protocol):
    def __call__(
        self,
        task: dict[str, Any],
        model: Model,
        candidate_path: Path,
        *,
        session_dir: Path | None = None,
        max_steps: int = 12,
        test_timeout: float = 10.0,
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


def run_agent_task(
    task: dict[str, Any],
    model: Model,
    candidate_path: Path,
    *,
    task_source: str,
    evaluate: Callable[[str, float], dict[str, Any]],
    benchmark: str,
    session_dir: Path | None = None,
    max_steps: int = 12,
    test_timeout: float = 10.0,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = USER_PROMPT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one isolated Agent task, preserve its candidate, and evaluate it."""
    started = time.perf_counter()
    with TemporaryDirectory(prefix=f"strataevo_{benchmark}_agent_") as directory:
        workspace = Path(directory)
        task_path = workspace / "task.py"
        task_path.write_text(task_source, encoding="utf-8")
        task_path.chmod(0o444)
        session_id = safe_name(str(task["task_id"]))
        session_store = SessionStore(session_dir) if session_dir else None
        if session_dir:
            (session_dir / f"{session_id}.json").unlink(missing_ok=True)
        agent = Agent(
            model=model,
            tools=ToolRegistry(Workspace(workspace).tools()),
            system_prompt=system_prompt,
            max_steps=max_steps,
            approval=allow_all,
            session_store=session_store,
        )
        try:
            agent_result = agent.run(
                user_prompt,
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
        "entry_point": task.get("entry_point", ""),
        "prompt": task.get("prompt", ""),
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
        test_result = failed_result("agent_error", agent_error)
    elif not source:
        test_result = failed_result("missing_candidate", "solution.py was not created")
    else:
        test_result = evaluate(source, test_timeout)

    result = {
        "task_id": task["task_id"],
        "entry_point": task.get("entry_point", ""),
        "candidate_path": str(candidate_path) if source else None,
        "agent_stop_reason": generation["stop_reason"],
        "agent_steps": generation["steps"],
        "input_tokens": generation["usage"]["input_tokens"],
        "output_tokens": generation["usage"]["output_tokens"],
        "generation_seconds": generation["generation_seconds"],
        **test_result,
    }
    return generation, result


def run_benchmark(
    args: argparse.Namespace,
    tasks: list[dict[str, Any]],
    task_runner: TaskRunner,
    *,
    benchmark: str,
    description: str,
    dataset: str,
    split: str,
) -> int:
    """Run independent Agent tasks concurrently and persist common artifacts."""
    output_dir = Path(args.output_dir)
    if args.no_resume and output_dir.exists():
        shutil.rmtree(output_dir)
    candidates_dir = output_dir / "candidates"
    sessions_dir = output_dir / "sessions"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**vars(args), "benchmark": benchmark}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    generations_path = output_dir / "generations.jsonl"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"

    selected_task_ids = {str(task["task_id"]) for task in tasks}
    existing = [row for row in read_jsonl(results_path) if row.get("task_id") in selected_task_ids]
    completed = {str(row["task_id"]) for row in existing}
    rows = list(existing)
    model = OpenAICompatibleModel(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        timeout=args.model_timeout,
    )

    pending_tasks = [task for task in tasks if str(task["task_id"]) not in completed]
    with tqdm(
        total=len(tasks),
        initial=len(tasks) - len(pending_tasks),
        desc=description,
        unit="task",
        dynamic_ncols=True,
    ) as progress:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures: dict[Future[tuple[dict[str, Any], dict[str, Any]]], dict[str, Any]] = {}
            for task in pending_tasks:
                candidate = candidates_dir / f"{safe_name(str(task['task_id']))}.py"
                future = executor.submit(
                    task_runner,
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
                generation, result = future.result()
                append_jsonl(generations_path, generation)
                append_jsonl(results_path, result)
                rows.append(result)
                summary = write_summary(
                    summary_path,
                    rows,
                    args,
                    len(tasks),
                    benchmark=benchmark,
                    dataset=dataset,
                    split=split,
                )
                progress.set_postfix(
                    status=result["status"],
                    pass_at_1=f"{summary['pass_at_1']:.3f}",
                )
                progress.update()
                tqdm.write(
                    f"{task['task_id']}: {result['status']} | "
                    f"pass@1={summary['pass_at_1']:.3f} "
                    f"({summary['passed']}/{summary['evaluated']})"
                )

    summary = write_summary(
        summary_path,
        rows,
        args,
        len(tasks),
        benchmark=benchmark,
        dataset=dataset,
        split=split,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def failed_result(status: str, error: str) -> dict[str, Any]:
    return {
        "passed": False,
        "compile_passed": False,
        "status": status,
        "returncode": None,
        "stdout": "",
        "stderr": error,
        "test_seconds": 0.0,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    selected_tasks: int,
    *,
    benchmark: str,
    dataset: str,
    split: str,
) -> dict[str, Any]:
    passed = sum(bool(row.get("passed")) for row in rows)
    statuses: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    evaluated = len(rows)
    summary = {
        "benchmark": benchmark,
        "model": args.model,
        "base_url": args.base_url,
        "dataset": dataset,
        "split": split,
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


def safe_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_") or "task"
