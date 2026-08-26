"""Collect verifier-passing repairs for failed tasks."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tinyagent import Model, OpenAICompatibleModel

from ...runtime.evaluation import BENCHMARKS
from ...types import EvolutionConfig
from ...utils.io import read_jsonl, write_json, write_jsonl

REPAIR_SYSTEM_PROMPT = """You are repairing a failed coding-agent attempt.
Work only in the provided workspace. Read task.py, failure.txt, and previous_solution.py.
Identify the concrete cause reported by the verifier, then create a complete corrected solution.py.
Use the previous solution only as evidence; replace it when its approach is wrong. Run at most one
concise local check, then stop. The final answer must exist in solution.py."""

REPAIR_USER_PROMPT = (
    "Repair the failed solution using the task, previous attempt, and verifier feedback. "
    "Create solution.py, run at most one concise check, then return."
)


@dataclass(slots=True)
class RepairCollection:
    failed_tasks: list[str]
    repaired_tasks: list[str]
    still_failed_tasks: list[str]
    attempts: int
    successful_trajectories: int
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_failed_task_repairs(
    config: EvolutionConfig,
    evaluation_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str,
    model: Model | None = None,
    task_ids: set[str] | None = None,
    guidance: str = "",
) -> RepairCollection:
    """Retry selected failed tasks and retain attempts accepted by their verifier."""
    source = Path(evaluation_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    results = read_jsonl(source / "results.jsonl")
    failed = {
        str(row["task_id"]): row
        for row in results
        if not row.get("passed", row.get("status") == "pass")
        and (task_ids is None or str(row["task_id"]) in task_ids)
    }
    if not failed:
        collection = RepairCollection([], [], [], 0, 0, str(destination))
        _write_collection(destination, [], [], collection)
        return collection

    trajectory_repair = _trajectory_repair(config.benchmark, config.repo)
    if trajectory_repair:
        tasks = _benchmark_tasks(config.benchmark, source / "config.json", config.repo)
        run_agent_task = render_task = verify = None
    else:
        run_agent_task = _load_eval_symbol(config.repo, "eval.coding_agent", "run_agent_task")
        tasks, render_task, verify = _benchmark_adapter(
            config.benchmark,
            source / "config.json",
            config.repo,
        )
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    missing = sorted(failed.keys() - tasks_by_id.keys())
    if missing:
        raise ValueError(f"failed tasks are absent from benchmark data: {missing}")
    repair_model = model or OpenAICompatibleModel(
        model=model_name,
        base_url=config.base_url,
        temperature=config.repair_temperature,
        timeout=300.0,
    )

    generations: list[dict[str, Any]] = []
    repair_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.eval_workers) as executor:
        futures = {}
        for task_id, failure in failed.items():
            for attempt in range(1, config.repair_attempts + 1):
                arguments = (
                    tasks_by_id[task_id],
                    failure,
                    attempt,
                    destination,
                    repair_model,
                    config,
                    guidance,
                )
                if trajectory_repair:
                    future = executor.submit(trajectory_repair, *arguments)
                else:
                    future = executor.submit(
                        _repair_once,
                        *arguments[:5],
                        run_agent_task,
                        render_task,
                        verify,
                        *arguments[5:],
                    )
                futures[future] = (task_id, attempt)
        for future in as_completed(futures):
            generation, result = future.result()
            generations.append(generation)
            repair_results.append(result)

    generations.sort(key=lambda row: (str(row["task_id"]), int(row["repair_attempt"])))
    repair_results.sort(key=lambda row: (str(row["task_id"]), int(row["repair_attempt"])))
    repaired = sorted(
        {
            str(row["task_id"])
            for row in repair_results
            if row.get("passed", row.get("status") == "pass")
        }
    )
    still_failed = sorted(failed.keys() - set(repaired))
    collection = RepairCollection(
        failed_tasks=sorted(failed),
        repaired_tasks=repaired,
        still_failed_tasks=still_failed,
        attempts=len(repair_results),
        successful_trajectories=sum(bool(row.get("passed")) for row in repair_results),
        output_dir=str(destination),
    )
    _write_collection(destination, generations, repair_results, collection)
    return collection


def _repair_once(
    task: dict[str, Any],
    failure: dict[str, Any],
    attempt: int,
    output_dir: Path,
    model: Model,
    run_agent_task: Callable[..., tuple[dict[str, Any], dict[str, Any]]],
    render_task: Callable[[dict[str, Any]], str],
    verify: Callable[[dict[str, Any], str, float], dict[str, Any]],
    config: EvolutionConfig,
    guidance: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = str(task["task_id"])
    previous_path = failure.get("candidate_path")
    previous = ""
    if isinstance(previous_path, str) and Path(previous_path).is_file():
        previous = Path(previous_path).read_text(encoding="utf-8")
    feedback = "\n".join(
        [
            f"status: {failure.get('status', 'unknown')}",
            str(failure.get("stderr", "")),
            str(failure.get("stdout", "")),
        ]
    )[:12_000]
    candidate = output_dir / "candidates" / f"{_safe_name(task_id)}-{attempt:02d}.py"
    generation, result = run_agent_task(
        task,
        model,
        candidate,
        task_source=render_task(task),
        evaluate=lambda source, timeout: verify(task, source, timeout),
        benchmark=f"{config.benchmark}-repair",
        max_steps=config.benchmark_max_steps,
        test_timeout=config.test_timeout,
        system_prompt=REPAIR_SYSTEM_PROMPT,
        user_prompt=(
            f"{REPAIR_USER_PROMPT}\n\nEvolution guidance:\n{guidance}"
            if guidance
            else REPAIR_USER_PROMPT
        ),
        extra_files={
            "previous_solution.py": previous or "# No previous solution was produced.\n",
            "failure.txt": feedback,
        },
    )
    generation["repair_attempt"] = attempt
    result["repair_attempt"] = attempt
    return generation, result


def _benchmark_adapter(
    benchmark: str,
    config_path: Path,
    repo: str | Path,
) -> tuple[
    list[dict[str, Any]],
    Callable[[dict[str, Any]], str],
    Callable[[dict[str, Any], str, float], dict[str, Any]],
]:
    try:
        spec = BENCHMARKS[benchmark]
    except KeyError as error:
        raise ValueError(f"repair collection does not support benchmark: {benchmark}") from error
    run_module = _load_eval_module(repo, spec.module)
    evaluate_source = _load_eval_symbol(repo, spec.execution_module, "evaluate_source")
    return (
        _benchmark_tasks(benchmark, config_path, repo),
        run_module.render_task_file,
        lambda task, source, timeout: evaluate_source(task, source, timeout=timeout),
    )


def _benchmark_tasks(benchmark: str, config_path: Path, repo: str | Path) -> list[dict[str, Any]]:
    tasks_path = config_path.parent / "tasks.jsonl"
    if tasks_path.is_file():
        return read_jsonl(tasks_path)
    values = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        spec = BENCHMARKS[benchmark]
    except KeyError as error:
        raise ValueError(f"repair collection does not support benchmark: {benchmark}") from error
    return _load_eval_module(repo, spec.module).load_tasks(argparse.Namespace(**values))


def _trajectory_repair(benchmark: str, repo: str | Path) -> Callable[..., Any] | None:
    try:
        spec = BENCHMARKS[benchmark]
    except KeyError as error:
        raise ValueError(f"repair collection does not support benchmark: {benchmark}") from error
    return getattr(_load_eval_module(repo, spec.execution_module), "repair_task", None)


def _load_eval_module(repo: str | Path, module: str) -> Any:
    root = Path(repo).resolve()
    if not (root / "eval" / "__init__.py").is_file():
        raise FileNotFoundError(f"benchmark package not found under repository: {root / 'eval'}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    importlib.invalidate_caches()
    return importlib.import_module(module)


def _load_eval_symbol(repo: str | Path, module: str, name: str) -> Any:
    return getattr(_load_eval_module(repo, module), name)


def _write_collection(
    output_dir: Path,
    generations: list[dict[str, Any]],
    results: list[dict[str, Any]],
    collection: RepairCollection,
) -> None:
    write_jsonl(output_dir / "generations.jsonl", generations)
    write_jsonl(output_dir / "results.jsonl", results)
    write_json(output_dir / "summary.json", collection.to_dict())


def _safe_name(task_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in task_id) or "task"
