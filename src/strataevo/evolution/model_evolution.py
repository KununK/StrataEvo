"""Verifier-guided test-time LoRA evolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .diagnosis import Diagnosis
from .evaluation import BenchmarkEvaluator
from .io import read_jsonl
from .plan import EvolutionPlan
from .profile_evolution import ProfileEvolutionResult, evaluate_profile, task_changes
from .repair import RepairCollection, collect_failed_task_repairs
from .types import EvaluationReport, EvolutionConfig


class AdapterRuntime(Protocol):
    def activate(self, name: str, path: Path) -> None: ...

    def deactivate(self, name: str) -> None: ...


class VLLMAdapterRuntime:
    """Load and unload LoRA adapters through vLLM's runtime API."""

    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        root = base_url.rstrip("/")
        self.root = root[:-3] if root.endswith("/v1") else root
        self.timeout = timeout

    def activate(self, name: str, path: Path) -> None:
        self.deactivate(name)
        self._post(
            "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": str(path.resolve())},
        )

    def deactivate(self, name: str) -> None:
        try:
            self._post("/v1/unload_lora_adapter", {"lora_name": name})
        except RuntimeError as error:
            if "404" not in str(error) and "not found" not in str(error).lower():
                raise

    def _post(self, endpoint: str, payload: dict[str, Any]) -> None:
        request = urllib.request.Request(
            self.root + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout):
                return
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"vLLM adapter request failed ({error.code}): {detail}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise RuntimeError(f"vLLM adapter request failed: {error}") from error


def build_training_dataset(
    evaluation_dir: str | Path,
    repairs: RepairCollection,
    destination: str | Path,
    *,
    limit: int,
) -> dict[str, Any]:
    """Combine one successful repair per failed task with successful replay traces."""
    parent = Path(evaluation_dir)
    repair_dir = Path(repairs.output_dir)
    repair_results = {
        (str(row["task_id"]), int(row["repair_attempt"])): bool(
            row.get("passed", row.get("status") == "pass")
        )
        for row in read_jsonl(repair_dir / "results.jsonl", missing_ok=True)
    }
    repaired_examples: list[dict[str, Any]] = []
    selected_repairs: set[str] = set()
    for row in read_jsonl(repair_dir / "generations.jsonl", missing_ok=True):
        task_id = str(row["task_id"])
        key = (task_id, int(row["repair_attempt"]))
        if (
            task_id in selected_repairs
            or not repair_results.get(key)
            or not row.get("messages")
        ):
            continue
        repaired_examples.append(
            {"task_id": row["task_id"], "source": "repair", "messages": row["messages"]}
        )
        selected_repairs.add(task_id)

    parent_passed = {
        str(row["task_id"])
        for row in read_jsonl(parent / "results.jsonl", missing_ok=True)
        if row.get("passed", row.get("status") == "pass")
    }
    replay_examples = [
        {"task_id": row["task_id"], "source": "replay", "messages": row["messages"]}
        for row in read_jsonl(parent / "generations.jsonl", missing_ok=True)
        if str(row.get("task_id")) in parent_passed and row.get("messages")
    ]
    examples = [*repaired_examples, *replay_examples][:limit]
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    trained_repairs = [
        str(item["task_id"]) for item in examples if item["source"] == "repair"
    ]
    return {
        "examples": len(examples),
        "repair_examples": len(trained_repairs),
        "replay_examples": sum(item["source"] == "replay" for item in examples),
        "trained_repair_tasks": trained_repairs,
    }


def evolve_model(
    config: EvolutionConfig,
    generation: int,
    generation_dir: Path,
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    *,
    parent_model: str,
    parent_adapter: dict[str, Any] | None,
    diagnosis: Diagnosis,
    plan: EvolutionPlan,
    runtime: AdapterRuntime | None = None,
) -> ProfileEvolutionResult:
    """Train, load, evaluate, and either retain or remove one LoRA candidate."""
    runtime = runtime or VLLMAdapterRuntime(config.base_url)
    model_dir = generation_dir / "model"
    data_path = model_dir / "verified_trajectories.jsonl"
    guidance = _repair_guidance(diagnosis, plan)
    print("[evolution] collecting verifier-guided repairs for failed tasks", flush=True)
    repairs = collect_failed_task_repairs(
        config,
        parent_report.output_dir,
        model_dir / "repairs",
        model_name=parent_model,
        task_ids=set(diagnosis.affected_tasks),
        guidance=guidance,
    )
    training_data = build_training_dataset(
        parent_report.output_dir,
        repairs,
        data_path,
        limit=config.sft_max_samples,
    )
    print(
        f"[evolution] repairs={len(repairs.repaired_tasks)}/{len(repairs.failed_tasks)} "
        f"training_examples={training_data['examples']}",
        flush=True,
    )
    sample_count = int(training_data["examples"])
    candidate_name = f"{config.run_name}-generation-{generation:04d}"
    adapter_path = model_dir / "adapter"
    candidate = {
        "name": candidate_name,
        "path": str(adapter_path),
        "parent_model": parent_model,
        "parent_adapter": (
            {"name": parent_adapter["name"], "path": parent_adapter["path"]}
            if parent_adapter
            else None
        ),
        "training_examples": sample_count,
        "training_data": str(data_path),
        "repair_collection": repairs.to_dict(),
        "training_dataset": training_data,
        "planned_intervention": {
            "affected_tasks": diagnosis.affected_tasks,
            "hypothesis": plan.hypothesis,
            "intervention": plan.intervention,
        },
        "executed_intervention": {
            "targeted_tasks": repairs.failed_tasks,
            "repair_guidance": guidance,
            "repair_attempts": repairs.attempts,
            "repaired_tasks": repairs.repaired_tasks,
            "training": None,
        },
    }
    if not training_data["repair_examples"]:
        reason = (
            "selected plan affected no failed task"
            if not repairs.failed_tasks
            else "no selected failed task produced a verifier-passing repair trajectory"
        )
        return ProfileEvolutionResult(
            "rejected",
            "model_training_skipped",
            reason,
            None,
            None,
            candidate,
        )

    train_config = {
        "base_model": config.model,
        "parent_adapter_path": parent_adapter["path"] if parent_adapter else None,
        "data_path": str(data_path),
        "output_dir": str(adapter_path),
        "epochs": config.sft_epochs,
        "max_length": config.sft_max_length,
        "lora_rank": config.sft_lora_rank,
        "learning_rate": config.sft_learning_rate,
    }
    candidate["executed_intervention"]["training"] = train_config
    train_config_path = model_dir / "train_config.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text(
        json.dumps(train_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    train_log = model_dir / "train.log"
    _run_training(config, train_config_path, train_log)
    candidate["train_log"] = str(train_log)

    def activate(candidate_record: dict[str, Any] | None) -> None:
        if candidate_record:
            _activate_candidate(
                runtime,
                evaluator,
                candidate_name,
                adapter_path,
                parent_adapter,
            )
        else:
            _activate_parent(runtime, evaluator, config.model, parent_adapter, candidate_name)

    evaluation = evaluate_profile(
        parent_report,
        evaluator,
        model_dir,
        None,
        candidate,
        activate,
    )
    if evaluation.candidate_report:
        candidate["screening_task_changes"] = task_changes(
            parent_report.output_dir,
            evaluation.candidate_report.output_dir,
        )
    return ProfileEvolutionResult(
        evaluation.decision,
        evaluation.outcome_type,
        evaluation.reason,
        evaluation.candidate_report,
        evaluation.promotion_parent_report,
        candidate,
    )


def _repair_guidance(diagnosis: Diagnosis, plan: EvolutionPlan) -> str:
    return "\n".join(
        [
            f"Diagnosed problem: {diagnosis.problem}",
            f"Causal hypothesis: {plan.hypothesis}",
            f"Planned intervention: {plan.intervention}",
        ]
    )


def activate_saved_adapter(
    config: EvolutionConfig,
    evaluator: BenchmarkEvaluator,
    adapter: dict[str, Any] | None,
) -> None:
    if not adapter:
        evaluator.set_model(config.model)
        return
    VLLMAdapterRuntime(config.base_url).activate(adapter["name"], Path(adapter["path"]))
    evaluator.set_model(adapter["name"])


def _activate_parent(
    runtime: AdapterRuntime,
    evaluator: BenchmarkEvaluator,
    base_model: str,
    parent_adapter: dict[str, Any] | None,
    candidate_name: str,
) -> None:
    runtime.deactivate(candidate_name)
    if parent_adapter:
        runtime.activate(parent_adapter["name"], Path(parent_adapter["path"]))
        evaluator.set_model(parent_adapter["name"])
    else:
        evaluator.set_model(base_model)


def _activate_candidate(
    runtime: AdapterRuntime,
    evaluator: BenchmarkEvaluator,
    candidate_name: str,
    candidate_path: Path,
    parent_adapter: dict[str, Any] | None,
) -> None:
    if parent_adapter:
        runtime.deactivate(parent_adapter["name"])
    runtime.activate(candidate_name, candidate_path)
    evaluator.set_model(candidate_name)


def _run_training(config: EvolutionConfig, config_path: Path, log_path: Path) -> None:
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": config.sft_device,
        "PYTHONUNBUFFERED": "1",
    }
    command = [
        sys.executable,
        "-m",
        "strataevo.evolution.sft",
        "--config",
        str(config_path),
    ]
    print(
        f"[evolution] training LoRA on GPU {config.sft_device} -> {log_path.parent / 'adapter'}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"LoRA training failed; see {log_path}")
    print("[evolution] LoRA training completed", flush=True)
