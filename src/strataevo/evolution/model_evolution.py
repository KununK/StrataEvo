"""Verifier-guided test-time LoRA evolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .evaluation import BenchmarkEvaluator
from .types import EvaluationReport, EvolutionConfig


@dataclass(slots=True)
class ModelEvolutionResult:
    decision: str
    outcome_type: str
    reason: str
    candidate_report: EvaluationReport | None
    promotion_parent_report: EvaluationReport | None
    candidate: dict[str, Any]


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


def collect_verified_trajectories(
    evaluation_dir: str | Path,
    destination: str | Path,
    *,
    limit: int,
) -> int:
    """Write successful benchmark conversations as SFT examples."""
    source = Path(evaluation_dir)
    passed = {
        str(row["task_id"])
        for row in _read_jsonl(source / "results.jsonl")
        if row.get("passed", row.get("status") == "pass")
    }
    examples = []
    for row in _read_jsonl(source / "generations.jsonl"):
        if str(row.get("task_id")) not in passed or not row.get("messages"):
            continue
        examples.append({"task_id": row["task_id"], "messages": row["messages"]})
        if len(examples) >= limit:
            break
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in examples),
        encoding="utf-8",
    )
    return len(examples)


def evolve_model(
    config: EvolutionConfig,
    generation: int,
    generation_dir: Path,
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    *,
    parent_model: str,
    parent_adapter: dict[str, Any] | None,
    runtime: AdapterRuntime | None = None,
) -> ModelEvolutionResult:
    """Train, load, evaluate, and either retain or remove one LoRA candidate."""
    runtime = runtime or VLLMAdapterRuntime(config.base_url)
    model_dir = generation_dir / "model"
    data_path = model_dir / "verified_trajectories.jsonl"
    sample_count = collect_verified_trajectories(
        parent_report.output_dir,
        data_path,
        limit=config.sft_max_samples,
    )
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
    }
    if sample_count == 0:
        return ModelEvolutionResult(
            "rejected",
            "model_training_skipped",
            "no verifier-passing trajectories were available for SFT",
            None,
            None,
            candidate,
        )

    train_config = {
        "base_model": config.model,
        "parent_adapter_path": parent_adapter["path"] if parent_adapter else None,
        "data_path": str(data_path),
        "output_dir": str(adapter_path),
        "max_steps": config.sft_max_steps,
        "max_length": config.sft_max_length,
        "lora_rank": config.sft_lora_rank,
        "learning_rate": config.sft_learning_rate,
    }
    train_config_path = model_dir / "train_config.json"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_config_path.write_text(
        json.dumps(train_config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    train_log = model_dir / "train.log"
    _run_training(config, train_config_path, train_log)
    candidate["train_log"] = str(train_log)

    try:
        _activate_candidate(
            runtime,
            evaluator,
            candidate_name,
            adapter_path,
            parent_adapter,
        )
        candidate_report = evaluator.evaluate(model_dir / "screening" / "evaluation")
        if candidate_report.task_score <= parent_report.task_score:
            _restore_parent(
                runtime,
                evaluator,
                config.model,
                parent_adapter,
                candidate_name,
            )
            return ModelEvolutionResult(
                "rejected",
                "benchmark_rejected",
                (
                    f"candidate screening: pass@1 {candidate_report.task_score:.6f} "
                    f"did not exceed parent {parent_report.task_score:.6f}"
                ),
                candidate_report,
                None,
                candidate,
            )

        _activate_parent(runtime, evaluator, config.model, parent_adapter, candidate_name)
        fresh_parent = evaluator.evaluate(model_dir / "promotion" / "parent" / "evaluation")
        _activate_candidate(
            runtime,
            evaluator,
            candidate_name,
            adapter_path,
            parent_adapter,
        )
        confirmed = evaluator.evaluate(model_dir / "promotion" / "candidate" / "evaluation")
    except BaseException:
        _restore_parent(
            runtime,
            evaluator,
            config.model,
            parent_adapter,
            candidate_name,
        )
        raise
    if confirmed.task_score <= fresh_parent.task_score:
        _restore_parent(
            runtime,
            evaluator,
            config.model,
            parent_adapter,
            candidate_name,
        )
        return ModelEvolutionResult(
            "rejected",
            "benchmark_rejected",
            (
                f"fresh promotion comparison: pass@1 {confirmed.task_score:.6f} "
                f"did not exceed parent {fresh_parent.task_score:.6f}"
            ),
            confirmed,
            fresh_parent,
            candidate,
        )
    return ModelEvolutionResult(
        "accepted",
        "accepted",
        "fresh promotion comparison: pass@1 strictly improved",
        confirmed,
        fresh_parent,
        candidate,
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


def _restore_parent(
    runtime: AdapterRuntime,
    evaluator: BenchmarkEvaluator,
    base_model: str,
    parent_adapter: dict[str, Any] | None,
    candidate_name: str,
) -> None:
    _activate_parent(
        runtime,
        evaluator,
        base_model,
        parent_adapter,
        candidate_name,
    )


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows
