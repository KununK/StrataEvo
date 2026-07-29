"""Fixed evaluation environment used to compare consecutive agent generations."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .contract import EvaluationContract
from .evidence import CodingAgentEvidenceCollector
from .types import EvaluationReport, EvolutionConfig


class Evaluator(Protocol):
    contract: EvaluationContract

    def set_context(self, context: dict[str, Any] | None) -> None: ...

    def set_tool_profile(self, profile: dict[str, Any] | None) -> None: ...

    def evaluate(self, output_dir: Path) -> EvaluationReport: ...


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    name: str
    display_name: str
    module: str
    objective: str
    direct_paths: tuple[str, ...] = ("src/tinyagent",)
    deferred_paths: tuple[str, ...] = ("src/strataevo/evolution/mutator.py",)
    arguments: tuple[str, ...] = ()


def validation_commands(repo: Path) -> list[list[str]]:
    python = str(repo / ".venv" / "bin" / "python")
    ruff = str(repo / ".venv" / "bin" / "ruff")
    if not Path(python).is_file() or not Path(ruff).is_file():
        raise FileNotFoundError("expected .venv/bin/python and .venv/bin/ruff")
    return [
        [ruff, "check", "src", "tests", "eval"],
        [python, "-m", "pytest", "-q"],
        [python, "-m", "strataevo.evolution.cli", "--help"],
    ]


def run_commands(
    commands: list[list[str]], repo: Path, log_path: Path, *, timeout: float = 600.0
) -> tuple[bool, str]:
    outputs = []
    passed = True
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = completed.stdout + completed.stderr
            outputs.append(f"$ {' '.join(command)}\nexit_code={completed.returncode}\n{output}")
            if completed.returncode != 0:
                passed = False
                break
        except subprocess.TimeoutExpired as error:
            outputs.append(f"$ {' '.join(command)}\ntimeout={timeout}\n{error}")
            passed = False
            break
    text = "\n\n".join(outputs)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return passed, text


class BenchmarkEvaluator:
    """Run one Agent benchmark and normalize its artifacts for evolution."""

    def __init__(self, repo: Path, config: EvolutionConfig, spec: BenchmarkSpec) -> None:
        self.repo = repo
        self.config = config
        self.spec = spec
        self.model = config.model
        self.context: dict[str, Any] | None = None
        self.tool_profile: dict[str, Any] | None = None
        self.contract = EvaluationContract(
            benchmark=spec.display_name,
            objective=spec.objective,
            direct_paths=spec.direct_paths,
            deferred_paths=spec.deferred_paths,
        )

    def set_model(self, model: str) -> None:
        self.model = model

    def set_context(self, context: dict[str, Any] | None) -> None:
        self.context = context

    def set_tool_profile(self, profile: dict[str, Any] | None) -> None:
        self.tool_profile = profile

    def evaluate(self, output_dir: Path) -> EvaluationReport:
        print(
            f"[evolution] evaluating {self.spec.display_name} -> {output_dir}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            self.spec.module,
            "--model",
            self.model,
            "--base-url",
            self.config.base_url,
            "--output-dir",
            str(output_dir),
            "--offset",
            str(self.config.eval_offset),
            "--workers",
            str(self.config.eval_workers),
            "--max-steps",
            str(self.config.benchmark_max_steps),
            "--test-timeout",
            str(self.config.test_timeout),
            "--no-resume",
            *self.spec.arguments,
        ]
        if self.context:
            context_path = output_dir.parent / f"{output_dir.name}.context.json"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text(
                json.dumps(self.context, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command.extend(["--context-file", str(context_path)])
        if self.tool_profile:
            profile_path = output_dir.parent / f"{output_dir.name}.tools.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(
                json.dumps(self.tool_profile, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            command.extend(["--tool-profile-file", str(profile_path)])
        if self.config.eval_limit is not None:
            command.extend(["--limit", str(self.config.eval_limit)])
        log_path = output_dir.parent / f"{output_dir.name}.log"
        passed, output = run_commands([command], self.repo, log_path, timeout=7_200)
        summary_path = output_dir / "summary.json"
        if not passed or not summary_path.is_file():
            raise RuntimeError(f"{self.spec.display_name} failed; see {log_path}\n{output[-2000:]}")
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))
        evidence = CodingAgentEvidenceCollector(self.spec.name).collect_and_write(output_dir)
        task_score = float(metrics["pass_at_1"])
        evaluated = max(int(metrics.get("evaluated", 0)), 1)
        metrics["average_tokens"] = (
            metrics.get("total_input_tokens", 0) + metrics.get("total_output_tokens", 0)
        ) / evaluated
        metrics["evidence_path"] = str(output_dir / "evidence.json")
        metrics["evidence_signal_counts"] = evidence.signal_counts
        metrics["evaluation_contract"] = self.contract.to_dict()
        print(f"[evolution] pass@1={task_score:.4f}", flush=True)
        return EvaluationReport(task_score, metrics, str(output_dir), str(log_path))


HUMANEVAL = BenchmarkSpec(
    name="humaneval",
    display_name="HumanEval",
    module="eval.humaneval.run",
    objective="Improve the task-solving Tinyagent measured by HumanEval pass@1.",
)

MBPP = BenchmarkSpec(
    name="mbpp",
    display_name="MBPP",
    module="eval.mbpp.run",
    objective="Improve the task-solving Tinyagent measured by MBPP pass@1.",
)

BENCHMARKS = {
    HUMANEVAL.name: HUMANEVAL,
    MBPP.name: MBPP,
}


def create_evaluator(repo: Path, config: EvolutionConfig) -> BenchmarkEvaluator:
    try:
        spec = BENCHMARKS[config.benchmark]
    except KeyError as error:
        raise ValueError(f"unknown benchmark: {config.benchmark}") from error
    return BenchmarkEvaluator(repo, config, spec)
