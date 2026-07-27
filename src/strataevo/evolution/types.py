"""Persistent records shared by the self-evolution loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_MUTABLE_PATHS = ["src/tinyagent", "src/strataevo/evolution/mutator.py"]


@dataclass(slots=True)
class EvolutionConfig:
    repo: str
    run_name: str
    branch: str = "evo"
    benchmark: str = "humaneval"
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    mutator_max_steps: int = 200
    mutator_rounds: int = 5
    max_eval_attempts: int = 5
    mutable_paths: list[str] = field(default_factory=lambda: list(DEFAULT_MUTABLE_PATHS))
    eval_limit: int | None = None
    eval_offset: int = 0
    eval_workers: int = 4
    benchmark_max_steps: int = 12
    test_timeout: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionConfig:
        values = dict(data)
        for legacy in (
            "generations",
            "step_penalty",
            "token_penalty",
            "min_utility_delta",
            "max_score_drop",
        ):
            values.pop(legacy, None)
        return cls(**values)


@dataclass(slots=True)
class EvaluationReport:
    task_score: float
    metrics: dict[str, Any]
    output_dir: str
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        values = dict(data)
        values.pop("utility", None)
        return cls(**values)


@dataclass(slots=True)
class GenerationRecord:
    generation: int
    parent_commit: str
    resulting_commit: str | None
    decision: str
    outcome_type: str
    reason: str
    changed_paths: list[str]
    patch_path: str | None
    diagnosis_path: str
    diagnosed_layers: list[str]
    plan_path: str
    planned_layer: str
    plan_hypothesis: str
    parent_report: dict[str, Any]
    promotion_parent_report: dict[str, Any] | None
    candidate_report: dict[str, Any] | None
    agent_stop_reason: str
    agent_steps: int
    input_tokens: int
    output_tokens: int
    evaluation_attempts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
