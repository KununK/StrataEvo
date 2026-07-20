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
    generations: int = 1
    model: str = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    mutator_max_steps: int = 20
    mutable_paths: list[str] = field(default_factory=lambda: list(DEFAULT_MUTABLE_PATHS))
    eval_limit: int = 5
    eval_offset: int = 0
    eval_workers: int = 4
    benchmark_max_steps: int = 8
    test_timeout: float = 10.0
    step_penalty: float = 0.001
    token_penalty: float = 0.0000001
    min_utility_delta: float = 0.0
    max_score_drop: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionConfig:
        return cls(**data)


@dataclass(slots=True)
class EvaluationReport:
    task_score: float
    utility: float
    metrics: dict[str, Any]
    output_dir: str
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationReport:
        return cls(**data)


@dataclass(slots=True)
class GenerationRecord:
    generation: int
    parent_commit: str
    resulting_commit: str | None
    decision: str
    reason: str
    changed_paths: list[str]
    patch_path: str | None
    diagnosis_path: str
    diagnosed_layers: list[str]
    parent_report: dict[str, Any]
    candidate_report: dict[str, Any] | None
    agent_stop_reason: str
    agent_steps: int
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
