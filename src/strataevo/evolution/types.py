"""Small persistent types for the evolution loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_MUTABLE_PATHS = ["src/tinyagent"]


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
    model_evolution: bool = False
    force_layer: str | None = None
    sft_device: str = "1"
    sft_epochs: int = 1
    sft_max_samples: int = 32
    sft_max_length: int = 4096
    sft_lora_rank: int = 8
    sft_learning_rate: float = 1e-4
    repair_attempts: int = 2
    repair_temperature: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionConfig:
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


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
        return cls(
            task_score=float(data["task_score"]),
            metrics=dict(data["metrics"]),
            output_dir=str(data["output_dir"]),
            log_path=str(data["log_path"]),
        )


@dataclass(slots=True)
class GenerationRecord:
    generation: int
    parent_commit: str
    resulting_commit: str | None
    layer: str
    decision: str
    outcome: str
    reason: str
    parent_report: dict[str, Any]
    candidate_report: dict[str, Any] | None
    changed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
