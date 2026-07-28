"""Choose one testable intervention from structured evolution diagnoses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel

from .contract import EvaluationContract
from .diagnosis import DiagnosisReport, EvolutionLayer
from .io import write_json
from .memory import EvolutionMemoryEntry, memory_context
from .structured import request_json
from .types import DEFAULT_MUTABLE_PATHS, EvaluationReport, EvolutionConfig


class MetricDirection(StrEnum):
    INCREASE = "increase"
    DECREASE = "decrease"
    NON_DECREASING = "non_decreasing"
    NON_INCREASING = "non_increasing"


@dataclass(slots=True)
class ExpectedOutcome:
    metric: str
    direction: MetricDirection
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["direction"] = self.direction.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any], available_metrics: set[str]) -> ExpectedOutcome:
        metric = str(data.get("metric", "")).strip()
        reason = str(data.get("reason", "")).strip()
        if metric not in available_metrics:
            raise ValueError(f"plan references unavailable metric: {metric!r}")
        if not reason:
            raise ValueError("plan expected outcome reason must not be empty")
        try:
            direction = MetricDirection(str(data["direction"]))
        except (KeyError, ValueError) as error:
            raise ValueError("plan expected outcome has an invalid direction") from error
        return cls(metric, direction, reason)


@dataclass(slots=True)
class EvolutionPlan:
    target_diagnosis: int
    primary_layer: EvolutionLayer
    hypothesis: str
    intervention: str
    expected_outcomes: list[ExpectedOutcome]
    likely_files: list[str]
    expected_long_term_value: str
    prerequisites: list[str]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_diagnosis": self.target_diagnosis,
            "primary_layer": self.primary_layer.value,
            "hypothesis": self.hypothesis,
            "intervention": self.intervention,
            "expected_outcomes": [item.to_dict() for item in self.expected_outcomes],
            "likely_files": self.likely_files,
            "expected_long_term_value": self.expected_long_term_value,
            "prerequisites": self.prerequisites,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        diagnosis: DiagnosisReport,
        available_metrics: set[str],
        mutable_paths: list[str],
        model_evolution: bool = False,
        force_layer: EvolutionLayer | None = None,
    ) -> EvolutionPlan:
        target = data.get("target_diagnosis")
        if not isinstance(target, int) or isinstance(target, bool):
            raise ValueError("plan target_diagnosis must be an integer")
        if not 0 <= target < len(diagnosis.diagnoses):
            raise ValueError(f"plan target_diagnosis is out of range: {target}")
        try:
            primary_layer = EvolutionLayer(str(data["primary_layer"]))
        except (KeyError, ValueError) as error:
            raise ValueError("plan primary_layer is invalid") from error
        diagnosed_layer = diagnosis.diagnoses[target].primary_layer
        if primary_layer != diagnosed_layer:
            raise ValueError(
                f"plan layer {primary_layer.value!r} does not match selected diagnosis "
                f"layer {diagnosed_layer.value!r}"
            )
        if force_layer is not None and primary_layer is not force_layer:
            raise ValueError(f"plan must select forced layer {force_layer.value!r}")
        hypothesis = _required_text(data, "hypothesis")
        intervention = _required_text(data, "intervention")
        expected_long_term_value = _required_text(data, "expected_long_term_value")
        raw_outcomes = data.get("expected_outcomes")
        if not isinstance(raw_outcomes, list) or not raw_outcomes:
            raise ValueError("plan expected_outcomes must be a non-empty list")
        if len(raw_outcomes) > 6 or not all(isinstance(item, dict) for item in raw_outcomes):
            raise ValueError("plan expected_outcomes must contain one to six objects")
        try:
            confidence = float(data.get("confidence", 0.0))
        except (TypeError, ValueError) as error:
            raise ValueError("plan confidence must be numeric") from error
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("plan confidence must be between 0 and 1")
        likely_files = _string_list(data.get("likely_files", []), "likely_files")
        trains_adapter = primary_layer is EvolutionLayer.MODEL and model_evolution
        if not likely_files and not trains_adapter:
            raise ValueError("plan likely_files must not be empty")
        invalid_files = [path for path in likely_files if not _is_mutable_path(path, mutable_paths)]
        if invalid_files:
            raise ValueError(f"plan likely_files are outside mutable paths: {invalid_files}")
        return cls(
            target_diagnosis=target,
            primary_layer=primary_layer,
            hypothesis=hypothesis,
            intervention=intervention,
            expected_outcomes=[
                ExpectedOutcome.from_dict(item, available_metrics) for item in raw_outcomes
            ],
            likely_files=likely_files,
            expected_long_term_value=expected_long_term_value,
            prerequisites=_string_list(data.get("prerequisites", []), "prerequisites"),
            confidence=confidence,
        )


@dataclass(slots=True)
class EvolutionPlanReport:
    plan: EvolutionPlan
    available_metrics: dict[str, float]
    input_tokens: int
    output_tokens: int
    raw_output: str
    attempts: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "available_metrics": self.available_metrics,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "raw_output": self.raw_output,
            "attempts": self.attempts,
        }


class EvolutionPlanner:
    """Select one diagnosis and turn it into a measurable intervention."""

    def __init__(self, model: Model, *, repair_retries: int = 1) -> None:
        if repair_retries < 0:
            raise ValueError("repair_retries must be non-negative")
        self.model = model
        self.repair_retries = repair_retries

    def create_plan(
        self,
        diagnosis: DiagnosisReport,
        parent_report: EvaluationReport,
        history: list[EvolutionMemoryEntry] | None = None,
        *,
        mutable_paths: list[str] | None = None,
        existing_files: list[str] | None = None,
        evaluation_contract: EvaluationContract | None = None,
        model_evolution: bool = False,
        force_layer: EvolutionLayer | None = None,
    ) -> EvolutionPlanReport:
        mutable_paths = list(mutable_paths or DEFAULT_MUTABLE_PATHS)
        available_metrics = evaluation_metrics(parent_report.to_dict())
        if force_layer is not None and not any(
            item.primary_layer is force_layer for item in diagnosis.diagnoses
        ):
            raise ValueError(f"diagnosis contains no {force_layer.value!r} problem to force")
        payload = {
            "diagnoses": [item.to_dict() for item in diagnosis.diagnoses],
            "available_metrics": available_metrics,
            "promotion_policy": {
                "primary_metric": "task_score",
                "requires_strict_improvement": True,
                "parent_task_score": parent_report.task_score,
            },
            "repository": {
                "mutable_paths": mutable_paths,
                "existing_mutable_files": existing_files or [],
            },
            "evaluation_contract": (
                evaluation_contract.to_dict() if evaluation_contract is not None else None
            ),
            "model_evolution": {
                "enabled": model_evolution,
                "mechanism": "verifier-guided test-time LoRA SFT",
            },
            "forced_layer": force_layer.value if force_layer else None,
            "prior_evolution": memory_context(history or [], max_chars=12_000),
        }
        messages = [
            Message("system", PLANNER_SYSTEM_PROMPT),
            Message(
                "user",
                "Choose one evolution direction from the supplied diagnoses. "
                "Return only the requested JSON.\n\n"
                + json.dumps(payload, indent=2, ensure_ascii=False),
            ),
        ]

        response = request_json(
            self.model,
            messages,
            lambda data: EvolutionPlan.from_dict(
                data,
                diagnosis,
                set(available_metrics),
                mutable_paths,
                model_evolution,
                force_layer,
            ),
            label="evolution plan",
            repair_retries=self.repair_retries,
        )
        return EvolutionPlanReport(
            plan=response.value,
            available_metrics=available_metrics,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            raw_output=response.raw_output,
            attempts=response.attempts,
        )


PLANNER_SYSTEM_PROMPT = """You plan one intervention for a self-evolving software agent.
Select exactly one supplied diagnosis. Prefer a direction supported by concrete evidence, not
already disproven by prior evolution, and likely to produce a measurable improvement without
regressing task quality. A rejected prior attempt does not prove the whole direction is useless,
but repeating the same intervention requires new evidence or a materially different mechanism.
Treat prior outcome_type=no_change or validation_failed as an execution failure, not benchmark
evidence against its hypothesis. Use those records to choose a more executable intervention.
Treat deferred_change, mixed_change_scope, and unclassified_change as an evaluation-contract
mismatch: the intervention was not tested by the benchmark and must not be interpreted as a
negative task result.

The promotion policy uses task score. Prefer an intervention that can plausibly fix an actually
failed task. Efficiency observations from passed tasks may be useful evidence, but are not direct
proof of task-score improvement.

Some changes may enable future improvements without helping the current benchmark immediately.
Record that possibility in expected_long_term_value and prerequisites, but do not use speculative
future value as proof of current benefit. Every plan must still have one or more outcomes measurable
now using only names from available_metrics. Every likely_files entry must be an existing file from
repository.existing_mutable_files or a plausible new file below repository.mutable_paths. These
paths guide the mutation but do not narrow its configured write permissions.

The evaluation_contract is authoritative when present. Its direct_paths are loaded by the active
benchmark and can be evaluated now. Its deferred_paths affect later evolution but are not loaded by
the active benchmark. Plan only a change under direct_paths. Never claim that a deferred change is
validated by an immediate benchmark score. The one exception is a primary_layer=model plan when
model_evolution.enabled is true: it trains a LoRA adapter from verifier-passing trajectories and
must use an empty likely_files list because it changes weights rather than repository files.
When forced_layer is not null, select a diagnosis whose primary_layer exactly matches it.

Return exactly this JSON object:
{
  "target_diagnosis": 0,
  "primary_layer": "model|context|tools|architecture",
  "hypothesis": "causal, testable hypothesis",
  "intervention": "one focused behavioral change, not a detailed code patch",
  "expected_outcomes": [
    {
      "metric": "one exact key from available_metrics",
      "direction": "increase|decrease|non_decreasing|non_increasing",
      "reason": "why this metric tests the hypothesis"
    }
  ],
  "likely_files": ["advisory repository path"],
  "expected_long_term_value": "possible future capability, or explain why value is immediate",
  "prerequisites": ["condition needed for the future value"],
  "confidence": 0.0
}"""


def plan_evolution(
    config: EvolutionConfig,
    diagnosis: DiagnosisReport,
    parent_report: EvaluationReport,
    history: list[EvolutionMemoryEntry],
    destination: str | Path,
    evaluation_contract: EvaluationContract | None = None,
) -> EvolutionPlanReport:
    forced_layer = EvolutionLayer(config.force_layer) if config.force_layer else None
    eligible_paths = (
        list(evaluation_contract.direct_paths)
        if evaluation_contract is not None
        else config.mutable_paths
    )
    existing_files = mutable_source_files(config.repo, eligible_paths)
    model = OpenAICompatibleModel(
        model=config.model,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    report = EvolutionPlanner(model).create_plan(
        diagnosis,
        parent_report,
        history,
        mutable_paths=eligible_paths,
        existing_files=existing_files,
        evaluation_contract=evaluation_contract,
        model_evolution=config.model_evolution,
        force_layer=forced_layer,
    )
    write_json(destination, report.to_dict())
    return report


def evaluation_metrics(report: dict[str, Any]) -> dict[str, float]:
    values = {
        "task_score": float(report["task_score"]),
    }
    metrics = report.get("metrics") or {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.setdefault(str(name), float(value))
    for name, value in (metrics.get("evidence_signal_counts") or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[f"signal:{name}"] = float(value)
    return dict(sorted(values.items()))


def mutable_source_files(repo: str | Path, mutable_paths: list[str]) -> list[str]:
    root = Path(repo).resolve()
    files: list[str] = []
    for relative in mutable_paths:
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"mutable path escapes repository: {relative}")
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = target.rglob("*")
        else:
            candidates = []
        for candidate in candidates:
            if candidate.is_file() and "__pycache__" not in candidate.parts:
                files.append(candidate.relative_to(root).as_posix())
    return sorted(set(files))


def observe_expected_outcomes(
    plan: EvolutionPlan,
    parent_report: dict[str, Any],
    candidate_report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    parent_metrics = evaluation_metrics(parent_report)
    candidate_metrics = evaluation_metrics(candidate_report) if candidate_report else {}
    observations = []
    for outcome in plan.expected_outcomes:
        parent_value = _metric_value(parent_metrics, outcome.metric, report_exists=True)
        candidate_value = _metric_value(
            candidate_metrics, outcome.metric, report_exists=candidate_report is not None
        )
        observations.append(
            {
                **outcome.to_dict(),
                "parent_value": parent_value,
                "candidate_value": candidate_value,
                "satisfied": _satisfied(outcome.direction, parent_value, candidate_value),
            }
        )
    return observations


def _satisfied(
    direction: MetricDirection,
    parent: float | None,
    candidate: float | None,
) -> bool | None:
    if parent is None or candidate is None:
        return None
    comparisons = {
        MetricDirection.INCREASE: candidate > parent,
        MetricDirection.DECREASE: candidate < parent,
        MetricDirection.NON_DECREASING: candidate >= parent,
        MetricDirection.NON_INCREASING: candidate <= parent,
    }
    return comparisons[direction]


def _metric_value(metrics: dict[str, float], name: str, *, report_exists: bool) -> float | None:
    if name.startswith("signal:") and report_exists:
        return metrics.get(name, 0.0)
    return metrics.get(name)


def _required_text(data: dict[str, Any], field_name: str) -> str:
    value = str(data.get(field_name, "")).strip()
    if not value:
        raise ValueError(f"plan {field_name} must not be empty")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"plan {field_name} must be a list of strings")
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def _is_mutable_path(path: str, mutable_paths: list[str]) -> bool:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in path:
        return False
    return any(
        candidate == root or root in candidate.parents
        for root in (PurePosixPath(item) for item in mutable_paths)
    )
