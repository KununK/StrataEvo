"""Versioned prompt evolution evaluated without rewriting source code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel

from .contract import EvaluationContract
from .diagnosis import DiagnosisReport
from .evaluation import BenchmarkEvaluator
from .io import write_json
from .memory import EvolutionMemoryEntry, memory_context
from .plan import EvolutionPlanReport
from .structured import request_json
from .types import EvaluationReport, EvolutionConfig

MAX_CONTEXT_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class ContextProfile:
    system_prompt_addendum: str = ""
    task_prompt_addendum: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ContextProfile:
        values = data or {}
        system = values.get("system_prompt_addendum", "")
        task = values.get("task_prompt_addendum", "")
        if not isinstance(system, str) or not isinstance(task, str):
            raise ValueError("context addenda must be strings")
        profile = cls(system.strip(), task.strip())
        if len(profile.system_prompt_addendum) + len(profile.task_prompt_addendum) > (
            MAX_CONTEXT_CHARS
        ):
            raise ValueError(f"context profile exceeds {MAX_CONTEXT_CHARS} characters")
        return profile


@dataclass(slots=True)
class ContextEvolutionResult:
    decision: str
    outcome_type: str
    reason: str
    candidate_report: EvaluationReport | None
    promotion_parent_report: EvaluationReport | None
    candidate: dict[str, Any]
    input_tokens: int
    output_tokens: int


class ContextEvolver:
    def __init__(self, model: Model, *, repair_retries: int = 1) -> None:
        self.model = model
        self.repair_retries = repair_retries

    def create_candidate(
        self,
        parent: ContextProfile,
        diagnosis: DiagnosisReport,
        plan: EvolutionPlanReport,
        history: list[EvolutionMemoryEntry],
        contract: EvaluationContract,
    ) -> tuple[ContextProfile, dict[str, Any]]:
        selected = diagnosis.diagnoses[plan.plan.target_diagnosis]
        payload = {
            "parent_context": parent.to_dict(),
            "diagnosis": selected.to_dict(),
            "plan": plan.plan.to_dict(),
            "evaluation_contract": contract.to_dict(),
            "prior_evolution": memory_context(history, max_chars=8_000),
        }

        def parse_candidate(data: dict[str, Any]) -> ContextProfile:
            profile = ContextProfile.from_dict(data)
            if not (profile.system_prompt_addendum or profile.task_prompt_addendum):
                raise ValueError("context candidate must contain at least one addendum")
            return profile

        response = request_json(
            self.model,
            [
                Message("system", CONTEXT_EVOLVER_SYSTEM_PROMPT),
                Message(
                    "user",
                    "Create one general context candidate for the supplied plan. "
                    "Return only the requested JSON.\n\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False),
                ),
            ],
            parse_candidate,
            label="context candidate",
            repair_retries=self.repair_retries,
        )
        metadata = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "raw_output": response.raw_output,
            "attempts": response.attempts,
        }
        return response.value, metadata


CONTEXT_EVOLVER_SYSTEM_PROMPT = """You evolve the reusable context of a software agent.
Change only general instructions that can improve future tasks. Do not include task IDs,
task-specific solutions, hidden tests, or benchmark answers. Preserve the evaluation contract.

Return exactly:
{
  "system_prompt_addendum": "general instruction appended to the system prompt, or empty",
  "task_prompt_addendum": "general instruction appended to each task prompt, or empty"
}
At least one field must contain a focused change."""


def evolve_context(
    config: EvolutionConfig,
    generation: int,
    generation_dir: Path,
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    diagnosis: DiagnosisReport,
    plan: EvolutionPlanReport,
    history: list[EvolutionMemoryEntry],
    *,
    parent_context: dict[str, Any] | None,
    model_name: str,
    model: Model | None = None,
) -> ContextEvolutionResult:
    """Generate, evaluate, and retain or restore one prompt-context candidate."""
    context_dir = generation_dir / "context"
    parent = ContextProfile.from_dict(parent_context)
    write_json(context_dir / "parent.json", parent.to_dict())
    model = model or OpenAICompatibleModel(
        model=model_name,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    candidate, metadata = ContextEvolver(model).create_candidate(
        parent,
        diagnosis,
        plan,
        history,
        evaluator.contract,
    )
    if candidate == parent:
        return ContextEvolutionResult(
            "rejected",
            "no_change",
            "context candidate is identical to its parent",
            None,
            None,
            parent.to_dict(),
            metadata["input_tokens"],
            metadata["output_tokens"],
        )

    candidate_path = context_dir / "candidate.json"
    candidate_record = {**candidate.to_dict(), "path": str(candidate_path)}
    write_json(candidate_path, {**candidate.to_dict(), "generation": generation, **metadata})

    evaluator.set_context(candidate_record)
    try:
        screening = evaluator.evaluate(context_dir / "screening" / "evaluation")
        if screening.task_score <= parent_report.task_score:
            evaluator.set_context(parent_context)
            return ContextEvolutionResult(
                "rejected",
                "benchmark_rejected",
                (
                    f"candidate screening: pass@1 {screening.task_score:.6f} "
                    f"did not exceed parent {parent_report.task_score:.6f}"
                ),
                screening,
                None,
                candidate_record,
                metadata["input_tokens"],
                metadata["output_tokens"],
            )

        evaluator.set_context(parent_context)
        fresh_parent = evaluator.evaluate(context_dir / "promotion" / "parent" / "evaluation")
        evaluator.set_context(candidate_record)
        confirmed = evaluator.evaluate(context_dir / "promotion" / "candidate" / "evaluation")
    except BaseException:
        evaluator.set_context(parent_context)
        raise

    if confirmed.task_score <= fresh_parent.task_score:
        evaluator.set_context(parent_context)
        return ContextEvolutionResult(
            "rejected",
            "benchmark_rejected",
            (
                f"fresh promotion comparison: pass@1 {confirmed.task_score:.6f} "
                f"did not exceed parent {fresh_parent.task_score:.6f}"
            ),
            confirmed,
            fresh_parent,
            candidate_record,
            metadata["input_tokens"],
            metadata["output_tokens"],
        )
    return ContextEvolutionResult(
        "accepted",
        "accepted",
        "fresh promotion comparison: pass@1 strictly improved",
        confirmed,
        fresh_parent,
        candidate_record,
        metadata["input_tokens"],
        metadata["output_tokens"],
    )
