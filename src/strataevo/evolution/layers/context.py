"""Versioned prompt evolution evaluated without rewriting source code."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel

from ..diagnosis import DiagnosisReport
from ..memory import EvolutionMemoryEntry, memory_context
from ..plan import EvolutionPlanReport
from ..runtime.contract import EvaluationContract
from ..runtime.evaluation import BenchmarkEvaluator
from ..runtime.structured import request_json
from ..types import EvaluationReport, EvolutionConfig
from .profile import ProfileEvolutionResult, evolve_profile

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
        feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[ContextProfile, dict[str, Any]]:
        selected = diagnosis.diagnoses[plan.plan.target_diagnosis]
        payload = {
            "parent_context": parent.to_dict(),
            "diagnosis": selected.to_dict(),
            "plan": plan.plan.to_dict(),
            "evaluation_contract": contract.to_dict(),
            "prior_evolution": memory_context(history, max_chars=8_000),
            "refinement_feedback": (feedback or [])[-1:],
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
                    "When refinement_feedback is present, revise the prior candidate in response "
                    "to its measured outcome. "
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
) -> ProfileEvolutionResult:
    """Generate, evaluate, and retain or restore one prompt-context candidate."""
    context_dir = generation_dir / "context"
    parent = ContextProfile.from_dict(parent_context)
    model = model or OpenAICompatibleModel(
        model=model_name,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    evolver = ContextEvolver(model)

    def create_candidate(
        feedback: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate, metadata = evolver.create_candidate(
            parent,
            diagnosis,
            plan,
            history,
            evaluator.contract,
            feedback,
        )
        return candidate.to_dict(), metadata

    return evolve_profile(
        generation=generation,
        output_dir=context_dir,
        label="context",
        parent_report=parent_report,
        evaluator=evaluator,
        normalized_parent=parent.to_dict(),
        active_parent=parent_context,
        create_candidate=create_candidate,
        activate=evaluator.set_context,
        max_evaluations=config.max_eval_attempts,
    )
