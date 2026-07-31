"""Versioned evolution of tool descriptions exposed to the model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel, Workspace

from .contract import EvaluationContract
from .diagnosis import DiagnosisReport
from .evaluation import BenchmarkEvaluator
from .memory import EvolutionMemoryEntry, memory_context
from .plan import EvolutionPlanReport
from .profile_evolution import ProfileEvolutionResult, evolve_profile
from .structured import request_json
from .types import EvaluationReport, EvolutionConfig

MAX_TOOL_PROFILE_CHARS = 8_000


@dataclass(frozen=True, slots=True)
class ToolProfile:
    description_addenda: dict[str, str]

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"description_addenda": dict(self.description_addenda)}

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        known_tools: set[str],
    ) -> ToolProfile:
        values = (data or {}).get("description_addenda", {})
        if not isinstance(values, dict) or not all(
            isinstance(name, str) and isinstance(text, str) for name, text in values.items()
        ):
            raise ValueError("tool description_addenda must be an object of strings")
        addenda = {name: text.strip() for name, text in values.items() if text.strip()}
        unknown = set(addenda) - known_tools
        if unknown:
            raise ValueError(f"unknown tools: {', '.join(sorted(unknown))}")
        if sum(map(len, addenda.values())) > MAX_TOOL_PROFILE_CHARS:
            raise ValueError(f"tool profile exceeds {MAX_TOOL_PROFILE_CHARS} characters")
        return cls(addenda)


class ToolEvolver:
    def __init__(self, model: Model, *, repair_retries: int = 1) -> None:
        self.model = model
        self.repair_retries = repair_retries

    def create_candidate(
        self,
        parent: ToolProfile,
        tool_descriptions: dict[str, str],
        diagnosis: DiagnosisReport,
        plan: EvolutionPlanReport,
        history: list[EvolutionMemoryEntry],
        contract: EvaluationContract,
    ) -> tuple[ToolProfile, dict[str, Any]]:
        selected = diagnosis.diagnoses[plan.plan.target_diagnosis]
        payload = {
            "parent_tool_profile": parent.to_dict(),
            "available_tools": tool_descriptions,
            "diagnosis": selected.to_dict(),
            "plan": plan.plan.to_dict(),
            "evaluation_contract": contract.to_dict(),
            "prior_evolution": memory_context(history, max_chars=8_000),
        }
        known_tools = set(tool_descriptions)

        def parse_candidate(data: dict[str, Any]) -> ToolProfile:
            profile = ToolProfile.from_dict(data, known_tools)
            if not profile.description_addenda:
                raise ValueError("tool candidate must contain at least one description addendum")
            return profile

        response = request_json(
            self.model,
            [
                Message("system", TOOL_EVOLVER_SYSTEM_PROMPT),
                Message(
                    "user",
                    "Create one general tool-description candidate for the supplied plan. "
                    "Return only the requested JSON.\n\n"
                    + json.dumps(payload, indent=2, ensure_ascii=False),
                ),
            ],
            parse_candidate,
            label="tool candidate",
            repair_retries=self.repair_retries,
        )
        metadata = {
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "raw_output": response.raw_output,
            "attempts": response.attempts,
        }
        return response.value, metadata


TOOL_EVOLVER_SYSTEM_PROMPT = """You evolve how a software agent understands its tools.
Change only concise, general description addenda for existing tools. Clarify intended use,
sequencing, verification, or common failure recovery. Do not change tool names, arguments,
implementations, permissions, or approval requirements. Do not include benchmark answers.

Return exactly:
{
  "description_addenda": {
    "existing_tool_name": "focused reusable guidance appended to its description"
  }
}
Include only tools whose descriptions should change."""


def evolve_tools(
    config: EvolutionConfig,
    generation: int,
    generation_dir: Path,
    parent_report: EvaluationReport,
    evaluator: BenchmarkEvaluator,
    diagnosis: DiagnosisReport,
    plan: EvolutionPlanReport,
    history: list[EvolutionMemoryEntry],
    *,
    parent_profile: dict[str, Any] | None,
    model_name: str,
    model: Model | None = None,
) -> ProfileEvolutionResult:
    """Generate, evaluate, and retain or restore one tool-description profile."""
    tool_dir = generation_dir / "tools"
    tools = Workspace(config.repo).tools()
    descriptions = {item.name: item.description for item in tools}
    parent = ToolProfile.from_dict(parent_profile, set(descriptions))
    model = model or OpenAICompatibleModel(
        model=model_name,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    candidate, metadata = ToolEvolver(model).create_candidate(
        parent,
        descriptions,
        diagnosis,
        plan,
        history,
        evaluator.contract,
    )
    return evolve_profile(
        generation=generation,
        output_dir=tool_dir,
        label="tool",
        parent_report=parent_report,
        evaluator=evaluator,
        normalized_parent=parent.to_dict(),
        active_parent=parent_profile,
        candidate=candidate.to_dict(),
        metadata=metadata,
        activate=evaluator.set_tool_profile,
    )
