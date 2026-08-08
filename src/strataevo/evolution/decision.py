"""Choose one evidence-grounded intervention for the next generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from tinyagent import Message, Model, OpenAICompatibleModel

from .evidence import EvidenceBundle, TaskEvidence
from .memory import EvolutionMemoryEntry, memory_context
from .runtime.structured import request_json
from .types import EvaluationReport, EvolutionConfig
from .utils.io import write_json


class EvolutionLayer(StrEnum):
    MODEL = "model"
    CONTEXT = "context"
    TOOLS = "tools"
    ARCHITECTURE = "architecture"


@dataclass(slots=True)
class EvolutionDecision:
    layer: EvolutionLayer
    evidence: list[str]
    affected_tasks: list[str]
    hypothesis: str
    intervention: str
    likely_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["layer"] = self.layer.value
        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        known_tasks: set[str],
        config: EvolutionConfig,
    ) -> EvolutionDecision:
        layer = EvolutionLayer(str(data["layer"]))
        if config.force_layer and layer.value != config.force_layer:
            raise ValueError(f"decision must use forced layer {config.force_layer}")
        if layer == EvolutionLayer.MODEL and not config.model_evolution:
            raise ValueError("model evolution is disabled")

        evidence = _strings(data.get("evidence"))
        tasks = _strings(data.get("affected_tasks"))
        hypothesis = str(data.get("hypothesis", "")).strip()
        intervention = str(data.get("intervention", "")).strip()
        if not evidence or not tasks or not hypothesis or not intervention:
            raise ValueError(
                "decision requires evidence, affected tasks, hypothesis, and intervention"
            )
        if not set(tasks) <= known_tasks:
            raise ValueError("decision affected tasks must come from current evidence")

        files = _strings(data.get("likely_files"))
        if layer == EvolutionLayer.ARCHITECTURE:
            if not files or any(
                not _is_mutable(path, config.mutable_paths)
                or not (Path(config.repo) / path).is_file()
                for path in files
            ):
                raise ValueError("architecture decision requires existing mutable likely_files")
        else:
            files = []
        return cls(layer, evidence, tasks, hypothesis, intervention, files)


SYSTEM_PROMPT = """Choose one evidence-grounded failure mechanism and one intervention.
Select the layer whose executor can change the observed cause:
- model: knowledge, reasoning, or generated-code correctness;
- context: reusable instructions or task workflow;
- tools: observed misuse of an existing agent tool caused by its description or interface;
- architecture: observed agent-loop, state, context-management, or tool-orchestration behavior.
Python, library, mathematical, and data-structure operations inside generated code are not agent
tools. A task-specific wrong answer or assertion failure is model evidence unless the trace shows
a different layer's mechanism caused it. Architecture must name mutable src/tinyagent files; every
other layer must return empty likely_files. Choose model only when model_evolution_enabled is true.
A later observed tool call that directly explains an outcome is stronger evidence than an inferred
environment fault. For architecture, connect that event to agent-loop behavior in an existing
source file and state the smallest observable behavior change rather than inventing a subsystem.
If max_steps is reported with fewer steps than configured, diagnose premature loop termination
rather than normal budget exhaustion or artifact persistence.
An orphan_tool_result means a tool result lost its initiating assistant call; treat it as direct
architecture evidence of broken message-state transfer, not artifact persistence.
Use only current-task evidence; history only shows prior intervention outcomes. Do not repeat an
exact rejected intervention. Return JSON:
{"layer":"model|context|tools|architecture","evidence":["..."],
"affected_tasks":["..."],"hypothesis":"...","intervention":"...","likely_files":[]}"""


class EvolutionDecider:
    def __init__(self, model: Model, max_cases: int = 40) -> None:
        self.model = model
        self.max_cases = max_cases

    def decide(
        self,
        bundle: EvidenceBundle,
        report: EvaluationReport,
        history: list[EvolutionMemoryEntry],
        config: EvolutionConfig,
    ) -> EvolutionDecision:
        cases = sorted(
            bundle.cases,
            key=lambda item: (item.passed, -len(item.signals), -item.steps),
        )[: self.max_cases]
        payload = {
            "task_score": report.task_score,
            "summary": bundle.summary,
            "signals": bundle.signal_counts,
            "cases": [_compact_case(case) for case in cases],
            "prior_generations": memory_context(history),
            "forced_layer": config.force_layer,
            "model_evolution_enabled": config.model_evolution,
            "benchmark_max_steps": config.benchmark_max_steps,
            "mutable_paths": config.mutable_paths,
            "mutable_source_files": _mutable_source_files(config),
        }
        known_tasks = {case.task_id for case in cases}

        def parse(data: dict[str, Any]) -> EvolutionDecision:
            decision = EvolutionDecision.from_dict(data, known_tasks, config)
            if not config.force_layer:
                _validate_layer_capability(decision, cases)
            _validate_intervention_novelty(decision, history)
            return decision

        return request_json(
            self.model,
            [Message("system", SYSTEM_PROMPT), Message("user", json.dumps(payload))],
            parse,
            label="evolution decision",
            repair_retries=1,
        ).value


def decide_evolution(
    config: EvolutionConfig,
    report: EvaluationReport,
    destination: str | Path,
    history: list[EvolutionMemoryEntry],
) -> EvolutionDecision:
    evidence_path = Path(report.metrics["evidence_path"])
    bundle = EvidenceBundle.from_dict(json.loads(evidence_path.read_text(encoding="utf-8")))
    model = OpenAICompatibleModel(
        model=config.model, base_url=config.base_url, temperature=0.0, timeout=300.0
    )
    decision = EvolutionDecider(model).decide(bundle, report, history, config)
    write_json(destination, decision.to_dict())
    return decision


def _compact_case(case: TaskEvidence) -> dict[str, Any]:
    return {
        "task_id": case.task_id,
        "passed": case.passed,
        "status": case.status,
        "stop_reason": case.stop_reason,
        "signals": case.signals,
        "error": case.error[:500],
        "tool_events": [
            {
                "name": event.name,
                "arguments": event.arguments,
                "result": event.result[:300] if event.result else None,
            }
            for event in case.tool_events[:30]
        ],
    }


def _strings(value: Any) -> list[str]:
    return (
        list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if isinstance(value, list)
        else []
    )


def _is_mutable(path: str, roots: list[str]) -> bool:
    clean = path.strip("/")
    return any(
        clean == root.strip("/") or clean.startswith(root.strip("/") + "/") for root in roots
    )


def _mutable_source_files(config: EvolutionConfig) -> list[str]:
    repo = Path(config.repo).resolve()
    return sorted(
        str(path.relative_to(repo))
        for root in config.mutable_paths
        for path in (repo / root).rglob("*.py")
        if path.is_file()
    )


def _validate_layer_capability(
    decision: EvolutionDecision, cases: list[TaskEvidence]
) -> None:
    if decision.layer != EvolutionLayer.ARCHITECTURE:
        return
    selected = {case.task_id: case for case in cases}
    if any(
        _artifact_removed_by_tool(selected[task_id])
        for task_id in decision.affected_tasks
    ):
        raise ValueError(
            "an observed tool call removed the output artifact; choose tools, not architecture"
        )


def _artifact_removed_by_tool(case: TaskEvidence) -> bool:
    if not case.candidate_created or case.candidate_present:
        return False
    writes = {
        str(event.arguments.get("path", "")): event.index
        for event in case.tool_events
        if event.name == "write_file" and event.arguments.get("path")
    }
    return any(
        event.name == "run_shell"
        and any(
            token in str(event.arguments.get("command", ""))
            for token in ("rm ", "unlink ")
        )
        and any(
            path in str(event.arguments.get("command", "")) and event.index > write_index
            for path, write_index in writes.items()
        )
        for event in case.tool_events
    )


def _validate_intervention_novelty(
    decision: EvolutionDecision, history: list[EvolutionMemoryEntry]
) -> None:
    intervention = " ".join(decision.intervention.casefold().split())
    if any(
        entry.decision == "rejected"
        and intervention == " ".join(entry.intervention.casefold().split())
        for entry in history
    ):
        raise ValueError("intervention exactly repeats a rejected prior intervention")
