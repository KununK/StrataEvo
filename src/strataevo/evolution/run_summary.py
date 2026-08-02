"""Run-level summaries derived from persisted evolution memory."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .io import write_json
from .memory import EvolutionMemoryEntry


def write_run_summary(
    run_dir: Path,
    entries: list[EvolutionMemoryEntry],
    state: dict[str, Any],
) -> None:
    generations = [_generation_summary(entry) for entry in entries]
    planned_layers = Counter(item["planned_layer"] for item in generations)
    diagnosed_layers = Counter(
        layer for item in generations for layer in item["diagnosed_layers"]
    )
    outcomes = Counter(item["outcome_type"] for item in generations)
    decisions = Counter(item["decision"] for item in generations)
    alignments = Counter(item["causal_alignment"] for item in generations)
    summary = {
        "run_name": run_dir.name,
        "generation_count": len(generations),
        "baseline_task_score": state.get(
            "baseline_task_score",
            entries[0].parent_task_score if entries else None,
        ),
        "current_task_score": state.get("current_report", {}).get("task_score"),
        "diagnosed_layers": dict(sorted(diagnosed_layers.items())),
        "planned_layers": dict(sorted(planned_layers.items())),
        "outcomes": dict(sorted(outcomes.items())),
        "decisions": dict(sorted(decisions.items())),
        "causal_alignments": dict(sorted(alignments.items())),
        "generations": generations,
    }
    write_json(run_dir / "summary.json", summary)
    _write_markdown(run_dir / "summary.md", summary)


def _generation_summary(entry: EvolutionMemoryEntry) -> dict[str, Any]:
    semantic_noops = sum(
        attempt.get("outcome_type") == "semantic_noop" for attempt in entry.evaluation_attempts
    )
    trace = entry.causal_trace
    alignment = trace.get("alignment", {})
    return {
        "generation": entry.generation,
        "diagnosed_layers": list(
            dict.fromkeys(diagnosis.primary_layer for diagnosis in entry.diagnoses)
        ),
        "planned_layer": entry.plan.get("primary_layer"),
        "candidate_type": _candidate_type(entry),
        "hypothesis": entry.plan.get("hypothesis"),
        "intervention": entry.plan.get("intervention"),
        "decision": entry.decision,
        "outcome_type": entry.outcome_type,
        "hypothesis_verdict": entry.outcome.hypothesis_verdict,
        "intervention_verdict": entry.outcome.intervention_verdict,
        "parent_task_score": entry.parent_task_score,
        "candidate_task_score": entry.candidate_task_score,
        "score_delta": entry.outcome.score_delta,
        "changed_paths": entry.changed_paths,
        "semantic_noop_attempts": semantic_noops,
        "reason": entry.reason,
        "causal_trace": trace,
        "causal_alignment": alignment.get("status", "unavailable"),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = [
        "# Evolution Run Summary",
        "",
        f"- Run: `{summary['run_name']}`",
        f"- Generations: {summary['generation_count']}",
        f"- Baseline task score: {_score(summary['baseline_task_score'])}",
        f"- Current task score: {_score(summary['current_task_score'])}",
        f"- Diagnosed layers: `{summary['diagnosed_layers']}`",
        f"- Planned layers: `{summary['planned_layers']}`",
        f"- Decisions: `{summary['decisions']}`",
        f"- Causal alignments: `{summary['causal_alignments']}`",
        "",
        "| Generation | Layer | Hypothesis | Intervention | Decision | Outcome | "
        "H verdict | I verdict | Parent | Candidate |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for item in summary["generations"]:
        rows.append(
            f"| {item['generation']} | {item['planned_layer']} | {_cell(item['hypothesis'])} | "
            f"{_cell(item['intervention'])} | {item['decision']} | {item['outcome_type']} | "
            f"{item['hypothesis_verdict']} | {item['intervention_verdict']} | "
            f"{_score(item['parent_task_score'])} | {_score(item['candidate_task_score'])} |"
        )
    rows.extend(
        [
            "",
            "## Causal Trace",
            "",
            "Alignment is observational and checks the selected layer only.",
            "",
            "| Generation | Failure mechanism | Target | Executed change | Promotion | Alignment |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for item in summary["generations"]:
        trace = item["causal_trace"]
        rows.append(
            f"| {item['generation']} | {_cell(trace.get('failure_mechanism'))} | "
            f"{_cell(trace.get('target_component'))} | "
            f"{_cell(_executed_summary(trace.get('executed_change')))} | "
            f"{_cell(_promotion_summary(trace))} | "
            f"{item['causal_alignment']} |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_type(entry: EvolutionMemoryEntry) -> str:
    if entry.model_candidate is not None:
        return "model_adapter"
    if entry.context_candidate is not None:
        return "context_profile"
    if entry.tool_candidate is not None:
        return "tool_profile"
    return "source_patch" if entry.changed_paths else "none"


def _cell(value: Any) -> str:
    return str(value or "-").replace("|", "\\|").replace("\n", " ")


def _score(value: Any) -> str:
    return "-" if value is None else f"{float(value):.6f}"


def _executed_summary(value: Any) -> str:
    if not isinstance(value, dict):
        return "-"
    change_type = str(value.get("type", "none"))
    if change_type == "source_patch":
        return f"source_patch: {', '.join(value.get('changed_paths', []))}"
    if change_type == "tool_profile":
        return f"tool_profile: {', '.join(value.get('description_addenda', {}))}"
    return change_type


def _promotion_summary(trace: dict[str, Any]) -> str:
    observed = trace.get("observed_effect")
    promotion = observed.get("promotion") if isinstance(observed, dict) else None
    if not isinstance(promotion, dict):
        return "-"
    gain = promotion.get("passed_gain")
    required = promotion.get("required_pass_gain")
    return f"passed gain {gain}/{required}" if gain is not None else "score fallback"
