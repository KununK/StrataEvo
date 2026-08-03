"""Build compact causal traces for completed evolution generations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory import EvolutionMemoryEntry

PATCH_CONTEXT_CHARS = 1200
TASK_CONTEXT_LIMIT = 12
TEXT_CONTEXT_CHARS = 500


def executed_intervention(entry: EvolutionMemoryEntry) -> dict[str, Any]:
    if entry.model_candidate:
        executed = entry.model_candidate.get("executed_intervention")
        if isinstance(executed, dict):
            training = executed.get("training")
            return {
                "type": "model_adapter",
                "targeted_tasks": _limited_strings(executed.get("targeted_tasks")),
                "repaired_tasks": _limited_strings(executed.get("repaired_tasks")),
                "repair_attempts": executed.get("repair_attempts", 0),
                "repair_guidance": str(executed.get("repair_guidance", ""))[
                    :TEXT_CONTEXT_CHARS
                ],
                "training": (
                    {
                        key: training.get(key)
                        for key in ("epochs", "max_length", "lora_rank", "learning_rate")
                    }
                    if isinstance(training, dict)
                    else None
                ),
            }
        return {"type": "model_adapter"}
    if entry.context_candidate:
        return {
            "type": "context_profile",
            "system_prompt_addendum": str(
                entry.context_candidate.get("system_prompt_addendum", "")
            )[:TEXT_CONTEXT_CHARS],
            "task_prompt_addendum": str(
                entry.context_candidate.get("task_prompt_addendum", "")
            )[:TEXT_CONTEXT_CHARS],
        }
    if entry.tool_candidate:
        addenda = entry.tool_candidate.get("description_addenda")
        return {
            "type": "tool_profile",
            "description_addenda": (
                {
                    str(name): str(text)[:TEXT_CONTEXT_CHARS]
                    for name, text in list(addenda.items())[:8]
                }
                if isinstance(addenda, dict)
                else {}
            ),
        }
    if entry.changed_paths:
        return {
            "type": "source_patch",
            "changed_paths": entry.changed_paths,
            "patch_excerpt": entry.patch_excerpt[:PATCH_CONTEXT_CHARS],
        }
    return {"type": "none"}


def causal_trace(
    entry: EvolutionMemoryEntry,
    diagnosis: Any,
    promotion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = diagnosis.diagnoses[int(entry.plan.get("target_diagnosis", 0))]
    executed = executed_intervention(entry)
    planned_layer = str(entry.plan.get("primary_layer", ""))
    executed_layer = {
        "model_adapter": "model",
        "context_profile": "context",
        "tool_profile": "tools",
        "source_patch": "architecture",
    }.get(str(executed.get("type")))
    alignment = (
        "not_executed"
        if executed_layer is None
        else "layer_aligned"
        if executed_layer == planned_layer
        else "layer_mismatch"
    )
    target_component: Any = entry.plan.get("likely_files", [])
    if planned_layer != "architecture":
        target_component = {
            "model": "model_adapter",
            "context": "context_profile",
            "tools": "tool_profile",
        }.get(planned_layer, planned_layer)
    return {
        "evidence_events": selected.evidence,
        "failure_mechanism": selected.problem,
        "selected_layer": planned_layer,
        "target_component": target_component,
        "planned_intervention": entry.plan.get("intervention"),
        "executed_change": executed,
        "alignment": {
            "status": alignment,
            "scope": "layer_only",
            "executed_layer": executed_layer,
        },
        "observed_effect": {
            "decision": entry.decision,
            "outcome_type": entry.outcome_type,
            "score_delta": entry.outcome.score_delta,
            "promotion": promotion,
            "expected_outcomes": entry.outcome_observations,
        },
    }


def causal_trace_context(trace: dict[str, Any]) -> dict[str, Any]:
    if not trace:
        return {}
    executed = trace.get("executed_change")
    if isinstance(executed, dict):
        executed = dict(executed)
        if "patch_excerpt" in executed:
            executed["patch_excerpt"] = str(executed["patch_excerpt"])[
                :PATCH_CONTEXT_CHARS
            ]
    observed = trace.get("observed_effect")
    if isinstance(observed, dict):
        observed = {
            key: observed.get(key)
            for key in ("decision", "outcome_type", "score_delta", "promotion")
        }
    return {
        "evidence_events": [
            str(item)[:TEXT_CONTEXT_CHARS]
            for item in trace.get("evidence_events", [])[:4]
        ],
        "failure_mechanism": str(trace.get("failure_mechanism", ""))[
            :TEXT_CONTEXT_CHARS
        ],
        "selected_layer": trace.get("selected_layer"),
        "target_component": trace.get("target_component"),
        "planned_intervention": str(trace.get("planned_intervention", ""))[
            :TEXT_CONTEXT_CHARS
        ],
        "executed_change": executed,
        "alignment": trace.get("alignment", {}),
        "observed_effect": observed,
    }


def _limited_strings(value: Any) -> list[str]:
    return [str(item) for item in value[:TASK_CONTEXT_LIMIT]] if isinstance(value, list) else []
