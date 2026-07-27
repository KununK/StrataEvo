"""One meta-agent run that rewrites the implementation loaded by its next generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from tinyagent import (
    Agent,
    AgentResult,
    OpenAICompatibleModel,
    SessionStore,
    ToolRegistry,
    Usage,
    allow_all,
)

from .contract import EvaluationContract
from .diagnosis import DiagnosisReport
from .memory import EvolutionMemoryEntry, memory_context
from .plan import EvolutionPlanReport
from .types import EvaluationReport, EvolutionConfig
from .workspace import SelfWorkspace

SYSTEM_PROMPT = """You are StrataEvo, a self-improving software agent.
Improve the task-solving Agent measured by the active evaluation contract. Inspect the repository
and evaluation evidence, identify one concrete limitation, and make one coherent improvement.
The evaluator and tests are a read-only external environment. Never weaken them or fabricate
results. Keep interfaces compatible, run validation, and stop after producing a focused diff."""

MUTATOR_CONTEXT_LIMIT_CHARS = 60_000


def mutate(
    repo: Path,
    config: EvolutionConfig,
    generation: int,
    parent_report: EvaluationReport,
    diagnosis: DiagnosisReport,
    plan_report: EvolutionPlanReport,
    history: list[EvolutionMemoryEntry],
    generation_dir: Path,
    commands: list[list[str]],
    evaluation_contract: EvaluationContract,
    evaluate_candidate: Callable[[], str],
) -> AgentResult:
    print(f"[evolution] generation {generation}: inspecting and rewriting self", flush=True)
    workspace = SelfWorkspace(
        repo,
        config.mutable_paths,
        commands,
        candidate_evaluator=evaluate_candidate,
    )
    model = OpenAICompatibleModel(
        model=config.model,
        base_url=config.base_url,
        temperature=0.0,
        timeout=300.0,
    )
    agent = Agent(
        model=model,
        tools=ToolRegistry(workspace.tools()),
        system_prompt=SYSTEM_PROMPT,
        max_steps=config.mutator_max_steps,
        context_limit_chars=MUTATOR_CONTEXT_LIMIT_CHARS,
        approval=allow_all,
        session_store=SessionStore(generation_dir / "sessions"),
    )
    feedback = json.dumps(parent_report.to_dict(), indent=2, ensure_ascii=False)
    selected_diagnosis = diagnosis.diagnoses[plan_report.plan.target_diagnosis]
    diagnosed_problem = json.dumps(selected_diagnosis.to_dict(), indent=2, ensure_ascii=False)
    evolution_plan = json.dumps(plan_report.plan.to_dict(), indent=2, ensure_ascii=False)
    prior_evolution = json.dumps(memory_context(history), indent=2, ensure_ascii=False)
    mutable_paths = json.dumps(config.mutable_paths, ensure_ascii=False)
    contract = json.dumps(evaluation_contract.to_dict(), indent=2, ensure_ascii=False)
    prompt = f"""Create generation {generation} by improving your own implementation.

Current evaluation report:
{feedback}

Detailed trajectories and results are stored under:
{parent_report.output_dir}

Selected diagnosis:
{diagnosed_problem}

Evolution plan:
{evolution_plan}

Active evaluation contract:
{contract}

Relevant prior evolution outcomes:
{prior_evolution}

Execution constraints:
- Writable paths: {mutable_paths}
- Any version-controlled project file may be changed. Keep each candidate focused enough that its
  effect can be understood from the diff and evaluation result.
- Total model/tool steps available: {config.mutator_max_steps}
- Refinement rounds available: {config.mutator_rounds}
- Candidate benchmark evaluations available: {config.max_eval_attempts}
- Start the source edit within the first third of the budget.
- Work on one candidate until the current round ends. The controller will then validate and
  evaluate the current diff and return the result in this same session. You may also call
  evaluate_candidate yourself when ready; repeated evaluation of an unchanged patch is cached.
- Reserve enough steps for show_diff, evaluation feedback, and repairs.
- Prefer the smallest coherent change. Do not add a new subsystem when an existing prompt, tool,
  schema, or control-flow check can address the evidence.
- replace_text requires an exact match. After one mismatch, read the relevant lines and use
  replace_lines instead of repeatedly guessing whitespace.

Read the relevant implementation and evidence before editing. Execute one focused intervention
consistent with the plan. Treat the hypothesis as testable, verify its evidence against the
referenced trajectories, and use prior outcomes to avoid repeating an unchanged rejected approach.
likely_files are guidance rather than a permission boundary. Do not silently switch to another
diagnosis or bundle unrelated improvements. evaluate_candidate runs the fixed validation commands
before the benchmark. Your changes remain on disk while you refine them; the external controller
will restore the best evaluated candidate and either commit or roll it back."""
    return _run_refinement_session(agent, prompt, evaluate_candidate, config)


def _run_refinement_session(
    agent: Agent,
    initial_prompt: str,
    evaluate_candidate: Callable[[], str],
    config: EvolutionConfig,
) -> AgentResult:
    session_id = "generation-refinement"
    remaining_steps = config.mutator_max_steps
    total_steps = 0
    total_usage = Usage()
    latest: AgentResult | None = None
    prompt = initial_prompt

    for round_number in range(1, config.mutator_rounds + 1):
        rounds_left = config.mutator_rounds - round_number + 1
        agent.max_steps = _round_step_budget(remaining_steps, rounds_left)
        latest = agent.run(prompt, session_id=session_id)
        total_steps += latest.steps
        total_usage = total_usage + latest.usage
        remaining_steps -= latest.steps

        feedback = evaluate_candidate()
        data = _feedback_object(feedback)
        if data.get("candidate_task_score") == 1.0:
            stop_reason = "completed"
            break
        if data.get("evaluations_remaining") == 0 or remaining_steps <= 0:
            stop_reason = "max_steps" if remaining_steps <= 0 else "evaluation_limit"
            break
        if latest.output.strip().upper() == "FINALIZE" and data.get("outcome_type") in {
            "evaluated",
            "no_change",
        }:
            stop_reason = "completed"
            break

        prompt = f"""Candidate evaluation feedback for refinement round {round_number}:
{feedback}

Continue from the current working tree. Fix validation errors before changing direction. If the
candidate was benchmarked, inspect its evidence before deciding the next edit. Make a coherent
revision that responds to this feedback. If no further justified improvement remains, do not edit
and answer exactly FINALIZE. Treat working_tree_state as authoritative: when candidate_retained is
false, the submitted patch has been discarded and its validation result does not describe the
currently active code."""
    else:
        stop_reason = "round_limit"

    if latest is None:
        raise RuntimeError("refinement session produced no agent result")
    return AgentResult(
        latest.output,
        latest.messages,
        total_usage,
        total_steps,
        stop_reason,
    )


def _round_step_budget(remaining_steps: int, rounds_left: int) -> int:
    if remaining_steps <= 0 or rounds_left <= 0:
        raise ValueError("remaining_steps and rounds_left must be positive")
    return max(1, (remaining_steps + rounds_left - 1) // rounds_left)


def _feedback_object(feedback: str) -> dict:
    try:
        data = json.loads(feedback)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
