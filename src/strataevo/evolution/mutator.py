"""One meta-agent run that rewrites the implementation loaded by its next generation."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tinyagent import (
    AgentResult,
    Message,
    Model,
    OpenAICompatibleModel,
    ToolRegistry,
    Usage,
)

from .decision import EvolutionDecision
from .memory import EvolutionMemoryEntry, memory_context
from .runtime.workspace import SelfWorkspace
from .types import EvaluationReport, EvolutionConfig

SYSTEM_PROMPT = """You improve a task-solving Agent by testing the supplied EvolutionDecision.
Inspect its evidence and the relevant code, implement one focused intervention, evaluate it, and
refine that intervention from measured feedback. The evaluator and tests are read-only: never
weaken the task or fabricate results. Stop when no evidence-based improvement remains."""

MUTATOR_CONTEXT_LIMIT_CHARS = 60_000


class RefinementAgent(Protocol):
    max_steps: int

    def run(
        self,
        prompt: str,
        *,
        history: list[Message] | None = None,
    ) -> AgentResult: ...


@dataclass(slots=True)
class _MetaAgent:
    """Fixed tool loop used to repair the evolvable task Agent."""

    model: Model
    tools: ToolRegistry
    system_prompt: str
    max_steps: int
    context_limit_chars: int

    def run(self, prompt: str, *, history: list[Message] | None = None) -> AgentResult:
        messages = list(history or [])
        if not messages or messages[0].role != "system":
            messages.insert(0, Message("system", self.system_prompt))
        messages.append(Message("user", prompt))
        usage = Usage()
        for step in range(1, self.max_steps + 1):
            messages = _compact_meta_messages(messages, self.context_limit_chars)
            response = self.model.complete(messages, self.tools.schemas)
            usage = usage + response.usage
            assistant = response.message
            messages.append(assistant)
            if not assistant.tool_calls:
                return AgentResult(assistant.content, messages, usage, step, "completed")
            for call in assistant.tool_calls:
                item = self.tools.get(call.name)
                if item is None:
                    result = f"Error: unknown tool '{call.name}'"
                else:
                    try:
                        result = item.run(call.arguments)
                    except Exception as error:
                        result = f"Error: {type(error).__name__}: {error}"
                messages.append(Message("tool", result, tool_call_id=call.id, name=call.name))
        return AgentResult("", messages, usage, self.max_steps, "max_steps")


def mutate(
    repo: Path,
    config: EvolutionConfig,
    generation: int,
    parent_report: EvaluationReport,
    decision: EvolutionDecision,
    history: list[EvolutionMemoryEntry],
    commands: list[list[str]],
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
    agent = _MetaAgent(
        model=model,
        tools=ToolRegistry(workspace.tools()),
        system_prompt=SYSTEM_PROMPT,
        max_steps=config.mutator_max_steps,
        context_limit_chars=MUTATOR_CONTEXT_LIMIT_CHARS,
    )
    parent = json.dumps(
        {
            "task_score": parent_report.task_score,
            "evidence_path": parent_report.metrics.get("evidence_path"),
            "output_dir": parent_report.output_dir,
        },
        indent=2,
        ensure_ascii=False,
    )
    evolution_decision = json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)
    prior_evolution = json.dumps(memory_context(history), indent=2, ensure_ascii=False)
    mutable_paths = json.dumps(config.mutable_paths, ensure_ascii=False)
    prompt = f"""Implement generation {generation}.

Parent evaluation:
{parent}

Evolution decision:
{evolution_decision}

Relevant prior evolution outcomes:
{prior_evolution}

Boundaries:
- Modify only {mutable_paths}.
- Make the smallest coherent change that directly tests the decision's hypothesis and intervention.
- Change relevant runtime behavior; comments, types, formatting, or unrelated API edits alone do
  not implement an intervention.
- Inspect the referenced evidence before editing and stay on the selected failure mechanism.
- Prefer the smallest exact repair for a directly observed control-flow contradiction; do not
  invent task-specific artifact behavior. Preserve the observed source indentation.
- Use evaluate_candidate when the candidate is executable, then revise from its feedback.
- The controller retains the best evaluated improvement and otherwise restores the parent.

Candidate benchmark evaluations available: {config.max_eval_attempts}."""
    return _run_refinement_session(agent, prompt, evaluate_candidate, config)


def _run_refinement_session(
    agent: RefinementAgent,
    initial_prompt: str,
    evaluate_candidate: Callable[[], str],
    config: EvolutionConfig,
) -> AgentResult:
    remaining_steps = config.mutator_max_steps
    total_steps = 0
    total_usage = Usage()
    latest: AgentResult | None = None
    prompt = initial_prompt
    history = None

    for round_number in range(1, config.mutator_rounds + 1):
        rounds_left = config.mutator_rounds - round_number + 1
        agent.max_steps = _round_step_budget(remaining_steps, rounds_left)
        latest = agent.run(prompt, history=history)
        history = latest.messages
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
        if latest.output.strip().upper() == "FINALIZE" and data.get("outcome") in {
            "evaluated",
            "no_change",
        }:
            stop_reason = "completed"
            break

        prompt = f"""Evaluation feedback for refinement round {round_number}:
{feedback}

Treat the current files as authoritative. Repair validation errors or revise the same intervention
using this feedback and its evidence. Evaluate again when executable. If no justified revision
remains, do not edit and answer exactly FINALIZE."""
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


def _compact_meta_messages(messages: list[Message], limit: int) -> list[Message]:
    if _messages_size(messages) <= limit:
        return messages
    latest_user = max(
        (index for index, message in enumerate(messages) if message.role == "user"),
        default=0,
    )
    prefix = ([messages[0]] if messages[0].role == "system" else []) + [messages[latest_user]]
    groups: list[list[Message]] = []
    for message in messages[latest_user + 1 :]:
        if message.role == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    while len(groups) > 1 and _messages_size(
        [*prefix, *(message for group in groups for message in group)]
    ) > limit:
        groups.pop(0)
    return [*prefix, *(message for group in groups for message in group)]


def _messages_size(messages: list[Message]) -> int:
    return sum(
        len(json.dumps(message.to_dict(), ensure_ascii=False, default=str))
        for message in messages
    )
