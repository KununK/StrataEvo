"""One meta-agent run that rewrites the implementation loaded by its next generation."""

from __future__ import annotations

import json
from pathlib import Path

from tinyagent import (
    Agent,
    AgentResult,
    OpenAICompatibleModel,
    SessionStore,
    ToolRegistry,
    allow_all,
)

from .types import EvaluationReport, EvolutionConfig
from .workspace import SelfWorkspace

SYSTEM_PROMPT = """You are StrataEvo, a recursively self-improving software agent.
You are editing the source code that will implement your next generation. Inspect the
repository and evaluation evidence, identify one concrete limitation, and make one coherent
improvement. You may change any file exposed as evolvable, including your agent loop, tools,
model integration, and self-evolution logic. The evaluator and tests are an external
environment and are intentionally read-only. Do not optimize by weakening tests or fabricating
results. Keep interfaces compatible, run validation, and stop after producing a focused diff."""


def mutate(
    repo: Path,
    config: EvolutionConfig,
    generation: int,
    parent_report: EvaluationReport,
    generation_dir: Path,
    commands: list[list[str]],
) -> AgentResult:
    print(f"[evolution] generation {generation}: inspecting and rewriting self", flush=True)
    workspace = SelfWorkspace(repo, config.mutable_paths, commands)
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
        approval=allow_all,
        session_store=SessionStore(generation_dir / "sessions"),
    )
    feedback = json.dumps(parent_report.to_dict(), indent=2, ensure_ascii=False)
    prompt = f"""Create generation {generation} by improving your own implementation.

Current evaluation report:
{feedback}

Detailed trajectories and results are stored under:
{parent_report.output_dir}

Read the relevant implementation and evidence before editing. Make a focused change that is
expected to improve task score or reduce steps/tokens without reducing task score. Run the
fixed validation command after editing. Your changes remain on disk for the external
evolution controller to evaluate and either commit or roll back."""
    return agent.run(prompt, session_id=f"generation-{generation}")
