"""Verifier-guided BFCL trajectory repair."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tinyagent import Model

from .run import SYSTEM_PROMPT, run_agent_task

REPAIR_PROMPT = """A previous attempt failed the official BFCL checker.
Failure status: {status}
Verifier feedback: {feedback}
Evolution guidance: {guidance}
Complete the current task normally. Correct the tool names, arguments, ordering, or state reasoning
that caused the failure. Use actual tool results as evidence and do not mention this feedback."""


def evaluate_source(*_args, **_kwargs):
    raise NotImplementedError("BFCL does not support source-based Model repair collection")


def repair_task(
    task: dict[str, Any],
    failure: dict[str, Any],
    attempt: int,
    output_dir: Path,
    model: Model,
    config: Any,
    guidance: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retry one failed interaction; the official checker supplies the training label."""
    feedback = str(failure.get("stderr") or failure.get("stdout") or "No details.")[:4_000]
    prompt = REPAIR_PROMPT.format(
        status=failure.get("status", "unknown"),
        feedback=feedback,
        guidance=guidance or "Correct the observed failure without changing the task contract.",
    )
    generation, result = run_agent_task(
        task,
        model,
        output_dir / "unused",
        max_steps=config.benchmark_max_steps,
        system_prompt=SYSTEM_PROMPT,
        initial_guidance=prompt,
    )
    generation["repair_attempt"] = attempt
    result["repair_attempt"] = attempt
    return generation, result
