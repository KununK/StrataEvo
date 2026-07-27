"""Compile and execute one HumanEval candidate in a timeout-limited subprocess."""

from __future__ import annotations

from typing import Any

from eval.execution import execute_python


def build_test_program(task: dict[str, Any], source: str) -> str:
    return (
        source.rstrip()
        + "\n\n"
        + task["test"].rstrip()
        + "\n\n"
        + f"check({task['entry_point']})\n"
    )


def evaluate_source(
    task: dict[str, Any],
    source: str,
    *,
    timeout: float = 10.0,
    output_chars: int = 4_000,
) -> dict[str, Any]:
    """Run HumanEval's hidden check function against a candidate source file."""
    return execute_python(
        source,
        build_test_program(task, source),
        benchmark="humaneval",
        timeout=timeout,
        output_chars=output_chars,
    )
