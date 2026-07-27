"""Compile and execute one MBPP candidate in a timeout-limited subprocess."""

from __future__ import annotations

from typing import Any

from eval.execution import execute_python


def build_test_program(task: dict[str, Any], source: str) -> str:
    setup = [
        str(task.get("test_imports", "")).strip(),
        str(task.get("test_setup", "")).strip(),
    ]
    tests = [str(test).strip() for test in task["test_list"] if str(test).strip()]
    return (
        "\n".join(block for block in setup if block).rstrip()
        + "\n\n"
        + source.rstrip()
        + "\n\n"
        + "\n".join(tests).rstrip()
        + "\n"
    )


def evaluate_source(
    task: dict[str, Any],
    source: str,
    *,
    timeout: float = 10.0,
    output_chars: int = 4_000,
) -> dict[str, Any]:
    return execute_python(
        source,
        build_test_program(task, source),
        benchmark="mbpp",
        timeout=timeout,
        output_chars=output_chars,
    )
