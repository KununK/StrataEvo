"""Compile and execute one MBPP candidate in a timeout-limited subprocess."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


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
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as error:
        return {
            "passed": False,
            "compile_passed": False,
            "status": "syntax_error",
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "test_seconds": 0.0,
        }

    with tempfile.TemporaryDirectory(prefix="strataevo_mbpp_test_") as directory:
        program = Path(directory) / "candidate_check.py"
        program.write_text(build_test_program(task, source), encoding="utf-8")
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(program)],
                cwd=directory,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            return {
                "passed": False,
                "compile_passed": True,
                "status": "timeout",
                "returncode": None,
                "stdout": _text(error.stdout)[-output_chars:],
                "stderr": _text(error.stderr)[-output_chars:],
                "test_seconds": timeout,
            }

    stderr = completed.stderr[-output_chars:]
    if completed.returncode == 0:
        status = "pass"
    elif "AssertionError" in stderr:
        status = "assertion_error"
    else:
        status = "runtime_error"
    return {
        "passed": completed.returncode == 0,
        "compile_passed": True,
        "status": status,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-output_chars:],
        "stderr": stderr,
        "test_seconds": time.perf_counter() - started,
    }


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
