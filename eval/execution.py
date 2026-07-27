"""Execute generated Python against one benchmark test program."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


def execute_python(
    source: str,
    test_program: str,
    *,
    benchmark: str,
    timeout: float,
    output_chars: int,
) -> dict[str, Any]:
    try:
        compile(source, "<candidate>", "exec")
    except SyntaxError as error:
        return _result(
            "syntax_error",
            f"{type(error).__name__}: {error}",
            compile_passed=False,
        )

    with tempfile.TemporaryDirectory(prefix=f"strataevo_{benchmark}_test_") as directory:
        program = Path(directory) / "candidate_check.py"
        program.write_text(test_program, encoding="utf-8")
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
            result = _result(
                "timeout",
                _text(error.stderr)[-output_chars:],
                compile_passed=True,
            )
            result["stdout"] = _text(error.stdout)[-output_chars:]
            result["test_seconds"] = timeout
            return result

    stderr = completed.stderr[-output_chars:]
    status = (
        "pass"
        if completed.returncode == 0
        else "assertion_error"
        if "AssertionError" in stderr
        else "runtime_error"
    )
    return {
        "passed": completed.returncode == 0,
        "compile_passed": True,
        "status": status,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-output_chars:],
        "stderr": stderr,
        "test_seconds": time.perf_counter() - started,
    }


def _result(status: str, error: str, *, compile_passed: bool) -> dict[str, Any]:
    return {
        "passed": False,
        "compile_passed": compile_passed,
        "status": status,
        "returncode": None,
        "stdout": "",
        "stderr": error,
        "test_seconds": 0.0,
    }


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
