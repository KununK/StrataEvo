"""Describe what each evolution-layer executor can actually produce."""

from __future__ import annotations

from typing import Any


def executor_capabilities(
    model_evolution: bool,
    available_tools: list[str],
) -> dict[str, Any]:
    return {
        "architecture": {
            "operation": "source_patch",
            "scope": "benchmark-active repository files",
        },
        "model": {
            "enabled": model_evolution,
            "operation": (
                "verifier-guided repair collection followed by LoRA SFT"
                if model_evolution
                else "generic source mutation; model-weight evolution is disabled"
            ),
        },
        "context": {
            "operation": "versioned prompt addenda",
            "fields": ["system_prompt_addendum", "task_prompt_addendum"],
        },
        "tools": {
            "operation": "description addenda for existing Agent tools",
            "available_tools": sorted(set(available_tools)),
        },
    }
