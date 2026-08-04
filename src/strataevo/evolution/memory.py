"""Minimal append-only memory for completed generations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvolutionMemoryEntry:
    generation: int
    layer: str
    hypothesis: str
    intervention: str
    decision: str
    outcome: str
    reason: str
    parent_score: float
    candidate_score: float | None
    changed_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvolutionMemoryEntry:
        return cls(
            generation=int(data["generation"]),
            layer=str(data["layer"]),
            hypothesis=str(data["hypothesis"]),
            intervention=str(data["intervention"]),
            decision=str(data["decision"]),
            outcome=str(data["outcome"]),
            reason=str(data["reason"]),
            parent_score=float(data["parent_score"]),
            candidate_score=(
                float(data["candidate_score"]) if data.get("candidate_score") is not None else None
            ),
            changed_paths=[str(path) for path in data.get("changed_paths", [])],
        )

    def to_context_dict(self) -> dict[str, Any]:
        return self.to_dict()


class EvolutionMemory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[EvolutionMemoryEntry]:
        if not self.path.is_file():
            return []
        return [
            EvolutionMemoryEntry.from_dict(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def append(self, entry: EvolutionMemoryEntry) -> None:
        if any(item.generation == entry.generation for item in self.load()):
            raise ValueError(f"memory already contains generation {entry.generation}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    def latest(self, limit: int = 5) -> list[EvolutionMemoryEntry]:
        return self.load()[-limit:]

    def relevant(self, layers: set[str], limit: int = 5) -> list[EvolutionMemoryEntry]:
        matching = [entry for entry in self.load() if entry.layer in layers]
        return matching[-limit:]


def memory_context(entries: list[EvolutionMemoryEntry]) -> list[dict[str, Any]]:
    return [entry.to_context_dict() for entry in entries]
