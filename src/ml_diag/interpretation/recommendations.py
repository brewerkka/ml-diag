from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

INTERPRETATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StageExplanation:
    stage_name: str
    predicted: str
    confidence: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "predicted": self.predicted,
            "confidence": float(self.confidence),
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class Recommendation:
    action_name: str
    parameters: dict[str, Any]
    priority: int
    rationale: str
    target_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_name": self.action_name,
            "parameters": dict(self.parameters),
            "priority": int(self.priority),
            "rationale": self.rationale,
            "target_classes": list(self.target_classes),
        }


@dataclass(frozen=True)
class InterpretationResult:
    schema_version: str
    generated_at: str
    backend: str
    run_id: str
    final_class: str
    final_confidence: float
    summary: str
    explanation: str
    stage_explanations: list[StageExplanation]
    symptoms: list[str]
    recommendations: list[Recommendation]
    confidence_notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    patch_summary: dict[str, Any] | None = None
    raw_response: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "backend": self.backend,
            "run_id": self.run_id,
            "final_class": self.final_class,
            "final_confidence": float(self.final_confidence),
            "summary": self.summary,
            "explanation": self.explanation,
            "stage_explanations": [s.to_dict() for s in self.stage_explanations],
            "symptoms": list(self.symptoms),
            "recommendations": [r.to_dict() for r in self.recommendations],
            "confidence_notes": list(self.confidence_notes),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "patch_summary": self.patch_summary,
            "raw_response": self.raw_response,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = [
    "INTERPRETATION_SCHEMA_VERSION",
    "InterpretationResult",
    "Recommendation",
    "StageExplanation",
]
