from __future__ import annotations

from typing import Any

try:  
    from pydantic import BaseModel, ConfigDict, Field, ValidationError

    _PYDANTIC_AVAILABLE = True

except ImportError:                    
    _PYDANTIC_AVAILABLE = False

__all__ = [
    "validate_llm_response_payload",
    "LlmInterpretationResponse",
    "RecommendationModel",
    "StageExplanationModel",
    "PYDANTIC_AVAILABLE",
]

PYDANTIC_AVAILABLE = _PYDANTIC_AVAILABLE

if _PYDANTIC_AVAILABLE:

    class StageExplanationModel(BaseModel):
        stage_name: str = ""
        predicted: str = ""
        confidence: float = 0.0
        explanation: str = ""
        model_config = ConfigDict(extra="allow")

    class RecommendationModel(BaseModel):
        action_name: str = ""
        parameters: dict[str, Any] = Field(default_factory=dict)
        priority: int | None = None
        rationale: str = ""
        model_config = ConfigDict(extra="allow")

    class LlmInterpretationResponse(BaseModel):
        summary: str | None = None
        explanation: str | None = None
        stage_explanations: list[StageExplanationModel] = Field(default_factory=list)
        symptoms: list[str] = Field(default_factory=list)
        recommendations: list[RecommendationModel] = Field(default_factory=list)
        confidence_notes: list[str] = Field(default_factory=list)
        warnings: list[str] = Field(default_factory=list)
        limitations: list[str] = Field(default_factory=list)
        model_config = ConfigDict(extra="allow")


def validate_llm_response_payload(
    parsed: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not _PYDANTIC_AVAILABLE:
        return parsed, []
    if not isinstance(parsed, dict):
        return ({} if parsed is None else dict(parsed) if hasattr(parsed, "items") else {}), [
            "LLM payload was not a JSON object; schema validation skipped."
        ]
    try:
        LlmInterpretationResponse.model_validate(parsed)
        return parsed, []
    except ValidationError as exc:
        msgs: list[str] = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            kind = err.get("type", "unknown")
            msg = err.get("msg", "")
            msgs.append(f"LLM response schema mismatch at `{loc}` (type={kind}): {msg}")
        n_more = max(0, len(exc.errors()) - 5)
        if n_more:
            msgs.append(f"... and {n_more} more schema mismatch(es).")
        return parsed, msgs
