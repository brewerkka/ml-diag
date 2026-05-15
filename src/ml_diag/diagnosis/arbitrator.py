from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ml_diag.interpretation.arbitrator_prompts import (
    JSON_SCHEMA_BLOCK,
    PROMPT_VERSION,
    SYSTEM_PROMPT_ARBITRATOR,
    build_user_prompt,
    render_evidence_digest,
)
from ml_diag.interpretation.llm_interpreter import (
    _cache_load,
    _cache_store,
    _parse_json_block,
)
from ml_diag.labels import PRIMARY_LABELS
from ml_diag.utils.logging import get_logger

if TYPE_CHECKING:
    from ml_diag.evaluation.explanation import StructuredEvidence

_LOG = get_logger(__name__)

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b-instruct",
    "template": "(none)",
    "auto": "(chain)",
}

_DEFAULT_CHAIN: tuple[str, ...] = ("groq", "ollama", "template")


@dataclass(frozen=True)
class ArbitrationDecision:
    chosen_label: str
    chosen_source: Literal["flat", "cascade", "neither"]
    confidence: float
    reasoning: str
    alternative_label: str | None
    backend: str
    cached: bool
    raw_response: str
    label_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chosen_label": str(self.chosen_label),
            "chosen_source": str(self.chosen_source),
            "confidence": float(self.confidence),
            "reasoning": str(self.reasoning),
            "alternative_label": (
                None if self.alternative_label is None else str(self.alternative_label)
            ),
            "backend": str(self.backend),
            "cached": bool(self.cached),
            "raw_response": str(self.raw_response),
            "label_probabilities": {k: float(v) for k, v in self.label_probabilities.items()},
        }


@dataclass(frozen=True)
class ArbitratorConfig:
    backend: Literal["auto", "groq", "ollama", "template"] = "auto"
    cache_path: Path = Path(".cache/arbitrator")
    temperature: float = 0.0
    max_retries: int = 2
    timeout_s: float = 30.0
    model: str | None = None
    fallback_chain: tuple[str, ...] = _DEFAULT_CHAIN
    max_tokens: int = 600
    groq_api_key_env: str = "GROQ_API_KEY"

    def resolved_model(self, backend: str) -> str:
        if self.model:
            return self.model
        return _DEFAULT_MODELS.get(backend, "(unknown)")


_EPHEMERAL_EVIDENCE_FIELDS = ("generated_at",)


def _evidence_hash(evidence: StructuredEvidence) -> str:
    d = dict(evidence.to_dict())
    for k in _EPHEMERAL_EVIDENCE_FIELDS:
        d.pop(k, None)
    blob = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_key_arbitrator(
    *,
    run_id: str,
    flat_label: str,
    flat_proba: dict[str, float],
    cascade_label: str,
    cascade_proba: dict[str, float],
    evidence: StructuredEvidence,
    backend: str,
    model: str,
    inner_fold_index: int | None = None,
) -> str:
    body: dict[str, object] = {
        "prompt_version": PROMPT_VERSION,
        "kind": "arbitrator",
        "backend": backend,
        "model": model,
        "run_id": run_id,
        "flat_label": flat_label,
        "flat_proba": {k: round(float(v), 6) for k, v in sorted(flat_proba.items())},
        "cascade_label": cascade_label,
        "cascade_proba": {k: round(float(v), 6) for k, v in sorted(cascade_proba.items())},
        "evidence_hash": _evidence_hash(evidence),
    }
    if inner_fold_index is not None:
        body["inner_fold_index"] = int(inner_fold_index)
    payload = json.dumps(body, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _soft_label_probabilities(
    chosen_label: str,
    confidence: float,
) -> dict[str, float]:
    conf = max(0.0, min(1.0, float(confidence)))
    n_other = max(1, len(PRIMARY_LABELS) - 1)
    other_share = (1.0 - conf) / n_other
    return {cls: conf if cls == chosen_label else other_share for cls in PRIMARY_LABELS}


def _coerce_decision(
    parsed: dict[str, Any] | None,
    *,
    backend: str,
    raw_text: str,
    cached: bool,
) -> tuple[ArbitrationDecision | None, str | None]:
    if not isinstance(parsed, dict):
        return None, "response was not a JSON object"
    chosen_label = parsed.get("chosen_label")
    if not isinstance(chosen_label, str):
        return None, "chosen_label missing or not a string"
    if chosen_label not in PRIMARY_LABELS:
        return None, f"chosen_label {chosen_label!r} not in PRIMARY_LABELS"
    chosen_source = parsed.get("chosen_source")
    if chosen_source not in ("flat", "cascade", "neither"):
        return None, f"chosen_source {chosen_source!r} not in (flat,cascade,neither)"
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except (TypeError, ValueError):
        return None, "confidence not a float"
    confidence = max(0.0, min(1.0, confidence))
    reasoning = str(parsed.get("reasoning") or "").strip()
    if not reasoning:
        reasoning = "(no reasoning provided)"
    alt = parsed.get("alternative_label")
    if alt is not None and not isinstance(alt, str):
        alt = None
    if isinstance(alt, str) and alt not in PRIMARY_LABELS:
        alt = None
    return (
        ArbitrationDecision(
            chosen_label=chosen_label,
            chosen_source=chosen_source,                          
            confidence=confidence,
            reasoning=reasoning,
            alternative_label=alt,
            backend=backend,
            cached=cached,
            raw_response=raw_text,
            label_probabilities=_soft_label_probabilities(chosen_label, confidence),
        ),
        None,
    )


def _template_decision(
    *,
    flat_label: str,
    cached: bool,
) -> ArbitrationDecision:
    return ArbitrationDecision(
        chosen_label=str(flat_label),
        chosen_source="flat",
        confidence=0.5,
        reasoning="template fallback",
        alternative_label=None,
        backend="template",
        cached=cached,
        raw_response="",
        label_probabilities=_soft_label_probabilities(str(flat_label), 0.5),
    )


def _call_groq(
    *,
    user_prompt: str,
    config: ArbitratorConfig,
) -> tuple[dict[str, Any] | None, str, str | None]:
    try:
        from groq import Groq                                  
    except Exception as e:
        return None, "", f"groq SDK not installed: {e}"
    api_key = os.environ.get(config.groq_api_key_env)
    if not api_key:
        return None, "", f"{config.groq_api_key_env} not set"
    try:
        client = Groq(api_key=api_key, timeout=config.timeout_s, max_retries=0)
        resp = client.chat.completions.create(
            model=config.resolved_model("groq"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ARBITRATOR},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return None, "", f"Groq call failed: {e}"
    try:
        raw_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, "", f"Groq response shape unexpected: {e}"
    parsed = _parse_json_block(raw_text)
    if parsed is None:
        return None, raw_text, "Groq response was not valid JSON"
    return parsed, raw_text, None


_BACKEND_CALLERS = {
    "groq": _call_groq,
}


def _snap_to_allowed_label(
    decision: ArbitrationDecision,
    *,
    flat_label: str,
    cascade_label: str,
) -> ArbitrationDecision:
    if decision.chosen_label in (flat_label, cascade_label):
        expected_source = "flat" if decision.chosen_label == flat_label else "cascade"
        if decision.chosen_source != expected_source:
            decision = replace(
                decision,
                chosen_source=expected_source,                          
            )
        return decision
    src = decision.chosen_source
    if src == "cascade":
        snapped_label = cascade_label
        snapped_source = "cascade"
    else:
        snapped_label = flat_label
        snapped_source = "flat"
    snapped_reasoning = (
        f"[snapped to {snapped_source}: LLM picked '{decision.chosen_label}' "
        f"which is not in {{{flat_label}, {cascade_label}}}] " + decision.reasoning
    )
    snapped_conf = min(decision.confidence, 0.5)
    return replace(
        decision,
        chosen_label=snapped_label,
        chosen_source=snapped_source,                          
        confidence=snapped_conf,
        reasoning=snapped_reasoning,
        label_probabilities=_soft_label_probabilities(snapped_label, snapped_conf),
    )


def _try_backend(
    *,
    backend: str,
    user_prompt: str,
    config: ArbitratorConfig,
) -> tuple[ArbitrationDecision | None, str | None]:
    if backend == "template":
        return None, "template handled separately"
    caller = _BACKEND_CALLERS.get(backend)
    if caller is None:
        return None, f"unknown backend {backend!r}"
    last_err: str | None = None
    augmented_prompt = user_prompt
    for attempt in range(max(1, int(config.max_retries) + 1)):
        if attempt > 0:
            augmented_prompt = (
                user_prompt + "\n\nВАЖНО: предыдущий ответ не прошёл валидацию. "
                "Верни СТРОГО JSON по этой схеме, без пояснений вокруг:\n" + JSON_SCHEMA_BLOCK
            )
        parsed, raw_text, err = caller(
            user_prompt=augmented_prompt,
            config=config,
        )
        if err is None and parsed is not None:
            decision, validation_err = _coerce_decision(
                parsed,
                backend=backend,
                raw_text=raw_text,
                cached=False,
            )
            if decision is not None:
                return decision, None
            last_err = f"validation: {validation_err}"
        else:
            last_err = err or "unknown backend error"
        if attempt < int(config.max_retries):
            time.sleep(0.5)
    return None, last_err


def arbitrate_one(
    *,
    run_id: str,
    flat_label: str,
    flat_proba: dict[str, float],
    cascade_label: str,
    cascade_proba: dict[str, float],
    evidence: StructuredEvidence,
    config: ArbitratorConfig,
    inner_fold_index: int | None = None,
) -> ArbitrationDecision:
    if config.backend == "auto":
        chain = list(config.fallback_chain)
    else:
        chain = [config.backend]
    user_prompt = build_user_prompt(
        run_id=run_id,
        evidence_md=render_evidence_digest(evidence),
        flat_label=flat_label,
        flat_proba=flat_proba,
        cascade_label=cascade_label,
        cascade_proba=cascade_proba,
    )
    cache_dir = str(config.cache_path) if config.cache_path else None
    last_err: str | None = None
    for backend in chain:
        if backend == "template":
            tmpl_key = _cache_key_arbitrator(
                run_id=run_id,
                flat_label=flat_label,
                flat_proba=flat_proba,
                cascade_label=cascade_label,
                cascade_proba=cascade_proba,
                evidence=evidence,
                backend="template",
                model=config.resolved_model("template"),
                inner_fold_index=inner_fold_index,
            )
            cached = _cache_load(cache_dir, tmpl_key)
            if cached is not None:
                cached_label = str(cached.get("chosen_label") or flat_label)
                cached_conf = float(cached.get("confidence", 0.5))
                cached_probs = cached.get("label_probabilities") or {}
                if not cached_probs:
                    cached_probs = _soft_label_probabilities(cached_label, cached_conf)
                return ArbitrationDecision(
                    chosen_label=cached_label,
                    chosen_source=str(cached.get("chosen_source") or "flat"),                          
                    confidence=cached_conf,
                    reasoning=str(cached.get("reasoning") or "template fallback"),
                    alternative_label=cached.get("alternative_label"),
                    backend="template",
                    cached=True,
                    raw_response="",
                    label_probabilities={k: float(v) for k, v in cached_probs.items()},
                )
            decision = _template_decision(flat_label=flat_label, cached=False)
            _cache_store(cache_dir, tmpl_key, decision.to_dict())
            return decision
        model = config.resolved_model(backend)
        key = _cache_key_arbitrator(
            run_id=run_id,
            flat_label=flat_label,
            flat_proba=flat_proba,
            cascade_label=cascade_label,
            cascade_proba=cascade_proba,
            evidence=evidence,
            backend=backend,
            model=model,
            inner_fold_index=inner_fold_index,
        )
        cached = _cache_load(cache_dir, key)
        if cached is not None:
            cached_label = str(cached["chosen_label"])
            cached_conf = float(cached["confidence"])
            cached_probs = cached.get("label_probabilities") or {}
            if not cached_probs:
                cached_probs = _soft_label_probabilities(cached_label, cached_conf)
            return ArbitrationDecision(
                chosen_label=cached_label,
                chosen_source=str(cached["chosen_source"]),                          
                confidence=cached_conf,
                reasoning=str(cached.get("reasoning") or ""),
                alternative_label=cached.get("alternative_label"),
                backend=backend,
                cached=True,
                raw_response=str(cached.get("raw_response") or ""),
                label_probabilities={k: float(v) for k, v in cached_probs.items()},
            )
        decision, err = _try_backend(
            backend=backend,
            user_prompt=user_prompt,
            config=config,
        )
        if decision is not None:
            decision = _snap_to_allowed_label(
                decision,
                flat_label=flat_label,
                cascade_label=cascade_label,
            )
            _cache_store(cache_dir, key, decision.to_dict())
            return decision
        last_err = err
        _LOG.warning("Arbitrator backend %s failed (run=%s): %s", backend, run_id, err)
    _LOG.warning(
        "Arbitrator chain exhausted (run=%s); falling back to template. last_err=%s",
        run_id,
        last_err,
    )
    decision = _template_decision(flat_label=flat_label, cached=False)
    return decision


__all__ = [
    "ArbitrationDecision",
    "ArbitratorConfig",
    "arbitrate_one",
    "PROMPT_VERSION",
]
