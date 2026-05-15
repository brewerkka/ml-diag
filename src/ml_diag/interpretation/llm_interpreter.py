from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

from ml_diag.actions import (
    Action,
    ActionParameterError,
    list_actions,
    recommend_actions,
    validate_parameters,
)
from ml_diag.evaluation.explanation import StructuredEvidence
from ml_diag.interpretation.prompts import (
    build_user_prompt,
    prompt_version,
    system_prompt,
)
from ml_diag.interpretation.recommendations import (
    INTERPRETATION_SCHEMA_VERSION,
    InterpretationResult,
    Recommendation,
    StageExplanation,
    _now,
)
from ml_diag.interpretation.response_schemas import (
    validate_llm_response_payload,
)
from ml_diag.labels import HEALTHY, to_stage1
from ml_diag.models.inference import HierarchicalDiagnosis
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

PROMPT_VERSION = prompt_version("ru")

_DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "qwen2.5:7b-instruct",
    "template": "(none)",
    "auto": "(chain)",
}

DEFAULT_FALLBACK_CHAIN: tuple[str, ...] = ("groq", "ollama", "template")


@dataclass(frozen=True)
class InterpretationConfig:
    backend: str = "auto"
    model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1500
    max_recommendations: int = 3
    language: str = "ru"
    groq_api_key_env: str = "GROQ_API_KEY"
    ollama_url: str = "http://localhost:11434"
    ollama_timeout_sec: float = 60.0
    fallback_chain: tuple[str, ...] = DEFAULT_FALLBACK_CHAIN
    cache_dir: str | None = "results/llm_cache"
    api_key_env: str = ""

    def resolved_model(self) -> str:
        if self.model:
            return self.model
        return _DEFAULT_MODELS.get(self.backend, "(unknown)")


_TT: dict[str, dict[str, str]] = {
    "ru": {
        "stage_predicted": "Stage `{name}` предсказал `{pred}` с confidence {conf:.3f}.",
        "stage_low_conf": (
            "Решение на этом этапе само по себе неуверенное — соседние "
            "классы получили существенную вероятность."
        ),
        "stage1_healthy_end": "Каскад завершается на этом шаге без атрибуции fault'а.",
        "stage1_faulty_continue": (
            "Faulty — каскад продолжается к разделению data / optimisation/generalisation."
        ),
        "stage2_role": (
            "Stage 2 определяет, кроется ли ошибка в data pipeline или в "
            "регулятивно-оптимизационном режиме."
        ),
        "stage3_role": "Stage 3 уточняет конкретный режим ошибки внутри выбранной ветки.",
        "secondary_signal_prefix": "[вторичный сигнал, не доминировал в диагнозе] ",
        "low_composed_conf": (
            "Итоговая composed confidence низкая ({conf:.3f}); "
            "относитесь к диагнозу как к предварительному."
        ),
        "stage1_low_conf": "Stage 1 (healthy vs faulty) сам по себе неуверенный.",
        "stage2_low_conf": (
            "Решение Stage 2 (data vs opt/gen) слабое — обе ветки остаются правдоподобными."
        ),
        "strongest_alt": "Сильнейший альтернативный класс: `{cls}` (p={p:.3f}).",
        "lim_demonstrative": "Рекомендации носят демонстрационный и экспериментальный характер.",
        "lim_verify_config": (
            "Каждую рекомендацию следует сверять с исходной training-конфигурацией "
            "перед применением."
        ),
        "rat_reduce_lr": (
            "Замечена нестабильность оптимизации или резкие скачки лосса; "
            "снижение learning rate — каноничный первый шаг для режима `{cls}`."
        ),
        "rat_increase_capacity": (
            "Train loss не достиг достаточно низкого значения — это "
            "согласуется с underfitting; увеличение ёмкости модели — "
            "каноничный первый шаг."
        ),
        "rat_add_regularization": (
            "Виден большой разрыв между train и val; weight decay "
            "(и опционально dropout) уменьшает этот разрыв без сокращения "
            "ёмкости модели."
        ),
        "rat_early_stop": (
            "Validation-кривая ухудшается после своего минимума; остановка "
            "на лучшей val-эпохе — самая дешёвая мера противодействия."
        ),
        "rat_clean_label_noise": (
            "Свидетельства согласуются с повышенным уровнем шума в метках; "
            "уменьшение доли шумных меток в обучающей выборке — каноничный "
            "первый шаг."
        ),
        "rat_fix_split": (
            "Свидетельства согласуются с утечкой между train и validation; "
            "пересборка split’а с дедупликацией и новым seed устраняет "
            "корневую причину со стороны данных."
        ),
        "rat_retrain_with_seed": (
            "Повторное обучение с другим seed позволяет измерить, какая "
            "часть наблюдаемого поведения воспроизводима, а какая — "
            "вариативна по seed."
        ),
        "rat_observe_only": (
            "Корректирующие действия не указаны; рекомендуется собрать "
            "больше runs для уточнения картины."
        ),
        "summary_healthy_with_secondary": (
            "Run диагностирован как `healthy` (composed P = {conf:.3f}) "
            "несмотря на отдельные слабые / противоречивые fault-like "
            "сигналы (см. секцию вторичных сигналов). Корректирующие "
            "действия не требуются."
        ),
        "summary_healthy_clean": (
            "Run диагностирован как `healthy` (composed P = {conf:.3f}); "
            "корректирующие действия не требуются."
        ),
        "summary_faulty": (
            "Run диагностирован как `{cls}` (composed P = {conf:.3f}). "
            "Рекомендуются корректирующие действия, нацеленные на этот класс."
        ),
        "expl_cascade_class": (
            "Иерархический каскад относит этот run к классу `{cls}` (P = {conf:.3f}). "
        ),
        "expl_stage1": "Stage 1 выделил `{pred}` с P = {conf:.3f}. ",
        "expl_stage2": "Stage 2 направил run в ветку `{pred}` (P = {conf:.3f}). ",
        "expl_stage3": "Stage 3 присвоил листовой класс `{pred}` (P = {conf:.3f}). ",
        "expl_decisive_signals": "Подтверждающие сигналы: ",
        "expl_secondary_healthy": (
            "Вторичные / противоречивые сигналы (не доминировали в диагнозе): "
        ),
        "expl_secondary_faulty": "Альтернативные сигналы (отвергнуты каскадом): ",
        "expl_integrity_avail": "Доступны integrity-признаки: ",
        "expl_alt_hypotheses": "Альтернативные гипотезы: ",
    },
    "en": {
        "stage_predicted": "Stage `{name}` predicted `{pred}` with confidence {conf:.3f}.",
        "stage_low_conf": (
            "The decision at this stage is itself uncertain — neighbouring "
            "classes received substantial probability."
        ),
        "stage1_healthy_end": "The cascade terminates at this step with no fault attribution.",
        "stage1_faulty_continue": (
            "Faulty — the cascade continues to the data / optimisation-or-generalisation split."
        ),
        "stage2_role": (
            "Stage 2 decides whether the failure lies in the data pipeline or "
            "in the regularisation/optimisation regime."
        ),
        "stage3_role": "Stage 3 refines the specific failure mode within the chosen branch.",
        "secondary_signal_prefix": "[secondary signal, did not drive the diagnosis] ",
        "low_composed_conf": (
            "Final composed confidence is low ({conf:.3f}); treat the diagnosis as preliminary."
        ),
        "stage1_low_conf": "Stage 1 (healthy vs faulty) is itself uncertain.",
        "stage2_low_conf": (
            "Stage 2 (data vs opt/gen) decision is weak — both branches remain plausible."
        ),
        "strongest_alt": "Strongest alternative class: `{cls}` (p={p:.3f}).",
        "lim_demonstrative": "Recommendations are demonstrative and experimental in nature.",
        "lim_verify_config": (
            "Each recommendation should be verified against the original "
            "training configuration before being applied."
        ),
        "rat_reduce_lr": (
            "Optimisation instability or loss spikes are present; lowering the "
            "learning rate is the canonical first step for the `{cls}` regime."
        ),
        "rat_increase_capacity": (
            "Training loss did not reach a sufficiently low value, consistent "
            "with underfitting; increasing model capacity is the canonical "
            "first step."
        ),
        "rat_add_regularization": (
            "There is a large train-vs-val gap; weight decay (and optionally "
            "dropout) reduces this gap without shrinking the model."
        ),
        "rat_early_stop": (
            "The validation curve worsens after its minimum; stopping at the "
            "best validation epoch is the cheapest mitigation."
        ),
        "rat_clean_label_noise": (
            "Evidence is consistent with elevated label noise; reducing the "
            "share of noisy labels in the training set is the canonical first "
            "step."
        ),
        "rat_fix_split": (
            "Evidence is consistent with leakage between train and validation; "
            "rebuilding the split with deduplication and a new seed removes "
            "the data-side root cause."
        ),
        "rat_retrain_with_seed": (
            "Retraining with a different seed measures how much of the "
            "observed behaviour is reproducible vs seed-variant."
        ),
        "rat_observe_only": (
            "No corrective actions apply; collecting more runs is suggested to clarify the picture."
        ),
        "summary_healthy_with_secondary": (
            "The run is diagnosed as `healthy` (composed P = {conf:.3f}) "
            "despite isolated weak or contradictory fault-like signals "
            "(see the secondary signals section). No corrective actions "
            "are required."
        ),
        "summary_healthy_clean": (
            "The run is diagnosed as `healthy` (composed P = {conf:.3f}); "
            "no corrective actions are required."
        ),
        "summary_faulty": (
            "The run is diagnosed as `{cls}` (composed P = {conf:.3f}). "
            "Corrective actions targeted at this class are recommended."
        ),
        "expl_cascade_class": (
            "The hierarchical cascade assigns this run to class `{cls}` (P = {conf:.3f}). "
        ),
        "expl_stage1": "Stage 1 selected `{pred}` with P = {conf:.3f}. ",
        "expl_stage2": "Stage 2 routed the run into branch `{pred}` (P = {conf:.3f}). ",
        "expl_stage3": "Stage 3 assigned leaf class `{pred}` (P = {conf:.3f}). ",
        "expl_decisive_signals": "Supporting signals: ",
        "expl_secondary_healthy": (
            "Secondary / contradictory signals (did not drive the diagnosis): "
        ),
        "expl_secondary_faulty": "Alternative signals (rejected by the cascade): ",
        "expl_integrity_avail": "Integrity features available: ",
        "expl_alt_hypotheses": "Alternative hypotheses: ",
    },
}


def _norm_lang(lang: str | None) -> str:
    if not lang:
        return "ru"
    s = str(lang).lower().strip()
    return s if s in _TT else "ru"


def _t(lang: str | None, key: str, **fmt) -> str:
    table = _TT[_norm_lang(lang)]
    s = table.get(key) or _TT["ru"].get(key) or key
    return s.format(**fmt) if fmt else s


def _stage_explanation_template(
    stage_name: str,
    predicted: str,
    confidence: float,
    evidence: StructuredEvidence,
    *,
    language: str = "ru",
) -> str:
    bits: list[str] = []
    bits.append(_t(language, "stage_predicted", name=stage_name, pred=predicted, conf=confidence))
    if confidence < 0.6:
        bits.append(_t(language, "stage_low_conf"))
    if "stage1" in stage_name:
        if predicted == HEALTHY:
            bits.append(_t(language, "stage1_healthy_end"))
        else:
            bits.append(_t(language, "stage1_faulty_continue"))
    elif "stage2" in stage_name:
        bits.append(_t(language, "stage2_role"))
    elif "stage3" in stage_name:
        bits.append(_t(language, "stage3_role"))
    return " ".join(bits)


def _symptom_lines_from_evidence(
    evidence: StructuredEvidence,
    diagnosis_class: str,
    *,
    language: str = "ru",
) -> list[str]:
    from ml_diag.evaluation.explanation import classify_evidence_notes

    raw = (
        list(evidence.curve_evidence.notes)
        + list(evidence.integrity_evidence.notes)
        + list(evidence.diagnostic_notes)
    )
    decisive, secondary = classify_evidence_notes(raw, diagnosis_class)
    prefix = _t(language, "secondary_signal_prefix")
    out: list[str] = []
    seen: set[str] = set()
    for s in decisive:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for s in secondary:
        if s and s not in seen:
            seen.add(s)
            out.append(f"{prefix}{s}")
    return out


def _confidence_notes(
    diagnosis: HierarchicalDiagnosis,
    *,
    language: str = "ru",
) -> list[str]:
    notes: list[str] = []
    if diagnosis.final_confidence < 0.5:
        notes.append(_t(language, "low_composed_conf", conf=diagnosis.final_confidence))
    if diagnosis.stage1.confidence < 0.6:
        notes.append(_t(language, "stage1_low_conf"))
    if diagnosis.stage2 is not None and diagnosis.stage2.confidence < 0.6:
        notes.append(_t(language, "stage2_low_conf"))
    if diagnosis.alternative_hypotheses:
        cls, p = diagnosis.alternative_hypotheses[0]
        if p > 0.25:
            notes.append(_t(language, "strongest_alt", cls=cls, p=p))
    return notes


def _standard_limitations(*, language: str = "ru") -> list[str]:
    return [
        _t(language, "lim_demonstrative"),
        _t(language, "lim_verify_config"),
    ]


def _build_recommendations_template(
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    *,
    max_recommendations: int,
    language: str = "ru",
) -> list[Recommendation]:
    actions = recommend_actions(
        diagnosis.final_class,
        evidence.to_dict(),
        max_recommendations=max_recommendations,
    )
    out: list[Recommendation] = []
    for i, action in enumerate(actions, start=1):
        try:
            params = validate_parameters(action, {})
        except ActionParameterError as e:
            _LOG.warning("Skipping action %s with invalid defaults: %s", action.name, e)
            continue
        out.append(
            Recommendation(
                action_name=action.name,
                parameters=params,
                priority=i,
                rationale=_template_rationale(action, diagnosis, language=language),
                target_classes=list(action.target_classes),
            )
        )
    return out


def _template_rationale(
    action: Action,
    diagnosis: HierarchicalDiagnosis,
    *,
    language: str = "ru",
) -> str:
    cls = diagnosis.final_class
    rationale_keys = {
        "reduce_lr": ("rat_reduce_lr", {"cls": cls}),
        "increase_capacity": ("rat_increase_capacity", {}),
        "add_regularization": ("rat_add_regularization", {}),
        "early_stop": ("rat_early_stop", {}),
        "clean_label_noise": ("rat_clean_label_noise", {}),
        "fix_split": ("rat_fix_split", {}),
        "retrain_with_seed": ("rat_retrain_with_seed", {}),
        "observe_only": ("rat_observe_only", {}),
    }
    if action.name in rationale_keys:
        key, fmt = rationale_keys[action.name]
        return _t(language, key, **fmt)
    return action.description


def _build_explanation_template(
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    *,
    language: str = "ru",
) -> tuple[str, str]:
    from ml_diag.evaluation.explanation import classify_evidence_notes

    cls = diagnosis.final_class
    conf = diagnosis.final_confidence
    all_notes = (
        list(evidence.curve_evidence.notes)
        + list(evidence.integrity_evidence.notes)
        + list(evidence.diagnostic_notes)
    )
    decisive_notes, secondary_notes = classify_evidence_notes(all_notes, cls)
    if to_stage1(cls) == HEALTHY:
        if secondary_notes:
            summary = _t(language, "summary_healthy_with_secondary", conf=conf)
        else:
            summary = _t(language, "summary_healthy_clean", conf=conf)
    else:
        summary = _t(language, "summary_faulty", cls=cls, conf=conf)
    parts: list[str] = []
    parts.append(_t(language, "expl_cascade_class", cls=cls, conf=conf))
    s1 = diagnosis.stage1
    parts.append(_t(language, "expl_stage1", pred=s1.predicted, conf=s1.confidence))
    if diagnosis.stage2 is not None:
        s2 = diagnosis.stage2
        parts.append(_t(language, "expl_stage2", pred=s2.predicted, conf=s2.confidence))
    if diagnosis.stage3 is not None:
        s3 = diagnosis.stage3
        parts.append(_t(language, "expl_stage3", pred=s3.predicted, conf=s3.confidence))
    if decisive_notes:
        parts.append(_t(language, "expl_decisive_signals") + " ".join(decisive_notes) + " ")
    if secondary_notes:
        if to_stage1(cls) == HEALTHY:
            parts.append(_t(language, "expl_secondary_healthy") + " ".join(secondary_notes) + " ")
        else:
            parts.append(_t(language, "expl_secondary_faulty") + " ".join(secondary_notes) + " ")
    if evidence.integrity_evidence.columns:
        cols_present = [k for k, v in evidence.integrity_evidence.columns.items() if v is not None]
        if cols_present:
            parts.append(
                _t(language, "expl_integrity_avail")
                + ", ".join(cols_present[:6])
                + ("" if len(cols_present) <= 6 else ", …")
                + ". "
            )
    if diagnosis.alternative_hypotheses:
        alt_str = ", ".join(f"`{c}`={p:.3f}" for c, p in diagnosis.alternative_hypotheses)
        parts.append(_t(language, "expl_alt_hypotheses") + alt_str + ".")
    return summary, "".join(parts)


def _interpret_template(
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    *,
    config: InterpretationConfig,
    patch_summary: dict[str, Any] | None = None,
) -> InterpretationResult:
    lang = _norm_lang(config.language)
    summary, explanation = _build_explanation_template(diagnosis, evidence, language=lang)
    stage_explanations: list[StageExplanation] = []
    for s in (diagnosis.stage1, diagnosis.stage2, diagnosis.stage3):
        if s is None:
            continue
        stage_explanations.append(
            StageExplanation(
                stage_name=s.stage_name,
                predicted=s.predicted,
                confidence=s.confidence,
                explanation=_stage_explanation_template(
                    s.stage_name,
                    s.predicted,
                    s.confidence,
                    evidence,
                    language=lang,
                ),
            )
        )
    recommendations = _build_recommendations_template(
        diagnosis,
        evidence,
        max_recommendations=config.max_recommendations,
        language=lang,
    )
    return InterpretationResult(
        schema_version=INTERPRETATION_SCHEMA_VERSION,
        generated_at=_now(),
        backend="template",
        run_id=diagnosis.run_id,
        final_class=diagnosis.final_class,
        final_confidence=float(diagnosis.final_confidence),
        summary=summary,
        explanation=explanation,
        stage_explanations=stage_explanations,
        symptoms=_symptom_lines_from_evidence(evidence, diagnosis.final_class, language=lang),
        recommendations=recommendations,
        confidence_notes=_confidence_notes(diagnosis, language=lang),
        warnings=[],
        limitations=_standard_limitations(language=lang),
        patch_summary=patch_summary,
        raw_response=None,
    )


def backend_status(config: InterpretationConfig | None = None) -> dict[str, dict[str, Any]]:
    config = config or InterpretationConfig()
    out: dict[str, dict[str, Any]] = {}
    out["template"] = {"available": True, "reason": "always"}
    try:
        import groq              

        if os.environ.get(config.groq_api_key_env):
            out["groq"] = {"available": True, "reason": "ok"}
        else:
            out["groq"] = {"available": False, "reason": f"{config.groq_api_key_env} not set"}
    except Exception:
        out["groq"] = {"available": False, "reason": "groq SDK not installed"}
    try:
        import requests

        try:
            r = requests.get(f"{config.ollama_url}/api/tags", timeout=2.0)
            if r.status_code == 200:
                out["ollama"] = {"available": True, "reason": "ok"}
            else:
                out["ollama"] = {"available": False, "reason": f"HTTP {r.status_code}"}
        except Exception:
            out["ollama"] = {"available": False, "reason": "daemon not reachable"}
    except Exception:
        out["ollama"] = {"available": False, "reason": "requests not installed"}
    return out


def _cache_key(
    *,
    backend: str,
    model: str,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    patch_summary: dict[str, Any] | None,
    max_recs: int,
    language: str = "ru",
) -> str:
    import hashlib

    allowlist_signature = json.dumps(
        [{"name": a.name, "target_classes": list(a.target_classes)} for a in list_actions()],
        sort_keys=True,
        ensure_ascii=False,
    )
    payload = json.dumps(
        {
            "prompt_version": prompt_version(language),
            "backend": backend,
            "model": model,
            "diagnosis": diagnosis.to_dict(),
            "evidence": evidence.to_dict(),
            "patch_summary": patch_summary,
            "max_recs": max_recs,
            "allowlist": allowlist_signature,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_load(cache_dir: str | None, key: str) -> dict[str, Any] | None:
    if not cache_dir:
        return None
    p = Path(cache_dir) / f"{key}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_store(cache_dir: str | None, key: str, payload: dict[str, Any]) -> None:
    if not cache_dir:
        return
    try:
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{key}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        _LOG.warning("LLM cache write failed for key=%s", key[:12])


def _build_user_prompt(
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    patch_summary: dict[str, Any] | None,
    max_recommendations: int,
    language: str = "ru",
) -> str:
    return build_user_prompt(
        diagnosis_json=json.dumps(diagnosis.to_dict(), indent=2, ensure_ascii=False),
        evidence_json=json.dumps(evidence.to_dict(), indent=2, ensure_ascii=False),
        patch_summary_json=(
            json.dumps(patch_summary, indent=2, ensure_ascii=False) if patch_summary else "null"
        ),
        max_recommendations=max_recommendations,
        language=language,
    )


def _try_groq_call(
    *,
    config,
    diagnosis,
    evidence,
    patch_summary,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        from groq import Groq                                  
    except Exception as e:
        return None, None, f"groq SDK not installed: {e}"
    api_key = os.environ.get(config.groq_api_key_env)
    if not api_key:
        return None, None, f"{config.groq_api_key_env} is not set"
    lang = _norm_lang(config.language)
    user_prompt = _build_user_prompt(
        diagnosis,
        evidence,
        patch_summary,
        config.max_recommendations,
        language=lang,
    )
    try:
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model=config.resolved_model(),
            messages=[
                {"role": "system", "content": system_prompt(lang)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return None, None, f"Groq API call failed: {e}"
    try:
        raw_text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return None, None, f"Groq response had unexpected shape: {e}"
    parsed = _parse_json_block(raw_text)
    if parsed is None:
        return None, raw_text, "Groq response was not valid JSON"
    return parsed, raw_text, None


def _try_ollama_call(
    *,
    config,
    diagnosis,
    evidence,
    patch_summary,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        import requests                                  
    except Exception as e:
        return None, None, f"requests not installed: {e}"
    lang = _norm_lang(config.language)
    user_prompt = _build_user_prompt(
        diagnosis,
        evidence,
        patch_summary,
        config.max_recommendations,
        language=lang,
    )
    try:
        r = requests.post(
            f"{config.ollama_url}/api/chat",
            json={
                "model": config.resolved_model(),
                "messages": [
                    {"role": "system", "content": system_prompt(lang)},
                    {"role": "user", "content": user_prompt},
                ],
                "format": "json",
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                },
            },
            timeout=config.ollama_timeout_sec,
        )
    except Exception as e:
        return None, None, f"Ollama call failed (daemon unreachable?): {e}"
    if r.status_code != 200:
        return None, None, f"Ollama returned HTTP {r.status_code}: {r.text[:200]}"
    try:
        body = r.json()
        raw_text = body.get("message", {}).get("content", "").strip()
    except Exception as e:
        return None, None, f"Ollama response not valid JSON envelope: {e}"
    parsed = _parse_json_block(raw_text)
    if parsed is None:
        return None, raw_text, "Ollama model output was not valid JSON"
    return parsed, raw_text, None


_BACKEND_CALLERS = {
    "groq": _try_groq_call,
    "ollama": _try_ollama_call,
}


def _parse_json_block(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        first_nl = candidate.find("\n")
        if first_nl != -1:
            candidate = candidate[first_nl + 1 :]
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _coerce_recommendations(
    raw_recs: Any, *, allowlist: Sequence[Action], max_recommendations: int
) -> tuple[list[Recommendation], list[str]]:
    by_name = {a.name: a for a in allowlist}
    valid: list[Recommendation] = []
    warnings: list[str] = []
    if not isinstance(raw_recs, list):
        warnings.append("LLM did not return a recommendations list; using template recs.")
        return valid, warnings
    for i, rec in enumerate(raw_recs):
        if not isinstance(rec, dict):
            warnings.append(f"Recommendation #{i} is not a JSON object; dropped.")
            continue
        name = rec.get("action_name")
        if not isinstance(name, str) or name not in by_name:
            warnings.append(f"Recommendation #{i}: unknown action `{name!r}`; dropped.")
            continue
        action = by_name[name]
        try:
            params = validate_parameters(action, dict(rec.get("parameters") or {}))
        except ActionParameterError as e:
            warnings.append(f"Recommendation #{i} ({name}): invalid parameters — {e}; dropped.")
            continue
        try:
            priority = int(rec.get("priority", i + 1))
        except (TypeError, ValueError):
            priority = i + 1
        rationale = str(rec.get("rationale") or action.description)
        valid.append(
            Recommendation(
                action_name=name,
                parameters=params,
                priority=max(1, priority),
                rationale=rationale,
                target_classes=list(action.target_classes),
            )
        )
        if len(valid) >= max_recommendations:
            break
    return valid, warnings


def _normalize_llm_payload(
    parsed: dict[str, Any],
    *,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    template: InterpretationResult,
    backend_name: str,
    raw_text: str | None,
    patch_summary: dict[str, Any] | None,
    config: InterpretationConfig,
    extra_warnings: list[str] | None = None,
) -> InterpretationResult:
    rec_warnings: list[str] = list(extra_warnings or [])
    parsed, schema_warnings = validate_llm_response_payload(parsed)
    rec_warnings.extend(schema_warnings)
    recs, val_warnings = _coerce_recommendations(
        parsed.get("recommendations"),
        allowlist=list_actions(),
        max_recommendations=config.max_recommendations,
    )
    rec_warnings.extend(val_warnings)
    if not recs:
        recs = template.recommendations
        rec_warnings.append("No valid LLM recommendations; using template recommendations.")
    raw_stages = parsed.get("stage_explanations") or []
    stage_explanations: list[StageExplanation] = []
    if isinstance(raw_stages, list) and raw_stages:
        for entry in raw_stages:
            if not isinstance(entry, dict):
                continue
            stage_explanations.append(
                StageExplanation(
                    stage_name=str(entry.get("stage_name") or ""),
                    predicted=str(entry.get("predicted") or ""),
                    confidence=float(entry.get("confidence") or 0.0),
                    explanation=str(entry.get("explanation") or ""),
                )
            )
    if not stage_explanations:
        stage_explanations = template.stage_explanations
    summary = str(parsed.get("summary") or template.summary)
    explanation = str(parsed.get("explanation") or template.explanation)
    symptoms = parsed.get("symptoms") or template.symptoms
    if not isinstance(symptoms, list):
        symptoms = template.symptoms
    confidence_notes = parsed.get("confidence_notes") or template.confidence_notes
    if not isinstance(confidence_notes, list):
        confidence_notes = template.confidence_notes
    warnings_block = list(parsed.get("warnings") or [])
    warnings_block.extend(rec_warnings)
    limitations_block = parsed.get("limitations") or template.limitations
    if not isinstance(limitations_block, list):
        limitations_block = template.limitations
    _DEPRECATED_LIMITATION_PHRASES = ("Диагностическая система помечает класс ошибки",)
    limitations_block = [
        s
        for s in limitations_block
        if isinstance(s, str) and not any(p in s for p in _DEPRECATED_LIMITATION_PHRASES)
    ]
    return InterpretationResult(
        schema_version=INTERPRETATION_SCHEMA_VERSION,
        generated_at=_now(),
        backend=backend_name,
        run_id=diagnosis.run_id,
        final_class=diagnosis.final_class,
        final_confidence=float(diagnosis.final_confidence),
        summary=summary,
        explanation=explanation,
        stage_explanations=stage_explanations,
        symptoms=[str(s) for s in symptoms],
        recommendations=recs,
        confidence_notes=[str(s) for s in confidence_notes],
        warnings=[str(s) for s in warnings_block],
        limitations=[str(s) for s in limitations_block],
        patch_summary=patch_summary,
        raw_response=raw_text,
    )


def _interpret_via_backend(
    backend: str,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    *,
    config: InterpretationConfig,
    patch_summary: dict[str, Any] | None,
    template: InterpretationResult,
) -> tuple[InterpretationResult | None, str | None]:
    caller = _BACKEND_CALLERS.get(backend)
    if caller is None:
        return None, f"unknown LLM backend: {backend!r}"
    model = config.resolved_model()
    lang = _norm_lang(config.language)
    cache_key = _cache_key(
        backend=backend,
        model=model,
        diagnosis=diagnosis,
        evidence=evidence,
        patch_summary=patch_summary,
        max_recs=config.max_recommendations,
        language=lang,
    )
    cached = _cache_load(config.cache_dir, cache_key)
    parsed: dict[str, Any] | None = None
    raw_text: str | None = None
    if cached is not None and isinstance(cached.get("parsed"), dict):
        parsed = cached["parsed"]
        raw_text = cached.get("raw_response")
        _LOG.info("LLM cache hit (%s/%s, key=%s…)", backend, model, cache_key[:12])
    else:
        parsed, raw_text, error = caller(
            config=config,
            diagnosis=diagnosis,
            evidence=evidence,
            patch_summary=patch_summary,
        )
        if parsed is None:
            return None, error
        _cache_store(
            config.cache_dir,
            cache_key,
            {
                "backend": backend,
                "model": model,
                "prompt_version": prompt_version(lang),
                "raw_response": raw_text,
                "parsed": parsed,
            },
        )
    result = _normalize_llm_payload(
        parsed,
        diagnosis=diagnosis,
        evidence=evidence,
        template=template,
        backend_name=backend,
        raw_text=raw_text,
        patch_summary=patch_summary,
        config=config,
    )
    return result, None


def _wrap_template_with_warnings(
    template: InterpretationResult, extra_warnings: list[str]
) -> InterpretationResult:
    return InterpretationResult(
        schema_version=template.schema_version,
        generated_at=template.generated_at,
        backend="template",
        run_id=template.run_id,
        final_class=template.final_class,
        final_confidence=template.final_confidence,
        summary=template.summary,
        explanation=template.explanation,
        stage_explanations=template.stage_explanations,
        symptoms=template.symptoms,
        recommendations=template.recommendations,
        confidence_notes=template.confidence_notes,
        warnings=list(template.warnings) + list(extra_warnings),
        limitations=template.limitations,
        patch_summary=template.patch_summary,
        raw_response=None,
    )


def interpret(
    *,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    config: InterpretationConfig | None = None,
    patch_summary: dict[str, Any] | None = None,
) -> InterpretationResult:
    config = config or InterpretationConfig()
    template = _interpret_template(diagnosis, evidence, config=config, patch_summary=patch_summary)
    if config.backend == "template":
        return template
    if config.backend == "auto":
        warnings: list[str] = []
        for candidate in config.fallback_chain:
            if candidate == "template":
                return _wrap_template_with_warnings(template, warnings)
            cand_cfg = dc_replace(config, backend=candidate)
            result, error = _interpret_via_backend(
                candidate,
                diagnosis,
                evidence,
                config=cand_cfg,
                patch_summary=patch_summary,
                template=template,
            )
            if result is not None:
                return result
            warnings.append(f"backend `{candidate}` skipped: {error}")
        return _wrap_template_with_warnings(template, warnings)
    if config.backend in _BACKEND_CALLERS:
        result, error = _interpret_via_backend(
            config.backend,
            diagnosis,
            evidence,
            config=config,
            patch_summary=patch_summary,
            template=template,
        )
        if result is not None:
            return result
        return _wrap_template_with_warnings(
            template,
            [f"backend `{config.backend}` unavailable, fell back to template: {error}"],
        )
    raise ValueError(f"Unknown interpretation backend: {config.backend!r}")


def render_markdown(result: InterpretationResult) -> str:
    out: list[str] = []
    out.append(f"# Interpretation — run `{result.run_id}`")
    out.append("")
    out.append(f"- generated: {result.generated_at}")
    out.append(f"- backend:   `{result.backend}`")
    out.append(f"- schema:    `{result.schema_version}`")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append(
        f"**Final class:** `{result.final_class}` (confidence {result.final_confidence:.3f})"
    )
    out.append("")
    out.append(result.summary)
    out.append("")
    out.append("## Explanation")
    out.append("")
    out.append(result.explanation)
    out.append("")
    if result.stage_explanations:
        out.append("## Stage trace")
        out.append("")
        for s in result.stage_explanations:
            out.append(
                f"- **`{s.stage_name}`** → `{s.predicted}` "
                f"(P = {s.confidence:.3f}): {s.explanation}"
            )
        out.append("")
    if result.symptoms:
        out.append("## Symptoms / supporting evidence")
        out.append("")
        for s in result.symptoms:
            out.append(f"- {s}")
        out.append("")
    if result.recommendations:
        out.append("## Recommendations")
        out.append("")
        out.append("| priority | action | parameters | target classes | rationale |")
        out.append("|---:|---|---|---|---|")
        for r in result.recommendations:
            params_s = ", ".join(f"`{k}={v}`" for k, v in r.parameters.items()) or "—"
            tcs = ", ".join(f"`{c}`" for c in r.target_classes) or "any"
            out.append(f"| {r.priority} | `{r.action_name}` | {params_s} | {tcs} | {r.rationale} |")
        out.append("")
    if result.confidence_notes:
        out.append("## Confidence notes")
        out.append("")
        for n in result.confidence_notes:
            out.append(f"- {n}")
        out.append("")
    if result.warnings:
        out.append("## Warnings")
        out.append("")
        for w in result.warnings:
            out.append(f"- {w}")
        out.append("")
    if result.limitations:
        out.append("## Limitations")
        out.append("")
        for l in result.limitations:
            out.append(f"- {l}")
        out.append("")
    if result.patch_summary:
        out.append("## Patch context")
        out.append("")
        out.append("```json")
        out.append(json.dumps(result.patch_summary, indent=2, ensure_ascii=False))
        out.append("```")
        out.append("")
    return "\n".join(out)


__all__ = [
    "InterpretationConfig",
    "interpret",
    "render_markdown",
]
