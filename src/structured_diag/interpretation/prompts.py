from __future__ import annotations

from textwrap import dedent
from typing import Any

from structured_diag.actions import list_actions

_SYSTEM_PROMPT_RU = dedent("""
    Ты — помощник интерпретации в системе автоматизированной диагностики
    ошибок обучения ML-моделей. Твоя задача — превратить *уже готовый*
    структурированный диагноз и эвиденс в чёткое, корректное и удобное
    для инженера объяснение и список корректирующих рекомендаций.

    ЯЗЫК ОТВЕТА. Все свободные текстовые поля
    (`summary`, `explanation`, `stage_explanations[*].explanation`,
    `symptoms`, `rationale`, `confidence_notes`, `warnings`,
    `limitations`) ОБЯЗАТЕЛЬНО на русском языке. Технические
    идентификаторы (имена ключей JSON, имена action из allowlist,
    имена feature'ов вроде `val_loss_argmin_frac`) оставляй как есть.

    Жёсткие правила (читай внимательно):

    1. Нельзя менять финальный класс диагноза, предсказания stage'ей или
       какие-либо вероятности. Диагноз авторитетен и копируется как есть.
    2. Нельзя выдумывать новые корректирующие действия. Каждая
       рекомендация ОБЯЗАНА ссылаться на `action_name` из переданного
       allowlist'а; параметры действия ОБЯЗАНЫ соответствовать схеме
       (тип, диапазон).
    3. Нельзя заявлять уверенность выше, чем поддерживают вероятности
       каскада. Если `final_confidence < 0.6`, явно укажи неуверенность
       в `summary` и в `warnings`.
    4. Тон — профессиональный, ML-грамотный, лаконичный. Не нравоучение.
       Не выдумывай числовых значений, отсутствующих в эвиденсе.
    5. Рекомендации носят демонстрационный и экспериментальный характер.
       Это должно быть отражено в `limitations`.
    6. Вывод — строго один JSON-объект указанной схемы. Никакого текста
       до или после JSON.
    7. Если диагноз — `healthy`, но в эвиденсе присутствуют
       fault-like-сигналы (например, индикаторы leakage proxy на
       насыщенном датасете), используй формулировку вида
       «диагноз `healthy` несмотря на отдельные слабые / противоречивые
       fault-like-сигналы»; не утверждай противоречие диагнозу.

    Эвиденс будет содержать:
    - диагноз (финальный класс, stage trace, композиционные вероятности,
      альтернативные и отвергнутые гипотезы);
    - structured curve evidence (n_epochs, минимумы val_loss, train/val
      gap, признаки divergence);
    - structured integrity evidence (столбцы `di_*` если доступны);
    - топ контрибутирующих признаков на каждом stage;
    - закрытый allowlist корректирующих действий с параметрическими
      схемами.
    """).strip()

_USER_TEMPLATE_RU = dedent("""
    # Диагноз для интерпретации

    ```json
    {diagnosis_json}
    ```

    # Structured evidence

    ```json
    {evidence_json}
    ```

    # Допустимые корректирующие действия (allowlist)

    В `recommendations` разрешены только перечисленные значения
    `action_name`. Диапазоны и типы параметров обязательны к соблюдению.

    ```json
    {allowlist_json}
    ```

    # Опциональный patch summary (может быть null)

    ```json
    {patch_summary_json}
    ```

    # Требуемая схема вывода

    Верни один JSON-объект, точно соответствующий этой структуре
    (опциональные поля заполняй `null` или пустым списком):

    ```json
    {schema_example}
    ```

    Ограничения вывода:
    - `final_class` и stage-предсказания копируются дословно из диагноза.
    - `recommendations[*].action_name` ∈ allowlist.
    - `recommendations[*].parameters` соответствует типу и диапазону.
    - `priority` — целое число ≥ 1; меньше — важнее.
    - Не более {max_recs} рекомендаций.
    - `summary` — 1–2 предложения, **на русском**.
    - `explanation` — 3–6 предложений, **на русском**.
    - `symptoms` — плоский список коротких строк **на русском**.
    - `rationale` каждой рекомендации — 1–2 предложения **на русском**.
    - `warnings`, `limitations`, `confidence_notes` — **на русском**.
    """).strip()

_OUTPUT_SCHEMA_EXAMPLE_RU: dict[str, Any] = {
    "schema_version": "1.0",
    "backend": "groq",
    "run_id": "<скопировать из диагноза>",
    "final_class": "<скопировать из диагноза>",
    "final_confidence": 0.0,
    "summary": "Короткое резюме на русском в 1–2 предложениях.",
    "explanation": "Развёрнутое объяснение на русском в 3–6 предложениях. Упомяни stage trace, наиболее информативные признаки и что говорят integrity-сигналы.",
    "stage_explanations": [
        {
            "stage_name": "stage1_healthy_vs_faulty",
            "predicted": "faulty",
            "confidence": 0.0,
            "explanation": "1–3 предложения на русском о том, что увидел этот stage.",
        }
    ],
    "symptoms": ["короткий симптом на русском #1", "короткий симптом на русском #2"],
    "recommendations": [
        {
            "action_name": "<из allowlist>",
            "parameters": {"<param>": 0.0},
            "priority": 1,
            "rationale": "На русском: почему это действие адресует данный диагноз.",
            "target_classes": ["overfitting"],
        }
    ],
    "confidence_notes": ["заметка об уверенности на русском"],
    "warnings": ["предупреждение об ограничениях диагноза на русском"],
    "limitations": ["Рекомендации носят демонстрационный характер."],
}

_SYSTEM_PROMPT_EN = dedent("""
    You are an interpretation assistant in an automated diagnostic system
    for ML-model training failures. Your job is to turn an *already
    finalised* structured diagnosis and evidence object into a clear,
    correct and engineer-friendly explanation plus a list of corrective
    recommendations.

    OUTPUT LANGUAGE. All free-form text fields
    (`summary`, `explanation`, `stage_explanations[*].explanation`,
    `symptoms`, `rationale`, `confidence_notes`, `warnings`,
    `limitations`) MUST be written in English. Technical identifiers
    (JSON keys, action names from the allowlist, feature names like
    `val_loss_argmin_frac`) stay as-is.

    Hard rules (read carefully):

    1. You may not change the final diagnostic class, any stage
       prediction, or any probability. The diagnosis is authoritative
       and is copied verbatim.
    2. You may not invent new corrective actions. Each recommendation
       MUST reference an `action_name` from the provided allowlist;
       parameters MUST conform to the action's schema (type, range).
    3. You may not claim more confidence than the cascade probabilities
       support. If `final_confidence < 0.6`, explicitly mark uncertainty
       in `summary` and in `warnings`.
    4. Tone — professional, ML-literate, concise. No moralising. Do not
       fabricate numerical values that are not in the evidence.
    5. Recommendations are demonstrative and experimental in nature.
       This must be reflected in `limitations`.
    6. Output is strictly one JSON object matching the given schema. No
       text before or after the JSON.
    7. If the diagnosis is `healthy` but the evidence contains
       fault-like signals (e.g. leakage-proxy indicators on a saturated
       dataset), use phrasing such as "diagnosis `healthy` despite some
       weak or contradictory fault-like signals"; do not contradict the
       diagnosis.

    Evidence will contain:
    - diagnosis (final class, stage trace, composed probabilities,
      alternative and rejected hypotheses);
    - structured curve evidence (n_epochs, val_loss minima, train/val
      gap, divergence indicators);
    - structured integrity evidence (`di_*` columns when available);
    - top contributing features per stage;
    - the closed allowlist of corrective actions with parameter schemas.
    """).strip()

_USER_TEMPLATE_EN = dedent("""
    # Diagnosis to interpret

    ```json
    {diagnosis_json}
    ```

    # Structured evidence

    ```json
    {evidence_json}
    ```

    # Allowed corrective actions (allowlist)

    In `recommendations`, only the listed `action_name` values are
    allowed. Parameter types and ranges must be respected.

    ```json
    {allowlist_json}
    ```

    # Optional patch summary (may be null)

    ```json
    {patch_summary_json}
    ```

    # Required output schema

    Return exactly one JSON object matching this structure (fill
    optional fields with `null` or an empty list):

    ```json
    {schema_example}
    ```

    Output constraints:
    - `final_class` and stage predictions are copied verbatim from the diagnosis.
    - `recommendations[*].action_name` ∈ allowlist.
    - `recommendations[*].parameters` respects the declared type and range.
    - `priority` — integer ≥ 1; smaller is higher priority.
    - At most {max_recs} recommendations.
    - `summary` — 1–2 sentences, **in English**.
    - `explanation` — 3–6 sentences, **in English**.
    - `symptoms` — flat list of short strings **in English**.
    - `rationale` of every recommendation — 1–2 sentences **in English**.
    - `warnings`, `limitations`, `confidence_notes` — **in English**.
    """).strip()

_OUTPUT_SCHEMA_EXAMPLE_EN: dict[str, Any] = {
    "schema_version": "1.0",
    "backend": "groq",
    "run_id": "<copy from diagnosis>",
    "final_class": "<copy from diagnosis>",
    "final_confidence": 0.0,
    "summary": "Short 1-2 sentence summary in English.",
    "explanation": "Detailed 3-6 sentence explanation in English. Mention the stage trace, the most informative features and what the integrity signals say.",
    "stage_explanations": [
        {
            "stage_name": "stage1_healthy_vs_faulty",
            "predicted": "faulty",
            "confidence": 0.0,
            "explanation": "1-3 sentences in English about what this stage observed.",
        }
    ],
    "symptoms": ["short symptom in English #1", "short symptom in English #2"],
    "recommendations": [
        {
            "action_name": "<from allowlist>",
            "parameters": {"<param>": 0.0},
            "priority": 1,
            "rationale": "In English: why this action addresses the diagnosis.",
            "target_classes": ["overfitting"],
        }
    ],
    "confidence_notes": ["confidence note in English"],
    "warnings": ["diagnostic limitation warning in English"],
    "limitations": ["Recommendations are demonstrative in nature."],
}

_LANGS = {"ru", "en"}

_PROMPT_VERSIONS = {"ru": "ru-1.2", "en": "en-1.0"}

_SYSTEM_PROMPTS = {"ru": _SYSTEM_PROMPT_RU, "en": _SYSTEM_PROMPT_EN}

_USER_TEMPLATES = {"ru": _USER_TEMPLATE_RU, "en": _USER_TEMPLATE_EN}

_OUTPUT_SCHEMAS = {"ru": _OUTPUT_SCHEMA_EXAMPLE_RU, "en": _OUTPUT_SCHEMA_EXAMPLE_EN}


def _norm(language: str | None) -> str:
    if not language:
        return "ru"
    lang = str(language).lower().strip()
    return lang if lang in _LANGS else "ru"


def system_prompt(language: str | None = "ru") -> str:
    return _SYSTEM_PROMPTS[_norm(language)]


def prompt_version(language: str | None = "ru") -> str:
    return _PROMPT_VERSIONS[_norm(language)]


SYSTEM_PROMPT = _SYSTEM_PROMPT_RU


def build_user_prompt(
    *,
    diagnosis_json: str,
    evidence_json: str,
    patch_summary_json: str = "null",
    max_recommendations: int = 3,
    language: str | None = "ru",
) -> str:
    import json as _json

    lang = _norm(language)
    allowlist = [a.to_dict() for a in list_actions()]
    return _USER_TEMPLATES[lang].format(
        diagnosis_json=diagnosis_json,
        evidence_json=evidence_json,
        allowlist_json=_json.dumps(allowlist, indent=2, ensure_ascii=False),
        patch_summary_json=patch_summary_json,
        schema_example=_json.dumps(
            _OUTPUT_SCHEMAS[lang],
            indent=2,
            ensure_ascii=False,
        ),
        max_recs=max_recommendations,
    )


__all__ = [
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "prompt_version",
    "system_prompt",
]
