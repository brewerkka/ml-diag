from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structured_diag.evaluation.explanation import StructuredEvidence

PROMPT_VERSION = "ru-arb-1.2"

SYSTEM_PROMPT_ARBITRATOR = """\
Ты — арбитр диагностической системы для ML-обучений. Два независимых
классификатора (flat baseline и hierarchical cascade) разошлись в
диагнозе одного training run. Твоя задача — выбрать, кто из них прав.

Доступные классы: healthy, overfitting, underfitting, leakage, label_noise, instability.

ЖЁСТКИЕ ПРАВИЛА:
1. Отвечай ТОЛЬКО валидным JSON по предоставленной схеме.
2. chosen_label ОБЯЗАН быть равен либо flat_label, либо cascade_label —
   ровно одному из этих двух конкретных классов. Третий класс
   запрещён, даже если он кажется тебе более правдоподобным.
3. chosen_source = "flat" если ты выбрал flat_label, иначе "cascade".
   Третьего варианта нет в этой задаче.
4. На дизагрементах один из контуров почти всегда прав. Эмпирически
   на 8ds: flat правый в ~55% случаев, cascade — в ~40%. Поэтому если
   evidence неубедительный или нейтральный — отдавай предпочтение
   контуру с большей top-1 вероятностью.
5. confidence ∈ [0, 1] — твоя самооценка качества решения.
6. reasoning — 2-4 предложения на русском языке, опирающиеся
   на конкретные сигналы из evidence (curve gaps, integrity values,
   topи feature contributions).

ТИПИЧНЫЕ ПОДВОДНЫЕ КАМНИ:
- "near-zero train/val gap" сам по себе — это И healthy, И leakage
  одновременно. Решает обычно integrity evidence или dataset-specific
  паттерн, а не сам gap.
- "early val_loss minimum" — характерно для overfitting / instability;
  для leakage это не специфично.
- Если оба контура с низкой confidence и разница меньше 0.1 —
  выбирай flat (он эмпирически чаще прав).
"""

_DATASET_HINTS = (
    "Эмпирические подсказки по 8ds:\n"
    "  flat сильнее на: iris, credit-g, spambase\n"
    "  cascade сильнее на: wine, breast_cancer, Australian, banknote"
)

JSON_SCHEMA_BLOCK = (
    "{\n"
    '  "chosen_label": "<строго flat_label или cascade_label>",\n'
    '  "chosen_source": "flat" | "cascade",\n'
    '  "confidence": <0.0..1.0>,\n'
    '  "reasoning": "<кратко по-русски, 2-4 предложения>",\n'
    '  "alternative_label": "<или null>"\n'
    "}"
)


def _format_top_proba(
    proba: dict[str, float],
    k: int = 3,
) -> str:
    items = sorted(proba.items(), key=lambda kv: -float(kv[1]))[:k]
    return ", ".join(f"{c}={float(p):.3f}" for c, p in items)


def render_evidence_digest(
    evidence: StructuredEvidence,
    *,
    top_k_features: int = 5,
) -> str:
    lines: list[str] = []
    lines.append(
        f"final_class предсказан каскадом: `{evidence.final_class}` "
        f"(confidence {evidence.final_confidence:.3f})"
    )
    sorted_p = sorted(evidence.class_probabilities.items(), key=lambda kv: -kv[1])[:3]
    lines.append("class_probabilities (топ-3): " + ", ".join(f"{c}={p:.3f}" for c, p in sorted_p))
    feats = sorted(
        evidence.top_features,
        key=lambda f: -float(f.importance),
    )[:top_k_features]
    if feats:
        lines.append("Топ контрибьютеры:")
        for f in feats:
            v = "—" if (f.value is None) else f"{float(f.value):.4f}"
            lines.append(
                f"  - {f.column} = {v}  (imp={float(f.importance):.4f}, stage={f.source_stage})"
            )
    ce = evidence.curve_evidence
    curve_bits: list[str] = []
    if ce.n_epochs is not None:
        curve_bits.append(f"epochs={ce.n_epochs}")
    if ce.final_acc_gap is not None:
        curve_bits.append(f"final_acc_gap={ce.final_acc_gap:+.3f}")
    if ce.val_loss_argmin_frac is not None:
        curve_bits.append(f"val_loss_argmin_frac={ce.val_loss_argmin_frac:.2f}")
    if ce.diverged is not None:
        curve_bits.append(f"diverged={ce.diverged}")
    if curve_bits:
        lines.append("Curve evidence: " + ", ".join(curve_bits))
    if evidence.integrity_evidence.columns:
        ie_pairs = list(evidence.integrity_evidence.columns.items())[:3]
        lines.append("Integrity evidence: " + ", ".join(f"{k}={v}" for k, v in ie_pairs))
    notes = (
        list(evidence.curve_evidence.notes)
        + list(evidence.integrity_evidence.notes)
        + list(evidence.diagnostic_notes)
    )
    notes = [n for n in notes if n][:3]
    if notes:
        lines.append("Заметки:")
        for n in notes:
            lines.append(f"  - {n}")
    return "\n".join(lines)


def build_user_prompt(
    *,
    run_id: str,
    evidence_md: str,
    flat_label: str,
    flat_proba: dict[str, float],
    cascade_label: str,
    cascade_proba: dict[str, float],
) -> str:
    flat_top3 = _format_top_proba(flat_proba, k=3)
    cascade_top3 = _format_top_proba(cascade_proba, k=3)
    return (
        f"RUN: {run_id}\n\n"
        f"ДОКАЗАТЕЛЬСТВА (evidence layer):\n{evidence_md}\n\n"
        f"ПРЕДСКАЗАНИЕ flat baseline (random forest, 51 признак):\n"
        f"  flat_label = {flat_label}\n"
        f"  топ-3 вероятности: {flat_top3}\n\n"
        f"ПРЕДСКАЗАНИЕ hierarchical cascade (Stage 1 → 2 → 3):\n"
        f"  cascade_label = {cascade_label}\n"
        f"  composed marginal топ-3: {cascade_top3}\n\n"
        f"{_DATASET_HINTS}\n\n"
        f"ВЫБОР: chosen_label обязан быть РОВНО ОДНИМ из двух значений:\n"
        f'  - {flat_label}  (chosen_source="flat")\n'
        f'  - {cascade_label}  (chosen_source="cascade")\n'
        f"Любой иной класс будет отвергнут как невалидный.\n\n"
        f"Верни JSON:\n"
        f"{JSON_SCHEMA_BLOCK}\n"
    )


__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT_ARBITRATOR",
    "JSON_SCHEMA_BLOCK",
    "build_user_prompt",
    "render_evidence_digest",
]
