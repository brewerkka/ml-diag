from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from structured_diag.labels import (
    FAULTY,
    HEALTHY,
    LEAKAGE,
    PRIMARY_LABELS,
    to_stage1,
    to_stage2,
    to_stage3,
)
from structured_diag.models.inference import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
)


@dataclass(frozen=True)
class RowAttribution:
    run_id: str
    y_true: str
    final_class: str
    correct: bool
    stage_at_fault: str | None
    stage1_pred: str
    stage1_p_healthy: float
    stage1_p_faulty: float
    stage2_pred: str | None
    stage3_pred: str | None
    is_leakage_to_healthy: bool
    is_healthy_to_faulty: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "y_true": self.y_true,
            "final_class": self.final_class,
            "correct": bool(self.correct),
            "stage_at_fault": self.stage_at_fault,
            "stage1_pred": self.stage1_pred,
            "stage1_p_healthy": float(self.stage1_p_healthy),
            "stage1_p_faulty": float(self.stage1_p_faulty),
            "stage2_pred": self.stage2_pred,
            "stage3_pred": self.stage3_pred,
            "is_leakage_to_healthy": bool(self.is_leakage_to_healthy),
            "is_healthy_to_faulty": bool(self.is_healthy_to_faulty),
        }


def _attribute_one(d: HierarchicalDiagnosis, y_true: str) -> RowAttribution:
    final = d.final_class
    correct = final == y_true
    s1_pred = d.stage1.predicted
    p_h = float(d.stage1.probabilities.get(HEALTHY, 0.0))
    p_f = float(d.stage1.probabilities.get(FAULTY, max(0.0, 1.0 - p_h)))
    s2_pred = d.stage2.predicted if d.stage2 is not None else None
    s3_pred = d.stage3.predicted if d.stage3 is not None else None
    is_leak_to_hlth = (y_true == LEAKAGE) and (final == HEALTHY)
    is_hlth_to_faulty = (y_true == HEALTHY) and (final != HEALTHY)
    if correct:
        stage_at_fault = None
    else:
        y_s1 = to_stage1(y_true)
        if s1_pred != y_s1:
            stage_at_fault = "stage1"
        elif y_s1 == HEALTHY:
            stage_at_fault = "composition"
        else:
            y_s2 = to_stage2(y_true)
            if s2_pred is None or s2_pred != y_s2:
                stage_at_fault = "stage2"
            else:
                y_s3 = to_stage3(y_true)
                if s3_pred is None or s3_pred != y_s3:
                    stage_at_fault = "stage3"
                else:
                    stage_at_fault = "composition"
    return RowAttribution(
        run_id=str(d.run_id),
        y_true=str(y_true),
        final_class=str(final),
        correct=bool(correct),
        stage_at_fault=stage_at_fault,
        stage1_pred=str(s1_pred),
        stage1_p_healthy=p_h,
        stage1_p_faulty=p_f,
        stage2_pred=str(s2_pred) if s2_pred is not None else None,
        stage3_pred=str(s3_pred) if s3_pred is not None else None,
        is_leakage_to_healthy=is_leak_to_hlth,
        is_healthy_to_faulty=is_hlth_to_faulty,
    )


def attribute_errors(
    diagnoses: Sequence[HierarchicalDiagnosis],
    y_true: pd.Series | Sequence[str],
) -> list[RowAttribution]:
    if isinstance(y_true, pd.Series):
        labels = y_true.astype(str).tolist()
    else:
        labels = [str(v) for v in y_true]
    if len(labels) != len(diagnoses):
        raise ValueError(f"y_true length {len(labels)} does not match diagnoses {len(diagnoses)}")
    return [_attribute_one(d, t) for d, t in zip(diagnoses, labels)]


@dataclass(frozen=True)
class AttributionSummary:
    n: int
    n_correct: int
    n_errors: int
    by_stage: dict[str, int]
    by_class_and_stage: dict[str, dict[str, int]]
    leakage_to_healthy: dict[str, int]
    healthy_to_faulty: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "n_correct": int(self.n_correct),
            "n_errors": int(self.n_errors),
            "by_stage": dict(self.by_stage),
            "by_class_and_stage": {k: dict(v) for k, v in self.by_class_and_stage.items()},
            "leakage_to_healthy": dict(self.leakage_to_healthy),
            "healthy_to_faulty": dict(self.healthy_to_faulty),
        }


_STAGE_KEYS = ("stage1", "stage2", "stage3", "composition")


def summarize_attributions(rows: Sequence[RowAttribution]) -> AttributionSummary:
    n = len(rows)
    n_correct = sum(1 for r in rows if r.correct)
    n_errors = n - n_correct
    by_stage: dict[str, int] = {k: 0 for k in _STAGE_KEYS}
    by_class_and_stage: dict[str, dict[str, int]] = {
        cls: {k: 0 for k in _STAGE_KEYS} for cls in PRIMARY_LABELS
    }
    leak2h = {"n_total": 0, **{k: 0 for k in _STAGE_KEYS}}
    h2f = {"n_total": 0, **{k: 0 for k in _STAGE_KEYS}}
    for r in rows:
        if r.correct or r.stage_at_fault is None:
            continue
        st = r.stage_at_fault
        by_stage[st] = by_stage.get(st, 0) + 1
        cls_dict = by_class_and_stage.setdefault(r.y_true, {k: 0 for k in _STAGE_KEYS})
        cls_dict[st] = cls_dict.get(st, 0) + 1
        if r.is_leakage_to_healthy:
            leak2h["n_total"] += 1
            leak2h[st] = leak2h.get(st, 0) + 1
        if r.is_healthy_to_faulty:
            h2f["n_total"] += 1
            h2f[st] = h2f.get(st, 0) + 1
    return AttributionSummary(
        n=int(n),
        n_correct=int(n_correct),
        n_errors=int(n_errors),
        by_stage=by_stage,
        by_class_and_stage=by_class_and_stage,
        leakage_to_healthy=leak2h,
        healthy_to_faulty=h2f,
    )


@dataclass(frozen=True)
class DisagreementRow:
    run_id: str
    y_true: str
    flat_pred: str
    cascade_pred: str
    flat_correct: bool
    cascade_correct: bool
    stage1_pred: str
    stage1_p_healthy: float
    stage1_p_faulty: float
    stage2_pred: str | None
    stage3_pred: str | None
    top_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "y_true": self.y_true,
            "flat_pred": self.flat_pred,
            "cascade_pred": self.cascade_pred,
            "flat_correct": bool(self.flat_correct),
            "cascade_correct": bool(self.cascade_correct),
            "stage1_pred": self.stage1_pred,
            "stage1_p_healthy": float(self.stage1_p_healthy),
            "stage1_p_faulty": float(self.stage1_p_faulty),
            "stage2_pred": self.stage2_pred,
            "stage3_pred": self.stage3_pred,
            "top_evidence": list(self.top_evidence),
        }


def _row_top_evidence(
    *,
    cascade: HierarchicalCascade,
    feature_row: pd.Series,
    k: int,
) -> list[dict[str, Any]]:
    from structured_diag.evaluation.explanation import _top_features_for_stage

    contribs = _top_features_for_stage(
        stage_name=cascade.stage1.name,
        model=cascade.stage1.model,
        feature_columns=cascade.stage1.feature_columns,
        feature_row=feature_row,
        k=k,
    )
    return [
        {
            "column": c.column,
            "value": float(c.value) if np.isfinite(c.value) else None,
            "importance": float(c.importance),
        }
        for c in contribs
    ]


def find_disagreements(
    *,
    diagnoses: Sequence[HierarchicalDiagnosis],
    y_true: pd.Series,
    flat_pred: Sequence[str],
    X: pd.DataFrame,
    cascade: HierarchicalCascade,
    top_k_evidence: int = 5,
) -> tuple[list[DisagreementRow], list[DisagreementRow]]:
    if len(diagnoses) != len(y_true) or len(diagnoses) != len(flat_pred):
        raise ValueError("Mismatched lengths between diagnoses, y_true, flat_pred.")
    flat_arr = np.asarray(list(flat_pred), dtype=object)
    y_arr = (
        y_true.astype(str).to_numpy()
        if isinstance(y_true, pd.Series)
        else np.asarray(y_true, dtype=object)
    )
    flat_wins: list[DisagreementRow] = []
    cascade_wins: list[DisagreementRow] = []
    for i, d in enumerate(diagnoses):
        cascade_pred = d.final_class
        f_pred = str(flat_arr[i])
        truth = str(y_arr[i])
        if cascade_pred == f_pred:
            continue
        flat_correct = f_pred == truth
        cascade_correct = cascade_pred == truth
        if not (flat_correct ^ cascade_correct):
            continue
        feature_row = X.loc[d.run_id] if d.run_id in X.index else X.iloc[i]
        top = _row_top_evidence(
            cascade=cascade,
            feature_row=feature_row,
            k=top_k_evidence,
        )
        s1 = d.stage1
        row = DisagreementRow(
            run_id=str(d.run_id),
            y_true=truth,
            flat_pred=f_pred,
            cascade_pred=cascade_pred,
            flat_correct=flat_correct,
            cascade_correct=cascade_correct,
            stage1_pred=str(s1.predicted),
            stage1_p_healthy=float(s1.probabilities.get(HEALTHY, 0.0)),
            stage1_p_faulty=float(s1.probabilities.get(FAULTY, 0.0)),
            stage2_pred=str(d.stage2.predicted) if d.stage2 is not None else None,
            stage3_pred=str(d.stage3.predicted) if d.stage3 is not None else None,
            top_evidence=top,
        )
        if flat_correct and not cascade_correct:
            flat_wins.append(row)
        elif cascade_correct and not flat_correct:
            cascade_wins.append(row)
    return flat_wins, cascade_wins


def _fmt_pct(numer: int, denom: int) -> str:
    if denom <= 0:
        return "—"
    return f"{numer}/{denom} ({100 * numer / denom:.1f}%)"


def render_attribution_markdown(
    *,
    corpus_name: str,
    slice_summaries: Mapping[str, AttributionSummary],
) -> str:
    out: list[str] = []
    out.append(f"# Cascade error attribution — {corpus_name}")
    out.append("")
    out.append(
        "For every cascade error we identify the *first* stage whose "
        "internal prediction did not match the corresponding sub-label. "
        "`stage1` errors mean Stage 1 chose the wrong binary branch; "
        "`stage2` means Stage 1 was right (faulty case) but Stage 2 picked "
        "the wrong branch; `stage3` means Stage 1 and Stage 2 were both "
        "right but Stage 3 picked the wrong leaf; `composition` means "
        "every stage prediction matched its sub-label but the marginal "
        "argmax over PRIMARY_LABELS still landed on a different class "
        "(this is the rare soft-mass failure mode of the cascade)."
    )
    out.append("")
    for slice_name, summ in slice_summaries.items():
        out.append(f"## Slice: `{slice_name}`")
        out.append("")
        out.append(f"- n = {summ.n}, correct = {summ.n_correct}, errors = {summ.n_errors}")
        out.append("")
        out.append("### Errors by stage at fault")
        out.append("")
        out.append("| stage | count | share of errors |")
        out.append("|---|---:|---:|")
        for st in _STAGE_KEYS:
            cnt = summ.by_stage.get(st, 0)
            share = (cnt / summ.n_errors) if summ.n_errors else 0.0
            out.append(f"| {st} | {cnt} | {share:.1%} |")
        out.append("")
        out.append("### Errors by true class × stage")
        out.append("")
        header = "| true class | total errors | " + " | ".join(_STAGE_KEYS) + " |"
        sep = "|---|---:|" + "|".join(["---:" for _ in _STAGE_KEYS]) + "|"
        out.append(header)
        out.append(sep)
        for cls in PRIMARY_LABELS:
            row = summ.by_class_and_stage.get(cls, {})
            total = sum(row.values())
            cells = " | ".join(str(row.get(st, 0)) for st in _STAGE_KEYS)
            out.append(f"| {cls} | {total} | {cells} |")
        out.append("")
        out.append("### Critical buckets")
        out.append("")
        out.append("| bucket | total | stage1 | stage2 | stage3 | composition |")
        out.append("|---|---:|---:|---:|---:|---:|")
        l = summ.leakage_to_healthy
        h = summ.healthy_to_faulty
        out.append(
            f"| leakage→healthy | {l.get('n_total', 0)} | {l.get('stage1', 0)} | "
            f"{l.get('stage2', 0)} | {l.get('stage3', 0)} | {l.get('composition', 0)} |"
        )
        out.append(
            f"| healthy→faulty | {h.get('n_total', 0)} | {h.get('stage1', 0)} | "
            f"{h.get('stage2', 0)} | {h.get('stage3', 0)} | {h.get('composition', 0)} |"
        )
        out.append("")
    return "\n".join(out)


def render_disagreements_markdown(
    *,
    slice_name: str,
    flat_wins: Sequence[DisagreementRow],
    cascade_wins: Sequence[DisagreementRow],
    top_n: int = 20,
) -> str:
    out: list[str] = []
    out.append(f"## Disagreements — slice `{slice_name}`")
    out.append("")
    out.append(
        f"- flat correct, cascade wrong: **{len(flat_wins)}** rows (cascade regressions vs flat)"
    )
    out.append(
        f"- cascade correct, flat wrong: **{len(cascade_wins)}** rows (cascade wins vs flat)"
    )
    out.append("")

    def _render_block(title: str, rows: Sequence[DisagreementRow]) -> None:
        out.append(f"### {title} (showing up to {top_n})")
        out.append("")
        if not rows:
            out.append("_(none)_")
            out.append("")
            return
        out.append(
            "| run_id | y_true | flat_pred | cascade_pred | s1 (P_h, P_f) | "
            "s2 | s3 | top evidence (col=value, imp) |"
        )
        out.append("|---|---|---|---|---|---|---|---|")
        for r in rows[:top_n]:
            ev = "; ".join(
                (
                    f"{e['column']}={e['value']:.3g} (imp={e['importance']:.3g})"
                    if e.get("value") is not None
                    else f"{e['column']}=NaN (imp={e['importance']:.3g})"
                )
                for e in r.top_evidence[:3]
            )
            s2 = r.stage2_pred or "—"
            s3 = r.stage3_pred or "—"
            out.append(
                f"| `{r.run_id}` | {r.y_true} | {r.flat_pred} | {r.cascade_pred} "
                f"| {r.stage1_pred} ({r.stage1_p_healthy:.2f}, {r.stage1_p_faulty:.2f}) "
                f"| {s2} | {s3} | {ev} |"
            )
        out.append("")

    _render_block("Flat correct / cascade wrong", flat_wins)
    _render_block("Cascade correct / flat wrong", cascade_wins)
    return "\n".join(out)


__all__ = [
    "RowAttribution",
    "AttributionSummary",
    "DisagreementRow",
    "attribute_errors",
    "summarize_attributions",
    "find_disagreements",
    "render_attribution_markdown",
    "render_disagreements_markdown",
]
