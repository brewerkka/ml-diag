from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from structured_diag.evaluation.metrics import ClassificationReport, classification_report
from structured_diag.evaluation.reports import report_to_markdown
from structured_diag.labels import (
    HEALTHY,
    LEAKAGE,
    PRIMARY_LABELS,
    STAGE1_LABELS,
    STAGE2_LABELS,
    STAGE3_LABELS_BY_BRANCH,
    to_stage1,
    to_stage2,
    to_stage3,
)
from structured_diag.models.flat_baseline import FlatBaselineResult
from structured_diag.models.inference import HierarchicalCascade, diagnose_batch
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class SliceComparison:
    slice_name: str
    n_samples: int
    flat_report: ClassificationReport
    hier_report: ClassificationReport
    stage_reports: dict[str, ClassificationReport] = field(default_factory=dict)
    leakage_healthy_confusion: dict[str, Any] = field(default_factory=dict)
    error_propagation: dict[str, Any] = field(default_factory=dict)

    def deltas(self) -> dict[str, float]:
        return {
            "delta_accuracy": self.hier_report.accuracy - self.flat_report.accuracy,
            "delta_macro_f1": self.hier_report.macro_f1 - self.flat_report.macro_f1,
            "delta_weighted_f1": self.hier_report.weighted_f1 - self.flat_report.weighted_f1,
            "delta_ece": (
                None
                if self.hier_report.ece is None or self.flat_report.ece is None
                else self.hier_report.ece - self.flat_report.ece
            ),
        }

    def per_class_deltas(self) -> dict[str, float]:
        all_classes = sorted(
            set(self.flat_report.per_class_f1) | set(self.hier_report.per_class_f1)
        )
        return {
            cls: float(
                self.hier_report.per_class_f1.get(cls, 0.0)
                - self.flat_report.per_class_f1.get(cls, 0.0)
            )
            for cls in all_classes
        }


def _flat_predict(
    result: FlatBaselineResult, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray | None]:
    from structured_diag.utils import (
        align_features_to_schema,
        ensure_feature_matrix,
    )

    if result.feature_columns:
        X_aligned = align_features_to_schema(X, result.feature_columns)
    else:
        X_aligned = X
    arr = ensure_feature_matrix(X_aligned)
    preds = result.model.predict(arr)
    proba: np.ndarray | None = None
    if hasattr(result.model, "predict_proba"):
        try:
            proba = np.asarray(result.model.predict_proba(arr), dtype=float)
        except Exception:
            proba = None
    return np.asarray(preds, dtype=object), proba


def _hier_predict(
    cascade: HierarchicalCascade, X: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, list]:
    diags = diagnose_batch(cascade, X)
    finals = np.asarray([d.final_class for d in diags], dtype=object)
    classes = list(PRIMARY_LABELS)
    proba = np.zeros((len(diags), len(classes)), dtype=float)
    for i, d in enumerate(diags):
        for j, cls in enumerate(classes):
            proba[i, j] = float(d.class_probabilities.get(cls, 0.0))
    return finals, proba, diags


def _stage_wise_reports(diags: list, y_true: pd.Series) -> dict[str, ClassificationReport]:
    out: dict[str, ClassificationReport] = {}
    y_true_s1 = y_true.map(to_stage1)
    y_pred_s1 = pd.Series([d.stage1.predicted for d in diags], index=y_true.index)
    out["stage1"] = classification_report(y_true_s1, y_pred_s1, label_order=STAGE1_LABELS)
    faulty_mask = y_true.map(to_stage1) != HEALTHY
    y_true_s2 = y_true[faulty_mask].map(to_stage2)
    y_pred_s2_raw: list[str | None] = []
    for d, take in zip(diags, faulty_mask.tolist()):
        if not take:
            continue
        y_pred_s2_raw.append(d.stage2.predicted if d.stage2 else "<no_stage2>")
    if not y_true_s2.empty:
        out["stage2"] = classification_report(
            y_true_s2,
            pd.Series(y_pred_s2_raw, index=y_true_s2.index),
            label_order=STAGE2_LABELS,
        )
    for branch, vocab in STAGE3_LABELS_BY_BRANCH.items():
        mask = faulty_mask & (y_true.map(to_stage2) == branch)
        if not mask.any():
            continue
        y_true_s3 = y_true[mask].map(to_stage3)
        preds: list[str | None] = []
        for d, take in zip(diags, mask.tolist()):
            if not take:
                continue
            preds.append(d.stage3.predicted if d.stage3 else "<no_stage3>")
        out[f"stage3_{branch}"] = classification_report(
            y_true_s3,
            pd.Series(preds, index=y_true_s3.index),
            label_order=vocab,
        )
    return out


def _leakage_healthy_confusion(
    y_true: pd.Series, flat_pred: np.ndarray, hier_pred: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    y = y_true.values
    for contour, pred in (("flat", flat_pred), ("hierarchical", hier_pred)):
        leaked_called_healthy = int(((y == LEAKAGE) & (pred == HEALTHY)).sum())
        healthy_called_leaked = int(((y == HEALTHY) & (pred == LEAKAGE)).sum())
        n_leakage = int((y == LEAKAGE).sum())
        n_healthy = int((y == HEALTHY).sum())
        out[contour] = {
            "leakage_called_healthy": leaked_called_healthy,
            "healthy_called_leakage": healthy_called_leaked,
            "leakage_recall_loss_to_healthy": (
                leaked_called_healthy / n_leakage if n_leakage else None
            ),
            "healthy_false_alarm_to_leakage": (
                healthy_called_leaked / n_healthy if n_healthy else None
            ),
            "n_leakage": n_leakage,
            "n_healthy": n_healthy,
        }
    return out


def _error_propagation(diags: list, y_true: pd.Series) -> dict[str, Any]:
    n = len(diags)
    if n == 0:
        return {}
    y_true_s1 = y_true.map(to_stage1).values
    y_true_s2 = y_true.map(to_stage2).values
    y_true_s3 = y_true.map(to_stage3).values
    s1_correct = np.array([d.stage1.predicted == t for d, t in zip(diags, y_true_s1)])
    s2_eligible = np.array([d.stage2 is not None for d in diags])
    s2_correct = np.array(
        [d.stage2 is not None and d.stage2.predicted == t for d, t in zip(diags, y_true_s2)]
    )
    s3_eligible = np.array([d.stage3 is not None for d in diags])
    s3_correct = np.array(
        [d.stage3 is not None and d.stage3.predicted == t for d, t in zip(diags, y_true_s3)]
    )
    final_correct = np.array([d.final_class == t for d, t in zip(diags, y_true.values)])
    return {
        "n": int(n),
        "stage1_acc": float(s1_correct.mean()) if n else None,
        "stage2_acc_given_eligible": (
            float(s2_correct[s2_eligible].mean()) if s2_eligible.any() else None
        ),
        "stage3_acc_given_eligible": (
            float(s3_correct[s3_eligible].mean()) if s3_eligible.any() else None
        ),
        "stage1_only_correct_but_final_wrong": int(((s1_correct) & (~final_correct)).sum()),
        "stage1_wrong_runs": int((~s1_correct).sum()),
        "final_acc": float(final_correct.mean()),
    }


def compare_on_slice(
    *,
    slice_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    flat_result: FlatBaselineResult,
    cascade: HierarchicalCascade,
    label_order: Sequence[str] = PRIMARY_LABELS,
) -> SliceComparison:
    flat_pred, flat_proba = _flat_predict(flat_result, X)
    hier_pred, hier_proba, diags = _hier_predict(cascade, X)
    flat_rep = classification_report(
        y,
        flat_pred,
        y_proba=flat_proba,
        proba_classes=flat_result.classes if flat_result.classes else None,
        label_order=label_order,
    )
    hier_rep = classification_report(
        y,
        hier_pred,
        y_proba=hier_proba,
        proba_classes=list(label_order),
        label_order=label_order,
    )
    stage_reports = _stage_wise_reports(diags, y)
    lh = _leakage_healthy_confusion(y, flat_pred, hier_pred)
    ep = _error_propagation(diags, y)
    return SliceComparison(
        slice_name=slice_name,
        n_samples=int(len(y)),
        flat_report=flat_rep,
        hier_report=hier_rep,
        stage_reports=stage_reports,
        leakage_healthy_confusion=lh,
        error_propagation=ep,
    )


def compare_all_slices(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    flat_result: FlatBaselineResult,
    cascade: HierarchicalCascade,
    slices: Mapping[str, pd.Index],
) -> dict[str, SliceComparison]:
    out: dict[str, SliceComparison] = {}
    for slice_name, idx in slices.items():
        if len(idx) == 0:
            continue
        Xs = X.loc[idx]
        ys = y.loc[idx]
        if Xs.empty:
            continue
        out[slice_name] = compare_on_slice(
            slice_name=slice_name,
            X=Xs,
            y=ys,
            flat_result=flat_result,
            cascade=cascade,
        )
    return out


def _fmt(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    if np.isnan(x):
        return "—"
    return f"{x:.{digits}f}"


def _delta_arrow(delta: float | None, *, eps: float = 1e-4) -> str:
    if delta is None:
        return "—"
    if delta > eps:
        return f"▲ {delta:+.4f}"
    if delta < -eps:
        return f"▼ {delta:+.4f}"
    return "≈ 0"


def render_comparison_markdown(
    *,
    corpus_name: str,
    feature_source: str,
    flat_model_name: str,
    cascade_stages: list[str],
    comparisons: Mapping[str, SliceComparison],
) -> str:
    out: list[str] = []
    out.append("# Flat vs Hierarchical — comparison report")
    out.append("")
    out.append(f"- corpus: **{corpus_name}**")
    out.append(f"- feature source: `{feature_source}`")
    out.append(f"- flat model: `{flat_model_name}`")
    out.append(f"- hierarchical stages: {', '.join(f'`{s}`' for s in cascade_stages)}")
    out.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append(
        "Both contours are evaluated on the **same rows** with the **same** "
        "feature columns. Flat predicts a primary label directly; hierarchical "
        "composes Stage 1 / Stage 2 / Stage 3 marginals and picks argmax."
    )
    out.append("")
    out.append("## Headline — per slice")
    out.append("")
    out.append(
        "| slice | n | flat acc | hier acc | Δ acc | flat macro-F1 | hier macro-F1 | Δ macro-F1 | flat ECE | hier ECE | Δ ECE |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for slice_name, c in comparisons.items():
        d = c.deltas()
        out.append(
            f"| **{slice_name}** | {c.n_samples} "
            f"| {_fmt(c.flat_report.accuracy)} | {_fmt(c.hier_report.accuracy)} "
            f"| {_delta_arrow(d['delta_accuracy'])} "
            f"| {_fmt(c.flat_report.macro_f1)} | {_fmt(c.hier_report.macro_f1)} "
            f"| {_delta_arrow(d['delta_macro_f1'])} "
            f"| {_fmt(c.flat_report.ece)} | {_fmt(c.hier_report.ece)} "
            f"| {_delta_arrow(d['delta_ece'])} |"
        )
    out.append("")
    for slice_name, c in comparisons.items():
        out.append("---")
        out.append("")
        out.append(f"## Slice: `{slice_name}` (n = {c.n_samples})")
        out.append("")
        out.append("### Per-class F1")
        out.append("")
        out.append("| class | flat F1 | hier F1 | Δ F1 |")
        out.append("|---|---:|---:|---:|")
        deltas = c.per_class_deltas()
        for cls in sorted(deltas, key=lambda k: -deltas[k]):
            f_flat = c.flat_report.per_class_f1.get(cls, 0.0)
            f_hier = c.hier_report.per_class_f1.get(cls, 0.0)
            out.append(
                f"| {cls} | {_fmt(f_flat)} | {_fmt(f_hier)} | {_delta_arrow(f_hier - f_flat)} |"
            )
        out.append("")
        out.append("### Confusion matrices")
        out.append("")
        out.append("**Flat**")
        out.append("")
        out.append(report_to_markdown(c.flat_report, heading=f"Flat on `{slice_name}`"))
        out.append("**Hierarchical**")
        out.append("")
        out.append(report_to_markdown(c.hier_report, heading=f"Hierarchical on `{slice_name}`"))
        if c.stage_reports:
            out.append("### Stage-wise metrics (hierarchical)")
            out.append("")
            for stage_key, rep in c.stage_reports.items():
                out.append(report_to_markdown(rep, heading=f"`{stage_key}`"))
        if c.leakage_healthy_confusion:
            out.append("### Leakage ↔ healthy confusion")
            out.append("")
            out.append(
                "| contour | leakage→healthy | healthy→leakage | leakage recall loss | healthy false alarm | n_leakage | n_healthy |"
            )
            out.append("|---|---:|---:|---:|---:|---:|---:|")
            for contour, vals in c.leakage_healthy_confusion.items():
                out.append(
                    f"| {contour} | {vals['leakage_called_healthy']} | {vals['healthy_called_leakage']} "
                    f"| {_fmt(vals['leakage_recall_loss_to_healthy'])} "
                    f"| {_fmt(vals['healthy_false_alarm_to_leakage'])} "
                    f"| {vals['n_leakage']} | {vals['n_healthy']} |"
                )
            out.append("")
        if c.error_propagation:
            ep = c.error_propagation
            out.append("### Error propagation (hierarchical only)")
            out.append("")
            out.append("| metric | value |")
            out.append("|---|---:|")
            out.append(f"| n | {ep.get('n')} |")
            out.append(f"| Stage-1 accuracy | {_fmt(ep.get('stage1_acc'))} |")
            out.append(
                f"| Stage-2 accuracy (when stage1 said faulty) | {_fmt(ep.get('stage2_acc_given_eligible'))} |"
            )
            out.append(
                f"| Stage-3 accuracy (when stage2 ran) | {_fmt(ep.get('stage3_acc_given_eligible'))} |"
            )
            out.append(f"| Final accuracy | {_fmt(ep.get('final_acc'))} |")
            out.append(
                f"| Stage-1 right but final wrong | {ep.get('stage1_only_correct_but_final_wrong')} |"
            )
            out.append(f"| Stage-1 wrong runs | {ep.get('stage1_wrong_runs')} |")
            out.append("")
    out.append("---")
    out.append("")
    out.append("## Verdict (mechanical)")
    out.append("")
    out.append("This block is auto-generated by comparing macro-F1 on each slice.")
    out.append("")
    for slice_name, c in comparisons.items():
        d = c.deltas()
        if d["delta_macro_f1"] > 1e-3:
            verdict = f"hierarchical wins by macro-F1 (Δ = +{d['delta_macro_f1']:.4f})"
        elif d["delta_macro_f1"] < -1e-3:
            verdict = f"flat wins by macro-F1 (Δ = {d['delta_macro_f1']:.4f})"
        else:
            verdict = "tie on macro-F1"
        out.append(f"- **{slice_name}**: {verdict}; n = {c.n_samples}")
    out.append("")
    return "\n".join(out)


__all__ = [
    "SliceComparison",
    "compare_on_slice",
    "compare_all_slices",
    "render_comparison_markdown",
]
