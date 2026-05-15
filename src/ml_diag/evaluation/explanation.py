from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_diag.labels import (
    DATA_RELATED,
    FAULTY,
    HEALTHY,
    OPT_GEN_RELATED,
    PRIMARY_LABELS,
    to_stage2,
)
from ml_diag.models.inference import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

EVIDENCE_SCHEMA_VERSION = "1.0"

_NOTE_CLASS_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("leakage proxy", "leakage"),
    ("leakage signal", "leakage"),
    ("leakage evidence", "leakage"),
    ("train/val overlap", "leakage"),
    ("duplicate overlap", "leakage"),
    ("replay", "leakage"),
    ("split_hash", "leakage"),
    ("overfitting fingerprint", "overfitting"),
    ("overfitting evidence", "overfitting"),
    ("early-overfitting", "overfitting"),
    ("late-overfitting", "overfitting"),
    ("underfitting evidence", "underfitting"),
    ("underfitting", "underfitting"),
    ("label noise", "label_noise"),
    ("label_noise", "label_noise"),
    ("declared noise", "label_noise"),
    ("diverged", "instability"),
    ("divergence", "instability"),
    ("instability", "instability"),
    ("nan/inf", "instability"),
    ("loss spike", "instability"),
)


def _supports_class(note: str) -> str | None:
    if not note:
        return None
    low = note.lower()
    for kw, cls in _NOTE_CLASS_KEYWORDS:
        if kw in low:
            return cls
    return None


def classify_evidence_notes(notes: list[str], diagnosis_class: str) -> tuple[list[str], list[str]]:
    decisive: list[str] = []
    secondary: list[str] = []
    for note in notes:
        if not note:
            continue
        supports = _supports_class(note)
        if supports is None or supports == diagnosis_class:
            decisive.append(note)
        else:
            secondary.append(note)
    return decisive, secondary


@dataclass(frozen=True)
class StageTraceEntry:
    stage_name: str
    predicted: str
    confidence: float
    top_classes: list[tuple[str, float]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "predicted": self.predicted,
            "confidence": float(self.confidence),
            "top_classes": [(c, float(p)) for c, p in self.top_classes],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class FeatureContribution:
    column: str
    value: float
    importance: float
    source_stage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "value": float(self.value) if self.value is not None else None,
            "importance": float(self.importance),
            "source_stage": self.source_stage,
        }


@dataclass(frozen=True)
class CurveEvidence:
    n_epochs: int | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    val_loss_min: float | None = None
    val_loss_argmin_frac: float | None = None
    final_acc_gap: float | None = None
    max_acc_gap: float | None = None
    diverged: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_epochs": self.n_epochs,
            "final_train_loss": self.final_train_loss,
            "final_val_loss": self.final_val_loss,
            "val_loss_min": self.val_loss_min,
            "val_loss_argmin_frac": self.val_loss_argmin_frac,
            "final_acc_gap": self.final_acc_gap,
            "max_acc_gap": self.max_acc_gap,
            "diverged": self.diverged,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class IntegrityEvidence:
    columns: dict[str, float | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": {k: (None if v is None else float(v)) for k, v in self.columns.items()},
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class StructuredEvidence:
    schema_version: str
    generated_at: str
    run_id: str
    final_class: str
    final_confidence: float
    class_probabilities: dict[str, float]
    alternative_hypotheses: list[tuple[str, float]]
    rejected_hypotheses: list[tuple[str, float]]
    stage_trace: list[StageTraceEntry]
    top_features: list[FeatureContribution]
    curve_evidence: CurveEvidence
    integrity_evidence: IntegrityEvidence
    diagnostic_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "run_id": self.run_id,
            "final_class": self.final_class,
            "final_confidence": float(self.final_confidence),
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "alternative_hypotheses": [(c, float(p)) for c, p in self.alternative_hypotheses],
            "rejected_hypotheses": [(c, float(p)) for c, p in self.rejected_hypotheses],
            "stage_trace": [s.to_dict() for s in self.stage_trace],
            "top_features": [f.to_dict() for f in self.top_features],
            "curve_evidence": self.curve_evidence.to_dict(),
            "integrity_evidence": self.integrity_evidence.to_dict(),
            "diagnostic_notes": list(self.diagnostic_notes),
        }


def _safe_top_classes(prob: Mapping[str, float], k: int = 3) -> list[tuple[str, float]]:
    return sorted(((c, float(p)) for c, p in prob.items()), key=lambda kv: -kv[1])[:k]


def _stage_trace_from_diagnosis(d: HierarchicalDiagnosis) -> list[StageTraceEntry]:
    out: list[StageTraceEntry] = []
    out.append(
        StageTraceEntry(
            stage_name=d.stage1.stage_name,
            predicted=d.stage1.predicted,
            confidence=d.stage1.confidence,
            top_classes=_safe_top_classes(d.stage1.probabilities),
            notes=[],
        )
    )
    if d.stage2 is not None:
        out.append(
            StageTraceEntry(
                stage_name=d.stage2.stage_name,
                predicted=d.stage2.predicted,
                confidence=d.stage2.confidence,
                top_classes=_safe_top_classes(d.stage2.probabilities),
            )
        )
    if d.stage3 is not None:
        out.append(
            StageTraceEntry(
                stage_name=d.stage3.stage_name,
                predicted=d.stage3.predicted,
                confidence=d.stage3.confidence,
                top_classes=_safe_top_classes(d.stage3.probabilities),
            )
        )
    return out


def _unwrap_classifier(model: object) -> object:
    if hasattr(model, "calibrated_classifiers_"):
        try:
            inner = model.calibrated_classifiers_[0]
            if hasattr(inner, "estimator"):
                return _unwrap_classifier(inner.estimator)
            if hasattr(inner, "base_estimator"):
                return _unwrap_classifier(inner.base_estimator)
        except Exception:
            pass
    if hasattr(model, "named_steps"):
        for name in ("clf", "classifier", "model"):
            if name in getattr(model, "named_steps", {}):
                return _unwrap_classifier(model.named_steps[name])
    return model


def _feature_importances(model: object) -> np.ndarray | None:
    inner = _unwrap_classifier(model)
    if hasattr(inner, "feature_importances_"):
        return np.asarray(inner.feature_importances_, dtype=float)
    if hasattr(inner, "coef_"):
        coef = np.asarray(inner.coef_, dtype=float)
        if coef.ndim == 1:
            return np.abs(coef)
        return np.abs(coef).mean(axis=0)
    return None


def _top_features_for_stage(
    *,
    stage_name: str,
    model: object,
    feature_columns: Sequence[str],
    feature_row: pd.Series,
    k: int = 5,
) -> list[FeatureContribution]:
    importances = _feature_importances(model)
    if importances is None or len(importances) != len(feature_columns):
        return []
    contrib = np.zeros(len(feature_columns), dtype=float)
    for i, col in enumerate(feature_columns):
        try:
            v = float(feature_row.get(col, np.nan))
        except (TypeError, ValueError):
            v = float("nan")
        if not np.isfinite(v):
            contrib[i] = 0.0
        else:
            contrib[i] = float(importances[i]) * abs(v)
    order_source = contrib if np.any(contrib > 0) else importances
    order = np.argsort(-order_source)
    out: list[FeatureContribution] = []
    for idx in order[:k]:
        col = feature_columns[idx]
        v = feature_row.get(col, np.nan)
        try:
            val_f = float(v)
        except (TypeError, ValueError):
            val_f = float("nan")
        out.append(
            FeatureContribution(
                column=col,
                value=val_f,
                importance=float(order_source[idx]),
                source_stage=stage_name,
            )
        )
    return out


def _build_curve_evidence(feature_row: pd.Series) -> CurveEvidence:
    def get(col: str) -> float | None:
        v = feature_row.get(col, None)
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(f):
            return None
        return f

    notes: list[str] = []
    val_argmin_frac = get("val_loss_argmin_frac")
    if val_argmin_frac is not None and 0 <= val_argmin_frac < 0.5:
        notes.append(
            f"val_loss minimum reached at {val_argmin_frac:.0%} of training — "
            "early-overfitting fingerprint."
        )
    final_acc_gap = get("acc_final_gap")
    if final_acc_gap is not None and final_acc_gap > 0.15:
        notes.append(f"final train/val accuracy gap = {final_acc_gap:+.3f} — overfitting evidence.")
    if final_acc_gap is not None and abs(final_acc_gap) < 0.01:
        notes.append(
            "near-zero train/val gap — could indicate either healthy convergence or leakage."
        )
    diverged = bool(get("diverged"))
    if diverged:
        notes.append("history contains NaN/inf in train_loss or val_loss — training diverged.")
    return CurveEvidence(
        n_epochs=int(get("n_epochs")) if get("n_epochs") is not None else None,
        final_train_loss=get("train_loss_final"),
        final_val_loss=get("val_loss_final"),
        val_loss_min=get("val_loss_min"),
        val_loss_argmin_frac=val_argmin_frac,
        final_acc_gap=final_acc_gap,
        max_acc_gap=get("acc_max_gap"),
        diverged=diverged if diverged is not None else None,
        notes=notes,
    )


def _build_integrity_evidence(
    feature_row: pd.Series, integrity_columns: Iterable[str] | None
) -> IntegrityEvidence:
    if integrity_columns is None:
        return IntegrityEvidence()
    cols: dict[str, float | None] = {}
    for col in integrity_columns:
        if col in feature_row.index:
            v = feature_row[col]
            try:
                f = float(v)
            except (TypeError, ValueError):
                f = None
            else:
                if not np.isfinite(f):
                    f = None
            cols[col] = f
    notes: list[str] = []
    overlap = cols.get("di_train_val_overlap")
    if overlap is not None and overlap > 0.01:
        notes.append(f"train/val overlap = {overlap:.3f} — direct leakage signal.")
    sat = cols.get("di_proxy_saturation")
    lockstep = cols.get("di_proxy_train_val_lockstep")
    if sat is not None and lockstep is not None and sat > 0.95 and lockstep > 0.95:
        notes.append(
            f"saturation={sat:.3f} with high train/val lockstep "
            f"({lockstep:.3f}) — strong leakage proxy."
        )
    return IntegrityEvidence(columns=cols, notes=notes)


def _rejected_hypotheses(
    diagnosis: HierarchicalDiagnosis,
) -> list[tuple[str, float]]:
    rejected: list[tuple[str, float]] = []
    if diagnosis.stage1.predicted == HEALTHY:
        for cls in PRIMARY_LABELS:
            if cls != HEALTHY:
                rejected.append((cls, float(diagnosis.class_probabilities.get(cls, 0.0))))
    elif diagnosis.stage1.predicted == FAULTY and diagnosis.stage2 is not None:
        rejected.append((HEALTHY, float(diagnosis.class_probabilities.get(HEALTHY, 0.0))))
        chosen = diagnosis.stage2.predicted
        loser_branch = OPT_GEN_RELATED if chosen == DATA_RELATED else DATA_RELATED
        for cls in PRIMARY_LABELS:
            if cls == HEALTHY:
                continue
            if to_stage2(cls) == loser_branch:
                rejected.append((cls, float(diagnosis.class_probabilities.get(cls, 0.0))))
    return sorted(rejected, key=lambda kv: -kv[1])


def _global_diagnostic_notes(d: HierarchicalDiagnosis) -> list[str]:
    notes: list[str] = []
    if d.final_confidence < 0.5:
        notes.append(
            f"low final confidence ({d.final_confidence:.3f}) — treat the diagnosis as tentative."
        )
    if d.stage1.confidence < 0.6:
        notes.append("Stage 1 (healthy/faulty) is itself uncertain.")
    if d.stage2 is not None and d.stage2.confidence < 0.6:
        notes.append("Stage 2 branch decision is uncertain — both branches are plausible.")
    if d.alternative_hypotheses:
        top_alt, top_alt_p = d.alternative_hypotheses[0]
        if top_alt_p > 0.25:
            notes.append(f"strong alternative: `{top_alt}` (p={top_alt_p:.3f}).")
    return notes


def build_evidence(
    *,
    diagnosis: HierarchicalDiagnosis,
    feature_row: pd.Series,
    cascade: HierarchicalCascade | None = None,
    integrity_columns: Iterable[str] | None = None,
    top_k_features_per_stage: int = 5,
) -> StructuredEvidence:
    stage_trace = _stage_trace_from_diagnosis(diagnosis)
    top_features: list[FeatureContribution] = []
    if cascade is not None:
        for stage in (cascade.stage1, cascade.stage2, cascade.stage3_data, cascade.stage3_opt):
            if stage is None:
                continue
            top_features.extend(
                _top_features_for_stage(
                    stage_name=stage.name,
                    model=stage.model,
                    feature_columns=stage.feature_columns,
                    feature_row=feature_row,
                    k=top_k_features_per_stage,
                )
            )
    curve_evidence = _build_curve_evidence(feature_row)
    integrity_evidence = _build_integrity_evidence(feature_row, integrity_columns)
    rejected = _rejected_hypotheses(diagnosis)
    notes = _global_diagnostic_notes(diagnosis)
    notes.extend(curve_evidence.notes)
    notes.extend(integrity_evidence.notes)
    return StructuredEvidence(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        run_id=str(diagnosis.run_id),
        final_class=diagnosis.final_class,
        final_confidence=float(diagnosis.final_confidence),
        class_probabilities={k: float(v) for k, v in diagnosis.class_probabilities.items()},
        alternative_hypotheses=[(c, float(p)) for c, p in diagnosis.alternative_hypotheses],
        rejected_hypotheses=rejected,
        stage_trace=stage_trace,
        top_features=top_features,
        curve_evidence=curve_evidence,
        integrity_evidence=integrity_evidence,
        diagnostic_notes=notes,
    )


def render_markdown(ev: StructuredEvidence) -> str:
    out: list[str] = []
    out.append(f"# Structured evidence for run `{ev.run_id}`")
    out.append("")
    out.append(f"- **final class:** `{ev.final_class}` (confidence {ev.final_confidence:.3f})")
    out.append(f"- generated at: {ev.generated_at}")
    out.append(f"- schema: `{ev.schema_version}`")
    out.append("")
    out.append("## Stage trace")
    out.append("")
    out.append("| stage | predicted | confidence | top classes |")
    out.append("|---|---|---:|---|")
    for s in ev.stage_trace:
        tops = ", ".join(f"`{c}`={p:.3f}" for c, p in s.top_classes)
        out.append(f"| `{s.stage_name}` | `{s.predicted}` | {s.confidence:.3f} | {tops} |")
    out.append("")
    out.append("## Class probabilities")
    out.append("")
    out.append("| class | probability |")
    out.append("|---|---:|")
    for cls, p in sorted(ev.class_probabilities.items(), key=lambda kv: -kv[1]):
        marker = " **← chosen**" if cls == ev.final_class else ""
        out.append(f"| `{cls}` | {p:.4f}{marker} |")
    out.append("")
    if ev.alternative_hypotheses:
        out.append("## Alternative hypotheses")
        out.append("")
        for cls, p in ev.alternative_hypotheses:
            out.append(f"- `{cls}` (p={p:.3f})")
        out.append("")
    if ev.rejected_hypotheses:
        out.append("## Rejected hypotheses")
        out.append("")
        out.append("These were ruled out by an earlier stage of the cascade:")
        out.append("")
        for cls, p in ev.rejected_hypotheses:
            out.append(f"- `{cls}` (p={p:.3f})")
        out.append("")
    if ev.top_features:
        out.append("## Top contributing features (per stage)")
        out.append("")
        out.append("| stage | feature | value | importance |")
        out.append("|---|---|---:|---:|")
        for f in ev.top_features:
            v = (
                "—"
                if f.value is None or (isinstance(f.value, float) and np.isnan(f.value))
                else f"{f.value:.4f}"
            )
            out.append(f"| `{f.source_stage}` | `{f.column}` | {v} | {f.importance:.4f} |")
        out.append("")
    out.append("## Training-curve evidence")
    out.append("")
    ce = ev.curve_evidence
    out.append("| key | value |")
    out.append("|---|---:|")
    for key, val in (
        ("n_epochs", ce.n_epochs),
        ("final_train_loss", ce.final_train_loss),
        ("final_val_loss", ce.final_val_loss),
        ("val_loss_min", ce.val_loss_min),
        ("val_loss_argmin_frac", ce.val_loss_argmin_frac),
        ("final_acc_gap", ce.final_acc_gap),
        ("max_acc_gap", ce.max_acc_gap),
        ("diverged", ce.diverged),
    ):
        out.append(f"| `{key}` | {('—' if val is None else val)} |")
    if ce.notes:
        out.append("")
        for n in ce.notes:
            out.append(f"- {n}")
    out.append("")
    if ev.integrity_evidence.columns:
        out.append("## Data-integrity evidence")
        out.append("")
        out.append("| key | value |")
        out.append("|---|---:|")
        for k, v in ev.integrity_evidence.columns.items():
            out.append(f"| `{k}` | {('—' if v is None else f'{v:.4f}')} |")
        if ev.integrity_evidence.notes:
            out.append("")
            for n in ev.integrity_evidence.notes:
                out.append(f"- {n}")
        out.append("")
    if ev.diagnostic_notes:
        out.append("## Diagnostic notes")
        out.append("")
        for n in ev.diagnostic_notes:
            out.append(f"- {n}")
        out.append("")
    return "\n".join(out)


def render_json(ev: StructuredEvidence) -> dict[str, Any]:
    return ev.to_dict()


def write_evidence(
    ev: StructuredEvidence,
    *,
    md_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if md_path is not None:
        p = Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_markdown(ev), encoding="utf-8")
        out["md"] = p
    if json_path is not None:
        import json as _json

        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(ev.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        out["json"] = p
    return out


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "CurveEvidence",
    "FeatureContribution",
    "IntegrityEvidence",
    "StageTraceEntry",
    "StructuredEvidence",
    "build_evidence",
    "classify_evidence_notes",
    "render_json",
    "render_markdown",
    "write_evidence",
]
