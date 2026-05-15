from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ml_diag.labels import PRIMARY_LABELS

if TYPE_CHECKING:
    from ml_diag.diagnosis.conformal_layer import ConformalCalibrator
    from ml_diag.diagnosis.stacking_resolver import StackingMetaModel
    from ml_diag.evaluation.explanation import StructuredEvidence
    from ml_diag.models.flat_baseline import FlatBaselineResult
    from ml_diag.models.inference import HierarchicalCascade

POLICY_NAMES = (
    "agreement_or_cascade",
    "agreement_or_flat",
    "confidence_weighted",
    "llm_arbitrate",
    "stacking",
    "stacking_with_conformal",
)

PolicyName = Literal[
    "agreement_or_cascade",
    "agreement_or_flat",
    "confidence_weighted",
    "llm_arbitrate",
    "stacking",
    "stacking_with_conformal",
]


@dataclass(frozen=True)
class HybridDiagnosis:
    final_label: str
    final_confidence: float
    flat_label: str
    flat_confidence: float
    cascade_label: str
    cascade_confidence: float
    agreement: bool
    resolution_path: str
    flat_proba: dict[str, float] = field(default_factory=dict)
    cascade_proba: dict[str, float] = field(default_factory=dict)
    arbitration_trigger: str | None = None
    arbitration_chosen_source: str | None = None
    arbitration_reasoning: str | None = None
    arbitration_backend: str | None = None
    arbitration_cached: bool | None = None
    stacking_probabilities: dict[str, float] | None = None
    stacking_top_features: list[tuple[str, float]] | None = None
    conformal_prediction_set: list[str] | None = None
    conformal_set_size: int | None = None
    conformal_is_abstained: bool | None = None
    conformal_nonconformity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_label": str(self.final_label),
            "final_confidence": float(self.final_confidence),
            "flat_label": str(self.flat_label),
            "flat_confidence": float(self.flat_confidence),
            "cascade_label": str(self.cascade_label),
            "cascade_confidence": float(self.cascade_confidence),
            "agreement": bool(self.agreement),
            "resolution_path": str(self.resolution_path),
            "flat_proba": {k: float(v) for k, v in self.flat_proba.items()},
            "cascade_proba": {k: float(v) for k, v in self.cascade_proba.items()},
            "arbitration_trigger": self.arbitration_trigger,
            "arbitration_chosen_source": self.arbitration_chosen_source,
            "arbitration_reasoning": self.arbitration_reasoning,
            "arbitration_backend": self.arbitration_backend,
            "arbitration_cached": self.arbitration_cached,
            "stacking_probabilities": (
                None
                if self.stacking_probabilities is None
                else {k: float(v) for k, v in self.stacking_probabilities.items()}
            ),
            "stacking_top_features": (
                None
                if self.stacking_top_features is None
                else [[name, float(score)] for name, score in self.stacking_top_features]
            ),
            "conformal_prediction_set": (
                None
                if self.conformal_prediction_set is None
                else [str(c) for c in self.conformal_prediction_set]
            ),
            "conformal_set_size": (
                None if self.conformal_set_size is None else int(self.conformal_set_size)
            ),
            "conformal_is_abstained": (
                None if self.conformal_is_abstained is None else bool(self.conformal_is_abstained)
            ),
            "conformal_nonconformity": (
                None
                if self.conformal_nonconformity is None
                else float(self.conformal_nonconformity)
            ),
        }


@dataclass(frozen=True)
class HybridResolverConfig:
    policy: PolicyName
    alpha: float = 0.5
    arbitrator_backend: str = "auto"
    arbitrator_low_conf_trigger: float = 0.65
    arbitrator_cache: Path = field(default_factory=lambda: Path(".cache/arbitrator"))
    stacking_meta_model: StackingMetaModel | None = None
    stacking_oof_path: Path | None = None
    stacking_classifier: str = "lr"
    conformal_calibrator: ConformalCalibrator | None = None
    conformal_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.policy not in POLICY_NAMES:
            raise ValueError(f"Unknown policy {self.policy!r}; expected one of {POLICY_NAMES}")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha!r}")
        if not 0.0 <= float(self.arbitrator_low_conf_trigger) <= 1.0:
            raise ValueError(
                "arbitrator_low_conf_trigger must be in [0, 1]; "
                f"got {self.arbitrator_low_conf_trigger!r}"
            )
        if self.stacking_classifier not in ("lr", "gbm"):
            raise ValueError(
                f"stacking_classifier must be 'lr' or 'gbm'; got {self.stacking_classifier!r}"
            )


def _argmax_of(prob_row: pd.Series) -> tuple[str, float]:
    cls = str(prob_row.idxmax())
    return cls, float(prob_row.max())


def _validate_proba_frame(df: pd.DataFrame, *, name: str, atol: float = 1e-3) -> None:
    expected_cols = list(PRIMARY_LABELS)
    if list(df.columns) != expected_cols:
        raise ValueError(f"{name}: column order must match PRIMARY_LABELS; got {list(df.columns)}")
    if df.isna().any().any():
        bad = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name}: NaN values present in columns {bad}")
    sums = df.sum(axis=1).to_numpy(dtype=float)
    if not np.allclose(sums, 1.0, atol=atol):
        max_dev = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"{name}: row sums deviate from 1 by max {max_dev:.4g} (>{atol})")


def cascade_marginal_proba(
    cascade: HierarchicalCascade,
    X: pd.DataFrame,
) -> pd.DataFrame:
    from ml_diag.models.inference import diagnose_batch

    diags = diagnose_batch(cascade, X)
    cols = list(PRIMARY_LABELS)
    arr = np.zeros((len(diags), len(cols)), dtype=float)
    for i, d in enumerate(diags):
        for j, cls in enumerate(cols):
            arr[i, j] = float(d.class_probabilities.get(cls, 0.0))
    df = pd.DataFrame(arr, index=X.index, columns=cols)
    sums = df.sum(axis=1)
    nonzero = sums > 0
    if (~nonzero).any():
        df.loc[~nonzero, :] = 1.0 / len(cols)
    else:
        df = df.div(sums, axis=0)
    _validate_proba_frame(df, name="cascade_marginal_proba")
    return df


def flat_proba_aligned(
    flat_result: FlatBaselineResult,
    X: pd.DataFrame,
) -> pd.DataFrame:
    from ml_diag.utils import (
        align_features_to_schema,
        ensure_feature_matrix,
    )

    if flat_result.feature_columns:
        X_aligned = align_features_to_schema(X, flat_result.feature_columns)
    else:
        X_aligned = X
    arr = ensure_feature_matrix(X_aligned)
    if not hasattr(flat_result.model, "predict_proba"):
        raise RuntimeError(
            "Flat model does not expose predict_proba; hybrid resolvers require probabilities."
        )
    raw = np.asarray(flat_result.model.predict_proba(arr), dtype=float)
    cols = list(PRIMARY_LABELS)
    df = pd.DataFrame(0.0, index=X.index, columns=cols)
    if raw.ndim != 2 or raw.shape[0] != len(X):
        raise RuntimeError(
            f"Flat predict_proba returned shape {raw.shape}, expected ({len(X)}, k)."
        )
    for k, cls in enumerate(flat_result.classes):
        cls_str = str(cls)
        if cls_str in cols:
            df[cls_str] = raw[:, k]
    sums = df.sum(axis=1)
    nonzero = sums > 0
    if (~nonzero).any():
        df.loc[~nonzero, :] = 1.0 / len(cols)
    else:
        df = df.div(sums, axis=0)
    _validate_proba_frame(df, name="flat_proba_aligned")
    return df


def _row_to_dict(row: pd.Series) -> dict[str, float]:
    return {str(k): float(v) for k, v in row.items()}


def resolve_agreement_or_cascade(
    flat_proba: pd.Series,
    cascade_proba: pd.Series,
    config: HybridResolverConfig,
) -> HybridDiagnosis:
    flat_lbl, flat_conf = _argmax_of(flat_proba)
    casc_lbl, casc_conf = _argmax_of(cascade_proba)
    agree = flat_lbl == casc_lbl
    if agree:
        path = "agreement"
        final_lbl = flat_lbl
        final_conf = float(0.5 * flat_conf + 0.5 * casc_conf)
    else:
        path = "cascade_picked"
        final_lbl = casc_lbl
        final_conf = casc_conf
    return HybridDiagnosis(
        final_label=final_lbl,
        final_confidence=float(final_conf),
        flat_label=flat_lbl,
        flat_confidence=float(flat_conf),
        cascade_label=casc_lbl,
        cascade_confidence=float(casc_conf),
        agreement=agree,
        resolution_path=path,
        flat_proba=_row_to_dict(flat_proba),
        cascade_proba=_row_to_dict(cascade_proba),
    )


def resolve_agreement_or_flat(
    flat_proba: pd.Series,
    cascade_proba: pd.Series,
    config: HybridResolverConfig,
) -> HybridDiagnosis:
    flat_lbl, flat_conf = _argmax_of(flat_proba)
    casc_lbl, casc_conf = _argmax_of(cascade_proba)
    agree = flat_lbl == casc_lbl
    if agree:
        path = "agreement"
        final_lbl = flat_lbl
        final_conf = float(0.5 * flat_conf + 0.5 * casc_conf)
    else:
        path = "flat_picked"
        final_lbl = flat_lbl
        final_conf = flat_conf
    return HybridDiagnosis(
        final_label=final_lbl,
        final_confidence=float(final_conf),
        flat_label=flat_lbl,
        flat_confidence=float(flat_conf),
        cascade_label=casc_lbl,
        cascade_confidence=float(casc_conf),
        agreement=agree,
        resolution_path=path,
        flat_proba=_row_to_dict(flat_proba),
        cascade_proba=_row_to_dict(cascade_proba),
    )


def resolve_confidence_weighted(
    flat_proba: pd.Series,
    cascade_proba: pd.Series,
    config: HybridResolverConfig,
) -> HybridDiagnosis:
    cols = list(PRIMARY_LABELS)
    p_flat = np.asarray([float(flat_proba.get(c, 0.0)) for c in cols], dtype=float)
    p_casc = np.asarray([float(cascade_proba.get(c, 0.0)) for c in cols], dtype=float)
    a = float(config.alpha)
    p = a * p_flat + (1.0 - a) * p_casc
    idx = int(np.argmax(p))
    final_lbl = cols[idx]
    final_conf = float(p[idx])
    flat_lbl = cols[int(np.argmax(p_flat))]
    casc_lbl = cols[int(np.argmax(p_casc))]
    agree = flat_lbl == casc_lbl
    return HybridDiagnosis(
        final_label=final_lbl,
        final_confidence=final_conf,
        flat_label=flat_lbl,
        flat_confidence=float(p_flat[int(np.argmax(p_flat))]),
        cascade_label=casc_lbl,
        cascade_confidence=float(p_casc[int(np.argmax(p_casc))]),
        agreement=agree,
        resolution_path="weighted",
        flat_proba=_row_to_dict(flat_proba),
        cascade_proba=_row_to_dict(cascade_proba),
    )


def resolve_llm_arbitrate(
    flat_proba: pd.Series,
    cascade_proba: pd.Series,
    config: HybridResolverConfig,
    *,
    run_id: str | None = None,
    evidence: StructuredEvidence | None = None,
) -> HybridDiagnosis:
    flat_lbl, flat_conf = _argmax_of(flat_proba)
    casc_lbl, casc_conf = _argmax_of(cascade_proba)
    agree = flat_lbl == casc_lbl
    disagreement = not agree
    low_conf = min(flat_conf, casc_conf) < float(config.arbitrator_low_conf_trigger)
    needs_arbitration = disagreement or low_conf
    base_kwargs: dict[str, Any] = {
        "flat_label": flat_lbl,
        "flat_confidence": float(flat_conf),
        "cascade_label": casc_lbl,
        "cascade_confidence": float(casc_conf),
        "agreement": agree,
        "flat_proba": _row_to_dict(flat_proba),
        "cascade_proba": _row_to_dict(cascade_proba),
    }
    if not needs_arbitration:
        return HybridDiagnosis(
            **base_kwargs,
            final_label=flat_lbl,
            final_confidence=float(0.5 * flat_conf + 0.5 * casc_conf),
            resolution_path="agreement",
            arbitration_trigger="agreement",
            arbitration_chosen_source=None,
            arbitration_reasoning=None,
            arbitration_backend=None,
            arbitration_cached=None,
        )
    trigger = "disagreement" if disagreement else "low_conf_agreement"
    if run_id is None or evidence is None:
        return HybridDiagnosis(
            **base_kwargs,
            final_label=flat_lbl,
            final_confidence=float(flat_conf),
            resolution_path="llm_skipped",
            arbitration_trigger=trigger,
            arbitration_chosen_source="flat",
            arbitration_reasoning="evidence missing → deterministic flat fallback",
            arbitration_backend="template",
            arbitration_cached=False,
        )
    from ml_diag.diagnosis.arbitrator import (
        ArbitratorConfig,
        arbitrate_one,
    )

    arb_cfg = ArbitratorConfig(
        backend=str(config.arbitrator_backend),                          
        cache_path=Path(config.arbitrator_cache),
    )
    decision = arbitrate_one(
        run_id=str(run_id),
        flat_label=flat_lbl,
        flat_proba=_row_to_dict(flat_proba),
        cascade_label=casc_lbl,
        cascade_proba=_row_to_dict(cascade_proba),
        evidence=evidence,
        config=arb_cfg,
    )
    return HybridDiagnosis(
        **base_kwargs,
        final_label=str(decision.chosen_label),
        final_confidence=float(decision.confidence),
        resolution_path="llm_arbitrated",
        arbitration_trigger=trigger,
        arbitration_chosen_source=str(decision.chosen_source),
        arbitration_reasoning=str(decision.reasoning),
        arbitration_backend=str(decision.backend),
        arbitration_cached=bool(decision.cached),
    )


_RESOLVERS = {
    "agreement_or_cascade": resolve_agreement_or_cascade,
    "agreement_or_flat": resolve_agreement_or_flat,
    "confidence_weighted": resolve_confidence_weighted,
}


def resolve(
    flat_proba: pd.Series,
    cascade_proba: pd.Series,
    config: HybridResolverConfig,
    *,
    run_id: str | None = None,
    evidence: StructuredEvidence | None = None,
) -> HybridDiagnosis:
    if config.policy == "llm_arbitrate":
        return resolve_llm_arbitrate(
            flat_proba,
            cascade_proba,
            config,
            run_id=run_id,
            evidence=evidence,
        )
    fn = _RESOLVERS[config.policy]
    return fn(flat_proba, cascade_proba, config)


def resolve_batch(
    *,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    config: HybridResolverConfig,
    evidence_by_run: Mapping[str, StructuredEvidence] | None = None,
) -> list[HybridDiagnosis]:
    if list(flat_proba.columns) != list(PRIMARY_LABELS):
        raise ValueError("flat_proba columns must equal PRIMARY_LABELS")
    if list(cascade_proba.columns) != list(PRIMARY_LABELS):
        raise ValueError("cascade_proba columns must equal PRIMARY_LABELS")
    if not flat_proba.index.equals(cascade_proba.index):
        cascade_proba = cascade_proba.reindex(flat_proba.index)
    out: list[HybridDiagnosis] = []
    for rid in flat_proba.index:
        ev = None
        if evidence_by_run is not None:
            ev = evidence_by_run.get(str(rid))
        out.append(
            resolve(
                flat_proba.loc[rid],
                cascade_proba.loc[rid],
                config,
                run_id=str(rid),
                evidence=ev,
            )
        )
    return out


def build_stacking_diagnoses(
    *,
    config: HybridResolverConfig,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    cascade_stage_probs: pd.DataFrame,
    arbitrator_label_probs: pd.DataFrame,
    arbitrator_triggered: pd.Series,
    arbitrator_confidence: pd.Series,
) -> list[HybridDiagnosis]:
    from ml_diag.diagnosis.stacking_resolver import stacking_predict

    if config.stacking_meta_model is None:
        raise ValueError(
            "build_stacking_diagnoses called without a trained stacking_meta_model on the config"
        )
    pred, proba_df = stacking_predict(
        config.stacking_meta_model,
        flat_proba=flat_proba,
        cascade_proba=cascade_proba,
        cascade_stage_probs=cascade_stage_probs,
        arbitrator_label_probs=arbitrator_label_probs,
        arbitrator_triggered=arbitrator_triggered,
        arbitrator_confidence=arbitrator_confidence,
    )
    flat_argmax = flat_proba.idxmax(axis=1)
    casc_argmax = cascade_proba.idxmax(axis=1)
    flat_max = flat_proba.max(axis=1)
    casc_max = cascade_proba.max(axis=1)
    top_features = sorted(
        config.stacking_meta_model.feature_importances.items(),
        key=lambda kv: -float(kv[1]),
    )[:3]
    top_features_pairs = [(str(k), float(v)) for k, v in top_features]
    diags: list[HybridDiagnosis] = []
    for rid in flat_proba.index:
        f_lbl = str(flat_argmax.loc[rid])
        c_lbl = str(casc_argmax.loc[rid])
        diags.append(
            HybridDiagnosis(
                final_label=str(pred.loc[rid]),
                final_confidence=float(proba_df.loc[rid].max()),
                flat_label=f_lbl,
                flat_confidence=float(flat_max.loc[rid]),
                cascade_label=c_lbl,
                cascade_confidence=float(casc_max.loc[rid]),
                agreement=(f_lbl == c_lbl),
                resolution_path="stacking",
                flat_proba={c: float(flat_proba.at[rid, c]) for c in flat_proba.columns},
                cascade_proba={c: float(cascade_proba.at[rid, c]) for c in cascade_proba.columns},
                stacking_probabilities={c: float(proba_df.at[rid, c]) for c in proba_df.columns},
                stacking_top_features=list(top_features_pairs),
            )
        )
    return diags


def build_stacking_with_conformal_diagnoses(
    *,
    config: HybridResolverConfig,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    cascade_stage_probs: pd.DataFrame,
    arbitrator_label_probs: pd.DataFrame,
    arbitrator_triggered: pd.Series,
    arbitrator_confidence: pd.Series,
) -> list[HybridDiagnosis]:
    from ml_diag.diagnosis.conformal_layer import predict_with_conformal
    from ml_diag.diagnosis.stacking_resolver import stacking_predict

    if config.stacking_meta_model is None:
        raise ValueError(
            "build_stacking_with_conformal_diagnoses called without a "
            "trained stacking_meta_model on the config"
        )
    if config.conformal_calibrator is None:
        raise ValueError(
            "build_stacking_with_conformal_diagnoses called without a "
            "calibrated conformal_calibrator on the config"
        )
    pred, proba_df = stacking_predict(
        config.stacking_meta_model,
        flat_proba=flat_proba,
        cascade_proba=cascade_proba,
        cascade_stage_probs=cascade_stage_probs,
        arbitrator_label_probs=arbitrator_label_probs,
        arbitrator_triggered=arbitrator_triggered,
        arbitrator_confidence=arbitrator_confidence,
    )
    flat_argmax = flat_proba.idxmax(axis=1)
    casc_argmax = cascade_proba.idxmax(axis=1)
    flat_max = flat_proba.max(axis=1)
    casc_max = cascade_proba.max(axis=1)
    conformal_results = predict_with_conformal(
        proba_test=proba_df,
        calibrator=config.conformal_calibrator,
        run_ids=[str(r) for r in proba_df.index],
    )
    conformal_by_id = {r.run_id: r for r in conformal_results}
    top_features = sorted(
        config.stacking_meta_model.feature_importances.items(),
        key=lambda kv: -float(kv[1]),
    )[:3]
    top_features_pairs = [(str(k), float(v)) for k, v in top_features]
    diags: list[HybridDiagnosis] = []
    for rid in flat_proba.index:
        f_lbl = str(flat_argmax.loc[rid])
        c_lbl = str(casc_argmax.loc[rid])
        cf = conformal_by_id.get(str(rid))
        is_abstained = bool(cf.is_abstained) if cf is not None else False
        diags.append(
            HybridDiagnosis(
                final_label=str(pred.loc[rid]),
                final_confidence=float(proba_df.loc[rid].max()),
                flat_label=f_lbl,
                flat_confidence=float(flat_max.loc[rid]),
                cascade_label=c_lbl,
                cascade_confidence=float(casc_max.loc[rid]),
                agreement=(f_lbl == c_lbl),
                resolution_path=("conformal_abstained" if is_abstained else "stacking"),
                flat_proba={c: float(flat_proba.at[rid, c]) for c in flat_proba.columns},
                cascade_proba={c: float(cascade_proba.at[rid, c]) for c in cascade_proba.columns},
                stacking_probabilities={c: float(proba_df.at[rid, c]) for c in proba_df.columns},
                stacking_top_features=list(top_features_pairs),
                conformal_prediction_set=(list(cf.prediction_set) if cf else None),
                conformal_set_size=(int(cf.set_size) if cf else None),
                conformal_is_abstained=is_abstained,
                conformal_nonconformity=(float(cf.nonconformity) if cf else None),
            )
        )
    return diags


__all__ = [
    "HybridDiagnosis",
    "HybridResolverConfig",
    "POLICY_NAMES",
    "PolicyName",
    "build_stacking_diagnoses",
    "build_stacking_with_conformal_diagnoses",
    "cascade_marginal_proba",
    "flat_proba_aligned",
    "resolve",
    "resolve_agreement_or_cascade",
    "resolve_agreement_or_flat",
    "resolve_batch",
    "resolve_confidence_weighted",
    "resolve_llm_arbitrate",
]
