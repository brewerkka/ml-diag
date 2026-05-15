from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from ml_diag.labels import PRIMARY_LABELS

if TYPE_CHECKING:
    from ml_diag.diagnosis.oof_predictions import OOFPredictions


@dataclass(frozen=True)
class ConformalCalibrator:
    quantile: float
    alpha: float
    n_calibration: int
    classes: list[str]
    nonconformity_scores: np.ndarray
    score_method: Literal["lac", "aps"] = "lac"

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "quantile": float(self.quantile),
            "alpha": float(self.alpha),
            "target_coverage": float(1.0 - self.alpha),
            "n_calibration": int(self.n_calibration),
            "score_method": str(self.score_method),
            "classes": list(self.classes),
            "score_summary": {
                "min": (
                    float(np.min(self.nonconformity_scores))
                    if len(self.nonconformity_scores)
                    else 0.0
                ),
                "median": (
                    float(np.median(self.nonconformity_scores))
                    if len(self.nonconformity_scores)
                    else 0.0
                ),
                "p95": (
                    float(np.quantile(self.nonconformity_scores, 0.95))
                    if len(self.nonconformity_scores)
                    else 0.0
                ),
                "max": (
                    float(np.max(self.nonconformity_scores))
                    if len(self.nonconformity_scores)
                    else 0.0
                ),
            },
        }


@dataclass(frozen=True)
class ConformalResult:
    run_id: str
    prediction_set: list[str]
    set_size: int
    point_prediction: str
    point_confidence: float
    is_abstained: bool
    nonconformity: float
    proba: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": str(self.run_id),
            "prediction_set": list(self.prediction_set),
            "set_size": int(self.set_size),
            "point_prediction": str(self.point_prediction),
            "point_confidence": float(self.point_confidence),
            "is_abstained": bool(self.is_abstained),
            "nonconformity": float(self.nonconformity),
            "proba": {k: float(v) for k, v in self.proba.items()},
        }


def _lac_scores(
    proba: pd.DataFrame,
    y: pd.Series,
    classes: Sequence[str],
) -> np.ndarray:
    cls_to_col = {str(c): i for i, c in enumerate(classes)}
    P = proba.reindex(columns=list(classes)).to_numpy(dtype=float)
    n = P.shape[0]
    s = np.empty(n, dtype=float)
    y_arr = y.astype(str).to_numpy()
    for i in range(n):
        col = cls_to_col.get(str(y_arr[i]))
        if col is None:
            s[i] = 1.0
        else:
            s[i] = 1.0 - float(P[i, col])
    return s


def _conformal_quantile(s: np.ndarray, alpha: float) -> float:
    n = len(s)
    if n == 0:
        return 1.0
    level = float(np.ceil((n + 1) * (1 - float(alpha))) / n)
    level = min(level, 1.0)
    return float(np.quantile(s, level, method="higher"))


def calibrate_split_conformal(
    *,
    proba_oof: pd.DataFrame,
    y_oof: pd.Series,
    alpha: float = 0.05,
    score_method: Literal["lac", "aps"] = "lac",
) -> ConformalCalibrator:
    if score_method != "lac":
        raise NotImplementedError(
            "Only LAC nonconformity is implemented; APS is documented as a future extension."
        )
    classes = list(PRIMARY_LABELS)
    proba = proba_oof.reindex(columns=classes, fill_value=0.0)
    y = y_oof.reindex(proba.index)
    if y.isna().any():
        missing = int(y.isna().sum())
        raise ValueError(
            f"y_oof has {missing} rows missing relative to proba_oof; indexes must match by run_id"
        )
    s = _lac_scores(proba, y, classes)
    q_hat = _conformal_quantile(s, float(alpha))
    return ConformalCalibrator(
        quantile=float(q_hat),
        alpha=float(alpha),
        n_calibration=int(len(proba)),
        classes=classes,
        nonconformity_scores=s,
        score_method=score_method,
    )


def predict_with_conformal(
    *,
    proba_test: pd.DataFrame,
    calibrator: ConformalCalibrator,
    run_ids: list[str] | None = None,
) -> list[ConformalResult]:
    classes = list(calibrator.classes)
    P = proba_test.reindex(columns=classes, fill_value=0.0).to_numpy(dtype=float)
    rids = list(run_ids) if run_ids is not None else list(proba_test.index.astype(str))
    if len(rids) != len(P):
        raise ValueError(f"run_ids length {len(rids)} does not match proba_test rows {len(P)}")
    threshold = 1.0 - float(calibrator.quantile)
    out: list[ConformalResult] = []
    for i, rid in enumerate(rids):
        row = P[i]
        in_set_mask = row >= threshold - 1e-12
        if not in_set_mask.any():
            best = int(np.argmax(row))
            in_set_mask = np.zeros_like(in_set_mask)
            in_set_mask[best] = True
        prediction_set = [classes[j] for j in range(len(classes)) if in_set_mask[j]]
        argmax_idx = int(np.argmax(row))
        point_pred = classes[argmax_idx]
        point_conf = float(row[argmax_idx])
        nonconf = 1.0 - point_conf
        out.append(
            ConformalResult(
                run_id=str(rid),
                prediction_set=prediction_set,
                set_size=int(len(prediction_set)),
                point_prediction=point_pred,
                point_confidence=point_conf,
                is_abstained=(len(prediction_set) > 1),
                nonconformity=float(nonconf),
                proba={classes[j]: float(row[j]) for j in range(len(classes))},
            )
        )
    return out


def evaluate_conformal(
    *,
    results: Sequence[ConformalResult],
    y_test: pd.Series,
) -> dict[str, Any]:
    if len(results) != len(y_test):
        raise ValueError(f"results / y_test length mismatch: {len(results)} vs {len(y_test)}")
    y_arr = y_test.astype(str).to_numpy()
    n = len(results)
    in_set = np.array([y_arr[i] in r.prediction_set for i, r in enumerate(results)])
    abstained = np.array([r.is_abstained for r in results])
    set_sizes = np.array([r.set_size for r in results])
    point_correct = np.array([r.point_prediction == y_arr[i] for i, r in enumerate(results)])
    confident_mask = ~abstained
    n_confident = int(confident_mask.sum())
    cond_acc = float(point_correct[confident_mask].mean()) if n_confident > 0 else float("nan")
    per_class_coverage: dict[str, float] = {}
    per_class_abstain: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    for cls in PRIMARY_LABELS:
        m = y_arr == cls
        per_class_n[cls] = int(m.sum())
        if m.any():
            per_class_coverage[cls] = float(in_set[m].mean())
            per_class_abstain[cls] = float(abstained[m].mean())
        else:
            per_class_coverage[cls] = float("nan")
            per_class_abstain[cls] = float("nan")
    return {
        "n_test": int(n),
        "empirical_coverage": float(in_set.mean()),
        "target_coverage": (float(1.0 - results[0].nonconformity) if False else None),
        "coverage_gap": None,
        "abstain_rate": float(abstained.mean()),
        "n_abstained": int(abstained.sum()),
        "n_confident": n_confident,
        "conditional_accuracy": cond_acc,
        "average_set_size": float(set_sizes.mean()),
        "expected_set_size": float(set_sizes.mean()),
        "per_class_n": per_class_n,
        "per_class_coverage": per_class_coverage,
        "per_class_abstain_rate": per_class_abstain,
    }


def compute_meta_oof_probabilities(
    *,
    oof: OOFPredictions,
    y_train: pd.Series,
    classifier: str = "gbm",
    seed: int = 0,
    n_folds: int = 5,
) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedKFold

    from ml_diag.diagnosis.stacking_resolver import (
        _build_classifier,
        featurize,
    )

    X_meta = featurize(
        flat_proba=oof.flat_proba,
        cascade_proba=oof.cascade_proba,
        cascade_stage_probs=oof.cascade_stage_probs,
        arbitrator_label_probs=oof.arbitrator_label_probs,
        arbitrator_triggered=oof.arbitrator_triggered,
        arbitrator_confidence=oof.arbitrator_confidence,
    )
    y = y_train.reindex(X_meta.index)
    if y.isna().any():
        missing = int(y.isna().sum())
        raise ValueError(f"y_train has {missing} rows missing for the OOF index")
    classes = list(PRIMARY_LABELS)
    proba_oof_meta = pd.DataFrame(
        np.zeros((len(X_meta), len(classes)), dtype=float),
        index=X_meta.index,
        columns=classes,
    )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (tr, va) in enumerate(skf.split(X_meta, y)):
        clf = _build_classifier(classifier, seed=seed)
        clf.fit(X_meta.iloc[tr].values, y.iloc[tr].values)
        raw = clf.predict_proba(X_meta.iloc[va].values)
        clf_classes = [str(c) for c in clf.classes_]
        for k_local, cls in enumerate(clf_classes):
            if cls in classes:
                proba_oof_meta.iloc[va, classes.index(cls)] = raw[:, k_local]
    sums = proba_oof_meta.sum(axis=1).replace(0.0, 1.0)
    proba_oof_meta = proba_oof_meta.div(sums, axis=0)
    return proba_oof_meta


__all__ = [
    "ConformalCalibrator",
    "ConformalResult",
    "calibrate_split_conformal",
    "compute_meta_oof_probabilities",
    "evaluate_conformal",
    "predict_with_conformal",
]
