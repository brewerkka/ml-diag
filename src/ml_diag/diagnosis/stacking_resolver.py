from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from ml_diag.labels import PRIMARY_LABELS

_FLAT_COLS = [f"flat_p_{c}" for c in PRIMARY_LABELS]

_CASC_COLS = [f"casc_p_{c}" for c in PRIMARY_LABELS]

_STAGE_COLS = [
    "stage1_p_healthy",
    "stage1_p_faulty",
    "stage2_p_data",
    "stage2_p_optgen",
    "stage3_p_leakage",
    "stage3_p_label_noise",
    "stage3_p_overfitting",
    "stage3_p_underfitting",
    "stage3_p_instability",
]

_DERIVED_COLS = ["agreement_bit", "min_max_proba", "max_max_proba", "kl_flat_cascade"]

_ARB_COLS = [f"arb_p_{c}" for c in PRIMARY_LABELS]

_ARB_META_COLS = ["arb_triggered_bit", "arb_confidence"]

META_FEATURES: list[str] = (
    _FLAT_COLS + _CASC_COLS + _STAGE_COLS + _DERIVED_COLS + _ARB_COLS + _ARB_META_COLS
)


@dataclass(frozen=True)
class StackingMetaModel:
    classifier: Any
    classifier_name: str
    feature_columns: list[str]
    feature_means: dict[str, float]
    classes_: list[str]
    cv_score_macro_f1: float
    feature_importances: dict[str, float]
    n_train_rows: int
    n_features: int

    def to_summary_dict(self) -> dict[str, Any]:
        top = sorted(
            self.feature_importances.items(),
            key=lambda kv: -float(kv[1]),
        )[:10]
        return {
            "classifier": self.classifier_name,
            "cv_score_macro_f1": float(self.cv_score_macro_f1),
            "feature_importances_top10": [[name, float(score)] for name, score in top],
            "n_train_rows": int(self.n_train_rows),
            "n_features": int(self.n_features),
        }


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=1)


def featurize(
    *,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    cascade_stage_probs: pd.DataFrame,
    arbitrator_label_probs: pd.DataFrame,
    arbitrator_triggered: pd.Series,
    arbitrator_confidence: pd.Series,
    feature_means: dict[str, float] | None = None,
) -> pd.DataFrame:
    primary = list(PRIMARY_LABELS)
    fp = flat_proba.reindex(columns=primary)
    cp = cascade_proba.reindex(columns=primary)
    sp = cascade_stage_probs.reindex(columns=_STAGE_COLS)
    ap = arbitrator_label_probs.reindex(columns=primary).fillna(0.0)
    at = arbitrator_triggered.astype(float)
    ac = arbitrator_confidence.astype(float)
    flat_max = fp.max(axis=1).values
    casc_max = cp.max(axis=1).values
    flat_argmax = fp.idxmax(axis=1)
    casc_argmax = cp.idxmax(axis=1)
    agreement_bit = (flat_argmax.values == casc_argmax.values).astype(float)
    min_max = np.minimum(flat_max, casc_max)
    max_max = np.maximum(flat_max, casc_max)
    kl = _kl_divergence(fp.values, cp.values)
    out = pd.DataFrame(index=fp.index)
    for col, vals in zip(_FLAT_COLS, fp.values.T):
        out[col] = vals
    for col, vals in zip(_CASC_COLS, cp.values.T):
        out[col] = vals
    for col in _STAGE_COLS:
        out[col] = sp[col].values
    out["agreement_bit"] = agreement_bit
    out["min_max_proba"] = min_max
    out["max_max_proba"] = max_max
    out["kl_flat_cascade"] = kl
    for col, vals in zip(_ARB_COLS, ap.values.T):
        out[col] = vals
    out["arb_triggered_bit"] = at.values
    out["arb_confidence"] = ac.values
    if feature_means is None:
        means = {
            col: float(np.nanmean(out[col].values)) if out[col].notna().any() else 0.0
            for col in out.columns
        }
    else:
        means = dict(feature_means)
    for col in out.columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(means.get(col, 0.0))
    if list(out.columns) != META_FEATURES:
        out = out[META_FEATURES]
    return out


def _build_classifier(
    name: str,
    *,
    seed: int,
):
    if name == "lr":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
        )
    if name == "gbm":
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=seed,
        )
    raise ValueError(f"Unknown stacking classifier {name!r}; expected 'lr' or 'gbm'")


def _feature_importance_vector(clf, classes_: list[str]) -> np.ndarray:
    coef = getattr(clf, "coef_", None)
    if coef is not None:
        return np.abs(np.asarray(coef, dtype=float)).sum(axis=0)
    fi = getattr(clf, "feature_importances_", None)
    if fi is not None:
        return np.asarray(fi, dtype=float)
    raise RuntimeError("Classifier does not expose coef_ or feature_importances_")


def train_stacking_meta(
    oof,
    y_train: pd.Series,
    *,
    classifier: Literal["lr", "gbm"] = "lr",
    seed: int = 0,
    n_cv_folds: int = 5,
) -> StackingMetaModel:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    from ml_diag.diagnosis.oof_predictions import OOFPredictions              

    X_meta = featurize(
        flat_proba=oof.flat_proba,
        cascade_proba=oof.cascade_proba,
        cascade_stage_probs=oof.cascade_stage_probs,
        arbitrator_label_probs=oof.arbitrator_label_probs,
        arbitrator_triggered=oof.arbitrator_triggered,
        arbitrator_confidence=oof.arbitrator_confidence,
    )
    feature_means = {col: float(X_meta[col].mean()) for col in X_meta.columns}
    y = y_train.reindex(X_meta.index)
    if y.isna().any():
        missing = int(y.isna().sum())
        raise ValueError(
            f"y_train has {missing} rows missing for the OOF index; "
            "are train/y derived from the same canonical split?"
        )
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    fold_scores: list[float] = []
    for fold, (tr, va) in enumerate(skf.split(X_meta, y)):
        clf = _build_classifier(classifier, seed=seed)
        clf.fit(X_meta.iloc[tr].values, y.iloc[tr].values)
        pred = clf.predict(X_meta.iloc[va].values)
        fold_scores.append(
            float(
                f1_score(
                    y.iloc[va].values,
                    pred,
                    average="macro",
                    labels=list(PRIMARY_LABELS),
                    zero_division=0,
                )
            )
        )
    cv_macro_f1 = float(np.mean(fold_scores))
    clf = _build_classifier(classifier, seed=seed)
    clf.fit(X_meta.values, y.values)
    classes_ = [str(c) for c in clf.classes_]
    importances_arr = _feature_importance_vector(clf, classes_)
    importances = {col: float(importances_arr[i]) for i, col in enumerate(X_meta.columns)}
    return StackingMetaModel(
        classifier=clf,
        classifier_name=str(classifier),
        feature_columns=list(X_meta.columns),
        feature_means=feature_means,
        classes_=classes_,
        cv_score_macro_f1=cv_macro_f1,
        feature_importances=importances,
        n_train_rows=int(len(X_meta)),
        n_features=int(len(X_meta.columns)),
    )


def stacking_predict(
    meta: StackingMetaModel,
    *,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    cascade_stage_probs: pd.DataFrame,
    arbitrator_label_probs: pd.DataFrame,
    arbitrator_triggered: pd.Series,
    arbitrator_confidence: pd.Series,
) -> tuple[pd.Series, pd.DataFrame]:
    X_meta = featurize(
        flat_proba=flat_proba,
        cascade_proba=cascade_proba,
        cascade_stage_probs=cascade_stage_probs,
        arbitrator_label_probs=arbitrator_label_probs,
        arbitrator_triggered=arbitrator_triggered,
        arbitrator_confidence=arbitrator_confidence,
        feature_means=meta.feature_means,
    )
    X_meta = X_meta.reindex(columns=meta.feature_columns)
    proba = meta.classifier.predict_proba(X_meta.values)
    proba_df = pd.DataFrame(
        proba,
        index=X_meta.index,
        columns=list(meta.classes_),
    )
    proba_df = proba_df.reindex(columns=list(PRIMARY_LABELS), fill_value=0.0)
    pred = proba_df.idxmax(axis=1)
    return pred, proba_df


__all__ = [
    "META_FEATURES",
    "StackingMetaModel",
    "featurize",
    "stacking_predict",
    "train_stacking_meta",
]
