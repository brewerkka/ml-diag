from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from structured_diag.evaluation import ClassificationReport, classification_report
from structured_diag.models.model_zoo import ModelSpec, default_zoo
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class FlatBaselineResult:
    model_name: str
    model: object
    classes: list[str]
    feature_columns: list[str]
    seed: int


def _split_train_test(
    X: pd.DataFrame, y: pd.Series, *, seed: int, n_splits: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(iter(skf.split(X, y)))
    return train_idx, test_idx


def _fit_one(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, seed: int):
    from structured_diag.utils import ensure_feature_matrix, ensure_label_array

    model = spec.factory(seed)
    if model is None:
        return None
    model.fit(ensure_feature_matrix(X), ensure_label_array(y))
    return model


def _predict_proba_safely(model, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        from structured_diag.utils import ensure_feature_matrix

        try:
            return np.asarray(model.predict_proba(ensure_feature_matrix(X)), dtype=float)
        except Exception:
            return None
    return None


def _model_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    return []


def train_flat_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int = 0,
    candidate_models: Sequence[ModelSpec] | None = None,
    do_internal_split: bool = True,
) -> FlatBaselineResult:
    candidate_models = list(candidate_models or default_zoo())
    if do_internal_split:
        train_idx, _ = _split_train_test(X, y, seed=seed)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    else:
        X_tr, y_tr = X, y
    best_score = -1.0
    best_name: str | None = None
    best_model = None
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for spec in candidate_models:
        scores: list[float] = []
        last_failure: str | None = None
        for fold_train, fold_val in skf.split(X_tr, y_tr):
            model = _fit_one(
                spec,
                X_tr.iloc[fold_train],
                y_tr.iloc[fold_train],
                seed=seed,
            )
            if model is None:
                last_failure = "factory returned None"
                break
            try:
                from structured_diag.utils import ensure_feature_matrix

                preds = model.predict(ensure_feature_matrix(X_tr.iloc[fold_val]))
                rep = classification_report(y_tr.iloc[fold_val], preds)
                scores.append(rep.macro_f1)
            except Exception as e:  # noqa: BLE001
                last_failure = str(e)
                break
        if not scores:
            _LOG.info("Skipping model %s: %s", spec.name, last_failure or "no scores")
            continue
        mean_score = float(np.mean(scores))
        _LOG.info("Flat baseline candidate %s: CV macro-F1 = %.4f", spec.name, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_name = spec.name
            best_model = _fit_one(spec, X_tr, y_tr, seed=seed)
    if best_model is None or best_name is None:
        raise RuntimeError("No flat baseline candidate could be trained.")
    _LOG.info("Best flat baseline: %s (CV macro-F1 = %.4f)", best_name, best_score)
    return FlatBaselineResult(
        model_name=best_name,
        model=best_model,
        classes=_model_classes(best_model),
        feature_columns=list(X.columns),
        seed=seed,
    )


def evaluate_on_slices(
    result: FlatBaselineResult,
    X: pd.DataFrame,
    y: pd.Series,
    slices: dict[str, pd.Index],
) -> dict[str, ClassificationReport]:
    from structured_diag.utils import (
        align_features_to_schema,
        ensure_feature_matrix,
    )

    reports: dict[str, ClassificationReport] = {}
    for slice_name, idx in slices.items():
        if len(idx) == 0:
            continue
        Xs = X.loc[idx]
        ys = y.loc[idx]
        if Xs.empty:
            continue
        if result.feature_columns:
            Xs = align_features_to_schema(Xs, result.feature_columns)
        preds = result.model.predict(ensure_feature_matrix(Xs))
        proba = _predict_proba_safely(result.model, Xs)
        rep = classification_report(
            ys,
            preds,
            y_proba=proba,
            proba_classes=result.classes,
            label_order=result.classes if result.classes else None,
        )
        reports[slice_name] = rep
    return reports


def slices_from_partition(
    partition_table: pd.DataFrame,
    feature_index: Iterable[str],
    *,
    holdout_index: Iterable[str] | None = None,
) -> dict[str, pd.Index]:
    feature_set = set(feature_index)
    pt = (
        partition_table.set_index("run_id")
        if "run_id" in partition_table.columns
        else partition_table
    )
    full = pd.Index([rid for rid in pt.index if rid in feature_set])
    core_ids = pd.Index([rid for rid in full if pt.loc[rid, "slice"] == "core"])
    ext_ids = pd.Index([rid for rid in full if pt.loc[rid, "slice"] == "extended"])
    if holdout_index is not None:
        holdout = pd.Index(holdout_index)
        full = full.intersection(holdout)
        core_ids = core_ids.intersection(holdout)
        ext_ids = ext_ids.intersection(holdout)
    return {"full": full, "core": core_ids, "extended": ext_ids}


__all__ = [
    "FlatBaselineResult",
    "train_flat_baseline",
    "evaluate_on_slices",
    "slices_from_partition",
]
