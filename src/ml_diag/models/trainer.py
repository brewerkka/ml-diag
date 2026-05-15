from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml_diag.evaluation import (
    ClassificationReport,
    classification_report,
    report_to_markdown,
)
from ml_diag.models.model_zoo import ModelSpec, default_zoo
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class StageTrainResult:
    stage_name: str
    model_name: str
    model: object
    classes: list[str]
    feature_columns: list[str]
    seed: int
    cv_scores: dict[str, float]
    test_run_ids: list[str]
    test_reports: dict[str, ClassificationReport]
    n_train: int
    n_test: int
    is_calibrated: bool = False
    calibration_method: str | None = None

    def summary_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "model_name": self.model_name,
            "classes": self.classes,
            "feature_columns": self.feature_columns,
            "seed": self.seed,
            "cv_scores": self.cv_scores,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "is_calibrated": self.is_calibrated,
            "calibration_method": self.calibration_method,
            "test_reports": {k: v.to_dict() for k, v in self.test_reports.items()},
        }


def _safe_n_splits(n_samples: int, default: int = 5, *, min_per_class: int = 2) -> int:
    if n_samples < 2 * min_per_class:
        return 2
    return min(default, max(2, n_samples // min_per_class))


def _stratified_first_fold(
    X: pd.DataFrame, y: pd.Series, *, seed: int, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return next(iter(skf.split(X, y)))


def _fit(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, seed: int):
    model = spec.factory(seed)
    if model is None:
        return None
    from ml_diag.utils import ensure_feature_matrix, ensure_label_array

    model.fit(ensure_feature_matrix(X), ensure_label_array(y))
    return model


def _model_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(c) for c in model.classes_]
    if hasattr(model, "named_steps") and "clf" in getattr(model, "named_steps", {}):
        clf = model.named_steps["clf"]
        if hasattr(clf, "classes_"):
            return [str(c) for c in clf.classes_]
    return []


def _predict_proba_safely(model, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        from ml_diag.utils import ensure_feature_matrix

        try:
            return np.asarray(model.predict_proba(ensure_feature_matrix(X)), dtype=float)
        except Exception:
            return None
    return None


def train_stage(
    *,
    stage_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    label_vocab: Sequence[str],
    partition_table: pd.DataFrame | None = None,
    candidate_models: Sequence[ModelSpec] | None = None,
    seed: int = 0,
    holdout_run_ids: Sequence[str] | None = None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> StageTrainResult:
    if X.empty:
        raise RuntimeError(f"[{stage_name}] empty feature matrix; nothing to train.")
    if y.nunique() < 2:
        raise RuntimeError(
            f"[{stage_name}] only one class present in y ({y.unique().tolist()}). "
            "Skip this stage in the caller."
        )
    candidate_models = list(candidate_models or default_zoo())
    if holdout_run_ids is not None:
        holdout_set = {str(r) for r in holdout_run_ids}
        idx_str = X.index.astype(str)
        is_holdout = idx_str.isin(holdout_set)
        if is_holdout.all():
            raise RuntimeError(
                f"[{stage_name}] every row is in holdout — nothing left to train on."
            )
        if not is_holdout.any():
            _LOG.warning(
                "[%s] holdout_run_ids did not intersect this stage's subset; "
                "falling back to internal split.",
                stage_name,
            )
            n_outer = _safe_n_splits(len(X))
            train_idx, test_idx = _stratified_first_fold(X, y, seed=seed, n_splits=n_outer)
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        else:
            X_tr = X.loc[~is_holdout]
            y_tr = y.loc[~is_holdout]
            X_te = X.loc[is_holdout]
            y_te = y.loc[is_holdout]
            if y_tr.nunique() < 2:
                raise RuntimeError(
                    f"[{stage_name}] only one class left in train fold after "
                    f"applying holdout: {y_tr.unique().tolist()}."
                )
    else:
        n_outer = _safe_n_splits(len(X))
        train_idx, test_idx = _stratified_first_fold(X, y, seed=seed, n_splits=n_outer)
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    n_inner = _safe_n_splits(len(X_tr))
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    cv_scores: dict[str, float] = {}
    best_score = -1.0
    best_name: str | None = None
    best_model: object | None = None
    for spec in candidate_models:
        scores: list[float] = []
        try:
            for f_tr, f_va in skf.split(X_tr, y_tr):
                model = _fit(spec, X_tr.iloc[f_tr], y_tr.iloc[f_tr], seed=seed)
                if model is None:
                    raise RuntimeError("factory returned None")
                from ml_diag.utils import ensure_feature_matrix

                preds = model.predict(ensure_feature_matrix(X_tr.iloc[f_va]))
                rep = classification_report(y_tr.iloc[f_va], preds, label_order=label_vocab)
                scores.append(rep.macro_f1)
        except Exception as e:                
            _LOG.info("[%s] candidate %s skipped: %s", stage_name, spec.name, e)
            continue
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        cv_scores[spec.name] = mean_score
        _LOG.info("[%s] %s CV macro-F1 = %.4f", stage_name, spec.name, mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_name = spec.name
            best_model = _fit(spec, X_tr, y_tr, seed=seed)
    if best_model is None or best_name is None:
        raise RuntimeError(f"[{stage_name}] no candidate model could be trained.")
    is_calibrated = False
    if calibrate:
        try:
            from sklearn.calibration import CalibratedClassifierCV

            from ml_diag.utils import ensure_feature_matrix, ensure_label_array

            n_cal_folds = min(5, _safe_n_splits(len(X_tr)))
            if n_cal_folds < 2:
                _LOG.warning(
                    "[%s] not enough samples for calibration (%d) — skipping.",
                    stage_name,
                    len(X_tr),
                )
            else:
                calibrated = CalibratedClassifierCV(
                    best_model,
                    method=calibration_method,
                    cv=n_cal_folds,
                )
                calibrated.fit(
                    ensure_feature_matrix(X_tr),
                    ensure_label_array(y_tr),
                )
                best_model = calibrated
                is_calibrated = True
                _LOG.info(
                    "[%s] calibrated best=%s with %s, cv=%d",
                    stage_name,
                    best_name,
                    calibration_method,
                    n_cal_folds,
                )
        except Exception as e:                
            _LOG.warning(
                "[%s] calibration failed (%s) — using uncalibrated model.",
                stage_name,
                e,
            )
    test_run_ids = list(X_te.index.astype(str))
    test_reports: dict[str, ClassificationReport] = {}

    def _report_on(idx: pd.Index, key: str) -> None:
        if len(idx) == 0:
            return
        Xs = X_te.loc[idx]
        ys = y_te.loc[idx]
        if Xs.empty:
            return
        from ml_diag.utils import ensure_feature_matrix

        preds = best_model.predict(ensure_feature_matrix(Xs))
        proba = _predict_proba_safely(best_model, Xs)
        rep = classification_report(
            ys,
            preds,
            y_proba=proba,
            proba_classes=_model_classes(best_model),
            label_order=label_vocab,
        )
        test_reports[key] = rep

    _report_on(X_te.index, "full")
    if partition_table is not None and not partition_table.empty:
        pt = (
            partition_table.set_index("run_id")
            if "run_id" in partition_table.columns
            else partition_table
        )
        core_ids = pd.Index(
            [rid for rid in X_te.index if rid in pt.index and pt.loc[rid, "slice"] == "core"]
        )
        ext_ids = pd.Index(
            [rid for rid in X_te.index if rid in pt.index and pt.loc[rid, "slice"] == "extended"]
        )
        _report_on(core_ids, "core")
        _report_on(ext_ids, "extended")
    _LOG.info(
        "[%s] best=%s CV=%.4f n_train=%d n_test=%d",
        stage_name,
        best_name,
        best_score,
        len(X_tr),
        len(X_te),
    )
    return StageTrainResult(
        stage_name=stage_name,
        model_name=best_name,
        model=best_model,
        classes=_model_classes(best_model),
        feature_columns=list(X.columns),
        seed=seed,
        cv_scores=cv_scores,
        test_run_ids=test_run_ids,
        test_reports=test_reports,
        n_train=int(len(X_tr)),
        n_test=int(len(X_te)),
        is_calibrated=is_calibrated,
        calibration_method=calibration_method if is_calibrated else None,
    )


def save_stage_artifacts(
    result: StageTrainResult,
    out_dir: str | Path,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    model_path = out_dir / f"{result.stage_name}.joblib"
    try:
        import joblib

        joblib.dump(
            {
                "model": result.model,
                "model_name": result.model_name,
                "classes": result.classes,
                "feature_columns": result.feature_columns,
                "seed": result.seed,
                "stage_name": result.stage_name,
            },
            model_path,
        )
        paths["model"] = model_path
    except ImportError:
        _LOG.warning("joblib not installed; skipping model dump for %s", result.stage_name)
    summary_path = out_dir / f"{result.stage_name}.summary.json"
    summary_path.write_text(json.dumps(result.summary_dict(), indent=2), encoding="utf-8")
    paths["summary_json"] = summary_path
    md_lines: list[str] = []
    md_lines.append(f"# Stage report: `{result.stage_name}`")
    md_lines.append("")
    md_lines.append(f"- best model: `{result.model_name}`")
    md_lines.append(f"- seed: {result.seed}")
    md_lines.append(f"- n_train: {result.n_train}, n_test: {result.n_test}")
    md_lines.append("- CV macro-F1 by candidate:")
    for cand, score in sorted(result.cv_scores.items(), key=lambda kv: -kv[1]):
        md_lines.append(f"    - `{cand}`: {score:.4f}")
    md_lines.append("")
    for slice_name, rep in result.test_reports.items():
        md_lines.append(report_to_markdown(rep, heading=f"Test slice: `{slice_name}`"))
    md_path = out_dir / f"{result.stage_name}.report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    paths["summary_md"] = md_path
    return paths


__all__ = ["StageTrainResult", "train_stage", "save_stage_artifacts"]
