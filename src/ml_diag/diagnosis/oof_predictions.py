from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml_diag.labels import (
    DATA_RELATED,
    FAULTY,
    HEALTHY,
    OPT_GEN_RELATED,
    PRIMARY_LABELS,
    STAGE1_LABELS,
    STAGE2_LABELS,
    STAGE3_LABELS_BY_BRANCH,
)
from ml_diag.utils.logging import get_logger

if TYPE_CHECKING:
    from ml_diag.diagnosis.arbitrator import ArbitratorConfig
    from ml_diag.models.inference import _StageModel

_LOG = get_logger(__name__)

STAGE_PROBA_COLS: tuple[str, ...] = (
    "stage1_p_healthy",
    "stage1_p_faulty",
    "stage2_p_data",
    "stage2_p_optgen",
    "stage3_p_leakage",
    "stage3_p_label_noise",
    "stage3_p_overfitting",
    "stage3_p_underfitting",
    "stage3_p_instability",
)

_DATA_LEAVES = STAGE3_LABELS_BY_BRANCH[DATA_RELATED]

_OPT_LEAVES = STAGE3_LABELS_BY_BRANCH[OPT_GEN_RELATED]


@dataclass(frozen=True)
class OOFPredictions:
    flat_proba: pd.DataFrame
    cascade_proba: pd.DataFrame
    cascade_stage_probs: pd.DataFrame
    arbitrator_label_probs: pd.DataFrame
    arbitrator_triggered: pd.Series
    arbitrator_confidence: pd.Series
    fold_assignments: pd.Series

    def index(self) -> pd.Index:
        return self.flat_proba.index

    def to_multiindex_frame(self) -> pd.DataFrame:
        meta = pd.DataFrame(
            {
                "arb_triggered": self.arbitrator_triggered.astype(int),
                "arb_confidence": self.arbitrator_confidence.astype(float),
                "fold": self.fold_assignments.astype(int),
            },
            index=self.flat_proba.index,
        )
        frames = {
            "flat": self.flat_proba,
            "cascade": self.cascade_proba,
            "stage_probs": self.cascade_stage_probs,
            "arbitrator": self.arbitrator_label_probs,
            "meta": meta,
        }
        out = pd.concat(frames, axis=1)
        out.columns = pd.MultiIndex.from_tuples(
            [(g, c) for g, frame in frames.items() for c in frame.columns],
            names=["group", "name"],
        )
        return out

    @classmethod
    def from_multiindex_frame(cls, df: pd.DataFrame) -> OOFPredictions:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Expected DataFrame with MultiIndex columns")
        flat = df["flat"]
        casc = df["cascade"]
        stp = df["stage_probs"]
        arb = df["arbitrator"]
        meta = df["meta"]
        return cls(
            flat_proba=flat[list(PRIMARY_LABELS)].copy(),
            cascade_proba=casc[list(PRIMARY_LABELS)].copy(),
            cascade_stage_probs=stp[list(STAGE_PROBA_COLS)].copy(),
            arbitrator_label_probs=arb[list(PRIMARY_LABELS)].copy(),
            arbitrator_triggered=meta["arb_triggered"].astype(bool),
            arbitrator_confidence=meta["arb_confidence"].astype(float),
            fold_assignments=meta["fold"].astype(int),
        )


def _stage_model_from_result(result) -> _StageModel:
    from ml_diag.models.inference import _StageModel

    return _StageModel(
        name=result.stage_name,
        model=result.model,
        classes=result.classes,
        feature_columns=result.feature_columns,
    )


def _predict_stage_proba(
    stage: _StageModel,
    X: pd.DataFrame,
    *,
    vocab: Sequence[str],
) -> pd.DataFrame:
    from ml_diag.models.inference import _proba_or_onehot, _row_for_stage

    rows: list[dict[str, float]] = []
    for _, row in X.iterrows():
        arr = _row_for_stage(row, stage)
        proba = _proba_or_onehot(stage.model, arr, stage.classes or list(vocab))
        rows.append({c: float(proba.get(c, 0.0)) for c in vocab})
    return pd.DataFrame(rows, index=X.index, columns=list(vocab))


def _build_cascade_predictions(
    *,
    stage1_res,
    stage2_res,
    stage3d_res,
    stage3o_res,
    X_holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from ml_diag.models.inference import (
        StagePrediction,
        _compose_class_probabilities,
    )

    s1 = _stage_model_from_result(stage1_res)
    s2 = _stage_model_from_result(stage2_res) if stage2_res is not None else None
    s3d = _stage_model_from_result(stage3d_res) if stage3d_res is not None else None
    s3o = _stage_model_from_result(stage3o_res) if stage3o_res is not None else None
    s1_p = _predict_stage_proba(s1, X_holdout, vocab=STAGE1_LABELS)
    s2_p = (
        _predict_stage_proba(s2, X_holdout, vocab=STAGE2_LABELS)
        if s2 is not None
        else pd.DataFrame(np.nan, index=X_holdout.index, columns=list(STAGE2_LABELS))
    )
    s3d_p = (
        _predict_stage_proba(s3d, X_holdout, vocab=_DATA_LEAVES)
        if s3d is not None
        else pd.DataFrame(np.nan, index=X_holdout.index, columns=list(_DATA_LEAVES))
    )
    s3o_p = (
        _predict_stage_proba(s3o, X_holdout, vocab=_OPT_LEAVES)
        if s3o is not None
        else pd.DataFrame(np.nan, index=X_holdout.index, columns=list(_OPT_LEAVES))
    )
    stage_probs = pd.DataFrame(
        {
            "stage1_p_healthy": s1_p[HEALTHY],
            "stage1_p_faulty": s1_p[FAULTY],
            "stage2_p_data": s2_p[DATA_RELATED],
            "stage2_p_optgen": s2_p[OPT_GEN_RELATED],
            "stage3_p_leakage": s3d_p["leakage"],
            "stage3_p_label_noise": s3d_p["label_noise"],
            "stage3_p_overfitting": s3o_p["overfitting"],
            "stage3_p_underfitting": s3o_p["underfitting"],
            "stage3_p_instability": s3o_p["instability"],
        }
    )
    composed_rows: list[dict[str, float]] = []
    for rid in X_holdout.index:
        s1_proba = {HEALTHY: float(s1_p.at[rid, HEALTHY]), FAULTY: float(s1_p.at[rid, FAULTY])}
        s1_pred = max(s1_proba, key=s1_proba.get)
        s1_obj = StagePrediction(
            stage_name=s1.name,
            predicted=s1_pred,
            confidence=s1_proba[s1_pred],
            probabilities=s1_proba,
        )
        s2_obj = None
        if s2 is not None:
            s2_proba = {
                DATA_RELATED: float(s2_p.at[rid, DATA_RELATED]),
                OPT_GEN_RELATED: float(s2_p.at[rid, OPT_GEN_RELATED]),
            }
            s2_pred = max(s2_proba, key=s2_proba.get)
            s2_obj = StagePrediction(
                stage_name=s2.name,
                predicted=s2_pred,
                confidence=s2_proba[s2_pred],
                probabilities=s2_proba,
            )
        s3d_obj = None
        if s3d is not None:
            s3d_proba = {leaf: float(s3d_p.at[rid, leaf]) for leaf in _DATA_LEAVES}
            s3d_pred = max(s3d_proba, key=s3d_proba.get) if s3d_proba else "leakage"
            s3d_obj = StagePrediction(
                stage_name=s3d.name,
                predicted=s3d_pred,
                confidence=s3d_proba.get(s3d_pred, 0.0),
                probabilities=s3d_proba,
            )
        s3o_obj = None
        if s3o is not None:
            s3o_proba = {leaf: float(s3o_p.at[rid, leaf]) for leaf in _OPT_LEAVES}
            s3o_pred = max(s3o_proba, key=s3o_proba.get) if s3o_proba else "overfitting"
            s3o_obj = StagePrediction(
                stage_name=s3o.name,
                predicted=s3o_pred,
                confidence=s3o_proba.get(s3o_pred, 0.0),
                probabilities=s3o_proba,
            )
        composed = _compose_class_probabilities(s1_obj, s2_obj, s3d_obj, s3o_obj)
        composed_rows.append(composed)
    composed_df = pd.DataFrame(composed_rows, index=X_holdout.index, columns=list(PRIMARY_LABELS))
    return composed_df, stage_probs


def _maybe_arbitrate_holdout(
    *,
    X_holdout: pd.DataFrame,
    flat_proba: pd.DataFrame,
    cascade_proba: pd.DataFrame,
    flat_classes: list[str],
    cascade,
    arbitrator_config: ArbitratorConfig | None,
    low_conf_trigger: float,
    inner_fold_index: int | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = list(PRIMARY_LABELS)
    label_probs = pd.DataFrame(0.0, index=X_holdout.index, columns=cols)
    triggered = pd.Series(False, index=X_holdout.index)
    confidence = pd.Series(0.0, index=X_holdout.index)
    if arbitrator_config is None:
        return label_probs, triggered, confidence
    from ml_diag.diagnosis.arbitrator import arbitrate_one
    from ml_diag.evaluation.explanation import build_evidence
    from ml_diag.models.inference import diagnose_one

    flat_argmax = flat_proba.idxmax(axis=1)
    cascade_argmax = cascade_proba.idxmax(axis=1)
    flat_max = flat_proba.max(axis=1)
    cascade_max = cascade_proba.max(axis=1)
    for rid in X_holdout.index:
        f_lbl = str(flat_argmax.loc[rid])
        c_lbl = str(cascade_argmax.loc[rid])
        disagreement = f_lbl != c_lbl
        low_conf = min(float(flat_max.loc[rid]), float(cascade_max.loc[rid])) < low_conf_trigger
        if not (disagreement or low_conf):
            continue
        if cascade is None:
            continue
        diag = diagnose_one(cascade, run_id=str(rid), x_row=X_holdout.loc[rid])
        ev = build_evidence(
            diagnosis=diag,
            feature_row=X_holdout.loc[rid],
            cascade=cascade,
        )
        decision = arbitrate_one(
            run_id=str(rid),
            flat_label=f_lbl,
            flat_proba={c: float(flat_proba.at[rid, c]) for c in cols},
            cascade_label=c_lbl,
            cascade_proba={c: float(cascade_proba.at[rid, c]) for c in cols},
            evidence=ev,
            config=arbitrator_config,
            inner_fold_index=inner_fold_index,
        )
        for c in cols:
            label_probs.at[rid, c] = float(decision.label_probabilities.get(c, 0.0))
        triggered.loc[rid] = True
        confidence.loc[rid] = float(decision.confidence)
    return label_probs, triggered, confidence


def generate_oof_predictions(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 0,
    arbitrator_config: ArbitratorConfig | None = None,
    arbitrator_low_conf_trigger: float = 0.0,
    include_catboost: bool = True,
) -> OOFPredictions:
    from ml_diag.diagnosis.hybrid_resolver import flat_proba_aligned
    from ml_diag.models import stage1 as _stage1_mod
    from ml_diag.models import stage2 as _stage2_mod
    from ml_diag.models import stage3 as _stage3_mod
    from ml_diag.models.flat_baseline import (
        train_flat_baseline,
    )
    from ml_diag.models.inference import HierarchicalCascade
    from ml_diag.models.model_zoo import default_zoo

    n = len(X)
    if len(y) != n:
        raise ValueError("X and y must have equal length")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cols = list(PRIMARY_LABELS)
    flat_proba = pd.DataFrame(np.nan, index=X.index, columns=cols)
    cascade_proba = pd.DataFrame(np.nan, index=X.index, columns=cols)
    cascade_stage_probs = pd.DataFrame(
        np.nan,
        index=X.index,
        columns=list(STAGE_PROBA_COLS),
    )
    arb_label_probs = pd.DataFrame(0.0, index=X.index, columns=cols)
    arb_triggered = pd.Series(False, index=X.index)
    arb_confidence = pd.Series(0.0, index=X.index)
    fold_assignments = pd.Series(-1, index=X.index, dtype=int)
    zoo = default_zoo(include_catboost=include_catboost)
    for fold_idx, (train_pos, holdout_pos) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_pos], y.iloc[train_pos]
        X_ho, y_ho = X.iloc[holdout_pos], y.iloc[holdout_pos]
        train_ids = set(X_tr.index)
        holdout_ids = set(X_ho.index)
        leaked = train_ids & holdout_ids
        assert not leaked, (
            f"OOF leak in fold {fold_idx}: {len(leaked)} ids appear in both train "
            "and holdout indices"
        )
        _LOG.info(
            "[oof fold %d/%d] training flat + cascade on %d rows, predicting on %d holdout rows",
            fold_idx + 1,
            n_splits,
            len(X_tr),
            len(X_ho),
        )
        flat_res = train_flat_baseline(
            X_tr,
            y_tr,
            seed=seed,
            candidate_models=zoo,
            do_internal_split=False,
        )
        f_p = flat_proba_aligned(flat_res, X_ho)
        flat_proba.loc[X_ho.index, :] = f_p[cols].values
        s1_res = _stage1_mod.train(
            X_tr,
            y_tr,
            seed=seed,
            calibrate=False,
        )
        s2_res = _stage2_mod.train(
            X_tr,
            y_tr,
            seed=seed,
            calibrate=False,
        )
        s3d_res = _stage3_mod.train_data_related(
            X_tr,
            y_tr,
            seed=seed,
            calibrate=False,
        )
        s3o_res = _stage3_mod.train_opt_gen(
            X_tr,
            y_tr,
            seed=seed,
            calibrate=False,
        )
        composed_df, stage_probs_df = _build_cascade_predictions(
            stage1_res=s1_res,
            stage2_res=s2_res,
            stage3d_res=s3d_res,
            stage3o_res=s3o_res,
            X_holdout=X_ho,
        )
        cascade_proba.loc[X_ho.index, :] = composed_df[cols].values
        cascade_stage_probs.loc[X_ho.index, :] = stage_probs_df.values
        fold_assignments.loc[X_ho.index] = fold_idx
        if arbitrator_config is not None:
            transient_cascade = HierarchicalCascade(
                stage1=_stage_model_from_result(s1_res),
                stage2=_stage_model_from_result(s2_res),
                stage3_data=_stage_model_from_result(s3d_res),
                stage3_opt=_stage_model_from_result(s3o_res),
                feature_columns=list(s1_res.feature_columns),
                stage1_healthy_threshold=0.5,
                threshold_source="oof",
            )
            ho_label_probs, ho_triggered, ho_conf = _maybe_arbitrate_holdout(
                X_holdout=X_ho,
                flat_proba=f_p,
                cascade_proba=composed_df,
                flat_classes=flat_res.classes,
                cascade=transient_cascade,
                arbitrator_config=arbitrator_config,
                low_conf_trigger=arbitrator_low_conf_trigger,
                inner_fold_index=int(fold_idx),
            )
            arb_label_probs.loc[X_ho.index, :] = ho_label_probs[cols].values
            arb_triggered.loc[X_ho.index] = ho_triggered.loc[X_ho.index].values
            arb_confidence.loc[X_ho.index] = ho_conf.loc[X_ho.index].values
            n_trig = int(ho_triggered.sum())
            _LOG.info(
                "[oof fold %d/%d] arbitrator triggered on %d / %d holdout rows",
                fold_idx + 1,
                n_splits,
                n_trig,
                len(X_ho),
            )
    if (fold_assignments < 0).any():
        missing = fold_assignments[fold_assignments < 0].index.tolist()
        raise RuntimeError(f"OOF generation skipped {len(missing)} rows: {missing[:5]}…")
    if flat_proba.isna().any().any():
        raise RuntimeError("flat_proba has NaN — OOF generation incomplete")
    if cascade_proba.isna().any().any():
        raise RuntimeError("cascade_proba has NaN — OOF generation incomplete")
    return OOFPredictions(
        flat_proba=flat_proba,
        cascade_proba=cascade_proba,
        cascade_stage_probs=cascade_stage_probs,
        arbitrator_label_probs=arb_label_probs,
        arbitrator_triggered=arb_triggered,
        arbitrator_confidence=arb_confidence,
        fold_assignments=fold_assignments,
    )


def write_oof_parquet(
    oof: OOFPredictions,
    path: str | Path,
) -> Path:
    df = oof.to_multiindex_frame()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p)
    return p


def read_oof_parquet(path: str | Path) -> OOFPredictions:
    df = pd.read_parquet(Path(path))
    return OOFPredictions.from_multiindex_frame(df)


__all__ = [
    "OOFPredictions",
    "STAGE_PROBA_COLS",
    "generate_oof_predictions",
    "read_oof_parquet",
    "write_oof_parquet",
]
