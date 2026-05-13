from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from structured_diag.labels import (
    DATA_RELATED,
    FAULTY,
    HEALTHY,
    OPT_GEN_RELATED,
    PRIMARY_LABELS,
    STAGE3_LABELS_BY_BRANCH,
)
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class StagePrediction:
    stage_name: str
    predicted: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "predicted": self.predicted,
            "confidence": float(self.confidence),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
        }


@dataclass(frozen=True)
class HierarchicalDiagnosis:
    run_id: str
    final_class: str
    final_confidence: float
    stage1: StagePrediction
    stage2: StagePrediction | None
    stage3: StagePrediction | None
    class_probabilities: dict[str, float]
    alternative_hypotheses: list[tuple[str, float]]

    def to_dict(self) -> dict[str, Any]:
        d = {
            "run_id": self.run_id,
            "final_class": self.final_class,
            "final_confidence": float(self.final_confidence),
            "stage1": self.stage1.to_dict(),
            "stage2": self.stage2.to_dict() if self.stage2 else None,
            "stage3": self.stage3.to_dict() if self.stage3 else None,
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "alternative_hypotheses": [
                {"class": c, "probability": float(p)} for c, p in self.alternative_hypotheses
            ],
        }
        return d


@dataclass(frozen=True)
class _StageModel:
    name: str
    model: object
    classes: list[str]
    feature_columns: list[str]


def _load_one(path: Path) -> _StageModel | None:
    if not path.is_file():
        return None
    try:
        import joblib
    except ImportError as e:
        raise RuntimeError("joblib is required to load saved stage artifacts.") from e
    bundle = joblib.load(path)
    return _StageModel(
        name=bundle.get("stage_name", path.stem),
        model=bundle["model"],
        classes=[str(c) for c in bundle.get("classes", [])],
        feature_columns=list(bundle.get("feature_columns", [])),
    )


def _resolve_artifacts_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Hierarchical artifacts dir not found: {p}")
    return p


@dataclass(frozen=True)
class HierarchicalCascade:
    stage1: _StageModel
    stage2: _StageModel | None
    stage3_data: _StageModel | None
    stage3_opt: _StageModel | None
    feature_columns: list[str]
    stage1_healthy_threshold: float = 0.5
    threshold_source: str = "default"

    @property
    def stages_available(self) -> list[str]:
        out: list[str] = ["stage1"]
        if self.stage2 is not None:
            out.append("stage2")
        if self.stage3_data is not None:
            out.append("stage3_data_related")
        if self.stage3_opt is not None:
            out.append("stage3_optimization_or_generalization_related")
        return out


def load_cascade(artifacts_dir: str | Path) -> HierarchicalCascade:
    d = _resolve_artifacts_dir(artifacts_dir)
    s1 = _load_one(d / "stage1_healthy_vs_faulty.joblib")
    if s1 is None:
        raise FileNotFoundError(
            f"Stage 1 artifact not found in {d}. Run scripts/run_hierarchical_train.py first."
        )
    s2 = _load_one(d / "stage2_faulty_data_vs_optgen.joblib")
    s3d = _load_one(d / "stage3_data_related.joblib")
    s3o = _load_one(d / "stage3_optimization_or_generalization_related.joblib")
    canonical = list(s1.feature_columns)
    for stage in (s2, s3d, s3o):
        if stage is None:
            continue
        if list(stage.feature_columns) != canonical:
            _LOG.warning(
                "Feature columns differ between stage1 and %s — inference will "
                "reindex to stage1's order.",
                stage.name,
            )
    threshold = 0.5
    threshold_source = "default"
    cfg_path = d / "cascade_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if "stage1_healthy_threshold" in cfg:
                threshold = float(cfg["stage1_healthy_threshold"])
                threshold_source = "config_file"
                _LOG.info(
                    "Cascade loaded with stage1_healthy_threshold=%.3f from %s",
                    threshold,
                    cfg_path.name,
                )
        except Exception as e:
            _LOG.warning("Could not parse %s: %s", cfg_path, e)
    return HierarchicalCascade(
        stage1=s1,
        stage2=s2,
        stage3_data=s3d,
        stage3_opt=s3o,
        feature_columns=canonical,
        stage1_healthy_threshold=threshold,
        threshold_source=threshold_source,
    )


def _proba_or_onehot(model: object, X_row: np.ndarray, classes: Sequence[str]) -> dict[str, float]:
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X_row.reshape(1, -1))[0], dtype=float)
            return {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception:
            pass
    pred = str(model.predict(X_row.reshape(1, -1))[0])
    out = {str(c): 0.0 for c in classes}
    if pred in out:
        out[pred] = 1.0
    else:
        out[pred] = 1.0
    return out


def _argmax_class(prob: Mapping[str, float]) -> tuple[str, float]:
    if not prob:
        raise ValueError("Empty probability dict.")
    cls, p = max(prob.items(), key=lambda kv: kv[1])
    return cls, float(p)


def _stage_pred(stage: _StageModel, X_row: np.ndarray) -> StagePrediction:
    proba = _proba_or_onehot(stage.model, X_row, stage.classes or [])
    pred, conf = _argmax_class(proba)
    return StagePrediction(
        stage_name=stage.name,
        predicted=pred,
        confidence=conf,
        probabilities=proba,
    )


def _branch_for_stage2(label: str) -> str:
    if label in (DATA_RELATED, OPT_GEN_RELATED):
        return label
    if "data" in label.lower():
        return DATA_RELATED
    return OPT_GEN_RELATED


def _compose_class_probabilities(
    s1: StagePrediction,
    s2: StagePrediction | None,
    s3_data: StagePrediction | None,
    s3_opt: StagePrediction | None,
) -> dict[str, float]:
    out = {label: 0.0 for label in PRIMARY_LABELS}
    p_healthy = float(s1.probabilities.get(HEALTHY, 0.0))
    p_faulty = float(s1.probabilities.get(FAULTY, 1.0 - p_healthy))
    out[HEALTHY] = p_healthy
    if s2 is None:
        leaves = [l for l in PRIMARY_LABELS if l != HEALTHY]
        if leaves:
            share = p_faulty / len(leaves)
            for l in leaves:
                out[l] = share
        return _normalize(out)
    p_data = float(s2.probabilities.get(DATA_RELATED, 0.0))
    p_opt = float(s2.probabilities.get(OPT_GEN_RELATED, 0.0))
    if p_data == 0.0 and p_opt == 0.0:
        if s2.predicted == DATA_RELATED:
            p_data = 1.0
        else:
            p_opt = 1.0
    s2_total = p_data + p_opt or 1.0
    p_data /= s2_total
    p_opt /= s2_total
    data_leaves = STAGE3_LABELS_BY_BRANCH[DATA_RELATED]
    if s3_data is not None:
        for leaf in data_leaves:
            out[leaf] = p_faulty * p_data * float(s3_data.probabilities.get(leaf, 0.0))
    else:
        share = p_faulty * p_data / max(1, len(data_leaves))
        for leaf in data_leaves:
            out[leaf] = share
    opt_leaves = STAGE3_LABELS_BY_BRANCH[OPT_GEN_RELATED]
    if s3_opt is not None:
        for leaf in opt_leaves:
            out[leaf] = p_faulty * p_opt * float(s3_opt.probabilities.get(leaf, 0.0))
    else:
        share = p_faulty * p_opt / max(1, len(opt_leaves))
        for leaf in opt_leaves:
            out[leaf] = share
    return _normalize(out)


def _normalize(prob: dict[str, float]) -> dict[str, float]:
    s = sum(max(0.0, v) for v in prob.values())
    if s <= 0:
        return prob
    return {k: max(0.0, v) / s for k, v in prob.items()}


def _align_features(X: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    from structured_diag.utils import align_features_to_schema

    df = align_features_to_schema(X, feature_columns, fill_value=np.nan)
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
    return df


def _row_for_stage(
    x_row: pd.Series | pd.DataFrame | np.ndarray,
    stage: _StageModel,
) -> np.ndarray:
    from structured_diag.utils import align_features_to_schema

    df = align_features_to_schema(x_row, stage.feature_columns, fill_value=np.nan)
    arr = df.iloc[0].to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def diagnose_one(
    cascade: HierarchicalCascade,
    run_id: str,
    x_row: pd.Series | pd.DataFrame | np.ndarray,
    *,
    top_k_alternatives: int = 3,
) -> HierarchicalDiagnosis:
    if isinstance(x_row, np.ndarray):
        if x_row.ndim != 1:
            raise ValueError("ndarray x_row must be 1-D")
        if x_row.size != len(cascade.stage1.feature_columns):
            raise ValueError(
                f"ndarray x_row has {x_row.size} entries, but stage1 expects "
                f"{len(cascade.stage1.feature_columns)}; pass a "
                "pandas.Series for safe alignment."
            )
        x_row = pd.Series(x_row, index=cascade.stage1.feature_columns)
    arr_s1 = _row_for_stage(x_row, cascade.stage1)
    s1_raw_proba = _proba_or_onehot(cascade.stage1.model, arr_s1, cascade.stage1.classes or [])
    p_healthy = float(s1_raw_proba.get(HEALTHY, 0.0))
    if p_healthy >= cascade.stage1_healthy_threshold:
        s1_predicted = HEALTHY
        s1_confidence = p_healthy
    else:
        s1_predicted = FAULTY
        s1_confidence = float(s1_raw_proba.get(FAULTY, 1.0 - p_healthy))
    s1 = StagePrediction(
        stage_name=cascade.stage1.name,
        predicted=s1_predicted,
        confidence=s1_confidence,
        probabilities=s1_raw_proba,
    )
    s2: StagePrediction | None = None
    s3_data: StagePrediction | None = None
    s3_opt: StagePrediction | None = None
    if s1.predicted == FAULTY and cascade.stage2 is not None:
        arr_s2 = _row_for_stage(x_row, cascade.stage2)
        s2 = _stage_pred(cascade.stage2, arr_s2)
        if cascade.stage3_data is not None:
            arr_s3d = _row_for_stage(x_row, cascade.stage3_data)
            s3_data = _stage_pred(cascade.stage3_data, arr_s3d)
        if cascade.stage3_opt is not None:
            arr_s3o = _row_for_stage(x_row, cascade.stage3_opt)
            s3_opt = _stage_pred(cascade.stage3_opt, arr_s3o)
    composed = _compose_class_probabilities(s1, s2, s3_data, s3_opt)
    final_class, final_conf = _argmax_class(composed)
    alternatives = [
        (cls, prob)
        for cls, prob in sorted(composed.items(), key=lambda kv: -kv[1])
        if cls != final_class
    ][:top_k_alternatives]
    s3_reported: StagePrediction | None = None
    if s2 is not None:
        if _branch_for_stage2(s2.predicted) == DATA_RELATED:
            s3_reported = s3_data
        else:
            s3_reported = s3_opt
    return HierarchicalDiagnosis(
        run_id=str(run_id),
        final_class=final_class,
        final_confidence=final_conf,
        stage1=s1,
        stage2=s2,
        stage3=s3_reported,
        class_probabilities=composed,
        alternative_hypotheses=alternatives,
    )


def diagnose_batch(
    cascade: HierarchicalCascade,
    X: pd.DataFrame,
    *,
    top_k_alternatives: int = 3,
) -> list[HierarchicalDiagnosis]:
    out: list[HierarchicalDiagnosis] = []
    for run_id, row in X.iterrows():
        out.append(
            diagnose_one(
                cascade,
                run_id=str(run_id),
                x_row=row,
                top_k_alternatives=top_k_alternatives,
            )
        )
    return out


def diagnoses_to_dataframe(diags: Iterable[HierarchicalDiagnosis]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for d in diags:
        row: dict[str, Any] = {
            "run_id": d.run_id,
            "final_class": d.final_class,
            "final_confidence": d.final_confidence,
            "stage1_pred": d.stage1.predicted,
            "stage1_conf": d.stage1.confidence,
            "stage2_pred": d.stage2.predicted if d.stage2 else None,
            "stage2_conf": d.stage2.confidence if d.stage2 else None,
            "stage3_pred": d.stage3.predicted if d.stage3 else None,
            "stage3_conf": d.stage3.confidence if d.stage3 else None,
        }
        for cls, prob in d.class_probabilities.items():
            row[f"p_{cls}"] = prob
        for i, (cls, prob) in enumerate(d.alternative_hypotheses, start=1):
            row[f"alt_{i}_class"] = cls
            row[f"alt_{i}_prob"] = prob
        rows.append(row)
    return pd.DataFrame(rows)


def diagnoses_to_jsonl(diags: Iterable[HierarchicalDiagnosis], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in diags:
            f.write(json.dumps(d.to_dict(), ensure_ascii=False))
            f.write("\n")
    return out_path


__all__ = [
    "HierarchicalDiagnosis",
    "StagePrediction",
    "HierarchicalCascade",
    "load_cascade",
    "diagnose_one",
    "diagnose_batch",
    "diagnoses_to_dataframe",
    "diagnoses_to_jsonl",
]
