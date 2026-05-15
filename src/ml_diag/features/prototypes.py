from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml_diag.data import CorpusManifest, load_manifest, load_run
from ml_diag.features.run_features import FeatureTable, build_feature_table
from ml_diag.labels import PRIMARY_LABELS
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


@dataclass(frozen=True)
class PrototypeBank:
    feature_columns: list[str]
    classes: list[str]
    prototypes: dict[str, np.ndarray]
    train_mean: np.ndarray
    train_std: np.ndarray
    prototypes_z: dict[str, np.ndarray] = field(default_factory=dict)

    def has_class(self, c: str) -> bool:
        return c in self.prototypes


def _zscore(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def build_prototype_bank(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    classes: Iterable[str] = PRIMARY_LABELS,
    eps: float = 1e-9,
) -> PrototypeBank:
    if X_train.empty or y_train.empty:
        raise RuntimeError("Empty training data — cannot build prototypes.")
    if not X_train.index.equals(y_train.index):
        y_train = y_train.reindex(X_train.index)
    feature_columns = list(X_train.columns)
    train_mean = X_train.values.mean(axis=0)
    train_std = X_train.values.std(axis=0) + eps
    prototypes: dict[str, np.ndarray] = {}
    prototypes_z: dict[str, np.ndarray] = {}
    for cls in classes:
        mask = (y_train == cls).values
        if not mask.any():
            _LOG.info("No training rows for class %r; prototype skipped.", cls)
            continue
        proto = X_train.values[mask].mean(axis=0)
        prototypes[cls] = proto
        prototypes_z[cls] = _zscore(proto, train_mean, train_std)
    if not prototypes:
        raise RuntimeError("Could not build any prototypes — every class is empty.")
    return PrototypeBank(
        feature_columns=feature_columns,
        classes=list(prototypes),
        prototypes=prototypes,
        train_mean=train_mean,
        train_std=train_std,
        prototypes_z=prototypes_z,
    )


def normalized_euclidean(X: pd.DataFrame, bank: PrototypeBank) -> pd.DataFrame:
    Xa = _align(X, bank.feature_columns).values
    Xz = _zscore(Xa, bank.train_mean, bank.train_std)
    out = {}
    for cls in bank.classes:
        diff = Xz - bank.prototypes_z[cls]
        out[f"proto_neuc_{cls}"] = np.sqrt((diff * diff).sum(axis=1))
    return pd.DataFrame(out, index=X.index)


def _pearson_distance(X: np.ndarray, p: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=1, keepdims=True)
    pc = p - p.mean()
    num = (Xc * pc).sum(axis=1)
    denom = np.sqrt((Xc**2).sum(axis=1) * (pc**2).sum() + 1e-30)
    corr = num / denom
    return 1.0 - corr


def correlation_distance(X: pd.DataFrame, bank: PrototypeBank) -> pd.DataFrame:
    Xa = _align(X, bank.feature_columns).values
    out = {}
    for cls in bank.classes:
        out[f"proto_corr_{cls}"] = _pearson_distance(Xa, bank.prototypes[cls])
    return pd.DataFrame(out, index=X.index)


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    n, m = a.size, b.size
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, np.inf)
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m]) / max(1, n + m)


def _curve_for_run(run_dir: Path, column: str) -> np.ndarray:
    rec = load_run(run_dir)
    if column not in rec.history.columns:
        return np.array([], dtype=float)
    return rec.history[column].to_numpy(dtype=float)


def build_curve_prototypes(
    manifest: CorpusManifest,
    train_run_ids: Iterable[str],
    y_train: pd.Series,
    *,
    column: str = "val_loss",
    classes: Iterable[str] = PRIMARY_LABELS,
) -> dict[str, np.ndarray]:
    by_class_curves: dict[str, list[np.ndarray]] = {c: [] for c in classes}
    train_set = set(map(str, train_run_ids))
    for run_dir in manifest.run_dirs:
        rid = str(run_dir.name)
        if rid not in train_set:
            continue
        cls = y_train.get(rid)
        if cls is None or cls not in by_class_curves:
            continue
        curve = _curve_for_run(run_dir, column)
        if curve.size == 0:
            continue
        by_class_curves[cls].append(curve)
    out: dict[str, np.ndarray] = {}
    for cls, curves in by_class_curves.items():
        if not curves:
            continue
        n = min(c.size for c in curves)
        if n == 0:
            continue
        stacked = np.stack([c[:n] for c in curves], axis=0)
        out[cls] = np.nanmedian(stacked, axis=0)
    return out


def dtw_distance_to_prototypes(
    manifest: CorpusManifest,
    run_ids: Iterable[str],
    curve_prototypes: Mapping[str, np.ndarray],
    *,
    column: str = "val_loss",
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    index: list[str] = []
    runs_by_id = {rd.name: rd for rd in manifest.run_dirs}
    for rid in run_ids:
        rd = runs_by_id.get(str(rid))
        if rd is None:
            continue
        curve = _curve_for_run(rd, column)
        row = {f"proto_dtw_{cls}": _dtw(curve, p) for cls, p in curve_prototypes.items()}
        rows.append(row)
        index.append(str(rid))
    return pd.DataFrame(rows, index=pd.Index(index, name="run_id"))


def _align(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        for c in missing:
            X = X.assign(**{c: np.nan})
    X = X[feature_columns]
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)
    return X


@dataclass(frozen=True)
class PrototypeFeatureTable:
    base: FeatureTable
    bank: PrototypeBank
    df: pd.DataFrame
    prototype_columns: list[str]

    def aligned_xy(
        self,
        *,
        only_prototype: bool = False,
        include_base: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=["primary_label"])
        cols: list[str] = []
        if include_base and not only_prototype:
            cols.extend([c for c in self.base.feature_columns if c in df.columns])
        if only_prototype or include_base:
            cols.extend([c for c in self.prototype_columns if c not in cols])
        if not cols:
            raise RuntimeError("No feature columns selected.")
        X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)
        return X, df["primary_label"].astype(str)


def build_prototype_features(
    corpus: str | Path | CorpusManifest,
    *,
    train_run_ids: Iterable[str],
    base_table: FeatureTable | None = None,
    include_dtw: bool = False,
    dtw_curve_column: str = "val_loss",
) -> PrototypeFeatureTable:
    manifest = corpus if isinstance(corpus, CorpusManifest) else load_manifest(corpus)
    base = base_table or build_feature_table(manifest)
    df = base.df.copy()
    if "primary_label" not in df.columns:
        raise RuntimeError("Base feature table must contain a primary_label column.")
    train_set = set(map(str, train_run_ids))
    train_mask = df.index.astype(str).isin(train_set)
    if not train_mask.any():
        raise RuntimeError("No train rows in feature table — cannot build prototypes.")
    feature_cols = base.feature_columns
    bank = build_prototype_bank(
        df.loc[train_mask, feature_cols],
        df.loc[train_mask, "primary_label"].astype(str),
    )
    neuc = normalized_euclidean(df[feature_cols], bank)
    corr = correlation_distance(df[feature_cols], bank)
    new_blocks = [neuc, corr]
    if include_dtw:
        curve_protos = build_curve_prototypes(
            manifest,
            train_run_ids=train_set,
            y_train=df.loc[train_mask, "primary_label"].astype(str),
            column=dtw_curve_column,
        )
        if curve_protos:
            dtw_df = dtw_distance_to_prototypes(
                manifest,
                run_ids=df.index.astype(str),
                curve_prototypes=curve_protos,
                column=dtw_curve_column,
            )
            dtw_df = dtw_df.reindex(df.index)
            new_blocks.append(dtw_df)
    proto_block = pd.concat(new_blocks, axis=1)
    full = pd.concat([df, proto_block], axis=1)
    proto_cols = list(proto_block.columns)
    return PrototypeFeatureTable(
        base=base,
        bank=bank,
        df=full,
        prototype_columns=proto_cols,
    )


__all__ = [
    "PrototypeBank",
    "PrototypeFeatureTable",
    "build_prototype_bank",
    "build_prototype_features",
    "correlation_distance",
    "dtw_distance_to_prototypes",
    "build_curve_prototypes",
    "normalized_euclidean",
]
