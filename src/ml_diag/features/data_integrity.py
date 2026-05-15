from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_diag.data import CorpusManifest, load_manifest, load_run
from ml_diag.features.run_features import FeatureTable, build_feature_table
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

_META_KEYS: dict[str, tuple[str, ...]] = {
    "di_train_val_overlap": (
        "train_val_overlap",
        "trainval_overlap",
        "duplicate_overlap_ratio",
    ),
    "di_duplicate_index_overlap": (
        "duplicate_index_overlap",
        "duplicate_indices_ratio",
        "exact_duplicates_ratio",
    ),
    "di_replay_evidence": (
        "replay_evidence",
        "augmentation_replay",
        "replay_indicator",
    ),
    "di_mi_target_features": (
        "mi_target_features",
        "mutual_information_features_target",
        "mi_features_target",
    ),
    "di_split_hash_collision": (
        "split_hash_collision",
        "split_hash_collisions",
        "split_integrity_collision_rate",
    ),
    "di_label_noise_rate_declared": (
        "label_noise_rate",
        "declared_noise_rate",
        "noise_ratio",
    ),
    "di_split_seed": (
        "split_seed",
        "data_split_seed",
    ),
}


@dataclass(frozen=True)
class DataIntegrityFeatureTable:
    base: FeatureTable
    df: pd.DataFrame
    integrity_columns: list[str]
    proxy_columns: list[str]
    meta_columns: list[str]

    def aligned_xy(
        self,
        *,
        include_base: bool = True,
        only_integrity: bool = False,
    ) -> tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=["primary_label"])
        cols: list[str] = []
        if include_base and not only_integrity:
            cols.extend([c for c in self.base.feature_columns if c in df.columns])
        cols.extend([c for c in self.integrity_columns if c not in cols])
        if not cols:
            raise RuntimeError("No feature columns selected.")
        X = df[cols].copy().replace([np.inf, -np.inf], np.nan)
        for c in X.columns:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].median() if X[c].notna().any() else 0.0)
        return X, df["primary_label"].astype(str)


def _coerce_scalar(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _meta_features_for_run(meta: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for new_name, candidates in _META_KEYS.items():
        for key in candidates:
            if key in meta:
                out[new_name] = _coerce_scalar(meta[key])
                if out[new_name] is not None:
                    break
        else:
            out[new_name] = None
    return out


def _proxy_features_from_curves(rec_history: pd.DataFrame, base_row: pd.Series) -> dict[str, float]:
    proxies: dict[str, float] = {}
    val_acc_max = float(base_row.get("val_acc_max", np.nan))
    train_acc_final = float(base_row.get("train_acc_final", np.nan))
    acc_final_gap = float(base_row.get("acc_final_gap", np.nan))
    proxies["di_proxy_saturation"] = float(val_acc_max if np.isfinite(val_acc_max) else np.nan)
    proxies["di_proxy_zero_gap_ceiling"] = float(
        max(0.0, val_acc_max - abs(acc_final_gap))
        if np.isfinite(val_acc_max) and np.isfinite(acc_final_gap)
        else np.nan
    )
    if np.isfinite(train_acc_final) and np.isfinite(val_acc_max):
        denom = max(1e-6, 1.0 - val_acc_max)
        raw = (1.0 - abs(train_acc_final - val_acc_max)) / denom
        proxies["di_proxy_train_val_lockstep"] = float(np.clip(raw, 0.0, 50.0))
    else:
        proxies["di_proxy_train_val_lockstep"] = float("nan")
    if "val_loss" in rec_history.columns:
        v = rec_history["val_loss"].to_numpy(dtype=float)
        if v.size > 4:
            unique = float(len(np.unique(np.round(v[np.isfinite(v)], 6)))) / max(
                1, np.isfinite(v).sum()
            )
            proxies["di_proxy_val_loss_uniqueness"] = unique
        else:
            proxies["di_proxy_val_loss_uniqueness"] = float("nan")
    else:
        proxies["di_proxy_val_loss_uniqueness"] = float("nan")
    if "val_loss" in rec_history.columns:
        v = rec_history["val_loss"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 4:
            head = v[: max(2, v.size // 5)]
            proxies["di_proxy_early_val_flatness"] = float(np.clip(np.std(head), 0.0, 50.0))
        else:
            proxies["di_proxy_early_val_flatness"] = float("nan")
    else:
        proxies["di_proxy_early_val_flatness"] = float("nan")

    def _both_curves(col_a: str, col_b: str) -> tuple[np.ndarray, np.ndarray] | None:
        if col_a not in rec_history.columns or col_b not in rec_history.columns:
            return None
        a = rec_history[col_a].to_numpy(dtype=float)
        b = rec_history[col_b].to_numpy(dtype=float)
        if a.size == 0 or b.size == 0:
            return None
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            return None
        return a[mask], b[mask]

    pair = _both_curves("train_acc", "val_acc")
    if pair is not None:
        ta, va = pair
        proxies["di_proxy_val_above_train_acc_frac"] = float((va > ta).mean())
        proxies["di_proxy_val_acc_minus_train_acc_mean"] = float((va - ta).mean())
    else:
        proxies["di_proxy_val_above_train_acc_frac"] = float("nan")
        proxies["di_proxy_val_acc_minus_train_acc_mean"] = float("nan")
    pair = _both_curves("train_loss", "val_loss")
    if pair is not None:
        tl, vl = pair
        proxies["di_proxy_val_below_train_loss_frac"] = float((vl < tl).mean())
        per_epoch_ratio = np.clip(vl / np.maximum(tl, 1e-9), 0.0, 50.0)
        per_epoch_ratio = per_epoch_ratio[np.isfinite(per_epoch_ratio)]
        proxies["di_proxy_loss_ratio_mean"] = (
            float(per_epoch_ratio.mean()) if per_epoch_ratio.size else float("nan")
        )
    else:
        proxies["di_proxy_val_below_train_loss_frac"] = float("nan")
        proxies["di_proxy_loss_ratio_mean"] = float("nan")
    pair = _both_curves("train_acc", "val_acc")
    if pair is not None:
        ta, va = pair
        if ta.std() > 1e-9 and va.std() > 1e-9:
            corr = float(np.corrcoef(ta, va)[0, 1])
            proxies["di_proxy_acc_lockstep_corr"] = corr
        else:
            proxies["di_proxy_acc_lockstep_corr"] = float("nan")
    else:
        proxies["di_proxy_acc_lockstep_corr"] = float("nan")
    if "val_acc" in rec_history.columns:
        va = rec_history["val_acc"].to_numpy(dtype=float)
        va = va[np.isfinite(va)]
        if va.size > 1:
            target = 0.9 * va.max()
            reached = np.where(va >= target)[0]
            if reached.size:
                proxies["di_proxy_epoch_to_90pct_val_acc_frac"] = float(reached[0]) / float(
                    va.size - 1
                )
            else:
                proxies["di_proxy_epoch_to_90pct_val_acc_frac"] = 1.0
        else:
            proxies["di_proxy_epoch_to_90pct_val_acc_frac"] = float("nan")
    else:
        proxies["di_proxy_epoch_to_90pct_val_acc_frac"] = float("nan")
    if "val_acc" in rec_history.columns:
        va = rec_history["val_acc"].to_numpy(dtype=float)
        va = va[np.isfinite(va)]
        if va.size >= 2:
            proxies["di_proxy_best_minus_final_val_acc"] = float(va.max() - va[-1])
        else:
            proxies["di_proxy_best_minus_final_val_acc"] = float("nan")
    else:
        proxies["di_proxy_best_minus_final_val_acc"] = float("nan")
    return proxies


def build_data_integrity_features(
    corpus: str | Path | CorpusManifest,
    *,
    base_table: FeatureTable | None = None,
) -> DataIntegrityFeatureTable:
    manifest = corpus if isinstance(corpus, CorpusManifest) else load_manifest(corpus)
    base = base_table or build_feature_table(manifest, with_di_proxy=False)
    df = base.df.copy()
    if "primary_label" not in df.columns:
        raise RuntimeError("Base feature table must contain a primary_label column.")
    rows: list[dict[str, Any]] = []
    em_index = manifest.entry_metadata or {}
    for run_dir in manifest.run_dirs:
        rid = str(run_dir.name)
        if rid not in df.index:
            continue
        try:
            rec = load_run(run_dir, entry_metadata=em_index.get(rid))
        except Exception as e:
            _LOG.warning("Skipping run %s in integrity features: %s", run_dir, e)
            continue
        meta_feats = _meta_features_for_run(rec.meta)
        proxies = _proxy_features_from_curves(rec.history, df.loc[rid])
        row = {"run_id": rid, **meta_feats, **proxies}
        rows.append(row)
    if not rows:
        raise RuntimeError("No integrity rows produced — corpus appears empty.")
    int_df = pd.DataFrame(rows).set_index("run_id")
    full = df.join(int_df, how="left")
    meta_cols = list(_META_KEYS)
    proxy_cols = [c for c in int_df.columns if c.startswith("di_proxy_")]
    integrity_cols = meta_cols + proxy_cols
    return DataIntegrityFeatureTable(
        base=base,
        df=full,
        integrity_columns=integrity_cols,
        proxy_columns=proxy_cols,
        meta_columns=meta_cols,
    )


def leakage_vs_healthy_diagnostic(
    table: DataIntegrityFeatureTable,
) -> pd.DataFrame:
    df = table.df
    sub = df[df["primary_label"].isin(["healthy", "leakage"])].copy()
    if sub.empty:
        return pd.DataFrame()
    agg = (
        sub.groupby("primary_label")[table.integrity_columns].agg(["count", "mean", "std"]).round(4)
    )
    return agg


__all__ = [
    "DataIntegrityFeatureTable",
    "build_data_integrity_features",
    "leakage_vs_healthy_diagnostic",
]
