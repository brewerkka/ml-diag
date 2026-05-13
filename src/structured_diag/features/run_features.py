from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from structured_diag.data import (
    CorpusManifest,
    RunRecord,
    load_manifest,
    load_run,
    load_runs_table,
)
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

_ADAPTER_CANDIDATES: tuple[str, ...] = (
    "features.parquet",
    "features.csv",
    "run_features.parquet",
    "run_features.csv",
)


@dataclass(frozen=True)
class FeatureTable:
    corpus_name: str
    source: str
    df: pd.DataFrame

    @property
    def feature_columns(self) -> list[str]:
        return [c for c in self.df.columns if c not in ("primary_label", "run_id")]

    def aligned_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=["primary_label"])
        X = df[self.feature_columns].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        y = df["primary_label"].astype(str)
        return X, y


def _try_load_adapter(corpus_path: Path) -> pd.DataFrame | None:
    _CSV_READ_ERRORS: tuple[type[BaseException], ...] = (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
        OSError,
        UnicodeDecodeError,
    )
    _PARQUET_READ_ERRORS: tuple[type[BaseException], ...] = (
        ValueError,
        OSError,
        ImportError,
    )
    for name in _ADAPTER_CANDIDATES:
        path = corpus_path / name
        if not path.is_file():
            continue
        _LOG.info("Adapter loading features from %s", path)
        expected_errors = _PARQUET_READ_ERRORS if path.suffix == ".parquet" else _CSV_READ_ERRORS
        try:
            if path.suffix == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)
        except expected_errors as e:
            _LOG.info(
                "Adapter unavailable (path=%s reason=%s: %s); "
                "fallback_reason=adapter_read_failed; will use history-based fallback.",
                path,
                type(e).__name__,
                e,
            )
            return None
    _LOG.info(
        "No pre-computed feature table found in %s; "
        "fallback_reason=no_adapter_file; will use history-based fallback.",
        corpus_path,
    )
    return None


def _safe_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    y = values.astype(float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return 0.0
    x = x[mask]
    y = y[mask]
    return float(np.polyfit(x, y, 1)[0])


def _curve_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_final": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_mean": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_slope": math.nan,
            f"{prefix}_slope_first_half": math.nan,
            f"{prefix}_slope_second_half": math.nan,
            f"{prefix}_max_jump": math.nan,
            f"{prefix}_n_nan": 0.0,
        }
    n = values.size
    half = max(1, n // 2)
    return {
        f"{prefix}_final": float(values[-1]),
        f"{prefix}_min": float(np.nanmin(values)) if np.isfinite(values).any() else math.nan,
        f"{prefix}_max": float(np.nanmax(values)) if np.isfinite(values).any() else math.nan,
        f"{prefix}_mean": float(np.nanmean(values)) if np.isfinite(values).any() else math.nan,
        f"{prefix}_std": float(np.nanstd(values)) if np.isfinite(values).any() else math.nan,
        f"{prefix}_slope": _slope(values),
        f"{prefix}_slope_first_half": _slope(values[:half]),
        f"{prefix}_slope_second_half": _slope(values[half:]),
        f"{prefix}_max_jump": float(np.nanmax(np.abs(np.diff(values)))) if n > 1 else 0.0,
        f"{prefix}_n_nan": float(np.sum(~np.isfinite(values))),
    }


def _gap_features(train: np.ndarray, val: np.ndarray, prefix: str) -> dict[str, float]:
    if train.size == 0 or val.size == 0:
        return {
            f"{prefix}_final_gap": math.nan,
            f"{prefix}_max_gap": math.nan,
            f"{prefix}_mean_gap": math.nan,
        }
    n = min(train.size, val.size)
    diff = train[:n] - val[:n]
    return {
        f"{prefix}_final_gap": float(diff[-1]),
        f"{prefix}_max_gap": float(np.nanmax(diff)) if np.isfinite(diff).any() else math.nan,
        f"{prefix}_mean_gap": float(np.nanmean(diff)) if np.isfinite(diff).any() else math.nan,
    }


def _features_for_run(rec: RunRecord) -> dict[str, float]:
    h = rec.history
    train_loss_col = _safe_first_existing(h, ("train_loss", "loss"))
    val_loss_col = _safe_first_existing(h, ("val_loss", "valid_loss", "test_loss"))
    train_acc_col = _safe_first_existing(h, ("train_acc", "acc", "train_accuracy"))
    val_acc_col = _safe_first_existing(h, ("val_acc", "valid_acc", "val_accuracy", "test_acc"))
    train_loss = h[train_loss_col].to_numpy(dtype=float) if train_loss_col else np.array([])
    val_loss = h[val_loss_col].to_numpy(dtype=float) if val_loss_col else np.array([])
    train_acc = h[train_acc_col].to_numpy(dtype=float) if train_acc_col else np.array([])
    val_acc = h[val_acc_col].to_numpy(dtype=float) if val_acc_col else np.array([])
    feats: dict[str, float] = {
        "n_epochs": float(len(h)),
        "diverged": float(
            (np.isnan(train_loss).any() if train_loss.size else False)
            or (np.isnan(val_loss).any() if val_loss.size else False)
        ),
    }
    feats.update(_curve_features(train_loss, "train_loss"))
    feats.update(_curve_features(val_loss, "val_loss"))
    feats.update(_curve_features(train_acc, "train_acc"))
    feats.update(_curve_features(val_acc, "val_acc"))
    feats.update(_gap_features(train_acc, val_acc, "acc"))
    feats.update(_gap_features(val_loss, train_loss, "loss"))
    if val_loss.size:
        try:
            feats["val_loss_argmin_frac"] = float(np.nanargmin(val_loss)) / max(
                1, val_loss.size - 1
            )
        except ValueError:
            feats["val_loss_argmin_frac"] = math.nan
    else:
        feats["val_loss_argmin_frac"] = math.nan
    for key in ("lr", "learning_rate", "batch_size", "weight_decay", "n_params"):
        v = rec.meta.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            feats[f"meta_{key}"] = float(v)
    return feats


_DI_PROXY_B1_KEEP: tuple[str, ...] = (
    "di_proxy_val_above_train_acc_frac",
    "di_proxy_val_acc_minus_train_acc_mean",
    "di_proxy_val_below_train_loss_frac",
    "di_proxy_loss_ratio_mean",
    "di_proxy_acc_lockstep_corr",
    "di_proxy_epoch_to_90pct_val_acc_frac",
    "di_proxy_best_minus_final_val_acc",
)


def _build_fallback(
    manifest: CorpusManifest,
    *,
    with_di_proxy: bool = True,
) -> pd.DataFrame:
    proxy_extractor = None
    if with_di_proxy:
        from structured_diag.features.data_integrity import (
            _proxy_features_from_curves as proxy_extractor,
        )
    from structured_diag.data.run_loader import RunLoadError

    rows: list[dict[str, float | str | None]] = []
    em_index = manifest.entry_metadata or {}
    for run_dir in manifest.run_dirs:
        em = em_index.get(run_dir.name)
        try:
            rec = load_run(run_dir, entry_metadata=em)
        except (RunLoadError, FileNotFoundError, OSError) as e:
            _LOG.warning("Skipping run %s in fallback features: %s", run_dir, e)
            continue
        feats = _features_for_run(rec)
        if proxy_extractor is not None:
            try:
                proxies = proxy_extractor(rec.history, pd.Series(feats))
                kept = {k: proxies[k] for k in _DI_PROXY_B1_KEEP if k in proxies}
                feats.update(kept)
            except (KeyError, ValueError, TypeError, ArithmeticError) as e:
                _LOG.warning(
                    "di_proxy extraction failed for run %s: %s; row will "
                    "carry NaN in di_proxy_* columns and be median-imputed.",
                    rec.run_id,
                    e,
                )
        feats["run_id"] = rec.run_id
        feats["primary_label"] = rec.primary_label
        rows.append(feats)
    if not rows:
        raise RuntimeError(f"Fallback feature builder produced no rows for {manifest.name}")
    df = pd.DataFrame(rows).set_index("run_id")
    return df


def build_feature_table(
    corpus: str | Path | CorpusManifest,
    *,
    prefer_adapter: bool = True,
    with_di_proxy: bool = True,
) -> FeatureTable:
    if isinstance(corpus, CorpusManifest):
        manifest = corpus
    else:
        manifest = load_manifest(corpus)
    if prefer_adapter:
        adapter_df = _try_load_adapter(manifest.corpus_path)
    else:
        adapter_df = None
    if adapter_df is not None:
        if "run_id" not in adapter_df.columns:
            raise RuntimeError("Adapter feature table is missing required column `run_id`.")
        adapter_df = adapter_df.set_index("run_id")
        run_table = load_runs_table(manifest, skip_broken=True).set_index("run_id")
        if "primary_label" not in adapter_df.columns:
            adapter_df["primary_label"] = run_table["primary_label"]
        for col in list(adapter_df.columns):
            if col in ("primary_label",):
                continue
            if not pd.api.types.is_numeric_dtype(adapter_df[col]):
                _LOG.info("Dropping non-numeric feature column %r", col)
                adapter_df = adapter_df.drop(columns=[col])
        return FeatureTable(
            corpus_name=manifest.name,
            source=f"adapter:{[n for n in _ADAPTER_CANDIDATES if (manifest.corpus_path / n).is_file()][0]}",
            df=adapter_df,
        )
    df = _build_fallback(manifest, with_di_proxy=with_di_proxy)
    src = "fallback:history_summary+di_proxy" if with_di_proxy else "fallback:history_summary"
    return FeatureTable(
        corpus_name=manifest.name,
        source=src,
        df=df,
    )


__all__ = ["FeatureTable", "build_feature_table"]
