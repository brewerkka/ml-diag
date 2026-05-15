from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_diag.benchmark import partition_corpus
from ml_diag.data import CorpusManifest, load_manifest, load_run
from ml_diag.features.run_features import (
    FeatureTable,
    build_feature_table,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

_AGGS: tuple[str, ...] = ("mean", "std", "min", "max", "ptp")

_STABILITY_COLUMNS: tuple[str, ...] = (
    "val_loss_final",
    "val_loss_min",
    "val_acc_final",
    "val_acc_max",
    "train_acc_final",
    "acc_final_gap",
    "loss_final_gap",
    "val_loss_argmin_frac",
)


@dataclass(frozen=True)
class GroupedFeatureTable:
    corpus_name: str
    source: str
    df: pd.DataFrame
    group_to_run_ids: dict[str, list[str]]

    @property
    def feature_columns(self) -> list[str]:
        reserved = {"group_label", "n_runs", "slice", "n_unique_labels"}
        return [c for c in self.df.columns if c not in reserved]

    def aligned_xy(self) -> tuple[pd.DataFrame, pd.Series]:
        df = self.df.dropna(subset=["group_label"])
        X = df[self.feature_columns].copy().replace([np.inf, -np.inf], np.nan)
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0.0)
        y = df["group_label"].astype(str)
        return X, y


def _entry_id_for(
    run_dir: Path, *, manifest_entry_metadata: dict[str, dict] | None = None
) -> str | None:
    rid = run_dir.name
    if manifest_entry_metadata and rid in manifest_entry_metadata:
        eid = manifest_entry_metadata[rid].get("entry_id")
        if eid:
            return str(eid)
    try:
        rec = load_run(run_dir)
    except Exception as e:
        _LOG.warning("Could not load run %s for entry_id lookup: %s", run_dir, e)
        return None
    for key in ("entry_id", "config_id", "group_id"):
        v = rec.meta.get(key)
        if v is not None:
            return str(v)
    return None


def _majority_label(labels: list[str]) -> tuple[str | None, int]:
    labels = [l for l in labels if l is not None]
    if not labels:
        return None, 0
    counts = Counter(labels)
    most_common, _ = counts.most_common(1)[0]
    return most_common, len(counts)


def _stability_features(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {f"{prefix}_var": float("nan"), f"{prefix}_iqr": float("nan")}
    q75, q25 = np.percentile(finite, [75, 25])
    return {
        f"{prefix}_var": float(np.var(finite)),
        f"{prefix}_iqr": float(q75 - q25),
        f"{prefix}_range": float(np.max(finite) - np.min(finite)),
    }


def build_grouped_feature_table(
    corpus: str | Path | CorpusManifest,
    *,
    base_table: FeatureTable | None = None,
    use_partition: bool = True,
) -> GroupedFeatureTable:
    if isinstance(corpus, CorpusManifest):
        manifest = corpus
    else:
        manifest = load_manifest(corpus)
    base = base_table or build_feature_table(manifest)
    base_df = base.df.copy()
    em = manifest.entry_metadata or {}
    entry_ids: dict[str, str] = {}
    n_missing = 0
    for run_dir in manifest.run_dirs:
        rid = run_dir.name
        eid = _entry_id_for(run_dir, manifest_entry_metadata=em)
        if eid is None:
            n_missing += 1
            entry_ids[rid] = f"__solo__:{rid}"
        else:
            entry_ids[rid] = eid
    if n_missing:
        _LOG.warning(
            "%d/%d runs lack entry_id; they will form singleton groups.",
            n_missing,
            len(manifest.run_dirs),
        )
    base_df["entry_id"] = base_df.index.map(lambda rid: entry_ids.get(str(rid), f"__solo__:{rid}"))
    slice_by_run: dict[str, str] = {}
    if use_partition:
        try:
            partition = partition_corpus(manifest, skip_broken=True)
            slice_by_run = dict(
                zip(partition.table["run_id"].astype(str), partition.table["slice"])
            )
        except Exception as e:
            _LOG.warning("Could not compute partition for grouped features: %s", e)
            slice_by_run = {}
    group_rows: list[dict[str, Any]] = []
    group_to_run_ids: dict[str, list[str]] = {}
    primary_label_col = "primary_label" if "primary_label" in base_df.columns else None
    feature_cols = [c for c in base_df.columns if c not in ("entry_id", "primary_label")]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(base_df[c])]
    for entry_id, group in base_df.groupby("entry_id"):
        run_ids = list(group.index.astype(str))
        group_to_run_ids[str(entry_id)] = run_ids
        agg_row: dict[str, Any] = {"n_runs": int(len(group))}
        for col in feature_cols:
            values = group[col].to_numpy(dtype=float)
            if values.size == 0:
                continue
            finite = values[np.isfinite(values)]
            agg_row[f"{col}__mean"] = float(np.mean(finite)) if finite.size else float("nan")
            agg_row[f"{col}__std"] = float(np.std(finite)) if finite.size else float("nan")
            agg_row[f"{col}__min"] = float(np.min(finite)) if finite.size else float("nan")
            agg_row[f"{col}__max"] = float(np.max(finite)) if finite.size else float("nan")
            agg_row[f"{col}__ptp"] = (
                float(np.max(finite) - np.min(finite)) if finite.size else float("nan")
            )
        for col in _STABILITY_COLUMNS:
            if col not in group.columns:
                continue
            agg_row.update(
                _stability_features(group[col].to_numpy(dtype=float), prefix=f"stab_{col}")
            )
        labels = group[primary_label_col].astype(str).tolist() if primary_label_col else []
        majority, n_unique = _majority_label(labels)
        agg_row["group_label"] = majority
        agg_row["n_unique_labels"] = int(n_unique)
        if slice_by_run:
            slices_in_group = {slice_by_run.get(rid) for rid in run_ids}
            slices_in_group.discard(None)
            if slices_in_group == {"core"}:
                agg_row["slice"] = "core"
            elif slices_in_group == {"extended"}:
                agg_row["slice"] = "extended"
            elif slices_in_group:
                agg_row["slice"] = "mixed"
            else:
                agg_row["slice"] = None
        group_rows.append({"entry_id": str(entry_id), **agg_row})
    df = pd.DataFrame(group_rows).set_index("entry_id")
    _LOG.info(
        "Grouped feature table for %s: %d groups (avg group size = %.2f).",
        manifest.name,
        len(df),
        df["n_runs"].mean() if not df.empty else 0.0,
    )
    return GroupedFeatureTable(
        corpus_name=manifest.name,
        source=f"grouped({base.source})",
        df=df,
        group_to_run_ids=group_to_run_ids,
    )


def grouped_slices(
    table: GroupedFeatureTable,
    *,
    holdout_index: pd.Index | None = None,
) -> dict[str, pd.Index]:
    df = table.df
    out: dict[str, pd.Index] = {"full": df.index}
    if "slice" in df.columns:
        out["core"] = df.index[df["slice"] == "core"]
        out["extended"] = df.index[df["slice"] == "extended"]
        out["mixed"] = df.index[df["slice"] == "mixed"]
    if holdout_index is not None:
        h = pd.Index(holdout_index)
        out = {k: idx.intersection(h) for k, idx in out.items()}
    return out


__all__ = [
    "GroupedFeatureTable",
    "build_grouped_feature_table",
    "grouped_slices",
]
