from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_diag.data.manifest_loader import CorpusManifest, load_manifest
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

DEFAULT_META_FILENAME = "meta.json"

DEFAULT_HISTORY_FILENAME = "history.csv"


class RunLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    meta: dict[str, Any]
    history: pd.DataFrame
    dataset: str | None = None
    labels: tuple[str, ...] = ()
    severity: str | None = None
    is_multi_label: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_history_rows(self) -> int:
        return int(len(self.history))

    @property
    def primary_label(self) -> str | None:
        return self.labels[0] if self.labels else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RunLoadError(f"Required file missing: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise RunLoadError(f"Invalid JSON at {path}: {e}") from e
    if not isinstance(data, dict):
        raise RunLoadError(f"Expected JSON object at {path}, got {type(data).__name__}.")
    return data


def _read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RunLoadError(f"Required file missing: {path}")
    try:
        df = pd.read_csv(path)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
        OSError,
        UnicodeDecodeError,
    ) as e:
        raise RunLoadError(f"Could not parse history CSV at {path}: {e}") from e
    if df.empty:
        raise RunLoadError(f"History CSV is empty: {path}")
    return df


def _coerce_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(x) for x in value if x is not None)
    raise RunLoadError(f"Unsupported label value: {value!r}")


def _extract_label_block(meta: dict[str, Any]) -> tuple[tuple[str, ...], bool]:
    labels: tuple[str, ...] = ()
    for key in ("labels", "diagnosis", "label", "fault_labels"):
        if key in meta:
            labels = _coerce_labels(meta[key])
            if labels:
                break
    declared_multi = bool(meta.get("multi_label", False))
    is_multi = declared_multi or len(labels) > 1
    return labels, is_multi


def _extract_dataset(meta: dict[str, Any]) -> str | None:
    for key in ("dataset", "dataset_name"):
        v = meta.get(key)
        if isinstance(v, str):
            return v
    cfg = meta.get("config")
    if isinstance(cfg, dict):
        task = meta.get("task")
        if isinstance(task, str) and isinstance(cfg.get(task), dict):
            v = cfg[task].get("dataset")
            if isinstance(v, str):
                return v
        for sub in cfg.values():
            if isinstance(sub, dict) and isinstance(sub.get("dataset"), str):
                return sub["dataset"]
    return None


def load_run(
    run_dir: str | Path,
    meta_filename: str = DEFAULT_META_FILENAME,
    history_filename: str = DEFAULT_HISTORY_FILENAME,
    *,
    entry_metadata: dict[str, Any] | None = None,
) -> RunRecord:
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise RunLoadError(f"Run directory does not exist: {run_dir}")
    meta = _read_json(run_dir / meta_filename)
    history = _read_history(run_dir / history_filename)
    labels, is_multi = _extract_label_block(meta)
    dataset = _extract_dataset(meta)
    severity_raw = meta.get("severity") or meta.get("fault_severity")
    severity = str(severity_raw).strip().lower() if severity_raw is not None else None
    run_id = str(meta.get("run_id") or run_dir.name)
    if entry_metadata:
        em_labels = _coerce_labels(entry_metadata.get("primary_label"))
        if em_labels:
            labels = em_labels
        em_sev = entry_metadata.get("severity")
        if em_sev is not None:
            severity = str(em_sev).strip().lower()
        if "is_multi_label" in entry_metadata:
            is_multi = bool(entry_metadata["is_multi_label"])
    promoted_keys = {
        "run_id",
        "labels",
        "label",
        "diagnosis",
        "fault_labels",
        "multi_label",
        "dataset",
        "dataset_name",
        "severity",
        "fault_severity",
    }
    extra = {k: v for k, v in meta.items() if k not in promoted_keys}
    if entry_metadata and "entry_id" in entry_metadata:
        extra.setdefault("entry_id", entry_metadata["entry_id"])
    return RunRecord(
        run_id=run_id,
        run_dir=run_dir,
        meta=meta,
        history=history,
        dataset=dataset if isinstance(dataset, str) or dataset is None else str(dataset),
        labels=labels,
        severity=severity,
        is_multi_label=is_multi,
        extra=extra,
    )


def _iter_run_dirs(corpus: CorpusManifest) -> Iterable[Path]:
    yield from corpus.run_dirs


def load_runs_table(
    corpus: str | Path | CorpusManifest,
    *,
    skip_broken: bool = False,
    meta_filename: str = DEFAULT_META_FILENAME,
    history_filename: str = DEFAULT_HISTORY_FILENAME,
) -> pd.DataFrame:
    if isinstance(corpus, (str, Path)):
        manifest = load_manifest(corpus)
    else:
        manifest = corpus
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for run_dir in _iter_run_dirs(manifest):
        em = manifest.entry_metadata.get(run_dir.name) if manifest.entry_metadata else None
        try:
            rec = load_run(
                run_dir,
                meta_filename=meta_filename,
                history_filename=history_filename,
                entry_metadata=em,
            )
        except RunLoadError as e:
            if not skip_broken:
                raise
            n_skipped += 1
            _LOG.warning("Skipping broken run %s: %s", run_dir, e)
            continue
        rows.append(
            {
                "run_id": rec.run_id,
                "run_dir": str(rec.run_dir),
                "dataset": rec.dataset,
                "primary_label": rec.primary_label,
                "labels": list(rec.labels),
                "n_labels": len(rec.labels),
                "is_multi_label": rec.is_multi_label,
                "severity": rec.severity,
                "n_history_rows": rec.n_history_rows,
                "entry_id": rec.extra.get("entry_id"),
            }
        )
    if not rows:
        raise RunLoadError(
            f"No runs could be loaded from corpus {manifest.corpus_path} (skipped {n_skipped})."
        )
    df = pd.DataFrame(rows)
    _LOG.info(
        "Built run table for %s: %d runs (%d skipped).",
        manifest.name,
        len(df),
        n_skipped,
    )
    return df
