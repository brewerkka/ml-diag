from __future__ import annotations

import csv
import json
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_diag.logging_sdk.schemas import (
    ALLOWED_STATUSES,
    DEFAULT_HISTORY_COLUMNS,
    REQUIRED_META_KEYS,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        import numpy as _np                                  

        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


class RunLoggerError(RuntimeError):
    pass


class RunLogger:
    def __init__(
        self,
        output_dir: str | Path,
        meta: Mapping[str, Any],
        *,
        history_columns: Iterable[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        for key in REQUIRED_META_KEYS:
            if key not in meta or meta[key] in (None, ""):
                raise RunLoggerError(
                    f"meta must include the required key {key!r}; got keys={list(meta.keys())}"
                )
        self.output_dir = Path(output_dir).expanduser().resolve()
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise RunLoggerError(f"output_dir exists and is not a directory: {self.output_dir}")
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not overwrite:
            existing_meta = self.output_dir / "meta.json"
            existing_history = self.output_dir / "history.csv"
            if existing_meta.exists() or existing_history.exists():
                raise RunLoggerError(
                    f"output_dir is not empty and overwrite=False: {self.output_dir}"
                )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.output_dir / "meta.json"
        self.history_path = self.output_dir / "history.csv"
        cols = tuple(history_columns) if history_columns else DEFAULT_HISTORY_COLUMNS
        if "epoch" not in cols:
            raise RunLoggerError("history_columns must include 'epoch'")
        self._history_columns: tuple[str, ...] = cols
        self._history_writer: csv.DictWriter | None = None
        self._history_file = None                            
        self._meta: dict[str, Any] = {str(k): _to_jsonable(v) for k, v in meta.items()}
        self._meta["status"] = "running"
        self._meta.setdefault("created_at", _now_iso())
        self._meta.setdefault("framework_version", None)
        self._meta.setdefault("python_version", sys.version.split()[0])
        self._meta.setdefault("tags", [])
        self._meta.setdefault("notes", None)
        self._meta["n_epochs_logged"] = 0
        self._meta["final_metrics"] = None
        self._meta["finalized_at"] = None
        self._meta["duration_sec"] = None
        self._t0 = time.time()
        self._finalized = False
        self._write_meta()
        self._open_history()
        _LOG.info("RunLogger initialised at %s (run_id=%s)", self.output_dir, self._meta["run_id"])

    def log_epoch(self, **fields: Any) -> None:
        if self._finalized:
            raise RunLoggerError("log_epoch called after finalize()")
        if "epoch" not in fields:
            raise RunLoggerError("log_epoch: 'epoch' is required")
        row: dict[str, Any] = {}
        for col in self._history_columns:
            v = fields.get(col)
            if isinstance(v, bool):
                row[col] = int(v)
            elif isinstance(v, float) and (v != v or v in (float("inf"), float("-inf"))):
                row[col] = ""
            elif v is None:
                row[col] = ""
            else:
                row[col] = v
        unknown = sorted(set(fields) - set(self._history_columns))
        if unknown:
            _LOG.warning(
                "RunLogger.log_epoch: ignoring unknown columns %s (known: %s)",
                unknown,
                list(self._history_columns),
            )
        assert self._history_writer is not None
        self._history_writer.writerow(row)
        if self._history_file is not None:
            self._history_file.flush()
        self._meta["n_epochs_logged"] = int(self._meta.get("n_epochs_logged", 0)) + 1

    def finalize(
        self,
        *,
        status: str = "completed",
        final_metrics: Mapping[str, Any] | None = None,
        notes: str | None = None,
    ) -> None:
        if self._finalized:
            return
        if status not in ALLOWED_STATUSES:
            raise RunLoggerError(f"unknown status {status!r}; allowed: {ALLOWED_STATUSES}")
        self._meta["status"] = status
        self._meta["finalized_at"] = _now_iso()
        self._meta["duration_sec"] = float(time.time() - self._t0)
        if final_metrics is not None:
            self._meta["final_metrics"] = _to_jsonable(dict(final_metrics))
        if notes is not None:
            self._meta["notes"] = str(notes)
        self._close_history()
        self._write_meta()
        self._finalized = True
        _LOG.info(
            "RunLogger finalised at %s (status=%s, epochs=%d, duration=%.1fs)",
            self.output_dir,
            status,
            int(self._meta["n_epochs_logged"]),
            float(self._meta["duration_sec"]),
        )

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def __enter__(self) -> RunLogger:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._finalized:
            try:
                self.finalize(status="crashed" if exc_type is not None else "completed")
            except Exception:
                _LOG.exception("RunLogger.finalize raised during __exit__")

    def _write_meta(self) -> None:
        head_keys = (
            "run_id",
            "dataset_name",
            "task_type",
            "model_name",
            "framework",
            "framework_version",
            "python_version",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epochs_planned",
            "seed",
            "status",
            "created_at",
            "finalized_at",
            "duration_sec",
            "n_epochs_logged",
            "final_metrics",
            "tags",
            "notes",
        )
        ordered: dict[str, Any] = {}
        for k in head_keys:
            if k in self._meta:
                ordered[k] = self._meta[k]
        for k, v in self._meta.items():
            if k not in ordered:
                ordered[k] = v
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.meta_path)

    def _open_history(self) -> None:
        if self._history_writer is not None:
            return
        self._history_file = self.history_path.open("w", encoding="utf-8", newline="")
        self._history_writer = csv.DictWriter(
            self._history_file,
            fieldnames=self._history_columns,
            restval="",
        )
        self._history_writer.writeheader()
        self._history_file.flush()

    def _close_history(self) -> None:
        if self._history_file is not None:
            try:
                self._history_file.flush()
                self._history_file.close()
            except Exception:
                _LOG.exception("RunLogger: failed to close history.csv")
        self._history_writer = None
        self._history_file = None


__all__ = ["RunLogger", "RunLoggerError"]
