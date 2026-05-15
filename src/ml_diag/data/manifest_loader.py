from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

DEFAULT_MANIFEST_FILENAME = "corpus.manifest.json"


class CorpusManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class CorpusManifest:
    corpus_path: Path
    name: str
    run_dirs: tuple[Path, ...]
    splits: dict[str, tuple[str, ...]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    entry_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return len(self.run_dirs)

    @property
    def has_splits(self) -> bool:
        return bool(self.splits)

    @property
    def has_entry_metadata(self) -> bool:
        return bool(self.entry_metadata)


def _normalize_run_entry(entry: Any, corpus_path: Path) -> Path:
    if isinstance(entry, str):
        candidate = Path(entry)
        return candidate if candidate.is_absolute() else corpus_path / candidate
    if isinstance(entry, dict):
        for key in ("path", "dir", "run_dir"):
            if key in entry:
                candidate = Path(entry[key])
                return candidate if candidate.is_absolute() else corpus_path / candidate
        if "run_id" in entry:
            return corpus_path / str(entry["run_id"])
    raise CorpusManifestError(
        f"Unsupported run entry shape: {entry!r}. "
        "Expected str, or dict with one of: path, dir, run_dir, run_id."
    )


def _normalize_splits(raw_splits: Any) -> dict[str, tuple[str, ...]]:
    if raw_splits is None:
        return {}
    if not isinstance(raw_splits, dict):
        raise CorpusManifestError(
            f"`splits` must be a dict mapping split_name -> list[run_id], got {type(raw_splits)}"
        )
    out: dict[str, tuple[str, ...]] = {}
    for split_name, ids in raw_splits.items():
        if not isinstance(ids, (list, tuple)):
            raise CorpusManifestError(f"`splits[{split_name!r}]` must be a list/tuple of run_ids.")
        out[str(split_name)] = tuple(str(x) for x in ids)
    return out


def load_manifest(
    corpus_path: str | Path,
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME,
) -> CorpusManifest:
    corpus_path = Path(corpus_path).expanduser().resolve()
    if not corpus_path.is_dir():
        raise CorpusManifestError(f"Corpus path is not a directory: {corpus_path}")
    manifest_path = corpus_path / manifest_filename
    if not manifest_path.is_file():
        raise CorpusManifestError(
            f"Manifest not found at {manifest_path}. Did you point --corpus at the right directory?"
        )
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CorpusManifestError(f"Manifest is not valid JSON ({manifest_path}): {e}") from e
    if not isinstance(raw, dict):
        raise CorpusManifestError(
            f"Manifest root must be a JSON object, got {type(raw).__name__} ({manifest_path})."
        )
    runs_raw = raw.get("runs")
    entries_raw = raw.get("entries")
    run_dirs: tuple[Path, ...]
    entry_metadata: dict[str, dict[str, Any]] = {}
    if runs_raw is not None:
        if not isinstance(runs_raw, list) or not runs_raw:
            raise CorpusManifestError(
                f"Manifest key `runs` must be a non-empty list ({manifest_path})."
            )
        run_dirs = tuple(_normalize_run_entry(e, corpus_path) for e in runs_raw)
    elif entries_raw is not None:
        if not isinstance(entries_raw, list) or not entries_raw:
            raise CorpusManifestError(
                f"Manifest key `entries` must be a non-empty list ({manifest_path})."
            )
        flat: list[Path] = []
        for entry in entries_raw:
            if not isinstance(entry, dict):
                raise CorpusManifestError(
                    f"Each `entries` item must be an object, got {type(entry).__name__}."
                )
            run_ids = entry.get("run_ids") or []
            if not isinstance(run_ids, list):
                raise CorpusManifestError(
                    f"`entries[*].run_ids` must be a list, got {type(run_ids).__name__}."
                )
            entry_id = str(entry.get("entry_id") or "")
            primary_label = entry.get("primary_label")
            severity = entry.get("severity")
            fault_plan = entry.get("fault_plan") or []
            n_faults = len(fault_plan) if isinstance(fault_plan, list) else 0
            for rid in run_ids:
                rid_s = str(rid)
                flat.append(corpus_path / rid_s)
                entry_metadata[rid_s] = {
                    "entry_id": entry_id,
                    "primary_label": primary_label,
                    "severity": severity,
                    "fault_plan": fault_plan,
                    "is_multi_label": n_faults > 1,
                }
        if not flat:
            raise CorpusManifestError(f"Manifest `entries` produced no runs ({manifest_path}).")
        run_dirs = tuple(flat)
    else:
        raise CorpusManifestError(
            f"Manifest is missing both `runs` and `entries` ({manifest_path})."
        )
    splits = _normalize_splits(raw.get("splits"))
    name = str(raw.get("name") or raw.get("corpus") or raw.get("corpus_id") or corpus_path.name)
    _LOG.info(
        "Loaded manifest %s: %d runs, splits=%s, entry_metadata=%s",
        manifest_path,
        len(run_dirs),
        sorted(splits) if splits else "none",
        len(entry_metadata),
    )
    return CorpusManifest(
        corpus_path=corpus_path,
        name=name,
        run_dirs=run_dirs,
        splits=splits,
        raw=raw,
        entry_metadata=entry_metadata,
    )


def manifest_for_single_run(run_dir: str | Path) -> CorpusManifest:
    rd = Path(run_dir).expanduser().resolve()
    if not rd.is_dir():
        raise CorpusManifestError(f"run_dir is not a directory: {rd}")
    if not (rd / "meta.json").is_file():
        raise CorpusManifestError(f"run_dir does not contain meta.json: {rd}")
    if not (rd / "history.csv").is_file():
        raise CorpusManifestError(f"run_dir does not contain history.csv: {rd}")
    parent = rd.parent
    return CorpusManifest(
        corpus_path=parent,
        name=f"{parent.name}_{rd.name}_singlerun",
        run_dirs=(rd,),
        splits={},
        raw={"single_run": True},
        entry_metadata={},
    )
