"""Thin orchestration helpers shared by the CLI and the Streamlit UI.

The Streamlit demo (``ui/app.py``) and ``scripts/run_full_case.py`` both
need the same pipeline: load the cascade, build features, run the
cascade end-to-end, build evidence, interpret, optionally write
artifacts. This module is the single source of truth for that
orchestration so neither caller duplicates pipeline logic.

Public surface
--------------
* :func:`list_corpus_run_ids` — enumerate run_ids in a corpus.
* :func:`load_run_meta_and_history` — read meta.json + history.csv
  for a run (corpus or stand-alone). Used by UI to render curves
  and overview cards before training-time data goes through the
  cascade.
* :class:`CaseResult` — typed bundle of the case's diagnosis,
  evidence, interpretation and (optionally) the on-disk artifact
  paths.
* :func:`diagnose_case` — the main orchestration entry point.

The helpers are deliberately *non-Streamlit-aware*; they raise
ordinary Python exceptions on misuse. The UI catches those and
renders user-friendly errors.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from structured_diag.data import (
    CorpusManifest,
    load_manifest,
    load_run,
    manifest_for_single_run,
)
from structured_diag.evaluation import (
    StructuredEvidence,
    build_evidence,
    write_case_outputs,
)
from structured_diag.features import (
    build_data_integrity_features,
    build_feature_table,
)
from structured_diag.interpretation import (
    InterpretationConfig,
    InterpretationResult,
    interpret,
)
from structured_diag.models import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
    diagnose_one,
    load_cascade,
)
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def list_corpus_run_ids(corpus: str | Path | CorpusManifest) -> list[str]:
    """Return every run_id contained in a corpus manifest.

    Works on the real ``corpus.manifest.json`` schema (entries → run_ids)
    and on the simpler ``runs`` schema. Used by the UI to populate the
    run-id dropdown.
    """
    manifest = corpus if isinstance(corpus, CorpusManifest) else load_manifest(corpus)
    return [rd.name for rd in manifest.run_dirs]


def load_run_meta_and_history(
    *,
    corpus: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
    """Read meta.json + history.csv for one run (either corpus-bound or stand-alone).

    Returns
    -------
    (meta, history, run_dir)
    """
    if run_dir is not None:
        rd = Path(run_dir).expanduser().resolve()
    else:
        if corpus is None or run_id is None:
            raise ValueError("Provide either run_dir, or both corpus and run_id.")
        rd = Path(corpus).expanduser().resolve() / run_id
    if not rd.is_dir():
        raise FileNotFoundError(f"run directory not found: {rd}")
    rec = load_run(rd)
    return dict(rec.meta), rec.history.copy(), rd


# ---------------------------------------------------------------------------
# Case orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseResult:
    """Typed bundle returned by :func:`diagnose_case`.

    Always present:
        diagnosis, evidence, interpretation, meta, history, run_dir,
        feature_source, integrity_columns_available.

    Present when ``write_outputs=True``:
        case_dir, files.

    Present when ``patch_summary`` is not ``None``:
        patch_summary.
    """

    diagnosis: HierarchicalDiagnosis
    evidence: StructuredEvidence
    interpretation: InterpretationResult
    meta: dict[str, Any]
    history: pd.DataFrame
    run_dir: Path
    feature_source: str
    integrity_columns_available: int
    case_dir: Path | None = None
    files: dict[str, Path] = field(default_factory=dict)
    patch_summary: dict[str, Any] | None = None


def diagnose_case(
    *,
    artifacts_dir: str | Path,
    corpus: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
    backend: str = "template",
    model: str | None = None,
    max_recommendations: int = 3,
    language: str = "ru",
    no_integrity: bool = False,
    out_dir: str | Path | None = None,
    write_outputs: bool = True,
    cascade: HierarchicalCascade | None = None,
    feature_table_cache: Mapping[str, Any] | None = None,
) -> CaseResult:
    """Diagnose one run end-to-end.

    Resolves the input mode (corpus + run_id, or stand-alone run_dir),
    builds features, runs the cascade, builds evidence, interprets, and
    optionally writes the canonical case directory.

    Parameters
    ----------
    artifacts_dir:
        Directory containing the trained cascade joblibs (output of
        ``scripts/run_hierarchical_train.py``).
    corpus, run_id:
        Use these together for a corpus-bound run.
    run_dir:
        Use this for a stand-alone run produced e.g. by
        :class:`structured_diag.logging_sdk.RunLogger`.
    backend, model, max_recommendations:
        Forwarded to :class:`InterpretationConfig`.
    no_integrity:
        Skip the integrity-feature layer (the diagnosis still runs but
        drops `di_*` columns).
    out_dir, write_outputs:
        When ``write_outputs`` is True, the canonical case folder is
        written to ``out_dir`` (default: ``results/cases/<run_id>/``).
    cascade, feature_table_cache:
        Optional pre-loaded cascade / pre-built feature table; the UI
        uses these to avoid reloading on every button click.
    """
    # -------- Resolve input mode -----------------------------------------
    if (run_dir is not None) == (run_id is not None):
        raise ValueError(
            "Provide exactly one of: run_dir (stand-alone) or run_id "
            "(corpus-bound). They are mutually exclusive."
        )
    if run_dir is not None:
        manifest = manifest_for_single_run(run_dir)
        rec = load_run(Path(run_dir))
        resolved_run_id = rec.run_id
        corpus_path_for_meta = Path(run_dir).expanduser().resolve().parent
        resolved_run_dir = Path(run_dir).expanduser().resolve()
    else:
        if corpus is None:
            raise ValueError("corpus must be provided when run_id is set.")
        manifest = load_manifest(corpus)
        resolved_run_id = run_id
        corpus_path_for_meta = Path(corpus).expanduser().resolve()
        resolved_run_dir = corpus_path_for_meta / run_id
        if not resolved_run_dir.is_dir():
            raise FileNotFoundError(f"run_id {run_id!r} not found in corpus {corpus!r}.")
    # -------- Cascade ----------------------------------------------------
    if cascade is None:
        cascade = load_cascade(artifacts_dir)
    # -------- Features ---------------------------------------------------
    if feature_table_cache is not None and feature_table_cache.get("base") is not None:
        base = feature_table_cache["base"]
    else:
        base = build_feature_table(manifest)
    integrity_columns: list[str] | None = None
    full_df = base.df
    if not no_integrity:
        try:
            di = build_data_integrity_features(manifest, base_table=base)
            integrity_columns = di.integrity_columns
            full_df = di.df
        except Exception as e:
            _LOG.warning("integrity features unavailable (%s); continuing without.", e)
    if resolved_run_id not in full_df.index:
        raise KeyError(
            f"run_id {resolved_run_id!r} not in the feature table; index={list(full_df.index)[:5]}…"
        )
    feature_cols = base.feature_columns
    # Pass the row as a Series (with column-name index) so per-stage
    # alignment in diagnose_one can reindex by name. Bypassing this with
    # ``.values`` would skip alignment and break on schema drift.
    x_row = full_df.loc[resolved_run_id, feature_cols]
    # -------- Diagnosis --------------------------------------------------
    diagnosis = diagnose_one(cascade, run_id=resolved_run_id, x_row=x_row)
    # -------- Evidence ---------------------------------------------------
    evidence = build_evidence(
        diagnosis=diagnosis,
        feature_row=full_df.loc[resolved_run_id],
        cascade=cascade,
        integrity_columns=integrity_columns,
    )
    # -------- Interpretation --------------------------------------------
    cfg = InterpretationConfig(
        backend=backend,
        model=model,
        max_recommendations=max_recommendations,
        language=language,
    )
    interpretation = interpret(diagnosis=diagnosis, evidence=evidence, config=cfg)
    # -------- Read meta + history (for UI rendering) --------------------
    meta = (
        json.loads((resolved_run_dir / "meta.json").read_text(encoding="utf-8"))
        if (resolved_run_dir / "meta.json").is_file()
        else {}
    )
    history = (
        pd.read_csv(resolved_run_dir / "history.csv")
        if (resolved_run_dir / "history.csv").is_file()
        else pd.DataFrame()
    )
    # -------- Persist (optional) ----------------------------------------
    case_dir = None
    files: dict[str, Path] = {}
    if write_outputs:
        case_dir = (
            Path(out_dir) if out_dir is not None else (Path("results/cases") / resolved_run_id)
        )
        files = write_case_outputs(
            case_dir,
            diagnosis=diagnosis,
            evidence=evidence,
            interpretation=interpretation,
            patch_summary=None,
            run_dir=resolved_run_dir,
            extras={
                "corpus_path": str(corpus_path_for_meta),
                "input_mode": "run_dir" if run_dir is not None else "corpus",
                "artifacts_path": str(artifacts_dir),
                "interpretation_backend": interpretation.backend,
                "feature_source": base.source,
                "integrity_columns_available": (len(integrity_columns) if integrity_columns else 0),
            },
        )
    return CaseResult(
        diagnosis=diagnosis,
        evidence=evidence,
        interpretation=interpretation,
        meta=meta,
        history=history,
        run_dir=resolved_run_dir,
        feature_source=base.source,
        integrity_columns_available=(len(integrity_columns) if integrity_columns else 0),
        case_dir=case_dir,
        files=files,
    )


__all__ = [
    "CaseResult",
    "diagnose_case",
    "list_corpus_run_ids",
    "load_run_meta_and_history",
]
