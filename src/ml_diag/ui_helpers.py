
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_diag.data import (
    CorpusManifest,
    load_manifest,
    load_run,
    manifest_for_single_run,
)
from ml_diag.evaluation import (
    StructuredEvidence,
    build_evidence,
    write_case_outputs,
)
from ml_diag.features import (
    build_data_integrity_features,
    build_feature_table,
)
from ml_diag.interpretation import (
    InterpretationConfig,
    InterpretationResult,
    interpret,
)
from ml_diag.models import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
    diagnose_one,
    load_cascade,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)


                                                                             
                   
                                                                             


def list_corpus_run_ids(corpus: str | Path | CorpusManifest) -> list[str]:
    manifest = corpus if isinstance(corpus, CorpusManifest) else load_manifest(corpus)
    return [rd.name for rd in manifest.run_dirs]


def load_run_meta_and_history(
    *,
    corpus: str | Path | None = None,
    run_id: str | None = None,
    run_dir: str | Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Path]:
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


                                                                             
                    
                                                                             


@dataclass(frozen=True)
class CaseResult:

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
                                                                           
    if cascade is None:
        cascade = load_cascade(artifacts_dir)
                                                                           
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
                                                                    
                                                                        
                                                                 
    x_row = full_df.loc[resolved_run_id, feature_cols]
                                                                           
    diagnosis = diagnose_one(cascade, run_id=resolved_run_id, x_row=x_row)
                                                                           
    evidence = build_evidence(
        diagnosis=diagnosis,
        feature_row=full_df.loc[resolved_run_id],
        cascade=cascade,
        integrity_columns=integrity_columns,
    )
                                                                          
    cfg = InterpretationConfig(
        backend=backend,
        model=model,
        max_recommendations=max_recommendations,
        language=language,
    )
    interpretation = interpret(diagnosis=diagnosis, evidence=evidence, config=cfg)
                                                                          
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
