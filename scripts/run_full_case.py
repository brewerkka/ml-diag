from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.data import (              
    load_run,
    manifest_for_single_run,
)
from ml_diag.evaluation import (              
    build_evidence,
    write_case_outputs,
)
from ml_diag.features import (              
    build_data_integrity_features,
    build_feature_table,
)
from ml_diag.interpretation import (              
    InterpretationConfig,
    interpret,
)
from ml_diag.models import diagnose_one, load_cascade              
from ml_diag.patch_eval import (              
    PatchCase,
    evaluate_patch,
)
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to a corpus directory (use together with --run-id).",
    )
    p.add_argument("--run-id", default=None, help="run_id within --corpus.")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a standalone run directory containing meta.json + history.csv "
        "(produced e.g. by ml_diag.logging_sdk.RunLogger).",
    )
    p.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Hierarchical artifacts dir (from run_hierarchical_train.py).",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Defaults to results/cases/<run_id>/")
    p.add_argument(
        "--backend", default="auto", choices=["auto", "template", "groq", "ollama"]
    )
    p.add_argument("--model", default=None, help="Override default per-backend model id.")
    p.add_argument("--max-recommendations", type=int, default=3)
    p.add_argument("--no-integrity", action="store_true")
    p.add_argument(
        "--no-curves", action="store_true", help="Skip curves.png even if matplotlib is installed."
    )
    p.add_argument("--patch-action", default=None, help="Allowlist action name for the patch case.")
    p.add_argument(
        "--patch-after", default=None, help="run_id of the after-run for the patch case."
    )
    p.add_argument("--patch-params", default="{}", help="JSON object of action parameters.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _maybe_meta(corpus: Path, run_id: str) -> dict[str, Any] | None:
    rd = corpus / run_id
    if not rd.is_dir():
        return None
    try:
        return load_run(rd).meta
    except Exception:
        return None


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    if args.run_dir is not None:
        if args.corpus is not None or args.run_id is not None:
            print(
                "ERROR: pass either --run-dir, or --corpus + --run-id, not both.", file=sys.stderr
            )
            return 2
        run_dir = Path(args.run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            print(f"ERROR: --run-dir does not exist: {run_dir}", file=sys.stderr)
            return 2
        manifest = manifest_for_single_run(run_dir)
        corpus_input = manifest
        try:
            rec = load_run(run_dir)
        except Exception as e:
            print(f"ERROR: cannot load run_dir: {e}", file=sys.stderr)
            return 2
        resolved_run_id = rec.run_id
        corpus_path_for_meta = run_dir.parent
    else:
        if args.corpus is None or args.run_id is None:
            print(
                "ERROR: in corpus mode you must pass both --corpus and --run-id.", file=sys.stderr
            )
            return 2
        corpus_input = args.corpus
        resolved_run_id = args.run_id
        corpus_path_for_meta = args.corpus
    out_dir = args.out_dir or (Path("results/cases") / resolved_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    cascade = load_cascade(args.artifacts)
    print(f"Loaded cascade: stages={cascade.stages_available}")
    base = build_feature_table(corpus_input)
    integrity_columns = None
    full_df = base.df
    if not args.no_integrity:
        try:
            di = build_data_integrity_features(corpus_input, base_table=base)
            integrity_columns = di.integrity_columns
            full_df = di.df
            print(f"Integrity columns available: {len(integrity_columns)}")
        except Exception as e:
            print(
                f"WARNING: integrity features unavailable ({e}); continuing without.",
                file=sys.stderr,
            )
    if resolved_run_id not in full_df.index:
        print(f"ERROR: run_id {resolved_run_id!r} not in feature table.", file=sys.stderr)
        return 2
    feature_cols = base.feature_columns
    x_row = full_df.loc[resolved_run_id, feature_cols]
    diagnosis = diagnose_one(cascade, run_id=resolved_run_id, x_row=x_row)
    print(f"Diagnosis: {diagnosis.final_class} (P = {diagnosis.final_confidence:.3f})")
    full_row = full_df.loc[resolved_run_id]
    evidence = build_evidence(
        diagnosis=diagnosis,
        feature_row=full_row,
        cascade=cascade,
        integrity_columns=integrity_columns,
    )
    cfg = InterpretationConfig(
        backend=args.backend,
        model=args.model,
        max_recommendations=args.max_recommendations,
    )
    interpretation = interpret(diagnosis=diagnosis, evidence=evidence, config=cfg)
    print(
        f"Interpretation: backend={interpretation.backend}, "
        f"recs={len(interpretation.recommendations)}, "
        f"warnings={len(interpretation.warnings)}"
    )
    patch_summary: dict[str, Any] | None = None
    if args.patch_action and args.patch_after:
        try:
            params = json.loads(args.patch_params)
        except json.JSONDecodeError as e:
            print(
                f"WARNING: --patch-params is not valid JSON ({e}); skipping patch step.",
                file=sys.stderr,
            )
        else:
            case = PatchCase(
                case_id=f"{resolved_run_id}__{args.patch_action}",
                before_run_id=resolved_run_id,
                after_run_id=args.patch_after,
                action_name=args.patch_action,
                action_parameters=params,
            )
            try:
                patch_report = evaluate_patch(
                    case=case,
                    cascade=cascade,
                    feature_table=base,
                    full_feature_df=full_df,
                    integrity_columns=integrity_columns,
                    before_meta=_maybe_meta(corpus_path_for_meta, resolved_run_id),
                    after_meta=_maybe_meta(corpus_path_for_meta, args.patch_after),
                )
            except Exception as e:
                print(f"WARNING: patch evaluation failed ({e}); skipping.", file=sys.stderr)
            else:
                patch_summary = patch_report.to_dict()
                print(
                    f"Patch outcome: {patch_report.outcome.status} "
                    f"(ΔP(healthy) = {patch_report.outcome.delta_p_healthy:+.4f})"
                )
    if args.run_dir is not None:
        run_dir_for_curves = Path(args.run_dir).expanduser().resolve()
    else:
        candidate = Path(corpus_path_for_meta) / resolved_run_id
        run_dir_for_curves = candidate if candidate.is_dir() else None
    if args.no_curves:
        run_dir_for_curves = None
    files = write_case_outputs(
        out_dir,
        diagnosis=diagnosis,
        evidence=evidence,
        interpretation=interpretation,
        patch_summary=patch_summary,
        run_dir=run_dir_for_curves,
        extras={
            "corpus_path": str(corpus_path_for_meta),
            "input_mode": "run_dir" if args.run_dir else "corpus",
            "artifacts_path": str(args.artifacts),
            "interpretation_backend": interpretation.backend,
            "feature_source": base.source,
            "integrity_columns_available": (len(integrity_columns) if integrity_columns else 0),
        },
    )
    print()
    print(f"Case directory: {out_dir}")
    for k, v in files.items():
        print(f"  [{k:24s}] {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
