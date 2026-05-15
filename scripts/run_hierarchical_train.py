from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.benchmark import partition_corpus              
from ml_diag.evaluation import report_to_markdown              
from ml_diag.features import build_feature_table              
from ml_diag.models import save_stage_artifacts, stage1, stage2, stage3              
from ml_diag.models.model_zoo import default_zoo              
from ml_diag.models.trainer import StageTrainResult              
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory where per-stage artifacts are written.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip per-stage isotonic calibration (Stage 47). "
        "По умолчанию каждый stage оборачивается в "
        "CalibratedClassifierCV с isotonic regression. "
        "На малых фолдах calibration может ухудшать качество "
        "— поставь этот флаг чтобы получить uncalibrated cascade.",
    )
    p.add_argument(
        "--calibration-method",
        default="isotonic",
        choices=["isotonic", "sigmoid"],
        help="`sigmoid` (Platt scaling) более устойчив на малых "
        "фолдах чем `isotonic`. По умолчанию isotonic.",
    )
    p.add_argument(
        "--drop-feature-prefix",
        default=None,
        help="Comma-separated list of feature-name prefixes to drop "
        "before training. Used for ablation studies. "
        "Пример: '--drop-feature-prefix di_proxy_' уберёт все "
        "Stage 46 (B1) leakage proxy features из обучения "
        "и оценки, оставив остальные.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _try_train(name: str, fn, *args, **kwargs) -> StageTrainResult | None:
    try:
        return fn(*args, **kwargs)
    except RuntimeError as e:
        print(f"WARNING: {name} skipped: {e}", file=sys.stderr)
        return None


def _write_combined_report(out_dir: Path, results: dict[str, StageTrainResult]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# Hierarchical diagnosis report")
    md.append("")
    md.append(
        "Per-stage models trained inside `ml_diag`. "
        "Each stage is trained on its own subset; see `docs/architecture.md`."
    )
    md.append("")
    for stage_key, result in results.items():
        md.append(f"## `{stage_key}`")
        md.append("")
        md.append(f"- best model: `{result.model_name}`")
        md.append(f"- n_train: {result.n_train}, n_test: {result.n_test}")
        md.append("- CV macro-F1 by candidate:")
        for cand, score in sorted(result.cv_scores.items(), key=lambda kv: -kv[1]):
            md.append(f"    - `{cand}`: {score:.4f}")
        md.append("")
        for slice_name, rep in result.test_reports.items():
            md.append(report_to_markdown(rep, heading=f"Test slice: `{slice_name}`"))
    path = out_dir / "hierarchical_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    if args.drop_feature_prefix:
        prefixes = tuple(p.strip() for p in args.drop_feature_prefix.split(",") if p.strip())
        if prefixes:
            dropped = [c for c in X.columns if c.startswith(prefixes)]
            if dropped:
                X = X.drop(columns=dropped)
                print(
                    f"Ablation: dropped {len(dropped)} features matching prefixes {prefixes}: "
                    f"{dropped[:5]}{'...' if len(dropped) > 5 else ''}"
                )
            else:
                print(f"Ablation: no features matched prefixes {prefixes} — no-op.")
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    from ml_diag.models.flat_baseline import _split_train_test

    _, test_idx = _split_train_test(X, y, seed=args.seed)
    canonical_test_run_ids = X.index[test_idx].astype(str).tolist()
    zoo = default_zoo(include_catboost=not args.no_catboost)
    common_kwargs = {
        "partition_table": pt,
        "seed": args.seed,
        "holdout_run_ids": canonical_test_run_ids,
        "calibrate": not args.no_calibrate,
        "calibration_method": args.calibration_method,
    }
    results: dict[str, StageTrainResult] = {}
    r = _try_train("stage1", stage1.train, X, y, **common_kwargs)
    if r is not None:
        results[stage1.STAGE_NAME] = r
    r = _try_train("stage2", stage2.train, X, y, **common_kwargs)
    if r is not None:
        results[stage2.STAGE_NAME] = r
    r = _try_train("stage3_data_related", stage3.train_data_related, X, y, **common_kwargs)
    if r is not None:
        results[stage3.STAGE_NAME_DATA] = r
    r = _try_train("stage3_opt_gen", stage3.train_opt_gen, X, y, **common_kwargs)
    if r is not None:
        results[stage3.STAGE_NAME_OPT_GEN] = r
    if not results:
        print("ERROR: no stage could be trained.", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_index: dict[str, dict[str, str]] = {}
    for stage_key, result in results.items():
        paths = save_stage_artifacts(result, args.out_dir)
        artifact_index[stage_key] = {k: str(v) for k, v in paths.items()}
    combined_md = _write_combined_report(args.out_dir, results)
    manifest_path = args.out_dir / "hierarchical_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "corpus": ftable.corpus_name,
                "feature_source": ftable.source,
                "seed": args.seed,
                "stages": {k: r.summary_dict() for k, r in results.items()},
                "artifacts": artifact_index,
                "combined_report": str(combined_md),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Trained stages: {list(results)}")
    for stage_key, result in results.items():
        print(f"\n[{stage_key}] best={result.model_name}")
        print(f"  n_train={result.n_train} n_test={result.n_test}")
        for slice_name, rep in result.test_reports.items():
            print(
                f"  slice={slice_name:9s} n={rep.n_samples:4d} "
                f"acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f}"
            )
    print(f"\nArtifacts in: {args.out_dir}")
    print(f"Combined report: {combined_md}")
    print(f"Manifest:        {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
