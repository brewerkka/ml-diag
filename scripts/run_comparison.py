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
from ml_diag.evaluation import (              
    compare_all_slices,
    render_comparison_markdown,
)
from ml_diag.features import build_feature_table              
from ml_diag.models import (              
    load_cascade,
    slices_from_partition,
    train_flat_baseline,
)
from ml_diag.models.flat_baseline import (              
    FlatBaselineResult,
    _split_train_test,
)
from ml_diag.models.model_zoo import default_zoo              
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--hier-artifacts", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--flat-model",
        type=Path,
        default=None,
        help="Optional joblib bundle from run_flat_baseline.py to reuse.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--drop-feature-prefix",
        default=None,
        help="Comma-separated list of feature-name prefixes to drop "
        "from both flat and cascade evaluation. Should match the "
        "value passed to run_hierarchical_train.py for a fair "
        "ablation comparison.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _maybe_load_flat(path: Path | None) -> FlatBaselineResult | None:
    if path is None:
        return None
    try:
        import joblib
    except ImportError:
        print("WARNING: joblib not available; refusing --flat-model.", file=sys.stderr)
        return None
    bundle = joblib.load(path)
    return FlatBaselineResult(
        model_name=bundle["model_name"],
        model=bundle["model"],
        classes=[str(c) for c in bundle.get("classes", [])],
        feature_columns=list(bundle.get("feature_columns", [])),
        seed=int(bundle.get("seed", 0)),
    )


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
                print(f"Ablation: dropped {len(dropped)} features matching prefixes {prefixes}.")
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    flat = _maybe_load_flat(args.flat_model)
    if flat is None:
        zoo = default_zoo(include_catboost=not args.no_catboost)
        flat = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
        print(f"Trained fresh flat baseline: {flat.model_name}")
    else:
        print(f"Loaded flat baseline: {flat.model_name}")
    cascade = load_cascade(args.hier_artifacts)
    print(f"Loaded cascade with stages: {cascade.stages_available}")
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = X.index[test_idx]
    slices = slices_from_partition(pt, X.index, holdout_index=test_run_ids)
    X_te = X.loc[test_run_ids]
    y_te = y.loc[test_run_ids]
    comparisons = compare_all_slices(
        X=X_te,
        y=y_te,
        flat_result=flat,
        cascade=cascade,
        slices={k: idx.intersection(test_run_ids) for k, idx in slices.items()},
    )
    if not comparisons:
        print("ERROR: no slices produced any rows.", file=sys.stderr)
        return 2
    md_text = render_comparison_markdown(
        corpus_name=ftable.corpus_name,
        feature_source=ftable.source,
        flat_model_name=flat.model_name,
        cascade_stages=cascade.stages_available,
        comparisons=comparisons,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        payload = {
            "corpus": ftable.corpus_name,
            "feature_source": ftable.source,
            "flat_model": flat.model_name,
            "cascade_stages": cascade.stages_available,
            "slices": {
                slice_name: {
                    "n_samples": c.n_samples,
                    "deltas": c.deltas(),
                    "per_class_deltas": c.per_class_deltas(),
                    "flat_report": c.flat_report.to_dict(),
                    "hier_report": c.hier_report.to_dict(),
                    "stage_reports": {k: r.to_dict() for k, r in c.stage_reports.items()},
                    "leakage_healthy_confusion": c.leakage_healthy_confusion,
                    "error_propagation": c.error_propagation,
                }
                for slice_name, c in comparisons.items()
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON     -> {args.out_json}")
    print("\nSummary (Δ = hier - flat):")
    for slice_name, c in comparisons.items():
        d = c.deltas()
        print(
            f"  {slice_name:9s} n={c.n_samples:4d} "
            f"acc {c.flat_report.accuracy:.3f} -> {c.hier_report.accuracy:.3f} (Δ {d['delta_accuracy']:+.4f})  "
            f"macro-F1 {c.flat_report.macro_f1:.3f} -> {c.hier_report.macro_f1:.3f} (Δ {d['delta_macro_f1']:+.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
