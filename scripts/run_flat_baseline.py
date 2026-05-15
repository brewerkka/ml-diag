from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from ml_diag.benchmark import partition_corpus              
from ml_diag.evaluation import (              
    render_flat_baseline_markdown,
    reports_to_json,
    write_report,
)
from ml_diag.features import build_feature_table              
from ml_diag.models import (              
    evaluate_on_slices,
    slices_from_partition,
    train_flat_baseline,
)
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--model-out",
        type=Path,
        default=None,
        help="Optional path to dump the trained model with joblib.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-catboost", action="store_true", help="Exclude CatBoost from the candidate zoo."
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    from ml_diag.models.model_zoo import default_zoo

    zoo = default_zoo(include_catboost=not args.no_catboost)
    result = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = X.index[test_idx]
    slices = slices_from_partition(pt, X.index, holdout_index=test_run_ids)
    reports = evaluate_on_slices(result, X, y, slices)
    md_text = render_flat_baseline_markdown(
        corpus_name=ftable.corpus_name,
        feature_source=ftable.source,
        model_name=result.model_name,
        reports=reports,
        extras={
            "seed": args.seed,
            "n_features": len(result.feature_columns),
            "n_classes": len(result.classes),
            "test_holdout_size": int(len(test_idx)),
        },
    )
    write_report(
        args.out_md,
        args.out_json,
        md_text=md_text,
        json_payload={
            "corpus": ftable.corpus_name,
            "feature_source": ftable.source,
            "model": result.model_name,
            "feature_columns": result.feature_columns,
            "classes": result.classes,
            "seed": args.seed,
            "reports": reports_to_json(reports),
        },
    )
    if args.model_out is not None:
        try:
            import joblib
        except ImportError:
            print("WARNING: joblib not installed; skipping --model-out.", file=sys.stderr)
        else:
            args.model_out.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": result.model,
                    "model_name": result.model_name,
                    "feature_columns": result.feature_columns,
                    "classes": result.classes,
                    "seed": args.seed,
                },
                args.model_out,
            )
            print(f"Wrote model -> {args.model_out}")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("Flat baseline summary:")
    print(f"  best model: {result.model_name}")
    print(f"  feature source: {ftable.source}")
    print(f"  classes: {result.classes}")
    for slice_name, rep in reports.items():
        print(
            f"  slice={slice_name:9s} n={rep.n_samples:4d} "
            f"acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f} ece={rep.ece}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
