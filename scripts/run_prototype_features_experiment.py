from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from ml_diag.benchmark import partition_corpus              
from ml_diag.evaluation import (              
    report_to_markdown,
)
from ml_diag.features import build_feature_table              
from ml_diag.features.prototypes import build_prototype_features              
from ml_diag.models import (              
    evaluate_on_slices,
    slices_from_partition,
    train_flat_baseline,
)
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.models.model_zoo import default_zoo              
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--include-dtw", action="store_true", help="Also compute DTW distances on val_loss curves."
    )
    p.add_argument("--dtw-column", default="val_loss")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _evaluate_variant(name, X, y, *, train_idx, test_run_ids, slices, seed, zoo):
    result = train_flat_baseline(X, y, seed=seed, candidate_models=zoo)
    reports = evaluate_on_slices(result, X, y, slices)
    return name, result, reports


def _render_md(*, corpus_name, source, n_proto_features, variants) -> str:
    out: list[str] = []
    out.append("# Prototype-distance features — ablation report")
    out.append("")
    out.append(f"- corpus: **{corpus_name}**")
    out.append(f"- base feature source: `{source}`")
    out.append(f"- prototype columns added: **{n_proto_features}**")
    out.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append(
        "Three variants are evaluated on the same train/test split. "
        "Prototypes are built **on training-fold rows only** to avoid leakage."
    )
    out.append("")
    out.append("## Headline (test fold, full slice)")
    out.append("")
    out.append("| variant | best model | acc | macro-F1 | weighted-F1 | ECE |")
    out.append("|---|---|---:|---:|---:|---:|")
    for name, model_name, reports in variants:
        rep = reports.get("full")
        if rep is None:
            continue
        out.append(
            f"| **{name}** | `{model_name}` "
            f"| {rep.accuracy:.4f} | {rep.macro_f1:.4f} | {rep.weighted_f1:.4f} "
            f"| {('-' if rep.ece is None else f'{rep.ece:.4f}')} |"
        )
    out.append("")
    for name, model_name, reports in variants:
        out.append(f"## Variant: `{name}` (best={model_name})")
        for slice_name, rep in reports.items():
            out.append(report_to_markdown(rep, heading=f"Slice: `{slice_name}`"))
    return "\n".join(out)


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    base = build_feature_table(args.corpus)
    X_base, y_base = base.aligned_xy()
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    train_idx, test_idx = _split_train_test(X_base, y_base, seed=args.seed)
    test_run_ids = X_base.index[test_idx]
    train_run_ids = X_base.index[train_idx]
    proto_table = build_prototype_features(
        args.corpus,
        train_run_ids=train_run_ids.astype(str).tolist(),
        base_table=base,
        include_dtw=args.include_dtw,
        dtw_curve_column=args.dtw_column,
    )
    print(f"Built {len(proto_table.prototype_columns)} prototype-distance columns.")
    zoo = default_zoo(include_catboost=not args.no_catboost)
    variants_outputs = []
    slices_full = slices_from_partition(pt, X_base.index, holdout_index=test_run_ids)
    name, res, reps = _evaluate_variant(
        "base",
        X_base,
        y_base,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=slices_full,
        seed=args.seed,
        zoo=zoo,
    )
    variants_outputs.append((name, res.model_name, reps))
    Xp, yp = proto_table.aligned_xy(only_prototype=True, include_base=False)
    slices_p = slices_from_partition(pt, Xp.index, holdout_index=test_run_ids)
    name, res, reps = _evaluate_variant(
        "proto-only",
        Xp,
        yp,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=slices_p,
        seed=args.seed,
        zoo=zoo,
    )
    variants_outputs.append((name, res.model_name, reps))
    Xc, yc = proto_table.aligned_xy(only_prototype=False, include_base=True)
    slices_c = slices_from_partition(pt, Xc.index, holdout_index=test_run_ids)
    name, res, reps = _evaluate_variant(
        "base+proto",
        Xc,
        yc,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=slices_c,
        seed=args.seed,
        zoo=zoo,
    )
    variants_outputs.append((name, res.model_name, reps))
    md = _render_md(
        corpus_name=base.corpus_name,
        source=base.source,
        n_proto_features=len(proto_table.prototype_columns),
        variants=variants_outputs,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    if args.out_json is not None:
        payload = {
            "corpus": base.corpus_name,
            "feature_source": base.source,
            "n_proto_features": len(proto_table.prototype_columns),
            "include_dtw": args.include_dtw,
            "variants": [
                {
                    "name": name,
                    "model": model_name,
                    "reports": {k: r.to_dict() for k, r in reps.items()},
                }
                for name, model_name, reps in variants_outputs
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON     -> {args.out_json}")
    print(f"Wrote markdown -> {args.out_md}")
    print("\nSummary (full slice, test fold):")
    for name, model_name, reps in variants_outputs:
        rep = reps.get("full")
        if rep is None:
            continue
        print(
            f"  {name:12s} model={model_name:18s} "
            f"acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
