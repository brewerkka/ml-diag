from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


import pandas as pd  # noqa: E402

from structured_diag.benchmark import partition_corpus  # noqa: E402
from structured_diag.evaluation import (  # noqa: E402
    attribute_errors,
    find_disagreements,
    render_attribution_markdown,
    render_disagreements_markdown,
    summarize_attributions,
)
from structured_diag.evaluation.compare_flat_vs_hier import _flat_predict  # noqa: E402
from structured_diag.features import build_feature_table  # noqa: E402
from structured_diag.models import (  # noqa: E402
    load_cascade,
    slices_from_partition,
    train_flat_baseline,
)
from structured_diag.models.flat_baseline import _split_train_test  # noqa: E402
from structured_diag.models.inference import diagnose_batch  # noqa: E402
from structured_diag.models.model_zoo import default_zoo  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--hier-artifacts", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--top-k-evidence", type=int, default=5, help="Number of top features per disagreement row."
    )
    p.add_argument(
        "--top-n-disagreements",
        type=int,
        default=20,
        help="Cap rows per disagreement table in the markdown "
        "report (the JSON sidecar carries everything).",
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
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = X.index[test_idx]
    X_te = X.loc[test_run_ids]
    y_te = y.loc[test_run_ids]
    zoo = default_zoo(include_catboost=not args.no_catboost)
    flat = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    print(f"Trained flat baseline: {flat.model_name}")
    cascade = load_cascade(args.hier_artifacts)
    print(
        f"Loaded cascade: stages={cascade.stages_available}, "
        f"stage1 threshold={cascade.stage1_healthy_threshold:.3f} "
        f"({cascade.threshold_source})"
    )
    diags = diagnose_batch(cascade, X_te)
    flat_pred, _flat_proba = _flat_predict(flat, X_te)
    slices = slices_from_partition(pt, X.index, holdout_index=test_run_ids)
    full_attribs = attribute_errors(diags, y_te)
    attribs_by_run = {a.run_id: a for a in full_attribs}
    slice_summaries = {}
    slice_disagreements = {}
    for slice_name, idx in slices.items():
        if len(idx) == 0:
            continue
        slice_set = set(idx)
        sub_attribs = [a for a in full_attribs if a.run_id in slice_set]
        slice_summaries[slice_name] = summarize_attributions(sub_attribs)
        sub_diags = [d for d in diags if d.run_id in slice_set]
        ids_in_order = [d.run_id for d in sub_diags]
        sub_y = y_te.loc[ids_in_order]
        sub_X = X_te.loc[ids_in_order]
        flat_pred_series = pd.Series(flat_pred, index=X_te.index)
        sub_flat_pred = flat_pred_series.loc[ids_in_order].tolist()
        flat_wins, cascade_wins = find_disagreements(
            diagnoses=sub_diags,
            y_true=sub_y,
            flat_pred=sub_flat_pred,
            X=sub_X,
            cascade=cascade,
            top_k_evidence=args.top_k_evidence,
        )
        slice_disagreements[slice_name] = (flat_wins, cascade_wins)
    md_parts: list[str] = []
    md_parts.append(
        render_attribution_markdown(
            corpus_name=ftable.corpus_name,
            slice_summaries=slice_summaries,
        )
    )
    md_parts.append("")
    md_parts.append("---")
    md_parts.append("")
    md_parts.append("# Disagreement diff (flat vs cascade)")
    md_parts.append("")
    md_parts.append(
        "Two buckets per slice: rows where flat got it right but cascade did "
        "not (cascade regressions vs flat), and the symmetric set (cascade "
        "wins). Each row carries Stage 1 probabilities and the top features "
        "by `importance × |value|` for Stage 1's model."
    )
    md_parts.append("")
    for slice_name, (fw, cw) in slice_disagreements.items():
        md_parts.append(
            render_disagreements_markdown(
                slice_name=slice_name,
                flat_wins=fw,
                cascade_wins=cw,
                top_n=args.top_n_disagreements,
            )
        )
        md_parts.append("")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_parts), encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        payload = {
            "corpus": ftable.corpus_name,
            "stage1_threshold": float(cascade.stage1_healthy_threshold),
            "threshold_source": cascade.threshold_source,
            "flat_model": flat.model_name,
            "n_test": int(len(X_te)),
            "slices": {
                slice_name: {
                    "summary": summ.to_dict(),
                    "per_row": [
                        attribs_by_run[rid].to_dict()
                        for rid in slices[slice_name]
                        if rid in attribs_by_run
                    ],
                    "disagreements": {
                        "flat_correct_cascade_wrong": [
                            r.to_dict() for r in slice_disagreements[slice_name][0]
                        ],
                        "cascade_correct_flat_wrong": [
                            r.to_dict() for r in slice_disagreements[slice_name][1]
                        ],
                    },
                }
                for slice_name, summ in slice_summaries.items()
            },
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("Error attribution summary (errors only):")
    for slice_name, summ in slice_summaries.items():
        l = summ.leakage_to_healthy
        h = summ.healthy_to_faulty
        print(
            f"  {slice_name:9s} n={summ.n:4d} errors={summ.n_errors:3d}  "
            f"by_stage={dict(summ.by_stage)}  "
            f"leak→hlth={l.get('n_total', 0)} (s1={l.get('stage1', 0)})  "
            f"hlth→faulty={h.get('n_total', 0)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
