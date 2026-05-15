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

from ml_diag.evaluation import (              
    report_to_markdown,
)
from ml_diag.features import (              
    build_grouped_feature_table,
    grouped_slices,
)
from ml_diag.models import (              
    evaluate_on_slices,
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
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _render_md(*, corpus_name: str, source: str, n_groups: int, model_name: str, reports) -> str:
    out: list[str] = []
    out.append("# Grouped baseline report (per entry_id)")
    out.append("")
    out.append(f"- corpus: **{corpus_name}**")
    out.append(f"- feature source: `{source}`")
    out.append(f"- groups: **{n_groups}**")
    out.append(f"- best model: `{model_name}`")
    out.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append(
        "Each row in this experiment is a hyperparameter configuration "
        "(`entry_id`), aggregated across its seeds. Features include "
        "mean/std/min/max/range over per-run features and curated "
        "stability metrics (variance, IQR, range) for convergence-state "
        "columns. Group label is computed by majority vote among the "
        "group's runs."
    )
    out.append("")
    for slice_name, rep in reports.items():
        out.append(report_to_markdown(rep, heading=f"Slice: `{slice_name}`"))
    out.append("")
    out.append(
        "Compare these numbers against `results/flat_baseline_report.md` "
        "(per-run, same corpus). If grouped numbers improve **monotonically** "
        "on `core`, group-level diagnosis pays off; if they only improve on "
        "`extended` it suggests the gain comes from label denoising rather "
        "than from a stronger signal."
    )
    return "\n".join(out)


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    gtable = build_grouped_feature_table(args.corpus)
    if gtable.df.empty:
        print("ERROR: grouped feature table is empty.", file=sys.stderr)
        return 2
    n_solo = sum(1 for k in gtable.group_to_run_ids if k.startswith("__solo__:"))
    print(
        f"Built grouped table for {gtable.corpus_name}: "
        f"{len(gtable.df)} groups, "
        f"{n_solo} singletons (no entry_id), "
        f"{len(gtable.feature_columns)} feature columns."
    )
    X, y = gtable.aligned_xy()
    if y.nunique() < 2:
        print(
            f"ERROR: only one unique group label ({y.unique().tolist()}); cannot train a baseline.",
            file=sys.stderr,
        )
        return 3
    zoo = default_zoo(include_catboost=not args.no_catboost)
    result = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_ids = X.index[test_idx]
    slices = grouped_slices(gtable, holdout_index=test_ids)
    reports = evaluate_on_slices(result, X, y, slices)
    md_text = _render_md(
        corpus_name=gtable.corpus_name,
        source=gtable.source,
        n_groups=len(gtable.df),
        model_name=result.model_name,
        reports=reports,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")
    if args.out_json is not None:
        payload = {
            "corpus": gtable.corpus_name,
            "feature_source": gtable.source,
            "n_groups": len(gtable.df),
            "n_singletons": n_solo,
            "model": result.model_name,
            "feature_columns": result.feature_columns,
            "classes": result.classes,
            "seed": args.seed,
            "reports": {k: r.to_dict() for k, r in reports.items()},
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON     -> {args.out_json}")
    print(f"Wrote markdown -> {args.out_md}")
    print()
    print("Grouped baseline summary:")
    print(f"  best model: {result.model_name}")
    print(f"  classes:    {result.classes}")
    for slice_name, rep in reports.items():
        print(
            f"  slice={slice_name:9s} n={rep.n_samples:4d} "
            f"acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f} ece={rep.ece}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
