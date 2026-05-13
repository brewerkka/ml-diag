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


from structured_diag.benchmark import partition_corpus  # noqa: E402
from structured_diag.evaluation import (  # noqa: E402
    report_to_markdown,
)
from structured_diag.features import (  # noqa: E402
    build_data_integrity_features,
    build_feature_table,
    leakage_vs_healthy_diagnostic,
)
from structured_diag.models import (  # noqa: E402
    evaluate_on_slices,
    slices_from_partition,
    train_flat_baseline,
)
from structured_diag.models.flat_baseline import _split_train_test  # noqa: E402
from structured_diag.models.model_zoo import default_zoo  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _leakage_healthy_focus(reports) -> dict[str, dict[str, float]]:
    rep = reports.get("full")
    if rep is None:
        return {}
    out: dict[str, dict[str, float]] = {}
    for cls in ("leakage", "healthy"):
        out[cls] = {
            "precision": rep.per_class_precision.get(cls, 0.0),
            "recall": rep.per_class_recall.get(cls, 0.0),
            "f1": rep.per_class_f1.get(cls, 0.0),
            "support": rep.per_class_support.get(cls, 0),
        }
    cm = rep.confusion_matrix
    labels = rep.confusion_labels
    if "leakage" in labels and "healthy" in labels:
        i_leak = labels.index("leakage")
        i_heal = labels.index("healthy")
        out["confusion"] = {
            "leakage_called_healthy": int(cm[i_leak][i_heal]),
            "healthy_called_leakage": int(cm[i_heal][i_leak]),
        }
    return out


def _render_md(*, corpus_name, source, integrity_columns, focus_table, variants) -> str:
    out: list[str] = []
    out.append("# Leakage-specific integrity features — experiment report")
    out.append("")
    out.append(f"- corpus: **{corpus_name}**")
    out.append(f"- base feature source: `{source}`")
    out.append(f"- integrity columns added: **{len(integrity_columns)}**")
    out.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    out.append("")
    out.append(
        "Leakage and healthy runs often look identical in *behaviour* "
        "(low train/val gap, high val acc on saturated datasets). The hypothesis "
        "is that **data-side integrity features** break that ambiguity."
    )
    out.append("")
    out.append("## Headline (full slice, test fold)")
    out.append("")
    out.append(
        "| variant | best model | acc | macro-F1 | leakage F1 | healthy F1 | leakage→healthy | healthy→leakage |"
    )
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, model_name, reports, focus in variants:
        rep = reports.get("full")
        if rep is None:
            continue
        conf = focus.get("confusion", {})
        out.append(
            f"| **{name}** | `{model_name}` | {rep.accuracy:.4f} | {rep.macro_f1:.4f} "
            f"| {focus.get('leakage', {}).get('f1', 0.0):.4f} "
            f"| {focus.get('healthy', {}).get('f1', 0.0):.4f} "
            f"| {conf.get('leakage_called_healthy', '—')} "
            f"| {conf.get('healthy_called_leakage', '—')} |"
        )
    out.append("")
    out.append("## Integrity feature distribution (leakage vs healthy)")
    out.append("")
    if focus_table.empty:
        out.append("_no leakage or healthy rows in the corpus_")
    else:
        out.append("Means and standard deviations of every `di_*` feature, grouped by class:")
        out.append("")
        out.append("```\n" + focus_table.to_string() + "\n```")
    out.append("")
    for name, model_name, reports, focus in variants:
        out.append("---")
        out.append(f"## Variant: `{name}` (best={model_name})")
        for slice_name, rep in reports.items():
            out.append(report_to_markdown(rep, heading=f"Slice: `{slice_name}`"))
    return "\n".join(out)


def _evaluate(name, X, y, *, train_idx, test_run_ids, slices, seed, zoo):
    result = train_flat_baseline(X, y, seed=seed, candidate_models=zoo)
    reports = evaluate_on_slices(result, X, y, slices)
    focus = _leakage_healthy_focus(reports)
    return name, result.model_name, reports, focus


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    base = build_feature_table(args.corpus)
    X_base, y_base = base.aligned_xy()
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    train_idx, test_idx = _split_train_test(X_base, y_base, seed=args.seed)
    test_run_ids = X_base.index[test_idx]
    di_table = build_data_integrity_features(args.corpus, base_table=base)
    print(
        f"Built {len(di_table.integrity_columns)} integrity columns "
        f"({len(di_table.meta_columns)} meta + {len(di_table.proxy_columns)} proxy)."
    )
    zoo = default_zoo(include_catboost=not args.no_catboost)
    sl_base = slices_from_partition(pt, X_base.index, holdout_index=test_run_ids)
    v1 = _evaluate(
        "base",
        X_base,
        y_base,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=sl_base,
        seed=args.seed,
        zoo=zoo,
    )
    Xc, yc = di_table.aligned_xy(include_base=True, only_integrity=False)
    sl_c = slices_from_partition(pt, Xc.index, holdout_index=test_run_ids)
    v2 = _evaluate(
        "base+integrity",
        Xc,
        yc,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=sl_c,
        seed=args.seed,
        zoo=zoo,
    )
    Xi, yi = di_table.aligned_xy(include_base=False, only_integrity=True)
    sl_i = slices_from_partition(pt, Xi.index, holdout_index=test_run_ids)
    v3 = _evaluate(
        "integrity-only",
        Xi,
        yi,
        train_idx=train_idx,
        test_run_ids=test_run_ids,
        slices=sl_i,
        seed=args.seed,
        zoo=zoo,
    )
    focus_table = leakage_vs_healthy_diagnostic(di_table)
    md = _render_md(
        corpus_name=base.corpus_name,
        source=base.source,
        integrity_columns=di_table.integrity_columns,
        focus_table=focus_table,
        variants=[v1, v2, v3],
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    if args.out_json is not None:
        payload = {
            "corpus": base.corpus_name,
            "feature_source": base.source,
            "integrity_columns": di_table.integrity_columns,
            "variants": [
                {
                    "name": name,
                    "model": model_name,
                    "reports": {k: r.to_dict() for k, r in reports.items()},
                    "leakage_healthy_focus": focus,
                }
                for name, model_name, reports, focus in (v1, v2, v3)
            ],
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON     -> {args.out_json}")
    print(f"Wrote markdown -> {args.out_md}")
    print("\nLeakage-specific summary (full slice, test fold):")
    for name, model_name, reports, focus in (v1, v2, v3):
        rep = reports.get("full")
        if rep is None:
            continue
        leak = focus.get("leakage", {})
        print(
            f"  {name:18s} model={model_name:18s} "
            f"acc={rep.accuracy:.4f} macro_f1={rep.macro_f1:.4f} "
            f"leakage_f1={leak.get('f1', 0.0):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
