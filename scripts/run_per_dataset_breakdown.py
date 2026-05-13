from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from structured_diag.evaluation import classification_report  # noqa: E402
from structured_diag.evaluation.compare_flat_vs_hier import _flat_predict  # noqa: E402
from structured_diag.features import build_feature_table  # noqa: E402
from structured_diag.labels import HEALTHY, LEAKAGE, PRIMARY_LABELS  # noqa: E402
from structured_diag.models import load_cascade, train_flat_baseline  # noqa: E402
from structured_diag.models.flat_baseline import _split_train_test  # noqa: E402
from structured_diag.models.inference import diagnose_batch  # noqa: E402
from structured_diag.models.model_zoo import default_zoo  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _build_run_to_dataset(corpus_dir: Path) -> pd.Series:
    manifest_path = corpus_dir / "corpus.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for entry in manifest.get("entries", []):
        ds = (entry.get("hparam_patch") or {}).get("dataset", "?")
        for rid in entry.get("run_ids", []):
            pairs.append((str(rid), str(ds)))
    if not pairs:
        raise RuntimeError(f"No (run_id, dataset) pairs found in {manifest_path}")
    s = pd.Series(
        data=[ds for _, ds in pairs],
        index=[rid for rid, _ in pairs],
        name="dataset",
    )
    return s[~s.index.duplicated(keep="last")]


def _per_dataset_metrics(
    *,
    y_true: pd.Series,
    flat_pred: np.ndarray,
    cascade_pred: np.ndarray,
    cascade_proba: np.ndarray,
) -> dict[str, Any]:
    label_order = list(PRIMARY_LABELS)
    flat_rep = classification_report(
        y_true,
        flat_pred,
        label_order=label_order,
    )
    hier_rep = classification_report(
        y_true,
        cascade_pred,
        y_proba=cascade_proba,
        proba_classes=label_order,
        label_order=label_order,
    )
    y_arr = y_true.to_numpy(dtype=object)
    leak_called_healthy_flat = int(((y_arr == LEAKAGE) & (flat_pred == HEALTHY)).sum())
    leak_called_healthy_hier = int(((y_arr == LEAKAGE) & (cascade_pred == HEALTHY)).sum())
    n_leakage = int((y_arr == LEAKAGE).sum())
    return {
        "n_samples": int(len(y_true)),
        "primary_label_distribution": dict(Counter(y_arr.tolist())),
        "n_leakage": n_leakage,
        "flat": {
            "accuracy": float(flat_rep.accuracy),
            "macro_f1": float(flat_rep.macro_f1),
            "weighted_f1": float(flat_rep.weighted_f1),
            "per_class_f1": {k: float(v) for k, v in flat_rep.per_class_f1.items()},
            "leakage_called_healthy": leak_called_healthy_flat,
            "confusion_matrix": flat_rep.confusion_matrix,
            "confusion_labels": list(flat_rep.confusion_labels),
        },
        "cascade": {
            "accuracy": float(hier_rep.accuracy),
            "macro_f1": float(hier_rep.macro_f1),
            "weighted_f1": float(hier_rep.weighted_f1),
            "per_class_f1": {k: float(v) for k, v in hier_rep.per_class_f1.items()},
            "ece": (None if hier_rep.ece is None else float(hier_rep.ece)),
            "leakage_called_healthy": leak_called_healthy_hier,
            "confusion_matrix": hier_rep.confusion_matrix,
            "confusion_labels": list(hier_rep.confusion_labels),
        },
        "delta_accuracy": float(hier_rep.accuracy - flat_rep.accuracy),
        "delta_macro_f1": float(hier_rep.macro_f1 - flat_rep.macro_f1),
        "delta_leakage_f1": float(
            hier_rep.per_class_f1.get(LEAKAGE, 0.0) - flat_rep.per_class_f1.get(LEAKAGE, 0.0)
        ),
    }


def _render_markdown(
    *,
    corpus_name: str,
    n_test: int,
    flat_model: str,
    cascade_default_threshold: float,
    datasets: dict[str, dict[str, Any]],
) -> str:
    out: list[str] = []
    out.append(f"# Per-dataset breakdown — {corpus_name}")
    out.append("")
    out.append(f"- n test = {n_test}")
    out.append(f"- flat model: `{flat_model}`")
    out.append(f"- cascade Stage 1 threshold: {cascade_default_threshold:.2f}")
    out.append(
        "- both contours evaluated on the **same canonical test fold** "
        "(StratifiedKFold(5).first_fold, seed=0)"
    )
    out.append("")
    out.append("## Headline table")
    out.append("")
    out.append(
        "| dataset | n | n_leakage | flat acc | hier acc | Δ acc | "
        "flat leakage F1 | hier leakage F1 | Δ leak F1 | hier L→H | flat L→H |"
    )
    out.append("|---|--:|--:|---:|---:|---:|---:|---:|---:|--:|--:|")
    for ds_name, row in sorted(datasets.items()):
        flat = row["flat"]
        hier = row["cascade"]
        out.append(
            f"| `{ds_name}` | {row['n_samples']} | {row['n_leakage']} "
            f"| {flat['accuracy']:.4f} | {hier['accuracy']:.4f} "
            f"| {row['delta_accuracy']:+.4f} "
            f"| {flat['per_class_f1'].get(LEAKAGE, 0.0):.4f} "
            f"| {hier['per_class_f1'].get(LEAKAGE, 0.0):.4f} "
            f"| {row['delta_leakage_f1']:+.4f} "
            f"| {hier['leakage_called_healthy']} | {flat['leakage_called_healthy']} |"
        )
    out.append("")
    out.append("## Per-class F1 — cascade")
    out.append("")
    classes = list(PRIMARY_LABELS)
    out.append("| dataset | n | " + " | ".join(classes) + " |")
    out.append("|---|--:|" + "|".join(["---:" for _ in classes]) + "|")
    for ds_name, row in sorted(datasets.items()):
        cells = " | ".join(f"{row['cascade']['per_class_f1'].get(c, 0.0):.4f}" for c in classes)
        out.append(f"| `{ds_name}` | {row['n_samples']} | {cells} |")
    out.append("")
    out.append("## Per-class F1 — flat")
    out.append("")
    out.append("| dataset | n | " + " | ".join(classes) + " |")
    out.append("|---|--:|" + "|".join(["---:" for _ in classes]) + "|")
    for ds_name, row in sorted(datasets.items()):
        cells = " | ".join(f"{row['flat']['per_class_f1'].get(c, 0.0):.4f}" for c in classes)
        out.append(f"| `{ds_name}` | {row['n_samples']} | {cells} |")
    out.append("")
    out.append("## Primary label distribution per dataset (test fold)")
    out.append("")
    out.append("| dataset | n | " + " | ".join(classes) + " |")
    out.append("|---|--:|" + "|".join(["--:" for _ in classes]) + "|")
    for ds_name, row in sorted(datasets.items()):
        dist = row.get("primary_label_distribution") or {}
        cells = " | ".join(str(dist.get(c, 0)) for c in classes)
        out.append(f"| `{ds_name}` | {row['n_samples']} | {cells} |")
    out.append("")
    return "\n".join(out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--hier-artifacts", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = X.index[test_idx]
    X_te = X.loc[test_run_ids]
    y_te = y.loc[test_run_ids]
    run_to_ds = _build_run_to_dataset(args.corpus)
    ds_te = run_to_ds.reindex(test_run_ids)
    missing = ds_te[ds_te.isna()]
    if not missing.empty:
        print(
            f"WARNING: {len(missing)} test rows have no dataset mapping; "
            f"they will be bucketed as `<unknown>`.",
            file=sys.stderr,
        )
        ds_te = ds_te.fillna("<unknown>")
    zoo = default_zoo(include_catboost=not args.no_catboost)
    flat = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    print(f"Trained flat baseline: {flat.model_name}")
    cascade = load_cascade(args.hier_artifacts)
    print(
        f"Loaded cascade: stages={cascade.stages_available}, "
        f"stage1 threshold={cascade.stage1_healthy_threshold:.3f} "
        f"({cascade.threshold_source})"
    )
    flat_pred_full, _ = _flat_predict(flat, X_te)
    flat_pred_full = pd.Series(flat_pred_full, index=X_te.index)
    diags = diagnose_batch(cascade, X_te)
    cascade_pred_full = pd.Series(
        [d.final_class for d in diags],
        index=X_te.index,
    )
    classes = list(PRIMARY_LABELS)
    cascade_proba_full = np.zeros((len(diags), len(classes)), dtype=float)
    for i, d in enumerate(diags):
        for j, cls in enumerate(classes):
            cascade_proba_full[i, j] = float(d.class_probabilities.get(cls, 0.0))
    cascade_proba_full = pd.DataFrame(
        cascade_proba_full,
        index=X_te.index,
        columns=classes,
    )
    datasets: dict[str, dict[str, Any]] = {}
    for ds_name in sorted(ds_te.unique()):
        idx = ds_te.index[ds_te == ds_name]
        if len(idx) == 0:
            continue
        sub_y = y_te.loc[idx]
        sub_flat = flat_pred_full.loc[idx].to_numpy(dtype=object)
        sub_hier = cascade_pred_full.loc[idx].to_numpy(dtype=object)
        sub_proba = cascade_proba_full.loc[idx].to_numpy(dtype=float)
        datasets[ds_name] = _per_dataset_metrics(
            y_true=sub_y,
            flat_pred=sub_flat,
            cascade_pred=sub_hier,
            cascade_proba=sub_proba,
        )
    md = _render_markdown(
        corpus_name=ftable.corpus_name,
        n_test=int(len(X_te)),
        flat_model=flat.model_name,
        cascade_default_threshold=float(cascade.stage1_healthy_threshold),
        datasets=datasets,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        payload = {
            "corpus": ftable.corpus_name,
            "n_test": int(len(X_te)),
            "seed": int(args.seed),
            "flat_model": flat.model_name,
            "cascade_stages": cascade.stages_available,
            "cascade_default_threshold": float(cascade.stage1_healthy_threshold),
            "datasets": datasets,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print(
        f"Per-dataset summary (n_test={len(X_te)}, default thr={cascade.stage1_healthy_threshold:.2f}):"
    )
    print(
        f"{'dataset':<35} {'n':>3} {'n_leak':>6}  "
        f"{'flat acc':>9} {'hier acc':>9} {'Δacc':>7}  "
        f"{'flat leakF1':>11} {'hier leakF1':>11} {'ΔleakF1':>8}  "
        f"{'L→H hier':>9}"
    )
    for ds_name, row in sorted(datasets.items()):
        flat = row["flat"]
        hier = row["cascade"]
        print(
            f"{ds_name:<35} {row['n_samples']:>3} {row['n_leakage']:>6}  "
            f"{flat['accuracy']:>9.4f} {hier['accuracy']:>9.4f} "
            f"{row['delta_accuracy']:>+7.4f}  "
            f"{flat['per_class_f1'].get(LEAKAGE, 0.0):>11.4f} "
            f"{hier['per_class_f1'].get(LEAKAGE, 0.0):>11.4f} "
            f"{row['delta_leakage_f1']:>+8.4f}  "
            f"{hier['leakage_called_healthy']:>9d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
