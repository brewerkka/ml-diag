from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from structured_diag.evaluation.metrics import (  # noqa: E402
    brier_score_multiclass,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram_bins,
)
from structured_diag.labels import PRIMARY_LABELS  # noqa: E402

POLICY_PROBA_GROUPS: dict[str, str] = {
    "flat": "flat",
    "cascade": "cascade",
    "llm_arbitrate": "arbitrator",
}


def _extract_policy_proba(
    df: pd.DataFrame, group: str, classes: list[str]
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray | None]:
    if group not in df.columns.get_level_values(0):
        return None, np.array([], dtype=object), None
    sub = df[group]
    missing = [c for c in classes if c not in sub.columns]
    if missing:
        return None, np.array([], dtype=object), None
    arr = sub[classes].to_numpy(dtype=float)
    row_sum = arr.sum(axis=1)
    mask = row_sum > 0
    if not mask.all():
        arr[~mask] = 1.0 / len(classes)
    idx = arr.argmax(axis=1)
    y_pred = np.array([classes[i] for i in idx], dtype=object)
    return arr, y_pred, mask


def _load_y_true_from_corpus(corpus_path: Path, run_ids: list[str]) -> np.ndarray:
    from structured_diag.features import build_feature_table

    ftable = build_feature_table(corpus_path)
    labels_by_run = ftable.df["primary_label"].astype(str).to_dict()
    y = [labels_by_run.get(rid, "") for rid in run_ids]
    if "" in y:
        n_missing = sum(1 for v in y if v == "")
        print(
            f"WARNING: {n_missing}/{len(y)} run_ids could not be matched to a label",
            file=sys.stderr,
        )
    return np.array(y, dtype=object)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", required=True, type=Path, help="Path to oof_predictions_*.parquet")
    p.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Path to the corpus directory (for run_id → primary_label lookup)",
    )
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
        help="If set, write a reliability-diagram grid PNG here.",
    )
    p.add_argument("--n-bins", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    df = pd.read_parquet(args.oof)
    if not isinstance(df.columns, pd.MultiIndex):
        print(
            "ERROR: OOF parquet does not have MultiIndex columns "
            "(expected groups: flat / cascade / arbitrator / stage_probs / meta).",
            file=sys.stderr,
        )
        return 1
    classes = list(PRIMARY_LABELS)
    run_ids = [str(r) for r in df.index.tolist()]
    y_true = _load_y_true_from_corpus(args.corpus, run_ids)
    rows: list[dict] = []
    bin_data: dict[str, dict] = {}
    for policy_name, group in POLICY_PROBA_GROUPS.items():
        proba, y_pred, mask = _extract_policy_proba(df, group, classes)
        if proba is None or len(y_pred) == 0:
            print(f"NOTE: policy {policy_name!r} has no proba columns; skipped.")
            continue
        if mask is None:
            triggered = np.ones(len(y_pred), dtype=bool)
        else:
            triggered = mask
        n_triggered = int(triggered.sum())
        if n_triggered == 0:
            print(f"NOTE: policy {policy_name!r} has no triggered rows; skipped.")
            continue
        proba_eff = proba[triggered]
        y_pred_eff = y_pred[triggered]
        y_true_eff = y_true[triggered]
        ece = expected_calibration_error(
            y_true_eff, y_pred_eff, proba_eff, classes=classes, n_bins=args.n_bins
        )
        mce = maximum_calibration_error(
            y_true_eff, y_pred_eff, proba_eff, classes=classes, n_bins=args.n_bins
        )
        brier = brier_score_multiclass(y_true_eff, proba_eff, classes=classes)
        bins = reliability_diagram_bins(y_true_eff, y_pred_eff, proba_eff, n_bins=args.n_bins)
        acc = float((y_pred_eff == y_true_eff).mean())
        rows.append(
            {
                "policy": policy_name,
                "n_total": int(len(y_true)),
                "n_triggered": n_triggered,
                "n_used": n_triggered,
                "accuracy": acc,
                "ECE": ece,
                "MCE": mce,
                "Brier": brier,
            }
        )
        bin_data[policy_name] = bins
    md = [
        "# Calibration report",
        "",
        f"Source: ``{args.oof.name}``; classes: {classes}",
        f"n_total = {int(len(y_true))} (OOF held-out predictions, 5-fold CV on train fold).",
        f"Bins: {args.n_bins}.",
        "",
        "**Triggered-rows-only reporting.** For the LLM arbitrator (which "
        "is only invoked on flat ↔ cascade disagreement) calibration is "
        "computed exclusively over rows where it actually produced a "
        "probability vector. The flat and cascade policies report over "
        "all rows, so their ``n_used`` equals ``n_total``.",
        "",
        "| Policy | n_used | Accuracy | ECE | MCE | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['policy']} | {r['n_used']} | {r['accuracy']:.4f} | "
            f"{r['ECE']:.4f} | {r['MCE']:.4f} | {r['Brier']:.4f} |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "* **ECE** (Expected Calibration Error): lower is better. ECE = 0 "
            "means confidence equals accuracy on every bin; ECE > 0.10 is a "
            "well-recognised miscalibration in the literature.",
            "* **MCE** (Maximum Calibration Error): worst-bin |conf − acc|. "
            "An MCE > 0.20 means at least one bin is badly miscalibrated, "
            "which is more informative for safety-critical deployment than "
            "the average ECE.",
            "* **Brier score**: proper scoring rule; lower is better. Decomposes "
            "(Murphy 1973) into reliability − resolution + uncertainty, "
            "letting us compare *calibration* across models with different "
            "confidence distributions.",
            "",
            "If ECE_cascade ≈ ECE_flat, the historical thesis claim "
            "'ECE сопоставим с flat baseline' is confirmed with numbers. "
            "If they differ by more than 0.02, that's a real finding to "
            "discuss in §3.3.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json:
        payload = {
            "source": str(args.oof),
            "classes": classes,
            "n_bins": args.n_bins,
            "policies": rows,
            "reliability_bins": bin_data,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    if args.out_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("WARNING: matplotlib not installed; skipping --out-png.")
        else:
            n_pol = len(bin_data)
            fig, axes = plt.subplots(
                1,
                n_pol,
                figsize=(4.5 * n_pol, 4),
                squeeze=False,
            )
            for ax, (policy, bins) in zip(axes[0], bin_data.items()):
                centers = bins["bin_centers"]
                conf = bins["bin_confidence"]
                acc = bins["bin_accuracy"]
                counts = bins["bin_count"]
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect")
                total = max(sum(counts), 1)
                widths = [0.08 + 0.04 * (c / total) for c in counts]
                ax.bar(centers, acc, width=widths, alpha=0.5, edgecolor="navy", label="empirical")
                ax.plot(
                    centers,
                    conf,
                    "o-",
                    color="crimson",
                    markersize=4,
                    linewidth=1,
                    label="confidence",
                )
                policy_row = next(r for r in rows if r["policy"] == policy)
                ax.set_title(
                    f"{policy}\nECE={policy_row['ECE']:.3f} "
                    f"MCE={policy_row['MCE']:.3f} "
                    f"Brier={policy_row['Brier']:.3f}",
                    fontsize=10,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Confidence (top-1)")
                ax.set_ylabel("Empirical accuracy")
                ax.legend(fontsize=8, loc="upper left")
                ax.grid(alpha=0.3)
            fig.suptitle(
                f"Reliability diagrams — {args.oof.stem}",
                fontsize=12,
            )
            fig.tight_layout()
            args.out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out_png, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"Wrote PNG      -> {args.out_png}")
    print()
    print("Calibration report summary:")
    for r in rows:
        print(
            f"  {r['policy']:14s}  n_used={r['n_used']:4d}  "
            f"acc={r['accuracy']:.4f}  "
            f"ECE={r['ECE']:.4f}  MCE={r['MCE']:.4f}  Brier={r['Brier']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
