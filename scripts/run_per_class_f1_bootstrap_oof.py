from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np              
import pandas as pd              

from ml_diag.features import build_feature_table              
from ml_diag.labels import PRIMARY_LABELS              


def _f1_for_class(y_true: np.ndarray, y_pred: np.ndarray, cls: str) -> float:
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    fp = int(((y_pred == cls) & (y_true != cls)).sum())
    fn = int(((y_pred != cls) & (y_true == cls)).sum())
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _bootstrap_class_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: str,
    *,
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    point = _f1_for_class(y_true, y_pred, cls)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        estimates[i] = _f1_for_class(y_true[idx], y_pred[idx], cls)
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "ci_width": float(hi - lo),
    }


def _argmax_per_row(df_group: pd.DataFrame, classes: list[str]) -> np.ndarray:
    arr = df_group[classes].to_numpy(dtype=float)
    row_sum = arr.sum(axis=1)
    bad = row_sum <= 0
    if bad.any():
        arr[bad] = 1.0 / len(classes)
    idx = arr.argmax(axis=1)
    return np.array([classes[i] for i in idx], dtype=object)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--tag", required=True, type=str, help="Short corpus tag for the report title (8ds/5ds/3ds)"
    )
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument(
        "--target-classes",
        default="leakage,healthy,overfitting,label_noise",
        help="Comma-separated list of classes for the CI table",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    df = pd.read_parquet(args.oof)
    if not isinstance(df.columns, pd.MultiIndex):
        print(
            "ERROR: OOF parquet must have MultiIndex columns (flat / cascade / arbitrator / meta).",
            file=sys.stderr,
        )
        return 1
    classes = list(PRIMARY_LABELS)
    ftable = build_feature_table(args.corpus)
    labels_by_run = ftable.df["primary_label"].astype(str).to_dict()
    y_true = np.array(
        [labels_by_run.get(str(rid), "") for rid in df.index],
        dtype=object,
    )
    valid = y_true != ""
    if not valid.all():
        print(f"NOTE: dropping {(~valid).sum()} unlabelled rows")
        df = df.loc[valid]
        y_true = y_true[valid]
    targets = [c.strip() for c in args.target_classes.split(",") if c.strip()]
    flat_pred = _argmax_per_row(df["flat"], classes)
    casc_pred = _argmax_per_row(df["cascade"], classes)
    arb_block = df["arbitrator"]
    arb_sum = arb_block[classes].sum(axis=1)
    triggered = arb_sum > 0
    arb_pred = _argmax_per_row(arb_block, classes)
    print(f"\nOOF predictions on {args.tag}:")
    print(f"  Total rows: {len(y_true)}")
    print(f"  Flat accuracy:    {(flat_pred == y_true).mean():.4f}")
    print(f"  Cascade accuracy: {(casc_pred == y_true).mean():.4f}")
    print(f"  Triggered rows (arbitrator): {int(triggered.sum())} of {len(y_true)}")
    if int(triggered.sum()) > 0:
        print(
            f"  Arbitrator accuracy (triggered): "
            f"{(arb_pred[triggered] == y_true[triggered]).mean():.4f}"
        )
    results: dict = {}
    for policy_name, y_pred in (("flat", flat_pred), ("cascade", casc_pred)):
        results[policy_name] = {}
        for cls in targets:
            results[policy_name][cls] = _bootstrap_class_f1(
                y_true,
                y_pred,
                cls,
                n_bootstrap=args.n_bootstrap,
                alpha=0.05,
                seed=args.seed,
            )
    if int(triggered.sum()) >= 10:
        results["arbitrator_triggered"] = {}
        y_true_arb = y_true[triggered]
        y_pred_arb = arb_pred[triggered]
        for cls in targets:
            results["arbitrator_triggered"][cls] = _bootstrap_class_f1(
                y_true_arb,
                y_pred_arb,
                cls,
                n_bootstrap=args.n_bootstrap,
                alpha=0.05,
                seed=args.seed,
            )
    md = [
        f"# Per-class F1 bootstrap CI on OOF predictions — {args.tag}",
        "",
        f"Source: ``{args.oof.name}`` (OOF predictions, n = {len(y_true)} train rows).",
        f"Bootstrap: n_resample = {args.n_bootstrap}, α = 0.05.",
        f"Target classes: {targets}.",
        "",
        "## Per-class F1 with 95 % bootstrap CI",
        "",
        "| Policy | " + " | ".join(f"F1({c}) [95% CI]" for c in targets) + " |",
        "|" + "---|" * (len(targets) + 1),
    ]
    for policy in ("flat", "cascade", "arbitrator_triggered"):
        if policy not in results:
            continue
        row = [policy]
        for cls in targets:
            b = results[policy][cls]
            row.append(f"{b['point']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]")
        md.append("| " + " | ".join(row) + " |")
    md.extend(
        [
            "",
            "## CI widths",
            "",
            "| Policy | " + " | ".join(f"width({c})" for c in targets) + " |",
            "|" + "---|" * (len(targets) + 1),
        ]
    )
    for policy in ("flat", "cascade", "arbitrator_triggered"):
        if policy not in results:
            continue
        row = [policy]
        for cls in targets:
            row.append(f"{results[policy][cls]['ci_width']:.3f}")
        md.append("| " + " | ".join(row) + " |")
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "**CI width is the diagnostic, not the point estimate.** When the "
            "CI width on a per-class F1 exceeds 0.20 at n = "
            f"{len(y_true)}, the policy ranking for that class is **not "
            "statistically resolved** at this sample size. Widening the test "
            "fold (n ≥ 1000) is required to discriminate between policies on "
            "individual minority classes.",
            "",
            "The **leakage class is the safety-critical priority** of the "
            "diagnostic system: false positives (healthy → leakage) cause "
            "unnecessary deployment delays, false negatives (leakage → healthy) "
            "release a leaky model. The CI on leakage F1 quantifies the "
            "epistemic uncertainty in deployment decisions.",
            "",
            "These CIs are computed on **OOF predictions** (n = "
            f"{len(y_true)}), not on the test fold (typically n = 160). OOF "
            "is methodologically equivalent for per-class F1 because each "
            "OOF row was predicted by a model that did not see it during "
            "training (5-fold leave-one-out structure). This gives **wider** "
            "yet **honest** CIs compared to the smaller test-fold-only "
            "analysis.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote markdown -> {args.out_md}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "stage": 80,
                    "tag": args.tag,
                    "source": str(args.oof),
                    "n_oof": int(len(y_true)),
                    "n_triggered_arb": int(triggered.sum()),
                    "n_bootstrap": int(args.n_bootstrap),
                    "alpha": 0.05,
                    "targets": targets,
                    "per_policy_per_class": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
