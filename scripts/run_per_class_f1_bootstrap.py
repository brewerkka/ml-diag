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


def _bootstrap_class_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cls: str,
    *,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
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
        "n_bootstrap": int(n_bootstrap),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hybrid-eval",
        required=True,
        type=Path,
        help="Path to hybrid_evaluation_*.json with per-row predictions",
    )
    p.add_argument(
        "--corpus",
        required=True,
        type=str,
        help="Short corpus tag for the report title (8ds / 5ds / 3ds)",
    )
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument(
        "--target-classes",
        default="leakage,healthy",
        help="Comma-separated list of classes to bootstrap",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.hybrid_eval.is_file():
        print(f"ERROR: {args.hybrid_eval} not found", file=sys.stderr)
        return 1
    d = json.loads(args.hybrid_eval.read_text(encoding="utf-8"))
    targets = [c.strip() for c in args.target_classes.split(",") if c.strip()]
    policies = list(d.get("baselines", {}).keys()) + list(d.get("hybrid", {}).keys())
    results: dict = {}
    for policy in policies:
        slc = d.get("baselines", {}).get(policy, {}).get("slices", {}).get("full") or d.get(
            "hybrid", {}
        ).get(policy, {}).get("slices", {}).get("full")
        if not slc:
            continue
        y_true_list = slc.get("y_true")
        y_pred_list = slc.get("y_pred")
        if not (y_true_list and y_pred_list):
            point_only = slc.get("per_class_f1", {})
            results[policy] = {
                "point_estimates_only": True,
                "per_class_f1_point": point_only,
            }
            continue
        y_true = np.array(y_true_list, dtype=object)
        y_pred = np.array(y_pred_list, dtype=object)
        per_class: dict = {}
        for cls in targets:
            per_class[cls] = _bootstrap_class_f1_ci(
                y_true,
                y_pred,
                cls,
                n_bootstrap=args.n_bootstrap,
                alpha=0.05,
                seed=args.seed,
            )
        results[policy] = {"per_class_f1_bootstrap": per_class}
    md = [
        f"# Per-class F1 bootstrap CI on {args.corpus}",
        "",
        f"Source: ``{args.hybrid_eval.name}``; n_bootstrap = "
        f"{args.n_bootstrap}; α = 0.05; classes = {targets}.",
        "",
    ]
    has_any_bootstrap = any("per_class_f1_bootstrap" in v for v in results.values())
    if not has_any_bootstrap:
        md.append(
            "**NOTE:** the source JSON does not expose per-row predictions; "
            "only point estimates are reported below. To obtain bootstrap CIs, "
            "re-run ``scripts/run_hybrid_evaluation.py`` with the "
            "``--include-per-row-predictions`` flag (planned for next release)."
        )
        md.append("")
        md.append("| Policy | " + " | ".join(f"F1({c})" for c in targets) + " |")
        md.append("|" + "---|" * (len(targets) + 1))
        for policy, r in results.items():
            point = r.get("per_class_f1_point", {})
            row = [policy] + [f"{point.get(c, 0):.4f}" for c in targets]
            md.append("| " + " | ".join(row) + " |")
    else:
        md.append("| Policy | " + " | ".join(f"F1({c}) [95% CI]" for c in targets) + " |")
        md.append("|" + "---|" * (len(targets) + 1))
        for policy, r in results.items():
            if "per_class_f1_bootstrap" in r:
                row = [policy]
                for cls in targets:
                    b = r["per_class_f1_bootstrap"][cls]
                    row.append(f"{b['point']:.4f} [{b['ci_low']:.4f}, {b['ci_high']:.4f}]")
                md.append("| " + " | ".join(row) + " |")
            else:
                row = [policy]
                point = r.get("per_class_f1_point", {})
                for cls in targets:
                    row.append(f"{point.get(cls, 0):.4f} (no CI)")
                md.append("| " + " | ".join(row) + " |")
    md.extend(
        [
            "",
            "## Why this matters",
            "",
            "The headline tables (Table 3 in §3.3) report accuracy and macro-F1 "
            "with bootstrap CIs, but per-class F1 is reported as point estimate "
            "only. For the **leakage class — the principal safety-critical class** "
            "of the diagnostic — a single 0.55 point estimate hides a potentially "
            "wide CI that materially affects deployment decisions. This report "
            "fills the gap.",
            "",
            "## Interpretation guide",
            "",
            "* If ``leakage F1 CI`` width exceeds 0.20 on n_test = 160, the "
            "policy ranking on this class is **not statistically resolved** "
            "at this sample size — re-running on a larger test fold "
            "(n_test ≥ 500) is required for definitive conclusions.",
            "* If two policies' CIs overlap, the per-class F1 difference is "
            "not significant; the headline argument should rely on overall "
            "accuracy or macro-F1.",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "stage": 80,
                    "corpus": args.corpus,
                    "source": str(args.hybrid_eval),
                    "n_bootstrap": int(args.n_bootstrap),
                    "alpha": 0.05,
                    "targets": targets,
                    "per_policy": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print(f"Per-class F1 bootstrap on {args.corpus}:")
    for policy, r in results.items():
        if "per_class_f1_bootstrap" in r:
            for cls, b in r["per_class_f1_bootstrap"].items():
                print(
                    f"  {policy:24s} F1({cls}) = {b['point']:.4f} "
                    f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}]"
                )
        else:
            print(f"  {policy:24s} (point estimates only — JSON has no per-row predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
