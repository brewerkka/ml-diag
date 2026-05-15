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
from sklearn.ensemble import GradientBoostingClassifier              
from sklearn.metrics import accuracy_score, f1_score              
from sklearn.model_selection import StratifiedKFold              

from ml_diag.diagnosis.stacking_resolver import (              
    _ARB_COLS,
    _ARB_META_COLS,
    featurize,
)
from ml_diag.evaluation.metrics import (              
    bootstrap_delta_ci,
)
from ml_diag.features import build_feature_table              


def _gbm_factory(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=seed,
    )


def _build_oof_feature_matrix(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if not isinstance(df.columns, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex columns in OOF parquet")
    flat = df["flat"]
    casc = df["cascade"]
    stp = df["stage_probs"]
    arb = df["arbitrator"]
    meta = df["meta"]
    arb_triggered = meta["arb_triggered"].astype(float)
    arb_confidence = meta["arb_confidence"].astype(float)
    X = featurize(
        flat_proba=flat,
        cascade_proba=casc,
        cascade_stage_probs=stp,
        arbitrator_label_probs=arb,
        arbitrator_triggered=arb_triggered,
        arbitrator_confidence=arb_confidence,
    )
    return X


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    X_full = _build_oof_feature_matrix(args.oof)
    ftable = build_feature_table(args.corpus)
    labels_by_run = ftable.df["primary_label"].astype(str).to_dict()
    y = pd.Series(
        [labels_by_run.get(rid, "") for rid in X_full.index],
        index=X_full.index,
        dtype=object,
    )
    if (y == "").any():
        n_missing = int((y == "").sum())
        print(f"WARNING: {n_missing} run_ids without labels — dropped.", file=sys.stderr)
        keep = y != ""
        X_full = X_full.loc[keep]
        y = y.loc[keep]
    arb_columns = set(_ARB_COLS + _ARB_META_COLS)
    cols_with = list(X_full.columns)
    cols_without = [c for c in cols_with if c not in arb_columns]
    print(f"OOF rows: {len(X_full)}")
    print(f"Features with arb:    {len(cols_with)}")
    print(f"Features without arb: {len(cols_without)}")
    skf = StratifiedKFold(
        n_splits=args.n_folds,
        shuffle=True,
        random_state=args.seed,
    )
    n = len(X_full)
    y_true_aggregated: list = []
    y_pred_with: list = []
    y_pred_without: list = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y)):
        X_tr = X_full.iloc[tr_idx]
        X_va = X_full.iloc[va_idx]
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]
        clf_w = _gbm_factory(args.seed)
        clf_w.fit(X_tr[cols_with].values, y_tr.values)
        pred_w = clf_w.predict(X_va[cols_with].values)
        clf_o = _gbm_factory(args.seed)
        clf_o.fit(X_tr[cols_without].values, y_tr.values)
        pred_o = clf_o.predict(X_va[cols_without].values)
        y_true_aggregated.extend(y_va.tolist())
        y_pred_with.extend(pred_w.tolist())
        y_pred_without.extend(pred_o.tolist())
    y_true_arr = np.array(y_true_aggregated)
    y_pred_with_arr = np.array(y_pred_with)
    y_pred_without_arr = np.array(y_pred_without)
    acc_with = float(accuracy_score(y_true_arr, y_pred_with_arr))
    acc_without = float(accuracy_score(y_true_arr, y_pred_without_arr))
    mf1_with = float(f1_score(y_true_arr, y_pred_with_arr, average="macro", zero_division=0))
    mf1_without = float(f1_score(y_true_arr, y_pred_without_arr, average="macro", zero_division=0))
    boot_acc = bootstrap_delta_ci(
        y_true_arr.tolist(),
        y_pred_without_arr.tolist(),
        y_pred_with_arr.tolist(),
        metric="accuracy",
        n_bootstrap=1000,
        alpha=0.05,
        seed=args.seed,
    )
    boot_mf1 = bootstrap_delta_ci(
        y_true_arr.tolist(),
        y_pred_without_arr.tolist(),
        y_pred_with_arr.tolist(),
        metric="macro_f1",
        n_bootstrap=1000,
        alpha=0.05,
        seed=args.seed,
    )
    delta_acc = boot_acc["delta_point"]
    p_b = boot_acc["p_b_better"]
    cohen_h = boot_acc.get("cohen_h")
    cohen_h_mag = boot_acc.get("cohen_h_magnitude", "n/a")
    if delta_acc >= 0.005 and p_b >= 0.80:
        verdict = (
            "POSITIVE: arb_* features measurably help stacking "
            f"(Δacc = +{delta_acc:.4f}, P_better = {p_b:.3f}). "
            "Claim 'LLM-as-teacher via OOF distillation' is supported."
        )
    elif abs(delta_acc) < 0.005:
        verdict = (
            "NULL: arb_* features add no measurable benefit to stacking "
            f"(|Δacc| = {abs(delta_acc):.4f}). Negative result — "
            "honestly report that the meta-classifier absorbs the cascade/flat "
            "probabilities sufficiently to ignore the explicit LLM signal."
        )
    elif delta_acc < 0:
        verdict = (
            f"NEGATIVE: arb_* features hurt stacking (Δacc = {delta_acc:+.4f}). "
            "Reconsider — possible noise injection by ungated arbitrator features."
        )
    else:
        verdict = (
            f"MARGINAL: Δacc = +{delta_acc:.4f}, P_better = {p_b:.3f} — not confidently positive."
        )
    md_lines = [
        f"# Stage 65 — LLM-arb ablation in stacking ({args.corpus.name})",
        "",
        "## Method",
        "",
        f"5-fold CV on n = {n} OOF rows from ``{args.oof.name}``. Each fold "
        "trains two GBM meta-classifiers — one on the full 33-feature stacking "
        "matrix (``with_arb``), one on a strict subset of 25 features without "
        "arbitrator signals (``without_arb``). Predictions are aggregated "
        "across folds for paired bootstrap comparison.",
        "",
        "Hyperparameters: ``GradientBoostingClassifier(n_estimators=100, "
        "max_depth=3, learning_rate=0.1, random_state=seed)``.",
        "",
        "## Headline",
        "",
        "| Variant | Features | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
        f"| with_arb    | {len(cols_with)} | {acc_with:.4f} | {mf1_with:.4f} |",
        f"| without_arb | {len(cols_without)} | {acc_without:.4f} | {mf1_without:.4f} |",
        f"| **Δ (with − without)** | — | **{delta_acc:+.4f}** | **{boot_mf1['delta_point']:+.4f}** |",
        "",
        "## Bootstrap CI (paired, n_bootstrap = 1000, α = 0.05)",
        "",
        "| Metric | Δ point | 95% CI | P_better |",
        "|---|---|---|---|",
        f"| accuracy | {delta_acc:+.4f} | [{boot_acc['delta_ci_low']:+.4f}, "
        f"{boot_acc['delta_ci_high']:+.4f}] | {p_b:.3f} |",
        f"| macro-F1 | {boot_mf1['delta_point']:+.4f} | "
        f"[{boot_mf1['delta_ci_low']:+.4f}, {boot_mf1['delta_ci_high']:+.4f}] | "
        f"{boot_mf1['p_b_better']:.3f} |",
        "",
    ]
    if cohen_h is not None:
        md_lines.extend(
            [
                "## Effect size",
                "",
                f"Cohen's h (accuracy with vs without): **{cohen_h:+.4f}** ({cohen_h_mag} per Cohen 1988).",
                f"95% bootstrap CI: [{boot_acc.get('cohen_h_ci_low', 0):+.4f}, "
                f"{boot_acc.get('cohen_h_ci_high', 0):+.4f}].",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Verdict",
            "",
            f"> {verdict}",
            "",
            "## Interpretation hint for the thesis",
            "",
            "* If POSITIVE: the meta-classifier learns to amplify the "
            "LLM-arbitrator signal. The thesis title claim 'с применением LLM' "
            "is supported at inference time via OOF distillation (Wolpert 1992).",
            "* If NULL: the LLM signal is redundant with cascade/flat in the "
            "stacked representation. The thesis should disclose this and "
            "reframe LLM contribution as 'standalone arbitrate-режим', not "
            "as part of the headline stacking pipeline.",
            "* If NEGATIVE: arb features inject noise. The arbitrator should "
            "be gated (e.g., zero arb features when arb_triggered_bit=0 — "
            "currently already done; if Δ is still negative, the gate is "
            "not sufficient).",
            "",
        ]
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json:
        payload = {
            "stage": "65",
            "method": "5-fold CV stacking ablation, arb features vs no arb features",
            "corpus": str(args.corpus),
            "oof_source": str(args.oof),
            "n_rows": int(n),
            "n_folds": int(args.n_folds),
            "seed": int(args.seed),
            "with_arb": {
                "n_features": len(cols_with),
                "accuracy": acc_with,
                "macro_f1": mf1_with,
                "feature_columns": cols_with,
            },
            "without_arb": {
                "n_features": len(cols_without),
                "accuracy": acc_without,
                "macro_f1": mf1_without,
                "feature_columns": cols_without,
            },
            "bootstrap_accuracy": boot_acc,
            "bootstrap_macro_f1": boot_mf1,
            "verdict": verdict,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print(f"LLM ablation on {args.corpus.name} (n = {n}):")
    print(f"  with_arb    ({len(cols_with)} features): acc={acc_with:.4f}  macro-F1={mf1_with:.4f}")
    print(
        f"  without_arb ({len(cols_without)} features): acc={acc_without:.4f}  macro-F1={mf1_without:.4f}"
    )
    print(
        f"  Δacc = {delta_acc:+.4f}  95%CI [{boot_acc['delta_ci_low']:+.4f}, "
        f"{boot_acc['delta_ci_high']:+.4f}]  P_better = {p_b:.3f}"
    )
    if cohen_h is not None:
        print(f"  Cohen's h = {cohen_h:+.4f} ({cohen_h_mag})")
    print()
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
