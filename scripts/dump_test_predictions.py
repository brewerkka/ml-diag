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

from ml_diag.diagnosis.conformal_layer import (              
    calibrate_split_conformal,
    compute_meta_oof_probabilities,
    predict_with_conformal,
)
from ml_diag.diagnosis.hybrid_resolver import (              
    cascade_marginal_proba,
    flat_proba_aligned,
)
from ml_diag.diagnosis.oof_predictions import (              
    STAGE_PROBA_COLS,
    read_oof_parquet,
)
from ml_diag.diagnosis.stacking_resolver import (              
    stacking_predict,
    train_stacking_meta,
)
from ml_diag.features import build_feature_table              
from ml_diag.models import train_flat_baseline              
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.models.inference import load_cascade              
from ml_diag.models.model_zoo import default_zoo              

REPO_ROOT = _REPO_ROOT


def _compute_cascade_stage_probs_test(cascade, X_test: pd.DataFrame) -> pd.DataFrame:
    from ml_diag.diagnosis.oof_predictions import _build_cascade_predictions

    class _FakeRes:
        def __init__(self, sm):
            self.stage_name = sm.name
            self.model = sm.model
            self.classes = list(sm.classes)
            self.feature_columns = list(sm.feature_columns)

    s1_res = _FakeRes(cascade.stage1)
    s2_res = _FakeRes(cascade.stage2) if cascade.stage2 is not None else None
    s3d_res = _FakeRes(cascade.stage3_data) if cascade.stage3_data is not None else None
    s3o_res = _FakeRes(cascade.stage3_opt) if cascade.stage3_opt is not None else None
    _composed_unused, stage_probs = _build_cascade_predictions(
        stage1_res=s1_res,
        stage2_res=s2_res,
        stage3d_res=s3d_res,
        stage3o_res=s3o_res,
        X_holdout=X_test,
    )
    return stage_probs[list(STAGE_PROBA_COLS)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=str(REPO_ROOT / "data" / "corpus" / "real_8ds_n5_multi"))
    p.add_argument("--hier-artifacts",
                   default=str(REPO_ROOT / "results" / "hierarchical" / "real_8ds_n5_multi"))
    p.add_argument("--stacking-oof",
                   default=str(REPO_ROOT / "results" / "oof_predictions_8ds.parquet"))
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "test_predictions_8ds_full.json"))
    p.add_argument(
        "--flat-model",
        default="random_forest",
        choices=["logreg", "random_forest", "gradient_boosting", "auto"],
        help="Force a specific flat baseline model (or 'auto' for CV-based selection). "
             "Default: random_forest — reproduces canonical Stage 57 headline numbers.",
    )
    args = p.parse_args()

    print(f"Loading feature table for {args.corpus}...")
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()

    print(f"Computing canonical train/test split (seed={args.seed})...")
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = list(X.index[test_idx])
    X_te = X.loc[test_run_ids]
    y_te = y.loc[test_run_ids]
    print(f"  train n={len(train_idx)}  test n={len(test_idx)}")

    zoo = default_zoo(include_catboost=False)
    if args.flat_model != "auto":
        zoo = [spec for spec in zoo if spec.name == args.flat_model]
        if not zoo:
            raise ValueError(f"--flat-model={args.flat_model!r} not found in default zoo")
        print(f"Training flat baseline (forced: {args.flat_model})...")
    else:
        print("Training flat baseline (auto-select from zoo)...")
    flat = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    print(f"  picked: {flat.model_name}")

    print(f"Loading cascade artifacts from {args.hier_artifacts}...")
    cascade = load_cascade(args.hier_artifacts)

    print("Computing base predictions on test fold...")
    flat_proba_df = flat_proba_aligned(flat, X_te)
    cascade_proba_df = cascade_marginal_proba(cascade, X_te)
    cascade_stage_probs_test = _compute_cascade_stage_probs_test(cascade, X_te)

    print(f"Loading OOF predictions from {args.stacking_oof}...")
    oof = read_oof_parquet(args.stacking_oof)
    y_oof = y.loc[oof.index()]
    print(f"  OOF rows: {len(oof.index())}")

    print("Training stacking GBM meta-classifier...")
    meta = train_stacking_meta(oof, y_oof, classifier="gbm", seed=args.seed)
    print(f"  CV macro-F1: {meta.cv_score_macro_f1:.4f}")

    print("Building EMPTY arbitrator inputs (standalone stacking, no LLM)...")
    classes = list(flat_proba_df.columns)
    n_te = len(X_te)
    empty_arb_proba = pd.DataFrame(
        np.full((n_te, len(classes)), 1.0 / len(classes)),
        index=X_te.index, columns=classes,
    )
    empty_arb_trig = pd.Series(np.zeros(n_te, dtype=float), index=X_te.index)
    empty_arb_conf = pd.Series(np.zeros(n_te, dtype=float), index=X_te.index)

    print("Applying meta to test fold...")
    pred_te, proba_te = stacking_predict(
        meta,
        flat_proba=flat_proba_df,
        cascade_proba=cascade_proba_df,
        cascade_stage_probs=cascade_stage_probs_test,
        arbitrator_label_probs=empty_arb_proba,
        arbitrator_triggered=empty_arb_trig,
        arbitrator_confidence=empty_arb_conf,
    )

    print(f"Computing OOF-of-OOF meta probabilities for conformal calibration (α={args.alpha})...")
    proba_oof_meta = compute_meta_oof_probabilities(
        oof=oof,
        y_train=y_oof,
        classifier="gbm",
        seed=args.seed,
        n_folds=5,
    )
    calibrator = calibrate_split_conformal(
        proba_oof=proba_oof_meta,
        y_oof=y_oof,
        alpha=args.alpha,
        score_method="lac",
    )
    print(f"  q_hat = {calibrator.quantile:.4f}")

    conformal_results = predict_with_conformal(
        proba_test=proba_te,
        calibrator=calibrator,
        run_ids=[str(r) for r in proba_te.index],
    )
    conformal_by_id = {r.run_id: r for r in conformal_results}

    print("Computing flat + cascade argmax predictions for comparison...")
    flat_pred = flat_proba_df.idxmax(axis=1)
    cascade_pred = cascade_proba_df.idxmax(axis=1)

    print("Building per-row dump...")
    rows = []
    for rid in test_run_ids:
        rid_s = str(rid)
        cf = conformal_by_id.get(rid_s)
        rows.append({
            "run_id": rid_s,
            "true_label": str(y_te.loc[rid]),
            "flat_pred": str(flat_pred.loc[rid]),
            "cascade_pred": str(cascade_pred.loc[rid]),
            "stacking_pred": str(pred_te.loc[rid]),
            "stacking_proba": {c: float(proba_te.at[rid, c]) for c in proba_te.columns},
            "prediction_set": list(cf.prediction_set) if cf else None,
            "set_size": int(cf.set_size) if cf else None,
            "is_abstained": bool(cf.is_abstained) if cf else False,
            "nonconformity": float(cf.nonconformity) if cf else None,
        })

    out = {
        "corpus": str(args.corpus),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "n_test": len(rows),
        "stacking_classifier": "gbm",
        "conformal": {
            "score_method": "lac",
            "q_hat": float(calibrator.quantile),
            "n_calibration": int(calibrator.n_calibration),
        },
        "predictions": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDumped {len(rows)} rows -> {out_path}")

    n_correct_stack = sum(1 for r in rows if r["stacking_pred"] == r["true_label"])
    n_correct_flat = sum(1 for r in rows if r["flat_pred"] == r["true_label"])
    n_abstained = sum(1 for r in rows if r["is_abstained"])
    print(f"  stacking acc: {n_correct_stack}/{len(rows)} = {n_correct_stack/len(rows):.4f}")
    print(f"  flat     acc: {n_correct_flat}/{len(rows)} = {n_correct_flat/len(rows):.4f}")
    print(f"  abstained:    {n_abstained}/{len(rows)} = {n_abstained/len(rows):.4f}")


if __name__ == "__main__":
    main()
