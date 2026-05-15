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
from sklearn.model_selection import StratifiedKFold              

from ml_diag.evaluation import classification_report              
from ml_diag.features import (              
    build_data_integrity_features,
    build_feature_table,
)
from ml_diag.labels import HEALTHY, to_stage1              
from ml_diag.models import load_cascade              
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.utils import setup_logging              

CANDIDATE_THRESHOLDS = [
    0.50,
    0.55,
    0.60,
    0.62,
    0.65,
    0.68,
    0.70,
    0.72,
    0.75,
    0.78,
    0.80,
    0.85,
    0.90,
]


def _stage1_predict(cascade, X: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    from ml_diag.models.inference import _proba_or_onehot, _row_for_stage

    preds = []
    p_h = []
    for _, row in X.iterrows():
        arr = _row_for_stage(row, cascade.stage1)
        proba = _proba_or_onehot(cascade.stage1.model, arr, cascade.stage1.classes or [])
        ph = float(proba.get(HEALTHY, 0.0))
        preds.append(HEALTHY if ph >= threshold else "faulty")
        p_h.append(ph)
    return np.array(preds, dtype=object), np.array(p_h, dtype=float)


def _out_of_fold_p_healthy(
    X_train_full: pd.DataFrame,
    y_train_primary: pd.Series,
    *,
    n_folds: int = 5,
    seed: int = 0,
    calibrate: bool = True,
) -> np.ndarray:
    from ml_diag.labels import STAGE1_LABELS
    from ml_diag.models.inference import _proba_or_onehot, _row_for_stage
    from ml_diag.models.stage1 import prepare as _stage1_prepare
    from ml_diag.models.trainer import train_stage

    inner_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_stage1 = y_train_primary.map(to_stage1)
    p_healthy = np.full(len(X_train_full), np.nan, dtype=float)
    for fold_idx, (tr, va) in enumerate(inner_skf.split(X_train_full, y_stage1)):
        X_tr = X_train_full.iloc[tr]
        y_tr = y_train_primary.iloc[tr]
        X_va = X_train_full.iloc[va]
        X_s, y_s = _stage1_prepare(X_tr, y_tr)
        result = train_stage(
            stage_name=f"stage1_oof_fold{fold_idx}",
            X=X_s,
            y=y_s,
            label_vocab=STAGE1_LABELS,
            seed=seed,
            calibrate=calibrate,
        )
        from ml_diag.models.inference import _StageModel

        stage_model = _StageModel(
            name=result.stage_name,
            model=result.model,
            classes=result.classes,
            feature_columns=result.feature_columns,
        )
        for j, (_, row) in enumerate(X_va.iterrows()):
            arr = _row_for_stage(row, stage_model)
            proba = _proba_or_onehot(stage_model.model, arr, stage_model.classes or [])
            p_healthy[va[j]] = float(proba.get(HEALTHY, 0.0))
        print(
            f"  [oof fold {fold_idx + 1}/{n_folds}] trained Stage 1, "
            f"predicted on {len(va)} held-out rows"
        )
    return p_healthy


def _evaluate_threshold_oof(
    p_healthy: np.ndarray,
    y_primary: pd.Series,
    threshold: float,
) -> dict:
    preds = np.where(p_healthy >= threshold, HEALTHY, "faulty").astype(object)
    y_binary = y_primary.map(to_stage1).values
    from ml_diag.evaluation import classification_report

    rep = classification_report(y_binary, preds, label_order=("healthy", "faulty"))
    is_leakage = (y_primary == "leakage").values
    is_healthy = (y_primary == "healthy").values
    leak_to_healthy = int(((preds == HEALTHY) & is_leakage).sum())
    healthy_to_faulty = int(((preds == "faulty") & is_healthy).sum())
    return {
        "threshold": float(threshold),
        "macro_f1": rep.macro_f1,
        "accuracy": rep.accuracy,
        "healthy_f1": rep.per_class_f1.get("healthy", 0.0),
        "faulty_f1": rep.per_class_f1.get("faulty", 0.0),
        "leakage_to_healthy": leak_to_healthy,
        "leakage_recall_loss": leak_to_healthy / max(1, int(is_leakage.sum())),
        "healthy_false_alarm": healthy_to_faulty / max(1, int(is_healthy.sum())),
        "n_leakage": int(is_leakage.sum()),
        "n_healthy": int(is_healthy.sum()),
    }


def _evaluate_threshold(
    *,
    cascade,
    X_val,
    y_val_binary,
    y_val_primary,
    threshold: float,
) -> dict:
    preds, _ = _stage1_predict(cascade, X_val, threshold)
    rep = classification_report(y_val_binary, preds, label_order=("healthy", "faulty"))
    is_leakage = (y_val_primary == "leakage").values
    is_healthy = (y_val_primary == "healthy").values
    leak_to_healthy = int(((preds == HEALTHY) & is_leakage).sum())
    healthy_to_faulty = int(((preds == "faulty") & is_healthy).sum())
    n_leak = int(is_leakage.sum())
    n_healthy = int(is_healthy.sum())
    return {
        "threshold": threshold,
        "macro_f1": rep.macro_f1,
        "accuracy": rep.accuracy,
        "healthy_f1": rep.per_class_f1.get("healthy", 0.0),
        "faulty_f1": rep.per_class_f1.get("faulty", 0.0),
        "leakage_to_healthy": leak_to_healthy,
        "leakage_recall_loss": leak_to_healthy / max(1, n_leak),
        "healthy_false_alarm": healthy_to_faulty / max(1, n_healthy),
        "n_leakage": n_leak,
        "n_healthy": n_healthy,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--artifacts", required=True, type=Path, help="Hierarchical artifacts dir.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--metric",
        default="cost_sensitive",
        choices=["macro_f1", "leakage_recall_loss", "cost_sensitive"],
        help=(
            "macro_f1: пик macro-F1 (balanced); "
            "leakage_recall_loss: минимизация leakage→healthy "
            "при ограничении healthy false alarm; "
            "cost_sensitive (default): минимизация "
            "w_leak·leakage_recall_loss + w_hlth_fa·healthy_FA, "
            "т.е. явно взвешенный trade-off."
        ),
    )
    p.add_argument(
        "--max-healthy-false-alarm",
        type=float,
        default=0.30,
        help="При оптимизации по leakage_recall_loss требование "
        "не превышать данный уровень healthy false alarm.",
    )
    p.add_argument(
        "--cost-w-leak",
        type=float,
        default=1.0,
        help="Вес ошибки leakage→healthy в cost_sensitive метрике "
        "(по умолчанию 1.0 — равные веса). Поставь >1 если "
        "ловить leakage важнее чем избежать false alarm.",
    )
    p.add_argument(
        "--cost-w-hlth-fa",
        type=float,
        default=1.0,
        help="Вес ошибки healthy→faulty (false alarm) в cost_sensitive.",
    )
    p.add_argument(
        "--out", type=Path, default=None, help="По умолчанию: <artifacts>/cascade_config.json"
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Число inner CV folds для out-of-fold predictions. "
        "Поставь --no-cv для legacy single-fold-mode.",
    )
    p.add_argument(
        "--no-cv",
        action="store_true",
        help="Использовать legacy single-fold-mode (1/5 train fold) вместо out-of-fold CV.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level="INFO")
    cascade = load_cascade(args.artifacts)
    print(
        f"Loaded cascade: stages={cascade.stages_available}, "
        f"current stage1 threshold={cascade.stage1_healthy_threshold:.3f}"
    )
    base = build_feature_table(args.corpus)
    try:
        di = build_data_integrity_features(args.corpus, base_table=base)
        full_df = di.df
    except Exception:
        full_df = base.df
    X_all, y_all = base.aligned_xy()
    feature_cols = base.feature_columns
    X_all = full_df.loc[X_all.index, feature_cols]
    train_idx, _ = _split_train_test(X_all, y_all, seed=args.seed)
    X_train_full = X_all.iloc[train_idx]
    y_train_full = y_all.iloc[train_idx]
    rows: list[dict] = []
    if args.no_cv:
        inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        inner_train_idx, inner_val_idx = next(iter(inner.split(X_train_full, y_train_full)))
        X_val = X_train_full.iloc[inner_val_idx]
        y_val_primary = y_train_full.iloc[inner_val_idx]
        y_val_binary = y_val_primary.map(to_stage1)
        print(
            f"[single-fold mode] Validation fold size: {len(X_val)} runs "
            f"(healthy={int((y_val_binary == 'healthy').sum())}, "
            f"faulty={int((y_val_binary == 'faulty').sum())})"
        )
        print()
        print(
            f"{'thr':>6}  {'macro_f1':>10}  {'acc':>7}  "
            f"{'leak→hlth':>10}  {'leak_recall_loss':>18}  {'hlth_FA':>9}"
        )
        for thr in CANDIDATE_THRESHOLDS:
            m = _evaluate_threshold(
                cascade=cascade,
                X_val=X_val,
                y_val_binary=y_val_binary,
                y_val_primary=y_val_primary,
                threshold=thr,
            )
            rows.append(m)
            print(
                f"{m['threshold']:6.2f}  {m['macro_f1']:10.4f}  "
                f"{m['accuracy']:7.4f}  {m['leakage_to_healthy']:10d}  "
                f"{m['leakage_recall_loss']:18.4f}  {m['healthy_false_alarm']:9.4f}"
            )
    else:
        print(
            f"[cv mode] Building out-of-fold P(healthy) on full train fold "
            f"({len(X_train_full)} runs, k={args.cv_folds})…"
        )
        p_healthy = _out_of_fold_p_healthy(
            X_train_full,
            y_train_full,
            n_folds=args.cv_folds,
            seed=args.seed,
        )
        n_oof = int(np.isfinite(p_healthy).sum())
        n_leak = int((y_train_full == "leakage").sum())
        n_hlth = int((y_train_full == "healthy").sum())
        print(f"  collected {n_oof} out-of-fold predictions (healthy={n_hlth}, leakage={n_leak})")
        print()
        print(
            f"{'thr':>6}  {'macro_f1':>10}  {'acc':>7}  "
            f"{'leak→hlth':>10}  {'leak_recall_loss':>18}  {'hlth_FA':>9}"
        )
        for thr in CANDIDATE_THRESHOLDS:
            m = _evaluate_threshold_oof(p_healthy, y_train_full, threshold=thr)
            rows.append(m)
            print(
                f"{m['threshold']:6.2f}  {m['macro_f1']:10.4f}  "
                f"{m['accuracy']:7.4f}  {m['leakage_to_healthy']:10d}  "
                f"{m['leakage_recall_loss']:18.4f}  {m['healthy_false_alarm']:9.4f}"
            )
    if args.metric == "macro_f1":
        best = max(rows, key=lambda r: r["macro_f1"])
        rationale = (
            f"Selected threshold {best['threshold']:.3f} maximises macro-F1 "
            f"({best['macro_f1']:.4f}) on validation fold."
        )
    elif args.metric == "leakage_recall_loss":
        feasible = [r for r in rows if r["healthy_false_alarm"] <= args.max_healthy_false_alarm]
        if not feasible:
            print(
                f"WARNING: no threshold meets healthy_false_alarm ≤ "
                f"{args.max_healthy_false_alarm:.2f}; falling back to argmax."
            )
            best = next(r for r in rows if r["threshold"] == 0.5)
            rationale = "Fell back to default 0.5 (no candidate met constraint)."
        else:
            best = min(feasible, key=lambda r: r["leakage_recall_loss"])
            rationale = (
                f"Selected threshold {best['threshold']:.3f} minimises "
                f"leakage→healthy recall loss ({best['leakage_recall_loss']:.4f}) "
                f"subject to healthy_false_alarm ≤ {args.max_healthy_false_alarm:.2f}."
            )
    else:
        w_leak = float(args.cost_w_leak)
        w_hlth = float(args.cost_w_hlth_fa)
        for r in rows:
            r["cost"] = w_leak * r["leakage_recall_loss"] + w_hlth * r["healthy_false_alarm"]
        best = min(rows, key=lambda r: r["cost"])
        rationale = (
            f"Selected threshold {best['threshold']:.3f} minimises "
            f"cost = {w_leak}·leakage_recall_loss + {w_hlth}·healthy_false_alarm "
            f"= {best['cost']:.4f} (leakage→healthy: "
            f"{best['leakage_to_healthy']}/{best['n_leakage']}, "
            f"healthy_FA: {best['healthy_false_alarm']:.2%})."
        )
    print()
    print("=" * 70)
    print(rationale)
    print("=" * 70)
    if args.metric == "cost_sensitive":
        print()
        print("Что выбрал бы cost_sensitive при других весах (для сравнения):")
        print(f"{'веса':>22}  {'thr':>5}  {'leak→h':>8}  {'hlth_FA':>9}  {'macro_F1':>9}")
        for wl, wh, label in [
            (1.0, 1.0, "balanced"),
            (2.0, 1.0, "leak ×2"),
            (3.0, 1.0, "leak ×3"),
            (1.0, 2.0, "FA ×2 (защита healthy)"),
        ]:
            scored = [
                {**r, "_c": wl * r["leakage_recall_loss"] + wh * r["healthy_false_alarm"]}
                for r in rows
            ]
            b = min(scored, key=lambda r: r["_c"])
            print(
                f"  w_leak={wl}, w_FA={wh:>3} ({label:>15})  "
                f"{b['threshold']:5.2f}  {b['leakage_to_healthy']:8d}  "
                f"{b['healthy_false_alarm']:9.4f}  {b['macro_f1']:9.4f}"
            )
        print()
        print("Если default не устраивает — перезапусти с явными весами, например:")
        print(
            "  python scripts/tune_stage1_threshold.py "
            "--metric cost_sensitive --cost-w-leak 2.0 --cost-w-hlth-fa 1.0 …"
        )
    out = args.out or (args.artifacts / "cascade_config.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "stage1_healthy_threshold": float(best["threshold"]),
        "tuning_metric": args.metric,
        "tuning_mode": "single_fold" if args.no_cv else "out_of_fold_cv",
        "tuning_n_points": (int(len(X_val)) if args.no_cv else int(len(X_train_full))),
        "tuning_rationale": rationale,
        "threshold_sweep": rows,
    }
    out.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out}")
    print(
        f"Каскад теперь будет использовать stage1_healthy_threshold = "
        f"{best['threshold']:.3f} при следующих вызовах load_cascade()."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
