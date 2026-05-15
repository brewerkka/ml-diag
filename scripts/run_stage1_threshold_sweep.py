from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np              
import pandas as pd              

from ml_diag.benchmark import partition_corpus              
from ml_diag.evaluation import classification_report              
from ml_diag.features import build_feature_table              
from ml_diag.labels import (              
    FAULTY,
    HEALTHY,
    LABEL_NOISE,
    LEAKAGE,
    PRIMARY_LABELS,
    to_stage1,
)
from ml_diag.models import (              
    load_cascade,
    slices_from_partition,
)
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.models.inference import diagnose_batch              
from ml_diag.utils import setup_logging              

THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

DEFAULT_COST_WEIGHTS = {
    "leakage_to_healthy": 3.0,
    "label_noise_to_healthy": 2.0,
    "other_faulty_to_healthy": 1.5,
    "healthy_to_faulty": 1.0,
}


def total_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: dict[str, float] = DEFAULT_COST_WEIGHTS,
) -> dict[str, float | int]:
    leak_to_h = int(((y_true == LEAKAGE) & (y_pred == HEALTHY)).sum())
    ln_to_h = int(((y_true == LABEL_NOISE) & (y_pred == HEALTHY)).sum())
    other_to_h = int(
        (
            (y_true != HEALTHY)
            & (y_true != LEAKAGE)
            & (y_true != LABEL_NOISE)
            & (y_pred == HEALTHY)
        ).sum()
    )
    h_to_f = int(((y_true == HEALTHY) & (y_pred != HEALTHY)).sum())
    cost = (
        weights["leakage_to_healthy"] * leak_to_h
        + weights["label_noise_to_healthy"] * ln_to_h
        + weights["other_faulty_to_healthy"] * other_to_h
        + weights["healthy_to_faulty"] * h_to_f
    )
    return {
        "leakage_to_healthy": leak_to_h,
        "label_noise_to_healthy": ln_to_h,
        "other_faulty_to_healthy": other_to_h,
        "healthy_to_faulty": h_to_f,
        "total_cost": float(cost),
    }


def _hier_predict_with_threshold(
    cascade,
    X: pd.DataFrame,
    threshold: float,
    *,
    hard_commit: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    from ml_diag.labels import HEALTHY as _HEALTHY

    tuned = replace(cascade, stage1_healthy_threshold=float(threshold), threshold_source="explicit")
    diags = diagnose_batch(tuned, X)
    classes = list(PRIMARY_LABELS)
    proba = np.zeros((len(diags), len(classes)), dtype=float)
    finals = np.empty(len(diags), dtype=object)
    for i, d in enumerate(diags):
        cls_probs = dict(d.class_probabilities)
        if hard_commit and d.stage1.predicted != _HEALTHY:
            cls_probs[_HEALTHY] = 0.0
            s = sum(max(0.0, v) for v in cls_probs.values())
            if s > 0:
                cls_probs = {k: max(0.0, v) / s for k, v in cls_probs.items()}
            final_cls = max(cls_probs.items(), key=lambda kv: kv[1])[0]
        else:
            final_cls = d.final_class
        finals[i] = final_cls
        for j, cls in enumerate(classes):
            proba[i, j] = float(cls_probs.get(cls, 0.0))
    return finals, proba


def _evaluate_one(
    *,
    threshold: float,
    cascade,
    X: pd.DataFrame,
    y: pd.Series,
    weights: dict[str, float],
    hard_commit: bool = True,
) -> dict[str, Any]:
    finals, proba = _hier_predict_with_threshold(
        cascade,
        X,
        threshold,
        hard_commit=hard_commit,
    )
    rep = classification_report(
        y,
        finals,
        y_proba=proba,
        proba_classes=list(PRIMARY_LABELS),
        label_order=list(PRIMARY_LABELS),
    )
    cost = total_cost(y.to_numpy(dtype=object), finals, weights=weights)
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "accuracy": float(rep.accuracy),
        "macro_f1": float(rep.macro_f1),
        "weighted_f1": float(rep.weighted_f1),
        "ece": (None if rep.ece is None else float(rep.ece)),
        **cost,
    }


def _build_oof_p_healthy(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    from sklearn.model_selection import StratifiedKFold

    from ml_diag.labels import STAGE1_LABELS
    from ml_diag.models.inference import (
        _proba_or_onehot,
        _row_for_stage,
        _StageModel,
    )
    from ml_diag.models.stage1 import prepare as _stage1_prepare
    from ml_diag.models.trainer import train_stage

    inner_skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_stage1 = y_train.map(to_stage1)
    p_healthy = np.full(len(X_train), np.nan, dtype=float)
    for fold_idx, (tr, va) in enumerate(inner_skf.split(X_train, y_stage1)):
        X_tr_inner = X_train.iloc[tr]
        y_tr_inner = y_train.iloc[tr]
        X_va_inner = X_train.iloc[va]
        X_s, y_s = _stage1_prepare(X_tr_inner, y_tr_inner)
        result = train_stage(
            stage_name=f"stage1_oof_fold{fold_idx}",
            X=X_s,
            y=y_s,
            label_vocab=STAGE1_LABELS,
            seed=seed,
            calibrate=False,
        )
        stage_model = _StageModel(
            name=result.stage_name,
            model=result.model,
            classes=result.classes,
            feature_columns=result.feature_columns,
        )
        for j, (_, row) in enumerate(X_va_inner.iterrows()):
            arr = _row_for_stage(row, stage_model)
            proba = _proba_or_onehot(stage_model.model, arr, stage_model.classes or [])
            p_healthy[va[j]] = float(proba.get(HEALTHY, 0.0))
        print(f"  [oof fold {fold_idx + 1}/{n_folds}] {len(va)} held-out rows")
    return p_healthy


def _evaluate_one_oof(
    *,
    threshold: float,
    p_healthy_oof: np.ndarray,
    y: pd.Series,
    weights: dict[str, float],
) -> dict[str, Any]:
    pred_binary = np.where(p_healthy_oof >= threshold, HEALTHY, FAULTY).astype(object)
    y_arr = y.to_numpy(dtype=object)
    y_binary = np.where(y_arr == HEALTHY, HEALTHY, FAULTY).astype(object)
    rep = classification_report(y_binary, pred_binary, label_order=(HEALTHY, FAULTY))
    cost = total_cost(y_arr, pred_binary, weights=weights)
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "accuracy": float(rep.accuracy),
        "macro_f1": float(rep.macro_f1),
        "weighted_f1": float(rep.weighted_f1),
        "ece": None,
        **cost,
    }


def _sweep_on_split(
    *,
    cascade,
    X: pd.DataFrame,
    y: pd.Series,
    slice_indices: dict[str, pd.Index],
    grid: list[float],
    weights: dict[str, float],
    hard_commit: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for slice_name, idx in slice_indices.items():
        if len(idx) == 0:
            continue
        Xs = X.loc[idx]
        ys = y.loc[idx]
        rows = []
        for thr in grid:
            rows.append(
                _evaluate_one(
                    threshold=thr,
                    cascade=cascade,
                    X=Xs,
                    y=ys,
                    weights=weights,
                    hard_commit=hard_commit,
                )
            )
        out[slice_name] = rows
    return out


def _sweep_oof_train(
    *,
    p_healthy_oof: np.ndarray,
    y_train: pd.Series,
    slice_indices: dict[str, pd.Index],
    grid: list[float],
    weights: dict[str, float],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    train_index = list(y_train.index)
    pos = {rid: i for i, rid in enumerate(train_index)}
    for slice_name, idx in slice_indices.items():
        if len(idx) == 0:
            continue
        positions = [pos[rid] for rid in idx if rid in pos]
        if not positions:
            continue
        sub_p = p_healthy_oof[positions]
        sub_y = y_train.loc[[train_index[i] for i in positions]]
        rows = [
            _evaluate_one_oof(
                threshold=thr,
                p_healthy_oof=sub_p,
                y=sub_y,
                weights=weights,
            )
            for thr in grid
        ]
        out[slice_name] = rows
    return out


def _pick_best_threshold(
    rows: list[dict[str, Any]],
    *,
    by: str = "total_cost",
    higher_is_better: bool = False,
) -> dict[str, Any]:
    if higher_is_better:
        return max(rows, key=lambda r: r[by])
    return min(rows, key=lambda r: r[by])


def _fmt_n(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def _render_sweep_table(rows: list[dict[str, Any]]) -> list[str]:
    out = []
    out.append(
        "| thr | acc | macro-F1 | ECE | leak→hlth | label_noise→hlth | "
        "other→hlth | hlth→faulty | total cost |"
    )
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        out.append(
            f"| {r['threshold']:.2f} | {_fmt_n(r['accuracy'])} | "
            f"{_fmt_n(r['macro_f1'])} | {_fmt_n(r['ece'])} | "
            f"{r['leakage_to_healthy']} | {r['label_noise_to_healthy']} | "
            f"{r['other_faulty_to_healthy']} | {r['healthy_to_faulty']} | "
            f"{r['total_cost']:.2f} |"
        )
    return out


def render_markdown(
    *,
    corpus_name: str,
    cascade_threshold_default: float,
    weights: dict[str, float],
    train_sweep: dict[str, list[dict[str, Any]]],
    test_sweep: dict[str, list[dict[str, Any]]],
    selected_threshold: float,
    selection_basis: str,
    test_at_selected: dict[str, dict[str, Any]],
    test_at_default: dict[str, dict[str, Any]],
    hard_commit: bool = True,
    oof_train: bool = True,
) -> str:
    out: list[str] = []
    out.append(f"# Stage 1 threshold sweep — {corpus_name}")
    out.append("")
    out.append(f"- threshold grid: `{THRESHOLD_GRID}`")
    out.append(f"- cascade default threshold: `{cascade_threshold_default:.2f}`")
    out.append(
        f"- routing semantics: **{'hard-commit' if hard_commit else 'soft (current production)'}**"
    )
    if hard_commit:
        out.append(
            "  Below-threshold rows have ``P(healthy)`` zeroed in the composed "
            "marginal so the final argmax falls inside the faulty cone. This "
            "is what actually trades healthy recall for fewer leakage→healthy "
            "false negatives."
        )
    else:
        out.append(
            "  Threshold only changes which sub-stages execute; the composed "
            "argmax keeps raw ``P(healthy)``. Confident-but-wrong Stage 1 "
            "predictions therefore stay invariant under the sweep."
        )
    out.append("- cost weights: " + ", ".join(f"`{k}={v}`" for k, v in weights.items()))
    out.append(
        "- selection basis: "
        f"**train fold cost minimum on slice `{selection_basis}`** "
        "(in-sample on the saved cascade — used as a validation surrogate; "
        "see notes)"
    )
    out.append(f"- selected threshold: **{selected_threshold:.2f}**")
    out.append("")
    out.append(
        "Cost is the weighted sum of Stage-1 boundary errors only "
        "(misses across the healthy/faulty axis). Errors that confuse one "
        "faulty leaf with another are not threshold-controllable and cost 0."
    )
    out.append("")
    out.append("## Test fold — sweep across slices")
    out.append("")
    for slice_name, rows in test_sweep.items():
        out.append(f"### `{slice_name}` (test, n={rows[0]['n']})")
        out.append("")
        out.extend(_render_sweep_table(rows))
        out.append("")
    if oof_train:
        out.append("## Train fold — sweep on OOF P(healthy) (selection basis)")
        out.append("")
        out.append(
            "_Each train row's ``P(healthy)`` was predicted by an inner-CV "
            "Stage 1 model that did not see the row during training. Cost "
            "depends only on the healthy/faulty routing decision under "
            "hard-commit, so OOF P(healthy) alone is sufficient for "
            "selection — no cascade retrain needed._"
        )
        out.append("")
    else:
        out.append("## Train fold — in-sample sweep (selection surrogate)")
        out.append("")
        out.append(
            "_The saved cascade was fit on the full train fold; these "
            "numbers are optimistic. Pass `--use-oof-train` for an honest "
            "OOF selection basis._"
        )
        out.append("")
    for slice_name, rows in train_sweep.items():
        out.append(f"### `{slice_name}` (train, n={rows[0]['n']})")
        out.append("")
        out.extend(_render_sweep_table(rows))
        out.append("")
    out.append("## Default vs selected (test fold)")
    out.append("")
    out.append(
        f"What changes on test when we move from the default threshold "
        f"({cascade_threshold_default:.2f}) to the cost-selected threshold "
        f"({selected_threshold:.2f})."
    )
    out.append("")
    out.append(
        "| slice | n | acc default → selected | macro-F1 | ECE | "
        "leak→hlth | hlth→faulty | total cost |"
    )
    out.append("|---|---:|---|---:|---|---:|---:|---|")
    for slice_name in test_at_default:
        d = test_at_default[slice_name]
        s = test_at_selected[slice_name]
        out.append(
            f"| {slice_name} | {d['n']} "
            f"| {d['accuracy']:.4f} → {s['accuracy']:.4f} "
            f"| {d['macro_f1']:.4f} → {s['macro_f1']:.4f} "
            f"| {_fmt_n(d['ece'])} → {_fmt_n(s['ece'])} "
            f"| {d['leakage_to_healthy']} → {s['leakage_to_healthy']} "
            f"| {d['healthy_to_faulty']} → {s['healthy_to_faulty']} "
            f"| {d['total_cost']:.2f} → {s['total_cost']:.2f} |"
        )
    out.append("")
    return "\n".join(out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--hier-artifacts", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--selection-slice",
        default="full",
        choices=["full", "core", "extended"],
        help="Which slice of the train fold to use for cost-based selection (default: full).",
    )
    p.add_argument("--cost-leak-to-healthy", type=float, default=3.0)
    p.add_argument("--cost-label-noise-to-healthy", type=float, default=2.0)
    p.add_argument("--cost-other-faulty-to-healthy", type=float, default=1.5)
    p.add_argument("--cost-healthy-to-faulty", type=float, default=1.0)
    p.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of inner CV folds for out-of-fold P(healthy) on the train fold (default: 5).",
    )
    p.add_argument(
        "--no-oof",
        action="store_true",
        help="Skip OOF computation; use the saved cascade's "
        "in-sample predictions on the train fold for "
        "selection (will memorize, gives 0 cost). Useful "
        "only as a debugging baseline.",
    )
    p.add_argument(
        "--soft-commit",
        action="store_true",
        help="Use the current cascade's soft-routing semantics "
        "(threshold only changes which sub-stages run; the "
        "composed argmax can still pick HEALTHY). Default is "
        "hard-commit, where below-threshold P(healthy) is "
        "zeroed in the composed marginal so the threshold "
        "actually trades healthy recall for leakage capture.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    weights = {
        "leakage_to_healthy": float(args.cost_leak_to_healthy),
        "label_noise_to_healthy": float(args.cost_label_noise_to_healthy),
        "other_faulty_to_healthy": float(args.cost_other_faulty_to_healthy),
        "healthy_to_faulty": float(args.cost_healthy_to_faulty),
    }
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    train_run_ids = X.index[train_idx]
    test_run_ids = X.index[test_idx]
    X_tr, y_tr = X.loc[train_run_ids], y.loc[train_run_ids]
    X_te, y_te = X.loc[test_run_ids], y.loc[test_run_ids]
    train_slices = slices_from_partition(pt, X.index, holdout_index=train_run_ids)
    test_slices = slices_from_partition(pt, X.index, holdout_index=test_run_ids)
    cascade = load_cascade(args.hier_artifacts)
    print(
        f"Loaded cascade: stages={cascade.stages_available}, "
        f"default threshold={cascade.stage1_healthy_threshold:.3f}"
    )
    hard_commit = not args.soft_commit
    use_oof = not args.no_oof
    p_healthy_oof = None
    if use_oof:
        print(f"Building OOF P(healthy) on train fold (n={len(X_tr)}, k={args.cv_folds})…")
        p_healthy_oof = _build_oof_p_healthy(
            X_train=X_tr,
            y_train=y_tr,
            n_folds=args.cv_folds,
            seed=args.seed,
        )
    print(
        f"Sweeping {len(THRESHOLD_GRID)} thresholds on train fold "
        f"(n={len(X_tr)}) and test fold (n={len(X_te)}); "
        f"hard_commit={hard_commit}, oof={use_oof}…"
    )
    if use_oof and p_healthy_oof is not None:
        train_sweep = _sweep_oof_train(
            p_healthy_oof=p_healthy_oof,
            y_train=y_tr,
            slice_indices=train_slices,
            grid=THRESHOLD_GRID,
            weights=weights,
        )
    else:
        train_sweep = _sweep_on_split(
            cascade=cascade,
            X=X_tr,
            y=y_tr,
            slice_indices=train_slices,
            grid=THRESHOLD_GRID,
            weights=weights,
            hard_commit=hard_commit,
        )
    test_sweep = _sweep_on_split(
        cascade=cascade,
        X=X_te,
        y=y_te,
        slice_indices=test_slices,
        grid=THRESHOLD_GRID,
        weights=weights,
        hard_commit=hard_commit,
    )
    sel_slice = args.selection_slice
    if sel_slice not in train_sweep:
        print(
            f"WARNING: selection slice '{sel_slice}' has no rows on train; falling back to 'full'."
        )
        sel_slice = "full"
    best = _pick_best_threshold(train_sweep[sel_slice], by="total_cost")
    selected_threshold = float(best["threshold"])
    print(
        f"Selected threshold {selected_threshold:.2f} (train {sel_slice} "
        f"cost = {best['total_cost']:.2f})."
    )
    default_threshold = float(cascade.stage1_healthy_threshold)
    test_at_default: dict[str, dict[str, Any]] = {}
    test_at_selected: dict[str, dict[str, Any]] = {}
    for slice_name, rows in test_sweep.items():
        closest_def = min(rows, key=lambda r: abs(r["threshold"] - default_threshold))
        closest_sel = min(rows, key=lambda r: abs(r["threshold"] - selected_threshold))
        test_at_default[slice_name] = closest_def
        test_at_selected[slice_name] = closest_sel
    md = render_markdown(
        corpus_name=ftable.corpus_name,
        cascade_threshold_default=default_threshold,
        weights=weights,
        train_sweep=train_sweep,
        test_sweep=test_sweep,
        selected_threshold=selected_threshold,
        selection_basis=sel_slice,
        test_at_selected=test_at_selected,
        test_at_default=test_at_default,
        hard_commit=hard_commit,
        oof_train=use_oof,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        payload = {
            "corpus": ftable.corpus_name,
            "cascade_default_threshold": default_threshold,
            "selected_threshold": selected_threshold,
            "selection_slice": sel_slice,
            "hard_commit": hard_commit,
            "oof_train": use_oof,
            "weights": weights,
            "threshold_grid": THRESHOLD_GRID,
            "train_sweep": train_sweep,
            "test_sweep": test_sweep,
            "test_at_default": test_at_default,
            "test_at_selected": test_at_selected,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("Test sweep — slice `full` (cost / leak→hlth / hlth→faulty):")
    for r in test_sweep.get("full", []):
        print(
            f"  thr={r['threshold']:.2f}  cost={r['total_cost']:6.2f}  "
            f"leak→hlth={r['leakage_to_healthy']:2d}  "
            f"hlth→faulty={r['healthy_to_faulty']:2d}  "
            f"acc={r['accuracy']:.4f}  macro_f1={r['macro_f1']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
