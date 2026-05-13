from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassificationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_support: dict[str, int]
    confusion_matrix: list[list[int]]
    confusion_labels: list[str]
    ece: float | None = None
    n_samples: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _as_array(x: Sequence[Any]) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy()
    return np.asarray(x)


def expected_calibration_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None,
    classes: Sequence[Any] | None = None,
    n_bins: int = 10,
) -> float | None:
    return _calibration_top1_error(y_true, y_pred, y_proba, n_bins=n_bins, mode="ece")


def maximum_calibration_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None,
    classes: Sequence[Any] | None = None,
    n_bins: int = 10,
) -> float | None:
    return _calibration_top1_error(y_true, y_pred, y_proba, n_bins=n_bins, mode="mce")


def _calibration_top1_error(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None,
    *,
    n_bins: int = 10,
    mode: str = "ece",
) -> float | None:
    if y_proba is None:
        return None
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(y_true_arr):
        return None
    confidences = proba.max(axis=1)
    correct = (y_pred_arr == y_true_arr).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confidences)
    if mode == "ece":
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            if not mask.any():
                continue
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correct[mask].mean())
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
        return float(ece)
    elif mode == "mce":
        gaps: list[float] = []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i == n_bins - 1:
                mask = (confidences >= lo) & (confidences <= hi)
            else:
                mask = (confidences >= lo) & (confidences < hi)
            if not mask.any():
                continue
            avg_conf = float(confidences[mask].mean())
            avg_acc = float(correct[mask].mean())
            gaps.append(abs(avg_conf - avg_acc))
        return float(max(gaps)) if gaps else 0.0
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def brier_score_multiclass(
    y_true: Sequence[Any],
    y_proba: np.ndarray | None,
    classes: Sequence[Any] | None = None,
) -> float | None:
    if y_proba is None:
        return None
    y_true_arr = _as_array(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(y_true_arr):
        return None
    if classes is None:
        classes = sorted(set(y_true_arr.tolist()))
    classes_list = list(classes)
    if proba.shape[1] != len(classes_list):
        return None
    cls_idx = {str(c): i for i, c in enumerate(classes_list)}
    n = len(y_true_arr)
    if n == 0:
        return 0.0
    y_onehot = np.zeros_like(proba)
    for i, y in enumerate(y_true_arr):
        col = cls_idx.get(str(y))
        if col is not None:
            y_onehot[i, col] = 1.0
    diff = proba - y_onehot
    return float(np.mean(np.sum(diff**2, axis=1)))


def reliability_diagram_bins(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: np.ndarray | None,
    *,
    n_bins: int = 10,
) -> dict[str, list[float]] | None:
    if y_proba is None:
        return None
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != len(y_true_arr):
        return None
    confidences = proba.max(axis=1)
    correct = (y_pred_arr == y_true_arr).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, conf_mean, acc_mean, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        centers.append(float((lo + hi) / 2))
        if mask.any():
            conf_mean.append(float(confidences[mask].mean()))
            acc_mean.append(float(correct[mask].mean()))
        else:
            conf_mean.append(float((lo + hi) / 2))
            acc_mean.append(0.0)
        counts.append(int(mask.sum()))
    return {
        "bin_centers": centers,
        "bin_confidence": conf_mean,
        "bin_accuracy": acc_mean,
        "bin_count": counts,
    }


def classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    y_proba: np.ndarray | None = None,
    proba_classes: Sequence[Any] | None = None,
    label_order: Sequence[Any] | None = None,
) -> ClassificationReport:
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    if label_order is None:
        labels = sorted(set(np.concatenate([y_true_arr, y_pred_arr]).tolist()))
    else:
        labels = list(label_order)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    macro_f1 = float(
        f1_score(y_true_arr, y_pred_arr, average="macro", labels=labels, zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true_arr, y_pred_arr, average="weighted", labels=labels, zero_division=0)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels).tolist()
    ece = expected_calibration_error(y_true_arr, y_pred_arr, y_proba, classes=proba_classes)
    return ClassificationReport(
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        per_class_f1={lbl: float(v) for lbl, v in zip(labels, f1)},
        per_class_precision={lbl: float(v) for lbl, v in zip(labels, precision)},
        per_class_recall={lbl: float(v) for lbl, v in zip(labels, recall)},
        per_class_support={lbl: int(v) for lbl, v in zip(labels, support)},
        confusion_matrix=cm,
        confusion_labels=[str(l) for l in labels],
        ece=ece,
        n_samples=int(len(y_true_arr)),
    )


def bootstrap_metric_ci(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    metric: str = "macro_f1",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    label_order: Sequence[Any] | None = None,
) -> dict[str, float]:
    y_true_arr = _as_array(y_true)
    y_pred_arr = _as_array(y_pred)
    n = len(y_true_arr)
    if n == 0:
        return {
            "point_estimate": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "alpha": alpha,
            "n_bootstrap": 0,
        }
    if metric == "macro_f1":

        def _score(yt, yp):
            return float(f1_score(yt, yp, average="macro", labels=label_order, zero_division=0))
    elif metric == "weighted_f1":

        def _score(yt, yp):
            return float(f1_score(yt, yp, average="weighted", labels=label_order, zero_division=0))
    elif metric == "accuracy":

        def _score(yt, yp):
            return float(accuracy_score(yt, yp))
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")
    rng = np.random.default_rng(seed)
    point = _score(y_true_arr, y_pred_arr)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        estimates[i] = _score(y_true_arr[idx], y_pred_arr[idx])
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point_estimate": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_delta_ci(
    y_true: Sequence[Any],
    y_pred_a: Sequence[Any],
    y_pred_b: Sequence[Any],
    *,
    metric: str = "macro_f1",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    label_order: Sequence[Any] | None = None,
    n_comparisons: int = 1,
) -> dict[str, float]:
    y_true_arr = _as_array(y_true)
    y_a = _as_array(y_pred_a)
    y_b = _as_array(y_pred_b)
    n = len(y_true_arr)
    if not (len(y_a) == n and len(y_b) == n):
        raise ValueError("y_true / y_pred_a / y_pred_b must have equal length")
    if n == 0:
        return {
            "delta_point": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "alpha": alpha,
            "n_bootstrap": 0,
            "p_b_better": float("nan"),
        }
    if metric == "macro_f1":

        def _score(yt, yp):
            return float(f1_score(yt, yp, average="macro", labels=label_order, zero_division=0))
    elif metric == "accuracy":

        def _score(yt, yp):
            return float(accuracy_score(yt, yp))
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")
    rng = np.random.default_rng(seed)
    point = _score(y_true_arr, y_b) - _score(y_true_arr, y_a)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        estimates[i] = _score(y_true_arr[idx], y_b[idx]) - _score(y_true_arr[idx], y_a[idx])
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_b_better = float((estimates > 0).mean())
    out: dict[str, float] = {
        "delta_point": point,
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "p_b_better": p_b_better,
    }
    if n_comparisons > 1:
        alpha_bf = float(alpha) / float(n_comparisons)
        out["alpha_bonferroni"] = alpha_bf
        out["n_comparisons"] = int(n_comparisons)
        out["p_b_better_bonferroni_significant"] = bool(p_b_better >= (1.0 - alpha_bf))
    if metric == "accuracy":
        p_a = float(accuracy_score(y_true_arr, y_a))
        p_b = float(accuracy_score(y_true_arr, y_b))
        h_point = _cohen_h(p_a, p_b)
        h_bs = np.empty(n_bootstrap, dtype=float)
        rng2 = np.random.default_rng(seed + 1)
        for i in range(n_bootstrap):
            idx = rng2.integers(0, n, n)
            p_a_b = float(accuracy_score(y_true_arr[idx], y_a[idx]))
            p_b_b = float(accuracy_score(y_true_arr[idx], y_b[idx]))
            h_bs[i] = _cohen_h(p_a_b, p_b_b)
        h_lo, h_hi = np.percentile(h_bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["cohen_h"] = float(h_point)
        out["cohen_h_ci_low"] = float(h_lo)
        out["cohen_h_ci_high"] = float(h_hi)
        out["cohen_h_magnitude"] = _cohen_h_magnitude(h_point)
    return out


def _cohen_h(p_a: float, p_b: float) -> float:
    pa = float(np.clip(p_a, 0.0, 1.0))
    pb = float(np.clip(p_b, 0.0, 1.0))
    return float(2.0 * (np.arcsin(np.sqrt(pb)) - np.arcsin(np.sqrt(pa))))


def _cohen_h_magnitude(h: float) -> str:
    a = abs(float(h))
    if a < 0.2:
        return "trivial"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def holm_bonferroni_adjust(
    p_better_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> list[dict[str, float | bool | int]]:
    m = len(p_better_values)
    if m == 0:
        return []
    pseudo_p = [1.0 - float(p) for p in p_better_values]
    order = sorted(range(m), key=lambda i: pseudo_p[i])
    accepted: dict[int, bool] = {}
    still_rejecting = True
    for k, idx in enumerate(order, start=1):
        threshold = alpha / (m - k + 1)
        if still_rejecting and pseudo_p[idx] <= threshold:
            accepted[idx] = True
        else:
            still_rejecting = False
            accepted[idx] = False
    rank_by_orig: dict[int, int] = {idx: k for k, idx in enumerate(order, start=1)}
    out: list[dict[str, float | bool | int]] = []
    for i, p_b in enumerate(p_better_values):
        rank = rank_by_orig[i]
        threshold = alpha / (m - rank + 1)
        out.append(
            {
                "p_b_better": float(p_b),
                "rank": int(rank),
                "holm_threshold_pseudo_p": float(threshold),
                "holm_threshold_p_better": float(1.0 - threshold),
                "significant": bool(accepted[i]),
                "alpha": float(alpha),
                "family_size": int(m),
            }
        )
    return out


def bootstrap_delta_ci_grouped(
    y_true: Sequence[Any],
    y_pred_a: Sequence[Any],
    y_pred_b: Sequence[Any],
    group_ids: Sequence[Any],
    *,
    metric: str = "macro_f1",
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    label_order: Sequence[Any] | None = None,
    n_comparisons: int = 1,
) -> dict[str, float]:
    y_true_arr = _as_array(y_true)
    y_a = _as_array(y_pred_a)
    y_b = _as_array(y_pred_b)
    g = np.asarray(list(group_ids))
    n = len(y_true_arr)
    if not (len(y_a) == n and len(y_b) == n and len(g) == n):
        raise ValueError("y_true / y_pred_a / y_pred_b / group_ids must have equal length")
    if n == 0:
        return {
            "delta_point": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "alpha": alpha,
            "n_bootstrap": 0,
            "p_b_better": float("nan"),
            "n_groups": 0,
        }
    if metric == "macro_f1":

        def _score(yt, yp):
            return float(f1_score(yt, yp, average="macro", labels=label_order, zero_division=0))
    elif metric == "accuracy":

        def _score(yt, yp):
            return float(accuracy_score(yt, yp))
    else:
        raise ValueError(f"Unsupported metric: {metric!r}")
    unique_groups, _ = np.unique(g, return_inverse=False), None
    unique_groups = list(unique_groups)
    rows_by_group: dict[Any, np.ndarray] = {gid: np.where(g == gid)[0] for gid in unique_groups}
    n_groups = len(unique_groups)
    rng = np.random.default_rng(seed)
    point = _score(y_true_arr, y_b) - _score(y_true_arr, y_a)
    estimates = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled_groups = rng.choice(n_groups, size=n_groups, replace=True)
        idx_parts = [rows_by_group[unique_groups[k]] for k in sampled_groups]
        idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=int)
        estimates[i] = _score(y_true_arr[idx], y_b[idx]) - _score(y_true_arr[idx], y_a[idx])
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_b_better = float((estimates > 0).mean())
    out: dict[str, float] = {
        "delta_point": point,
        "delta_ci_low": float(lo),
        "delta_ci_high": float(hi),
        "alpha": alpha,
        "n_bootstrap": n_bootstrap,
        "p_b_better": p_b_better,
        "n_groups": int(n_groups),
    }
    if n_comparisons > 1:
        alpha_bf = float(alpha) / float(n_comparisons)
        out["alpha_bonferroni"] = alpha_bf
        out["n_comparisons"] = int(n_comparisons)
        out["p_b_better_bonferroni_significant"] = bool(p_b_better >= (1.0 - alpha_bf))
    return out


__all__ = [
    "ClassificationReport",
    "bootstrap_delta_ci",
    "bootstrap_delta_ci_grouped",
    "bootstrap_metric_ci",
    "classification_report",
    "expected_calibration_error",
]
