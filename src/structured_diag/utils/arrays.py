from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def ensure_label_array(y: Any) -> list[str]:
    if y is None:
        return []
    if isinstance(y, (pd.Series, pd.Index)):
        return [str(v) for v in y.tolist()]
    if isinstance(y, np.ndarray):
        flat = y.ravel().tolist()
        return [str(v) for v in flat]
    try:
        from pandas.api.extensions import ExtensionArray  # type: ignore[import-not-found]

        if isinstance(y, ExtensionArray):
            return [str(v) for v in list(y)]
    except Exception:
        pass
    if isinstance(y, Iterable) and not isinstance(y, (str, bytes)):
        return [str(v) for v in y]
    return [str(y)]


def ensure_feature_matrix(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float, copy=True)
    else:
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
    if not arr.size:
        return arr
    finite = np.isfinite(arr)
    if not finite.all():
        for j in range(arr.shape[1]):
            col_finite = finite[:, j]
            if col_finite.all():
                continue
            if col_finite.any():
                med = float(np.median(arr[col_finite, j]))
            else:
                med = 0.0
            arr[~col_finite, j] = med
    return arr


def align_features_to_schema(
    X: Any,
    target_columns: Sequence[str],
    *,
    fill_value: float = float("nan"),
) -> pd.DataFrame:
    target = list(target_columns)
    if isinstance(X, pd.Series):
        df = X.to_frame().T
    elif isinstance(X, pd.DataFrame):
        df = X.copy()
    elif isinstance(X, np.ndarray):
        arr = np.atleast_2d(X)
        if arr.shape[1] != len(target):
            raise ValueError(
                f"ndarray feature matrix has {arr.shape[1]} columns, "
                f"expected {len(target)}; pass a Series/DataFrame to "
                "align by name instead."
            )
        return pd.DataFrame(arr, columns=target)
    else:
        raise TypeError(f"unsupported feature container: {type(X).__name__}")
    missing = [c for c in target if c not in df.columns]
    extra = [c for c in df.columns if c not in target]
    if missing:
        for c in missing:
            df[c] = fill_value
    df = df[target]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


__all__ = [
    "align_features_to_schema",
    "ensure_feature_matrix",
    "ensure_label_array",
]
