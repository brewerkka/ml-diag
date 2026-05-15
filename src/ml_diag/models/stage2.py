from __future__ import annotations

import pandas as pd

from ml_diag.labels import HEALTHY, STAGE2_LABELS, to_stage1, to_stage2
from ml_diag.models.trainer import StageTrainResult, train_stage

STAGE_NAME = "stage2_faulty_data_vs_optgen"


def prepare(X: pd.DataFrame, y_primary: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    mask = y_primary.notna() & (y_primary.map(to_stage1) != HEALTHY)
    X_s = X.loc[mask]
    y_s = y_primary.loc[mask].map(to_stage2)
    return X_s, y_s


def train(
    X: pd.DataFrame,
    y_primary: pd.Series,
    *,
    partition_table: pd.DataFrame | None = None,
    seed: int = 0,
    holdout_run_ids=None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> StageTrainResult:
    X_s, y_s = prepare(X, y_primary)
    return train_stage(
        stage_name=STAGE_NAME,
        X=X_s,
        y=y_s,
        label_vocab=STAGE2_LABELS,
        partition_table=partition_table,
        seed=seed,
        holdout_run_ids=holdout_run_ids,
        calibrate=calibrate,
        calibration_method=calibration_method,
    )


__all__ = ["STAGE_NAME", "prepare", "train"]
