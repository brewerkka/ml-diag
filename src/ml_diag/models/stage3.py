from __future__ import annotations

import pandas as pd

from ml_diag.labels import (
    DATA_RELATED,
    OPT_GEN_RELATED,
    STAGE3_LABELS_BY_BRANCH,
    to_stage2,
    to_stage3,
)
from ml_diag.models.trainer import StageTrainResult, train_stage

STAGE_NAME_DATA = "stage3_data_related"

STAGE_NAME_OPT_GEN = "stage3_optimization_or_generalization_related"


def _prepare_for_branch(
    X: pd.DataFrame, y_primary: pd.Series, branch: str
) -> tuple[pd.DataFrame, pd.Series]:
    mask = y_primary.notna() & (y_primary.map(to_stage2) == branch)
    X_s = X.loc[mask]
    y_s = y_primary.loc[mask].map(to_stage3)
    return X_s, y_s


def prepare_data_related(X: pd.DataFrame, y_primary: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    return _prepare_for_branch(X, y_primary, DATA_RELATED)


def prepare_opt_gen(X: pd.DataFrame, y_primary: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    return _prepare_for_branch(X, y_primary, OPT_GEN_RELATED)


def train_data_related(
    X: pd.DataFrame,
    y_primary: pd.Series,
    *,
    partition_table: pd.DataFrame | None = None,
    seed: int = 0,
    holdout_run_ids=None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> StageTrainResult:
    X_s, y_s = prepare_data_related(X, y_primary)
    return train_stage(
        stage_name=STAGE_NAME_DATA,
        X=X_s,
        y=y_s,
        label_vocab=STAGE3_LABELS_BY_BRANCH[DATA_RELATED],
        partition_table=partition_table,
        seed=seed,
        holdout_run_ids=holdout_run_ids,
        calibrate=calibrate,
        calibration_method=calibration_method,
    )


def train_opt_gen(
    X: pd.DataFrame,
    y_primary: pd.Series,
    *,
    partition_table: pd.DataFrame | None = None,
    seed: int = 0,
    holdout_run_ids=None,
    calibrate: bool = True,
    calibration_method: str = "isotonic",
) -> StageTrainResult:
    X_s, y_s = prepare_opt_gen(X, y_primary)
    return train_stage(
        stage_name=STAGE_NAME_OPT_GEN,
        X=X_s,
        y=y_s,
        label_vocab=STAGE3_LABELS_BY_BRANCH[OPT_GEN_RELATED],
        partition_table=partition_table,
        seed=seed,
        holdout_run_ids=holdout_run_ids,
        calibrate=calibrate,
        calibration_method=calibration_method,
    )


__all__ = [
    "STAGE_NAME_DATA",
    "STAGE_NAME_OPT_GEN",
    "prepare_data_related",
    "prepare_opt_gen",
    "train_data_related",
    "train_opt_gen",
]
