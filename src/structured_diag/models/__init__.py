from structured_diag.models import stage1, stage2, stage3  # noqa: F401  (re-exports)
from structured_diag.models.flat_baseline import (
    FlatBaselineResult,
    evaluate_on_slices,
    slices_from_partition,
    train_flat_baseline,
)
from structured_diag.models.inference import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
    StagePrediction,
    diagnose_batch,
    diagnose_one,
    diagnoses_to_dataframe,
    diagnoses_to_jsonl,
    load_cascade,
)
from structured_diag.models.model_zoo import ModelSpec, default_zoo
from structured_diag.models.trainer import (
    StageTrainResult,
    save_stage_artifacts,
    train_stage,
)

__all__ = [
    "FlatBaselineResult",
    "HierarchicalCascade",
    "HierarchicalDiagnosis",
    "ModelSpec",
    "StagePrediction",
    "StageTrainResult",
    "default_zoo",
    "diagnose_batch",
    "diagnose_one",
    "diagnoses_to_dataframe",
    "diagnoses_to_jsonl",
    "evaluate_on_slices",
    "load_cascade",
    "save_stage_artifacts",
    "slices_from_partition",
    "stage1",
    "stage2",
    "stage3",
    "train_flat_baseline",
    "train_stage",
]
