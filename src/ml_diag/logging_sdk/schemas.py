from __future__ import annotations

DEFAULT_META_KEYS: tuple[str, ...] = (
    "run_id",
    "dataset_name",
    "task_type",
    "model_name",
    "framework",
    "optimizer",
    "learning_rate",
    "batch_size",
    "epochs_planned",
    "seed",
    "status",
    "created_at",
    "finalized_at",
    "duration_sec",
    "n_epochs_logged",
    "final_metrics",
    "tags",
    "notes",
    "framework_version",
    "python_version",
)

REQUIRED_META_KEYS: tuple[str, ...] = ("run_id",)

ALLOWED_STATUSES: tuple[str, ...] = (
    "running",
    "completed",
    "failed",
    "crashed",
    "interrupted",
)

MINIMUM_HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
)

OPTIONAL_HISTORY_COLUMNS: tuple[str, ...] = (
    "grad_norm",
    "weight_norm",
    "lr",
    "step_time_sec",
    "loss_spike_flag",
)

DEFAULT_HISTORY_COLUMNS: tuple[str, ...] = (
    *MINIMUM_HISTORY_COLUMNS,
    *OPTIONAL_HISTORY_COLUMNS,
)

__all__ = [
    "ALLOWED_STATUSES",
    "DEFAULT_HISTORY_COLUMNS",
    "DEFAULT_META_KEYS",
    "MINIMUM_HISTORY_COLUMNS",
    "OPTIONAL_HISTORY_COLUMNS",
    "REQUIRED_META_KEYS",
]
