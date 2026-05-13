from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path

_SRC_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "src",
    Path("/Users/brewerka/Desktop/structured_diag/src"),
)

for _c in _SRC_CANDIDATES:
    if _c.is_dir() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import pandas as pd
from mlflow.tracking import MlflowClient

from structured_diag.logging_sdk import RunLogger

_DEFAULT_METRIC_PATTERNS: dict[str, tuple[str, ...]] = {
    "train_loss": (
        "train_loss",
        "training_loss",
        "loss",
        "Loss/train",
        "loss/train",
        "train/loss",
        "training/loss",
    ),
    "val_loss": (
        "val_loss",
        "validation_loss",
        "valid_loss",
        "Loss/val",
        "loss/val",
        "val/loss",
        "validation/loss",
    ),
    "train_acc": (
        "train_acc",
        "train_accuracy",
        "training_accuracy",
        "accuracy",
        "Accuracy/train",
        "accuracy/train",
        "train/accuracy",
        "training/accuracy",
    ),
    "val_acc": (
        "val_acc",
        "val_accuracy",
        "validation_accuracy",
        "valid_accuracy",
        "Accuracy/val",
        "accuracy/val",
        "val/accuracy",
        "validation/accuracy",
    ),
    "lr": (
        "lr",
        "learning_rate",
        "LearningRate",
        "optim/lr",
    ),
    "grad_norm": (
        "grad_norm",
        "GradNorm",
        "gradient_norm",
    ),
    "weight_norm": (
        "weight_norm",
        "WeightNorm",
    ),
}

_PARAM_LIFTS: dict[str, tuple[str, ...]] = {
    "dataset_name": ("dataset_name", "dataset", "data"),
    "model_name": ("model_name", "model", "architecture"),
    "framework": ("framework", "library"),
    "optimizer": ("optimizer", "optim"),
    "learning_rate": ("learning_rate", "lr", "initial_lr"),
    "batch_size": ("batch_size", "batchsize"),
    "seed": ("seed", "random_seed"),
}


def _parse_user_map(map_args: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in map_args:
        if "=" not in raw:
            raise SystemExit(f"--map expects `column=metric_key`, got {raw!r} (no '=' separator)")
        col, key = raw.split("=", 1)
        col, key = col.strip(), key.strip()
        if not col or not key:
            raise SystemExit(f"--map cannot have empty column or key: {raw!r}")
        out[col] = key
    return out


def _resolve_metrics(
    available: list[str],
    user_map: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    available_set = set(available)
    for col, override in user_map.items():
        if override not in available_set:
            raise SystemExit(
                f"--map column {col!r} points at metric {override!r}, "
                f"but it was not found in this MLflow run. "
                f"Available metric keys: {sorted(available)}"
            )
        resolved[col] = override
    for col, patterns in _DEFAULT_METRIC_PATTERNS.items():
        if col in resolved:
            continue
        for pat in patterns:
            if pat in available_set:
                resolved[col] = pat
                break
    return resolved


def _resolve_run_id(
    client: MlflowClient,
    *,
    run_id: str | None,
    experiment: str | None,
    run_name: str | None,
) -> str:
    if run_id and (experiment or run_name):
        raise SystemExit("Pass either --run-id, OR (--experiment + --run-name), not both.")
    if run_id:
        return run_id
    if not experiment or not run_name:
        raise SystemExit("Must pass either --run-id, or both --experiment and --run-name.")
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"MLflow experiment not found: {experiment!r}")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.`mlflow.runName` = '{run_name}'",
        max_results=2,
    )
    if not runs:
        raise SystemExit(f"No runs in experiment {experiment!r} with name {run_name!r}.")
    if len(runs) > 1:
        raise SystemExit(
            f"Multiple runs match name {run_name!r} in experiment "
            f"{experiment!r}; pass --run-id directly to disambiguate."
        )
    return runs[0].info.run_id


def _metric_history_to_series(
    client: MlflowClient,
    run_id: str,
    key: str,
) -> pd.Series:
    history = client.get_metric_history(run_id, key)
    if not history:
        return pd.Series(dtype=float)
    rows = []
    for h in history:
        rows.append(
            {
                "step": int(h.step) if h.step is not None else None,
                "ts": int(h.timestamp),
                "value": float(h.value),
            }
        )
    df = pd.DataFrame(rows)
    if df["step"].isna().any():
        df = df.sort_values("ts").reset_index(drop=True)
        df["step"] = range(len(df))
    df = df.groupby("step", as_index=False)["value"].last()
    return df.set_index("step")["value"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Defaults to whatever "
        "MLFLOW_TRACKING_URI / mlflow.get_tracking_uri() "
        "yields. Examples: 'file:./mlruns', "
        "'http://localhost:5000', "
        "'sqlite:///mlflow.db'.",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="MLflow run_id (32-character hex). Mutually exclusive with --experiment + --run-name.",
    )
    p.add_argument("--experiment", default=None, help="Experiment name (alternative to --run-id).")
    p.add_argument(
        "--run-name", default=None, help="Run name within the experiment (alternative to --run-id)."
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output directory for meta.json + history.csv."
    )
    p.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="COL=METRIC",
        help="Override an auto-resolved metric, e.g. '--map train_loss=Loss/train'. Repeatable.",
    )
    p.add_argument(
        "--list-metrics",
        action="store_true",
        help="List the metric keys for the resolved run and exit (no conversion).",
    )
    p.add_argument(
        "--dataset",
        default=None,
        help="Override dataset_name (otherwise lifted from MLflow params).",
    )
    p.add_argument(
        "--model", default=None, help="Override model_name (otherwise lifted from MLflow params)."
    )
    p.add_argument(
        "--framework",
        default=None,
        help="Override framework (otherwise lifted from MLflow params).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import mlflow

    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    rid = _resolve_run_id(
        client,
        run_id=args.run_id,
        experiment=args.experiment,
        run_name=args.run_name,
    )
    print(f"Using MLflow run_id: {rid}")
    print(f"  tracking_uri: {mlflow.get_tracking_uri()}")
    run = client.get_run(rid)
    metric_keys = sorted(run.data.metrics.keys())
    print(f"  found {len(metric_keys)} metric key(s).")
    if args.list_metrics:
        for k in metric_keys:
            print(f"    - {k}")
        return
    if not metric_keys:
        raise SystemExit("Run has no metrics — nothing to convert.")
    user_map = _parse_user_map(args.map)
    resolved = _resolve_metrics(metric_keys, user_map)
    if not resolved:
        raise SystemExit(
            "Could not auto-resolve any expected column.\n"
            f"Available metric keys: {metric_keys}\n"
            "Re-run with --map flags, e.g. "
            "'--map train_loss=Loss/train --map val_loss=Loss/val'."
        )
    print("Metric mapping (MLflow key → diagnostic column):")
    for col, key in resolved.items():
        print(f"    {key:40s}  →  {col}")
    series_per_col = {
        col: _metric_history_to_series(client, rid, key) for col, key in resolved.items()
    }
    aligned = pd.DataFrame(series_per_col)
    aligned.sort_index(inplace=True)
    if aligned.empty:
        raise SystemExit("All resolved metrics came back empty.")
    aligned.index.name = "step"
    aligned.reset_index(inplace=True)
    aligned["epoch"] = range(1, len(aligned) + 1)
    n_epochs = len(aligned)
    print(f"  built {n_epochs} per-step rows.")
    params = run.data.params or {}
    lifted: dict = {}
    for meta_key, candidates in _PARAM_LIFTS.items():
        for c in candidates:
            if c in params:
                v = params[c]
                if meta_key in ("learning_rate",):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                if meta_key in ("batch_size", "seed"):
                    try:
                        v = int(v)
                    except (ValueError, TypeError):
                        pass
                lifted[meta_key] = v
                break
    if args.dataset is not None:
        lifted["dataset_name"] = args.dataset
    if args.model is not None:
        lifted["model_name"] = args.model
    if args.framework is not None:
        lifted["framework"] = args.framework
    run_name = run.data.tags.get("mlflow.runName") or rid[:8]
    out_run_id = f"mlflow_{run_name}_{int(time.time())}"
    meta: dict = {
        "run_id": out_run_id,
        "task_type": "tabular",
    }
    meta.update({k: v for k, v in lifted.items() if v is not None and v != ""})
    meta["epochs_planned"] = n_epochs
    meta["notes"] = f"Converted from MLflow run {rid} (structured_diag.examples.convert_mlflow)."
    meta["mlflow_run_id"] = rid
    meta["mlflow_tracking_uri"] = mlflow.get_tracking_uri()
    if run_name and run_name != rid[:8]:
        meta["mlflow_run_name"] = run_name
    meta["mlflow_resolved_metrics"] = resolved
    out_dir = args.out.resolve()
    history_columns = ("epoch",) + tuple(c for c in resolved if c != "epoch")
    with RunLogger(
        output_dir=out_dir, meta=meta, history_columns=history_columns, overwrite=args.overwrite
    ) as logger:
        for _, row in aligned.iterrows():
            payload: dict = {"epoch": int(row["epoch"])}
            for col in resolved:
                v = row.get(col)
                if v is None:
                    continue
                if isinstance(v, float) and v != v:
                    continue
                payload[col] = float(v) if not isinstance(v, bool) else int(v)
            logger.log_epoch(**payload)
        finals: dict = {}
        for col in ("train_loss", "val_loss", "train_acc", "val_acc"):
            if col in aligned.columns:
                finals[f"{col}_final"] = float(aligned[col].iloc[-1])
        logger.finalize(status="completed", final_metrics=finals or None)
    print(f"\nRun written to {out_dir}")
    print("Files:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"    {p.name}")


if __name__ == "__main__":
    main()
