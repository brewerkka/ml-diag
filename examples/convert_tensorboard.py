from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path

_SRC_CANDIDATES = (
    Path(__file__).resolve().parent.parent / "src",
    Path("/Users/brewerka/Desktop/ml_diag/src"),
)

for _c in _SRC_CANDIDATES:
    if _c.is_dir() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ml_diag.logging_sdk import RunLogger

_DEFAULT_TAG_PATTERNS: dict[str, tuple[str, ...]] = {
    "train_loss": (
        "train_loss",
        "Loss/train",
        "loss/train",
        "Loss_train",
        "epoch_loss",
        "training/loss",
    ),
    "val_loss": (
        "val_loss",
        "Loss/val",
        "loss/val",
        "Loss_val",
        "epoch_val_loss",
        "validation/loss",
    ),
    "train_acc": (
        "train_acc",
        "Accuracy/train",
        "accuracy/train",
        "Accuracy_train",
        "epoch_accuracy",
        "training/accuracy",
    ),
    "val_acc": (
        "val_acc",
        "Accuracy/val",
        "accuracy/val",
        "Accuracy_val",
        "epoch_val_accuracy",
        "validation/accuracy",
    ),
    "lr": (
        "lr",
        "learning_rate",
        "LearningRate",
        "Learning_rate",
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


def _parse_user_map(map_args: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in map_args:
        if "=" not in raw:
            raise SystemExit(f"--map expects `column=tag`, got {raw!r} (no '=' separator)")
        col, tag = raw.split("=", 1)
        col = col.strip()
        tag = tag.strip()
        if not col or not tag:
            raise SystemExit(f"--map cannot have empty column or tag: {raw!r}")
        out[col] = tag
    return out


def _resolve_tags(
    available: list[str],
    user_map: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    available_set = set(available)
    for col, override in user_map.items():
        if override not in available_set:
            raise SystemExit(
                f"--map column {col!r} points at tag {override!r}, "
                f"but it was not found in the TB log. "
                f"Available tags: {sorted(available)}"
            )
        resolved[col] = override
    for col, patterns in _DEFAULT_TAG_PATTERNS.items():
        if col in resolved:
            continue
        for pat in patterns:
            if pat in available_set:
                resolved[col] = pat
                break
    return resolved


def _scalar_series(acc: EventAccumulator, tag: str) -> pd.Series:
    events = acc.Scalars(tag)
    df = pd.DataFrame([{"step": e.step, "value": float(e.value)} for e in events])
    if df.empty:
        return pd.Series(dtype=float)
    df = df.groupby("step", as_index=False)["value"].last()
    return df.set_index("step")["value"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tb-logdir",
        type=Path,
        required=True,
        help="TensorBoard log directory (contains events.out.tfevents.* files).",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output directory for meta.json + history.csv."
    )
    p.add_argument("--dataset", default="", help="Free-text dataset name (e.g. 'my_table_v3').")
    p.add_argument("--model", default="", help="Free-text model name.")
    p.add_argument("--framework", default="", help="e.g. pytorch, keras, tensorflow.")
    p.add_argument("--optimizer", default="", help="e.g. adam, sgd.")
    p.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Initial LR (used if no `lr` series is found).",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--run-id",
        default=None,
        help="Override run_id (defaults to tb-logdir basename + timestamp).",
    )
    p.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="COL=TAG",
        help="Override an auto-resolved tag, e.g. '--map train_loss=Loss/train'. Repeatable.",
    )
    p.add_argument(
        "--list-tags",
        action="store_true",
        help="Just list the scalar tags and exit (no conversion).",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.tb_logdir.is_dir():
        raise SystemExit(f"TB log dir not found: {args.tb_logdir}")
    print(f"Reading TensorBoard log dir: {args.tb_logdir}")
    acc = EventAccumulator(
        str(args.tb_logdir),
        size_guidance={"scalars": 0},
    )
    acc.Reload()
    available = list(acc.Tags().get("scalars", []))
    print(f"  found {len(available)} scalar tag(s).")
    if args.list_tags:
        for tag in sorted(available):
            print(f"    - {tag}")
        return
    if not available:
        raise SystemExit("No scalar tags found — is this a real TB log dir?")
    user_map = _parse_user_map(args.map)
    resolved = _resolve_tags(available, user_map)
    if not resolved:
        raise SystemExit(
            "Could not auto-resolve any expected column.\n"
            f"Available tags: {sorted(available)}\n"
            "Re-run with --map flags, e.g. "
            "'--map train_loss=Loss/train --map val_loss=Loss/val'."
        )
    print("Tag mapping (TB tag → diagnostic column):")
    for col, tag in resolved.items():
        print(f"    {tag:40s}  →  {col}")
    series_per_col = {col: _scalar_series(acc, tag) for col, tag in resolved.items()}
    aligned = pd.DataFrame(series_per_col)
    aligned.sort_index(inplace=True)
    if aligned.empty:
        raise SystemExit("All resolved tags came back empty.")
    aligned.index.name = "step"
    aligned.reset_index(inplace=True)
    aligned["epoch"] = range(1, len(aligned) + 1)
    n_epochs = len(aligned)
    print(f"  built {n_epochs} per-step rows.")
    run_id = args.run_id or (f"{args.tb_logdir.name}_{int(time.time())}")
    meta: dict = {
        "run_id": run_id,
        "task_type": "tabular",
    }
    for k, v in (
        ("dataset_name", args.dataset),
        ("model_name", args.model),
        ("framework", args.framework),
        ("optimizer", args.optimizer),
        ("learning_rate", args.learning_rate),
        ("batch_size", args.batch_size),
        ("seed", args.seed),
    ):
        if v is not None and v != "":
            meta[k] = v
    meta["epochs_planned"] = n_epochs
    meta["notes"] = (
        f"Converted from TensorBoard log dir {args.tb_logdir} "
        f"(ml_diag.examples.convert_tensorboard)."
    )
    meta["tb_logdir"] = str(args.tb_logdir.resolve())
    meta["tb_resolved_tags"] = resolved
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
