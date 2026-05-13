from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

from structured_diag.logging_sdk import RunLogger


def _fake_epoch(epoch: int, *, lr: float, rng: random.Random) -> dict:
    train_loss = 1.0 / (1.0 + epoch) + rng.gauss(0.0, 0.01)
    val_loss = 0.9 / (1.0 + 0.7 * epoch) + rng.gauss(0.0, 0.02)
    train_acc = 1.0 - math.exp(-0.2 * (epoch + 1)) + rng.gauss(0.0, 0.01)
    val_acc = 1.0 - math.exp(-0.18 * (epoch + 1)) + rng.gauss(0.0, 0.015)
    grad_norm = 2.5 / (1.0 + 0.1 * epoch) + rng.gauss(0.0, 0.05)
    weight_norm = 8.0 + 0.3 * epoch + rng.gauss(0.0, 0.1)
    step_time_sec = 0.25 + rng.gauss(0.0, 0.02)
    loss_spike_flag = 1 if abs(train_loss - val_loss) > 0.5 else 0
    time.sleep(0.01)
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "grad_norm": grad_norm,
        "weight_norm": weight_norm,
        "lr": lr,
        "step_time_sec": step_time_sec,
        "loss_spike_flag": loss_spike_flag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/exp_demo"),
        help="Output directory for the logged run.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow re-using a non-empty output directory."
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    with RunLogger(
        output_dir=args.out,
        meta={
            "run_id": args.out.name,
            "dataset_name": "synthetic_classification",
            "task_type": "classification",
            "model_name": "tiny_mlp",
            "framework": "pytorch",
            "framework_version": "2.x",
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs_planned": args.epochs,
            "seed": args.seed,
            "tags": ["example", "external-run"],
            "notes": "Synthetic example demonstrating the logging contract.",
        },
        overwrite=args.overwrite,
    ) as logger:
        for epoch in range(args.epochs):
            metrics = _fake_epoch(epoch, lr=1e-3, rng=rng)
            logger.log_epoch(**metrics)
        logger.finalize(
            status="completed",
            final_metrics={
                "best_val_acc": 0.81,
                "best_val_loss": 0.34,
                "best_epoch": 18,
            },
            notes="Training finished without errors.",
        )
    print(f"\nLogged run is ready at: {args.out}")
    print("Files:")
    for f in sorted(args.out.iterdir()):
        print(f"  {f}")
    print()
    print("Next step:")
    print("  python scripts/run_full_case.py \\")
    print(f"      --run-dir   {args.out} \\")
    print("      --artifacts results/hierarchical/<corpus>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
