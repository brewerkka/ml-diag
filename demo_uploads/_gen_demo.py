
from __future__ import annotations

import csv

import json

import math

import random

from pathlib import Path

OUT = Path(__file__).resolve().parent

random.seed(42)

def write_run(name: str, history_rows: list[dict], meta: dict) -> None:

    rd = OUT / name

    rd.mkdir(exist_ok=True)

    cols = list(history_rows[0].keys())

    with (rd / "history.csv").open("w", newline="", encoding="utf-8") as f:

        w = csv.DictWriter(f, fieldnames=cols)

        w.writeheader()

        w.writerows(history_rows)

    (rd / "meta.json").write_text(

        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"

    )

    print(f"wrote {rd}/{{history.csv, meta.json}}")

def jitter(x: float, scale: float = 0.01) -> float:

    return x + random.uniform(-scale, scale)

N1 = 22

hist1 = []

for ep in range(1, N1 + 1):

    train_loss = 0.045 + 0.85 * math.exp(-0.30 * ep) + jitter(0, 0.006)

    if ep <= 8:

        val_loss = 0.30 + 0.045 * (8 - ep) + jitter(0, 0.020)

    else:

        val_loss = 0.30 + 0.022 * (ep - 8) + jitter(0, 0.020)

    train_acc = min(0.992, 0.55 + (1 - math.exp(-0.22 * ep)) * 0.45 + jitter(0, 0.004))

    val_acc = min(0.795, 0.55 + (1 - math.exp(-0.30 * ep)) * 0.25 + jitter(0, 0.008))

    if ep > 12:

        val_acc -= 0.0035 * (ep - 12)

    hist1.append({

        "epoch": ep,

        "train_loss": round(train_loss, 4),

        "val_loss": round(val_loss, 4),

        "train_acc": round(train_acc, 4),

        "val_acc": round(val_acc, 4),

        "lr": 0.001,

        "grad_norm": round(0.5 + 0.05 * math.sin(ep) + jitter(0, 0.02), 4),

        "weight_norm": round(4.2 + 0.04 * ep, 4),

    })

meta1 = {

    "run_id": "demo_overfitting_2026_05_08",

    "task": "tabular",

    "dataset_name": "demo:synthetic_overfit",

    "model_name": "mlp_h128_h64_dropout_off",

    "framework": "pytorch",

    "optimizer": "adam",

    "learning_rate": 0.001,

    "batch_size": 64,

    "epochs_planned": N1,

    "n_epochs_logged": N1,

    "status": "completed",

    "notes": "Synthetic demo run designed to show overfitting: train_loss "

             "keeps falling, val_loss bottoms ~epoch 6 then climbs."

}

write_run("demo_overfitting", hist1, meta1)

N2 = 20

hist2 = []

for ep in range(1, N2 + 1):

    train_loss = max(0.045, 0.78 * math.exp(-0.21 * ep) + jitter(0, 0.008))

    val_loss = max(0.30, 0.85 * math.exp(-0.16 * ep) + 0.04 + jitter(0, 0.025))

    train_acc = min(0.985, 0.55 + (1 - math.exp(-0.18 * ep)) * 0.43 + jitter(0, 0.005))

    val_acc = max(0.50, 0.55 + (1 - math.exp(-0.18 * ep)) * 0.36 + jitter(0, 0.014))

    hist2.append({

        "epoch": ep,

        "train_loss": round(train_loss, 4),

        "val_loss": round(val_loss, 4),

        "train_acc": round(train_acc, 4),

        "val_acc": round(val_acc, 4),

        "lr": 0.0008,

        "grad_norm": round(0.4 + 0.03 * math.sin(ep) + jitter(0, 0.012), 4),

        "weight_norm": round(3.5 + 0.025 * ep, 4),

    })

meta2 = {

    "run_id": "demo_healthy_2026_05_08",

    "task": "tabular",

    "dataset_name": "demo:synthetic_healthy",

    "model_name": "mlp_h64_h32_dropout_0.2",

    "framework": "pytorch",

    "optimizer": "adam",

    "learning_rate": 0.0008,

    "batch_size": 64,

    "epochs_planned": N2,

    "n_epochs_logged": N2,

    "status": "completed",

    "notes": "Synthetic demo run with no pathologies: train and val track "

             "each other, both improve through the end."

}

write_run("demo_healthy", hist2, meta2)

