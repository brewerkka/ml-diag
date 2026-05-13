"""Stage 75 — MLflow / Weights & Biases trace parser → structured_diag schema.

Purpose
-------
Provide a scaffold for converting publicly-available MLflow runs or
Weights & Biases (wandb) projects into the structured_diag run-level
schema (``meta.json`` + ``history.csv``). Each parsed trace becomes
a "silver-labelled" run that can be appended to the corpus or used
in a holdout evaluation. The reviewer's pillar 2(b) — silver-label
dataset from MLflow / W&B — is partially closed by this script's
existence; full coverage requires actual API tokens and trace fetching.

Parsing strategies
------------------

  1. **MLflow tracking URI** — read via the official ``mlflow`` Python
     client. Each run's metric history is fetched via
     ``client.get_metric_history(run_id, key)``; tags and parameters
     come from ``client.get_run(run_id)``. Map metrics to
     ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc`` where
     names roughly match.

  2. **W&B public project** — read via the ``wandb`` Python client.
     Each run's history is fetched via ``run.history()``; tags and
     config from ``run.config``.

  3. **Local CSV directory** — fallback for datasets like the FAIR
     PyTorch examples repo: each subdir has its own ``history.csv``
     under a known schema. Pass the directory glob, iterate.

Silver-labelling rule
---------------------
For runs without an explicit human label, we use a simple rule
combining tags, run name, and final metric values:

  * tags or run_name contains "leak", "leakage", "data_leak"        → leakage
  * tags or run_name contains "overfit"                              → overfitting
  * final val_acc / train_acc < 0.7 and small gap                    → underfitting
  * final val_acc / train_acc < 0.5 from epoch 1                     → underfitting
  * train_acc - val_acc > 0.20 by final epoch                        → overfitting
  * val_acc stddev across last 10 epochs > 0.10                      → instability
  * val_acc converges to >=0.85, train_acc >=0.95, gap < 0.05        → healthy
  * anything else                                                    → "unlabeled"

Cases labelled "unlabeled" should be reviewed manually.

Output
------
For each parsed run: ``data/silver_corpus/<run_id>/{meta.json,history.csv}``.
Summary report: ``results/silver_corpus_summary.{md,json}``.

Note
----
This script does NOT make outbound HTTP calls unless explicitly
configured with credentials. By default, running it without arguments
prints the parsing protocol and exits.

Usage
-----
    # Discover protocol (no fetching)
    python scripts/parse_mlflow_traces.py

    # MLflow mode (requires mlflow client + tracking URI)
    python scripts/parse_mlflow_traces.py \\
        --mode mlflow \\
        --tracking-uri https://example-mlflow.org \\
        --experiment-name "my_experiment" \\
        --out-corpus data/silver_corpus/ \\
        --max-runs 100

    # W&B mode (requires WANDB_API_KEY env var)
    python scripts/parse_mlflow_traces.py \\
        --mode wandb \\
        --entity huggingface \\
        --project transformers \\
        --out-corpus data/silver_corpus/ \\
        --max-runs 100

    # Local CSV mode
    python scripts/parse_mlflow_traces.py \\
        --mode local \\
        --input-dir /path/to/csv_runs/ \\
        --out-corpus data/silver_corpus/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _silver_label(history, tags: dict, run_name: str) -> str:
    """Assign a silver label from training-dynamics + tags + name.

    See module docstring for the rule. The rule is intentionally simple
    so it can be auditable; production-grade silver labels would use
    structured_diag itself as a labelling oracle.
    """
    text = " ".join(
        [
            str(run_name or ""),
            " ".join(f"{k} {v}" for k, v in tags.items()),
        ]
    ).lower()
    if any(w in text for w in ("leak", "data_leak", "leakage")):
        return "leakage"
    if any(w in text for w in ("overfit",)):
        return "overfitting"
    if any(w in text for w in ("unstable", "instability", "diverg")):
        return "instability"
    if any(w in text for w in ("noise", "label_noise", "noisy_label")):
        return "label_noise"
    if history is None or len(history) < 3:
        return "unlabeled"
    try:
        import numpy as np

        last10 = history.tail(10) if hasattr(history, "tail") else history[-10:]
        if "val_acc" in history.columns:
            vacc = np.asarray(history["val_acc"], dtype=float)
            tacc = (
                np.asarray(history["train_acc"], dtype=float)
                if "train_acc" in history.columns
                else None
            )
            vacc_final = float(vacc[-1])
            if tacc is not None:
                tacc_final = float(tacc[-1])
                gap = tacc_final - vacc_final
                if vacc_final >= 0.85 and tacc_final >= 0.95 and abs(gap) < 0.05:
                    return "healthy"
                if gap > 0.20:
                    return "overfitting"
            vacc_last10_std = (
                float(np.std(last10["val_acc"])) if hasattr(last10, "__getitem__") else 0.0
            )
            if vacc_last10_std > 0.10:
                return "instability"
            if vacc_final < 0.65 and (tacc is None or vacc_final - float(tacc[0]) < 0.10):
                return "underfitting"
    except Exception:
        return "unlabeled"
    return "unlabeled"


def _parse_mlflow(args):
    """Parse MLflow runs (requires ``mlflow`` package)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("ERROR: mlflow package not installed. pip install mlflow", file=sys.stderr)
        return 1
    if not args.tracking_uri:
        print("ERROR: --tracking-uri required for --mode mlflow", file=sys.stderr)
        return 1
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = (
        client.get_experiment_by_name(args.experiment_name) if args.experiment_name else None
    )
    if experiment is None and args.experiment_name:
        print(f"ERROR: experiment {args.experiment_name!r} not found.", file=sys.stderr)
        return 1
    exp_id = experiment.experiment_id if experiment else "0"
    runs = client.search_runs([exp_id], max_results=args.max_runs)
    print(f"Found {len(runs)} runs in experiment {args.experiment_name!r}")
    out_dir = args.out_corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for run in runs:
        run_id = run.info.run_id
        try:
            import pandas as pd  # noqa: F401

            metrics_keys = run.data.metrics.keys()
            history_dict = {}
            for key in ("train_loss", "val_loss", "train_acc", "val_acc"):
                if key in metrics_keys:
                    hist = client.get_metric_history(run_id, key)
                    history_dict[key] = [m.value for m in hist]
            if not history_dict:
                continue

            history = pd.DataFrame(history_dict)
            history["epoch"] = range(len(history))
            label = _silver_label(history, run.data.tags or {}, run.info.run_name or "")
            run_dir = out_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(run_dir / "history.csv", index=False)
            (run_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "fault_labels": [label],
                        "source": "mlflow",
                        "tracking_uri": args.tracking_uri,
                        "tags": dict(run.data.tags or {}),
                        "params": dict(run.data.params or {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            rows.append({"run_id": run_id, "label": label, "n_epochs": len(history)})
        except Exception as e:
            print(f"  skip {run_id}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"Wrote {len(rows)} silver-labelled runs to {out_dir}")
    _write_summary(
        rows,
        out_dir.parent / "silver_corpus_summary.md",
        out_dir.parent / "silver_corpus_summary.json",
    )
    return 0


def _parse_wandb(args):
    """Parse W&B project (requires ``wandb`` package + WANDB_API_KEY)."""
    try:
        import wandb
    except ImportError:
        print("ERROR: wandb package not installed. pip install wandb", file=sys.stderr)
        return 1
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}"
    runs = api.runs(project_path)
    print(f"Found {len(runs)} runs in W&B project {project_path}")
    out_dir = args.out_corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    count = 0
    for run in runs:
        if count >= args.max_runs:
            break
        try:
            history = run.history(keys=["train_loss", "val_loss", "train_acc", "val_acc"])
            if history is None or len(history) < 3:
                continue
            label = _silver_label(history, dict(run.tags) if run.tags else {}, run.name or "")
            run_dir = out_dir / run.id
            run_dir.mkdir(parents=True, exist_ok=True)
            history.to_csv(run_dir / "history.csv", index=False)
            (run_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "run_id": run.id,
                        "fault_labels": [label],
                        "source": "wandb",
                        "project": project_path,
                        "tags": list(run.tags) if run.tags else [],
                        "config": dict(run.config or {}),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            rows.append({"run_id": run.id, "label": label, "n_epochs": len(history)})
            count += 1
        except Exception as e:
            print(f"  skip {run.id}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"Wrote {len(rows)} silver-labelled runs to {out_dir}")
    _write_summary(
        rows,
        out_dir.parent / "silver_corpus_summary.md",
        out_dir.parent / "silver_corpus_summary.json",
    )
    return 0


def _parse_local(args):
    """Parse a local directory of (run_dir / history.csv) layouts."""
    in_dir = args.input_dir
    if not in_dir or not in_dir.is_dir():
        print(f"ERROR: --input-dir {in_dir} not a directory", file=sys.stderr)
        return 1
    out_dir = args.out_corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    import pandas as pd

    for run_dir in sorted(in_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        h = run_dir / "history.csv"
        if not h.is_file():
            continue
        try:
            history = pd.read_csv(h)
            tags = {}
            run_name = run_dir.name
            label = _silver_label(history, tags, run_name)
            target = out_dir / run_dir.name
            target.mkdir(parents=True, exist_ok=True)
            history.to_csv(target / "history.csv", index=False)
            (target / "meta.json").write_text(
                json.dumps(
                    {
                        "run_id": run_dir.name,
                        "fault_labels": [label],
                        "source": "local",
                        "source_dir": str(run_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            rows.append({"run_id": run_dir.name, "label": label, "n_epochs": len(history)})
        except Exception as e:
            print(f"  skip {run_dir.name}: {type(e).__name__}: {e}", file=sys.stderr)
    print(f"Wrote {len(rows)} silver-labelled runs to {out_dir}")
    _write_summary(
        rows,
        out_dir.parent / "silver_corpus_summary.md",
        out_dir.parent / "silver_corpus_summary.json",
    )
    return 0


def _write_summary(rows: list[dict], out_md: Path, out_json: Path) -> None:
    from collections import Counter

    label_counts = Counter(r["label"] for r in rows)
    md = [
        "# Stage 75 — Silver corpus summary",
        "",
        f"Parsed {len(rows)} runs total.",
        "",
        "| Label | Count |",
        "|---|---|",
    ]
    for label, count in sorted(label_counts.items(), key=lambda kv: -kv[1]):
        md.append(f"| {label} | {count} |")
    md.extend(
        [
            "",
            "## Next step",
            "",
            "Append parsed runs to the corpus manifest:",
            "",
            "```bash",
            "python scripts/build_cv_corpus.py \\",
            "    --root-dir data/silver_corpus/ \\",
            "    --out      data/corpus/silver_corpus.manifest.json",
            "```",
            "",
            "Then re-train structured_diag on the combined corpus (or run "
            "as held-out evaluation only).",
            "",
        ]
    )
    out_md.write_text("\n".join(md), encoding="utf-8")
    out_json.write_text(
        json.dumps(
            {"stage": 75, "n_runs": len(rows), "label_counts": dict(label_counts), "rows": rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["mlflow", "wandb", "local", "protocol"], default="protocol")
    p.add_argument("--tracking-uri", default=None)
    p.add_argument("--experiment-name", default=None)
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default=None)
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--out-corpus", type=Path, default=_REPO_ROOT / "data/silver_corpus/")
    p.add_argument("--max-runs", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.mode == "protocol":
        print(__doc__)
        return 0
    if args.mode == "mlflow":
        return _parse_mlflow(args)
    if args.mode == "wandb":
        return _parse_wandb(args)
    if args.mode == "local":
        return _parse_local(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
