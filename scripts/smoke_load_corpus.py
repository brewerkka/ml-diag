from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.data import (                                      
    CorpusManifestError,
    RunLoadError,
    load_manifest,
    load_run,
    load_runs_table,
)
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--corpus",
        required=True,
        type=Path,
        help="Path to a corpus directory containing corpus.manifest.json.",
    )
    p.add_argument(
        "--skip-broken",
        action="store_true",
        help="Skip runs that fail to load instead of erroring out.",
    )
    p.add_argument(
        "--sample-n",
        type=int,
        default=1,
        help="How many sample runs to print in detail (default: 1).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _print_class_balance(labels: list[str | None], heading: str) -> None:
    counts = Counter(labels)
    total = sum(counts.values())
    print(f"\n{heading}")
    if total == 0:
        print("  (no labels)")
        return
    for label, n in counts.most_common():
        pct = 100.0 * n / total
        shown = label if label is not None else "<unlabeled>"
        print(f"  {shown:30s} {n:5d}  ({pct:5.1f}%)")


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    try:
        manifest = load_manifest(args.corpus)
    except CorpusManifestError as e:
        print(f"ERROR loading manifest: {e}", file=sys.stderr)
        return 2
    print("=" * 72)
    print(f"Corpus: {manifest.name}")
    print(f"Path:   {manifest.corpus_path}")
    print(f"Runs declared in manifest: {manifest.n_runs}")
    if manifest.has_splits:
        print("Splits:")
        for split, ids in manifest.splits.items():
            print(f"  {split:10s} {len(ids):5d} runs")
    else:
        print("Splits: (none recorded in manifest)")
    try:
        table = load_runs_table(manifest, skip_broken=args.skip_broken)
    except RunLoadError as e:
        print(f"ERROR loading runs: {e}", file=sys.stderr)
        return 3
    print(f"\nRuns loaded: {len(table)}")
    print(
        f"Multi-label runs: {int(table['is_multi_label'].sum())} "
        f"({100.0 * table['is_multi_label'].mean():.1f}%)"
    )
    _print_class_balance(
        table["primary_label"].tolist(),
        "Class balance (by primary_label):",
    )
    if "severity" in table.columns:
        _print_class_balance(
            table["severity"].tolist(),
            "Severity distribution:",
        )
    if "dataset" in table.columns:
        _print_class_balance(
            table["dataset"].tolist(),
            "Dataset distribution:",
        )
    n = max(0, min(args.sample_n, len(table)))
    if n > 0:
        print("\n" + "=" * 72)
        print(f"Sample runs (first {n}):")
        for _, row in table.head(n).iterrows():
            print("-" * 72)
            print(f"run_id   : {row['run_id']}")
            print(f"run_dir  : {row['run_dir']}")
            print(f"dataset  : {row['dataset']}")
            print(f"labels   : {row['labels']}")
            print(f"severity : {row['severity']}")
            print(f"n_epochs : {row['n_history_rows']}")
            try:
                rec = load_run(row["run_dir"])
                cols = ", ".join(rec.history.columns.astype(str)[:10])
                print(
                    f"history  : {rec.history.shape[0]} rows x "
                    f"{rec.history.shape[1]} cols  ({cols}{'…' if rec.history.shape[1] > 10 else ''})"
                )
            except RunLoadError as e:
                print(f"history  : <error: {e}>")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
