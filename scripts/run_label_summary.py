from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.data import load_runs_table              
from ml_diag.labels import (              
    PRIMARY_LABELS,
    STAGE3_LABELS_BY_BRANCH,
    label_distribution,
    validate_schema,
)
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path. If omitted, only prints to stdout.",
    )
    p.add_argument("--skip-broken", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _print_block(title: str, counts: dict[str, int]) -> None:
    total = sum(counts.values()) or 1
    print(f"\n{title} (total = {sum(counts.values())})")
    if not counts or sum(counts.values()) == 0:
        print("  (empty)")
        return
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {label:42s} {n:5d}  ({100.0 * n / total:5.1f}%)")


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    validate_schema()
    table = load_runs_table(args.corpus, skip_broken=args.skip_broken)
    primaries = table["primary_label"].dropna().tolist()
    unknown = sorted(set(primaries) - set(PRIMARY_LABELS))
    dist = label_distribution(primaries)
    print("=" * 72)
    print(f"Corpus runs (with primary_label set): {len(primaries)}")
    if unknown:
        print(
            f"WARNING: {len(unknown)} unknown label values "
            f"(skipped from per-stage counts): {unknown}"
        )
    _print_block("Stage 1 — healthy vs faulty", dist["stage1"])
    _print_block("Stage 2 — data_related vs optimization_or_generalization_related", dist["stage2"])
    print("\nStage 3 — leaf labels per Stage-2 branch:")
    for branch in STAGE3_LABELS_BY_BRANCH:
        _print_block(f"  branch = {branch}", dist["stage3"][branch])
    print("=" * 72)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "corpus": str(Path(args.corpus).resolve()),
            "n_runs_with_label": len(primaries),
            "unknown_labels": unknown,
            "distribution": dist,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote summary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
