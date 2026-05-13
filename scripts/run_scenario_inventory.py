from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structured_diag.scenarios import (  # noqa: E402
    build_inventory,
    validate_inventory,
)
from structured_diag.scenarios.inventory import write_inventory  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--min-leaf-support",
        type=int,
        default=5,
        help="Stage-3 leaves below this count count as warnings.",
    )
    p.add_argument(
        "--min-history-epochs",
        type=int,
        default=5,
        help="Runs shorter than this are reported as short-history.",
    )
    p.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    inv = build_inventory(
        args.corpus,
        min_leaf_support=args.min_leaf_support,
        min_history_epochs=args.min_history_epochs,
        skip_broken=True,
    )
    if args.out_md or args.out_json:
        write_inventory(inv, md_path=args.out_md, json_path=args.out_json)
        if args.out_md:
            print(f"Wrote markdown -> {args.out_md}")
        if args.out_json:
            print(f"Wrote JSON     -> {args.out_json}")
    print()
    print(f"Corpus: {inv.corpus_name}")
    print(f"Runs: {inv.n_runs}, entries: {inv.n_entries}")
    print(f"Multi-label: {inv.multi_label_count}, short-history: {inv.short_history_count}")
    print()
    print("Taxonomy coverage:")
    for label, n in inv.primary_label_counts.items():
        print(f"  {label:14s} {n:5d}")
    print()
    print("Branch coverage (faulty only):")
    for b, n in inv.branch_counts.items():
        print(f"  {b:42s} {n:5d}")
    print()
    print(f"Coverage gaps: {len(inv.gaps)}")
    n_blockers = sum(1 for g in inv.gaps if g.severity == "blocker")
    n_warnings = sum(1 for g in inv.gaps if g.severity == "warning")
    print(f"  blockers: {n_blockers}, warnings: {n_warnings}")
    for g in inv.gaps:
        print(f"  [{g.severity}] {g.kind}/{g.label}: {g.detail}")
    ok, errs = validate_inventory(inv, strict=args.strict)
    if not ok:
        print()
        print("VALIDATION FAILED:")
        for e in errs:
            print(f"  {e}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
