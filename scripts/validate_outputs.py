from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from structured_diag.evaluation import (  # noqa: E402
    CASE_OUTPUTS_SCHEMA_VERSION,
    REQUIRED_FILES,
    validate_case_dir,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", type=Path, help="Case directories to validate.")
    p.add_argument(
        "--all", action="store_true", help="Validate every directory under results/cases/."
    )
    p.add_argument("--cases-root", type=Path, default=Path("results/cases"))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    targets: list[Path] = list(args.paths)
    if args.all:
        if not args.cases_root.is_dir():
            print(f"ERROR: {args.cases_root} is not a directory.", file=sys.stderr)
            return 2
        targets.extend(p for p in args.cases_root.iterdir() if p.is_dir())
    if not targets:
        print("ERROR: nothing to validate. Pass a path or use --all.", file=sys.stderr)
        return 2
    print(f"Schema version: {CASE_OUTPUTS_SCHEMA_VERSION}")
    print(f"Required files per case: {', '.join(REQUIRED_FILES)}")
    print()
    n_ok = 0
    n_bad = 0
    for path in targets:
        ok, errs = validate_case_dir(path)
        marker = "✓" if ok else "✗"
        print(f"{marker} {path}")
        if not ok:
            for e in errs:
                print(f"    - {e}")
            n_bad += 1
        else:
            n_ok += 1
    print()
    print(f"OK: {n_ok}, FAILED: {n_bad}")
    return 0 if n_bad == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
