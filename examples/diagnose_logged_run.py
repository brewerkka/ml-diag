from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Standalone run directory produced by RunLogger "
        "(must contain meta.json + history.csv).",
    )
    parser.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Hierarchical cascade artifacts dir (from scripts/run_hierarchical_train.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write the case folder. Defaults to results/cases/<run_dir.name>/.",
    )
    parser.add_argument("--backend", default="template", choices=["template", "anthropic"])
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    cmd: list[str] = [
        sys.executable,
        str(repo_root / "scripts" / "run_full_case.py"),
        "--run-dir",
        str(args.run_dir),
        "--artifacts",
        str(args.artifacts),
        "--backend",
        args.backend,
    ]
    if args.out_dir is not None:
        cmd.extend(["--out-dir", str(args.out_dir)])
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
