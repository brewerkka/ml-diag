from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.features import build_feature_table              
from ml_diag.models import (              
    diagnose_batch,
    diagnoses_to_dataframe,
    diagnoses_to_jsonl,
    load_cascade,
)
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Directory produced by run_hierarchical_train.py.",
    )
    p.add_argument("--out-jsonl", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument(
        "--run-id", default=None, help="Optionally restrict inference to a single run_id."
    )
    p.add_argument(
        "--top-k", type=int, default=3, help="How many alternative hypotheses to keep (default: 3)."
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    cascade = load_cascade(args.artifacts)
    print(f"Loaded cascade with stages: {cascade.stages_available}")
    ftable = build_feature_table(args.corpus)
    X = ftable.df[ftable.feature_columns].copy()
    if args.run_id is not None:
        if args.run_id not in X.index:
            print(f"ERROR: run_id {args.run_id!r} not found in feature table.", file=sys.stderr)
            return 2
        X = X.loc[[args.run_id]]
    diagnoses = diagnose_batch(cascade, X, top_k_alternatives=args.top_k)
    print(f"Diagnosed {len(diagnoses)} runs. First {min(5, len(diagnoses))} shown:")
    for d in diagnoses[:5]:
        s2 = d.stage2.predicted if d.stage2 else "-"
        s3 = d.stage3.predicted if d.stage3 else "-"
        alt_str = ", ".join(f"{c}={p:.3f}" for c, p in d.alternative_hypotheses)
        print(
            f"  {d.run_id:30s} -> {d.final_class:14s} "
            f"(conf={d.final_confidence:.3f})  "
            f"s1={d.stage1.predicted:7s}({d.stage1.confidence:.2f}) "
            f"s2={s2:25s}({d.stage2.confidence if d.stage2 else 0.0:.2f}) "
            f"s3={s3:14s}  alts: [{alt_str}]"
        )
    if args.out_jsonl is not None:
        diagnoses_to_jsonl(diagnoses, args.out_jsonl)
        print(f"Wrote JSONL -> {args.out_jsonl}")
    if args.out_csv is not None:
        df = diagnoses_to_dataframe(diagnoses)
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote CSV   -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
