from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml_diag.evaluation import build_evidence              
from ml_diag.features import (              
    build_data_integrity_features,
    build_feature_table,
)
from ml_diag.interpretation import (              
    InterpretationConfig,
    interpret,
    render_markdown,
)
from ml_diag.models import diagnose_one, load_cascade              
from ml_diag.utils import setup_logging              


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--artifacts",
        required=True,
        type=Path,
        help="Directory produced by run_hierarchical_train.py",
    )
    p.add_argument("--run-id", default=None, help="Single run_id (else first --n).")
    p.add_argument("--n", type=int, default=3, help="When --run-id is omitted.")
    p.add_argument(
        "--backend", default="auto", choices=["auto", "template", "groq", "ollama"]
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override default model id for the chosen backend.",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--max-recommendations", type=int, default=3)
    p.add_argument("--no-integrity", action="store_true")
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--out-dir", type=Path, default=None, help="Used in batch mode (when --run-id is omitted)."
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    cascade = load_cascade(args.artifacts)
    base = build_feature_table(args.corpus)
    integrity_columns = None
    full_df = base.df
    if not args.no_integrity:
        try:
            di = build_data_integrity_features(args.corpus, base_table=base)
            integrity_columns = di.integrity_columns
            full_df = di.df
        except Exception as e:
            print(
                f"WARNING: integrity features unavailable ({e}); continuing without.",
                file=sys.stderr,
            )
    feature_cols = base.feature_columns
    X = full_df[feature_cols].copy()
    if args.run_id is not None:
        if args.run_id not in X.index:
            print(f"ERROR: run_id {args.run_id!r} not found in feature table.", file=sys.stderr)
            return 2
        run_ids = [args.run_id]
    else:
        run_ids = X.index.astype(str).tolist()[: max(1, args.n)]
    cfg = InterpretationConfig(
        backend=args.backend,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_recommendations=args.max_recommendations,
    )
    if args.out_dir is None and args.run_id is None:
        args.out_dir = Path("results/interpretation")
    n_done = 0
    for rid in run_ids:
        x_row = X.loc[rid]
        full_row = full_df.loc[rid]
        diagnosis = diagnose_one(cascade, run_id=rid, x_row=x_row)
        evidence = build_evidence(
            diagnosis=diagnosis,
            feature_row=full_row,
            cascade=cascade,
            integrity_columns=integrity_columns,
        )
        result = interpret(diagnosis=diagnosis, evidence=evidence, config=cfg)
        if args.run_id is not None:
            md_target = args.out_md
            json_target = args.out_json
        else:
            md_target = (args.out_dir / f"{rid}.md") if args.out_dir else None
            json_target = (args.out_dir / f"{rid}.json") if args.out_dir else None
        if md_target is not None:
            md_target.parent.mkdir(parents=True, exist_ok=True)
            md_target.write_text(render_markdown(result), encoding="utf-8")
            print(f"  [md]   {rid} -> {md_target}")
        if json_target is not None:
            json_target.parent.mkdir(parents=True, exist_ok=True)
            json_target.write_text(
                json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  [json] {rid} -> {json_target}")
        print(
            f"[{rid}] backend={result.backend} class={result.final_class} "
            f"conf={result.final_confidence:.3f} recs={len(result.recommendations)} "
            f"warnings={len(result.warnings)}"
        )
        n_done += 1
    if n_done == 0:
        print("ERROR: nothing was interpreted.", file=sys.stderr)
        return 3
    print(f"\nInterpreted {n_done} run(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
