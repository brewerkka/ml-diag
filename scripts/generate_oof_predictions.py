from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


from structured_diag.diagnosis.arbitrator import ArbitratorConfig  # noqa: E402
from structured_diag.diagnosis.oof_predictions import (  # noqa: E402
    generate_oof_predictions,
    write_oof_parquet,
)
from structured_diag.features import build_feature_table  # noqa: E402
from structured_diag.models.flat_baseline import _split_train_test  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output parquet path (e.g. results/oof_predictions_8ds.parquet)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--arbitrator-backend",
        default="auto",
        choices=["auto", "groq", "anthropic", "template", "none"],
        help="`none` skips arbitrator entirely.",
    )
    p.add_argument("--arbitrator-low-conf", type=float, default=0.0)
    p.add_argument("--arbitrator-cache", type=Path, default=Path(".cache/arbitrator"))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X_all, y_all = ftable.aligned_xy()
    train_idx, test_idx = _split_train_test(X_all, y_all, seed=args.seed)
    X_train = X_all.iloc[train_idx]
    y_train = y_all.iloc[train_idx]
    print(
        f"Generating OOF predictions over the canonical train fold "
        f"(n={len(X_train)}, n_splits={args.n_splits}, seed={args.seed})…"
    )
    arb_cfg: ArbitratorConfig | None = None
    if args.arbitrator_backend != "none":
        arb_cfg = ArbitratorConfig(
            backend=str(args.arbitrator_backend),  # type: ignore[arg-type]
            cache_path=Path(args.arbitrator_cache),
        )
        print(
            f"Arbitrator: backend={arb_cfg.backend}, "
            f"low_conf_trigger={args.arbitrator_low_conf}, "
            f"cache={arb_cfg.cache_path}"
        )
    else:
        print("Arbitrator: disabled (--arbitrator-backend none)")
    oof = generate_oof_predictions(
        X=X_train,
        y=y_train,
        n_splits=int(args.n_splits),
        seed=int(args.seed),
        arbitrator_config=arb_cfg,
        arbitrator_low_conf_trigger=float(args.arbitrator_low_conf),
        include_catboost=not args.no_catboost,
    )
    out_path = write_oof_parquet(oof, args.out)
    print(f"Wrote OOF parquet -> {out_path}")
    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    n_arb_calls = int(oof.arbitrator_triggered.sum())
    flat_pred = oof.flat_proba.idxmax(axis=1)
    cascade_pred = oof.cascade_proba.idxmax(axis=1)
    flat_acc = float((flat_pred.values == y_train.values).mean())
    cascade_acc = float((cascade_pred.values == y_train.values).mean())
    agreement_rate = float((flat_pred.values == cascade_pred.values).mean())
    fold_counts = oof.fold_assignments.value_counts().sort_index().to_dict()
    summary = {
        "corpus": ftable.corpus_name,
        "seed": int(args.seed),
        "n_splits": int(args.n_splits),
        "n_train_rows": int(len(X_train)),
        "fold_sizes": {int(k): int(v) for k, v in fold_counts.items()},
        "flat_oof_accuracy": flat_acc,
        "cascade_oof_accuracy": cascade_acc,
        "agreement_rate": agreement_rate,
        "n_arbitrator_calls": n_arb_calls,
        "n_arbitrator_low_conf": int(
            oof.arbitrator_triggered.sum() - (flat_pred != cascade_pred).sum().clip(0)
        ),
        "arbitrator_backend": str(args.arbitrator_backend),
        "arbitrator_low_conf_trigger": float(args.arbitrator_low_conf),
    }
    meta_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote OOF meta    -> {meta_path}")
    print()
    print("OOF base-learner accuracies on train fold:")
    print(f"  flat    : {flat_acc:.4f}")
    print(f"  cascade : {cascade_acc:.4f}")
    print(f"  agreement rate (flat == cascade): {agreement_rate:.3f}")
    print(
        f"  arbitrator calls: {n_arb_calls} "
        f"(disagreement={(flat_pred != cascade_pred).sum()}, "
        f"low-conf={n_arb_calls - int((flat_pred != cascade_pred).sum())})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
