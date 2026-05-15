from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np              
import pandas as pd              
from sklearn.tree import DecisionTreeClassifier              

from ml_diag.evaluation.metrics import classification_report              
from ml_diag.features import build_feature_table              
from ml_diag.models.flat_baseline import _split_train_test              
from ml_diag.utils.logging import get_logger, setup_logging              

_LOG = get_logger(__name__)

DEEPFD_DT_HYPERPARAMS: dict = {
    "max_depth": 10,
    "min_samples_leaf": 2,
    "criterion": "entropy",
    "ccp_alpha": 0.0,
}


@dataclass(frozen=True)
class DeepFDResult:
    corpus_name: str
    feature_source: str
    n_features: int
    n_classes: int
    n_train: int
    n_test: int
    seed: int
    test_run_ids: list[str]
    y_test_true: list[str]
    y_test_pred: list[str]
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_support: dict[str, int]
    feature_columns: list[str]
    feature_importances: dict[str, float]


def _train_and_eval_deepfd(X: pd.DataFrame, y: pd.Series, *, seed: int) -> DeepFDResult:
    train_idx, test_idx = _split_train_test(X, y, seed=seed)
    X_tr = X.iloc[train_idx]
    y_tr = y.iloc[train_idx]
    X_te = X.iloc[test_idx]
    y_te = y.iloc[test_idx]
    X_tr_clean = X_tr.replace([np.inf, -np.inf], np.nan)
    X_te_clean = X_te.replace([np.inf, -np.inf], np.nan)
    medians = X_tr_clean.median(numeric_only=True)
    X_tr_clean = X_tr_clean.fillna(medians)
    X_te_clean = X_te_clean.fillna(medians)
    clf = DecisionTreeClassifier(random_state=seed, **DEEPFD_DT_HYPERPARAMS)
    clf.fit(X_tr_clean.to_numpy(dtype=float), y_tr.to_numpy())
    y_pred = clf.predict(X_te_clean.to_numpy(dtype=float))
    rep = classification_report(y_te, pd.Series(y_pred, index=y_te.index))
    importances = {col: float(imp) for col, imp in zip(X.columns, clf.feature_importances_)}
    classes = list(rep.confusion_labels)
    return DeepFDResult(
        corpus_name="",
        feature_source="",
        n_features=int(X.shape[1]),
        n_classes=len(classes),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        seed=int(seed),
        test_run_ids=[str(r) for r in X_te.index.tolist()],
        y_test_true=[str(v) for v in y_te.tolist()],
        y_test_pred=[str(v) for v in y_pred.tolist()],
        accuracy=float(rep.accuracy),
        macro_f1=float(rep.macro_f1),
        per_class_f1={c: float(rep.per_class_f1.get(c, 0.0)) for c in classes},
        per_class_precision={c: float(rep.per_class_precision.get(c, 0.0)) for c in classes},
        per_class_recall={c: float(rep.per_class_recall.get(c, 0.0)) for c in classes},
        per_class_support={c: int(rep.per_class_support.get(c, 0)) for c in classes},
        feature_columns=list(X.columns),
        feature_importances=importances,
    )


_MD_HEADER = """# DeepFD-inspired baseline — {corpus_name}

**Method:** Decision tree (sklearn ``DecisionTreeClassifier``) with hyperparameters
mirroring DeepFD (Cao, Lin, Yang, Liu, Tian — *DeepFD: Automated Fault Diagnosis
and Localization for Deep Learning Programs*, ICSE 2022).

**Honesty caveat.** This is not a literal replication of DeepFD. The original
work extracts gradient-derived dynamic features per batch; our corpus exposes
only per-epoch training history. We use the closest in-corpus surrogate:
the same engineered feature matrix our flat baseline sees, classified by a
decision tree with the same depth budget and pruning policy DeepFD reports.

**Hyperparameters:** ``max_depth={max_depth}``, ``min_samples_leaf={min_samples_leaf}``,
``criterion="{criterion}"``.

**Setup**
- Corpus: ``{corpus_name}``
- Feature source: ``{feature_source}``
- Features: {n_features}
- Train: {n_train}, Test: {n_test}, Classes: {n_classes}
- Test fold: ``StratifiedKFold(5, shuffle=True, random_state={seed}).first``
- Seed: {seed}

## Headline

| Metric | Value |
|---|---|
| Accuracy | {accuracy:.4f} |
| Macro-F1 | {macro_f1:.4f} |

## Per-class F1

| Class | F1 | Precision | Recall | Support |
|---|---|---|---|---|
"""


def _render_md(result: DeepFDResult) -> str:
    md = _MD_HEADER.format(
        corpus_name=result.corpus_name,
        feature_source=result.feature_source,
        n_features=result.n_features,
        n_train=result.n_train,
        n_test=result.n_test,
        n_classes=result.n_classes,
        seed=result.seed,
        accuracy=result.accuracy,
        macro_f1=result.macro_f1,
        max_depth=DEEPFD_DT_HYPERPARAMS["max_depth"],
        min_samples_leaf=DEEPFD_DT_HYPERPARAMS["min_samples_leaf"],
        criterion=DEEPFD_DT_HYPERPARAMS["criterion"],
    )
    classes = sorted(result.per_class_f1.keys())
    for c in classes:
        md += (
            f"| {c} | {result.per_class_f1.get(c, 0.0):.3f} | "
            f"{result.per_class_precision.get(c, 0.0):.3f} | "
            f"{result.per_class_recall.get(c, 0.0):.3f} | "
            f"{result.per_class_support.get(c, 0)} |\n"
        )
    top10 = sorted(result.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:10]
    md += "\n## Top-10 feature importances\n\n"
    md += "| Feature | Importance |\n|---|---|\n"
    for name, imp in top10:
        md += f"| {name} | {imp:.4f} |\n"
    md += (
        "\n## Comparison hint\n\n"
        "To position this baseline against the project's own policies, "
        "compare ``accuracy`` and per-class ``leakage F1`` against the "
        "headline tables in ``cross_corpus_summary.md`` "
        "(``flat``, ``cascade``, ``stacking``, ``stacking_with_conformal``).\n"
    )
    return md


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    _LOG.info(
        "DeepFD baseline on %s: %d features, %d rows, %d classes",
        ftable.corpus_name,
        X.shape[1],
        X.shape[0],
        y.nunique(),
    )
    res = _train_and_eval_deepfd(X, y, seed=args.seed)
    res = DeepFDResult(
        **{**res.__dict__, "corpus_name": ftable.corpus_name, "feature_source": ftable.source}
    )
    md_text = _render_md(res)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "method": "deepfd_inspired_decision_tree",
                    "method_origin": "Cao et al. ICSE 2022 (surrogate replication)",
                    "hyperparameters": DEEPFD_DT_HYPERPARAMS,
                    "corpus": res.corpus_name,
                    "feature_source": res.feature_source,
                    "n_features": res.n_features,
                    "n_classes": res.n_classes,
                    "n_train": res.n_train,
                    "n_test": res.n_test,
                    "seed": res.seed,
                    "accuracy": res.accuracy,
                    "macro_f1": res.macro_f1,
                    "per_class_f1": res.per_class_f1,
                    "per_class_precision": res.per_class_precision,
                    "per_class_recall": res.per_class_recall,
                    "per_class_support": res.per_class_support,
                    "test_run_ids": res.test_run_ids,
                    "y_test_true": res.y_test_true,
                    "y_test_pred": res.y_test_pred,
                    "top_feature_importances": dict(
                        sorted(
                            res.feature_importances.items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:20]
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("DeepFD-inspired baseline summary:")
    print(f"  corpus:    {res.corpus_name}")
    print(f"  n_features: {res.n_features}")
    print(f"  n_test:     {res.n_test}")
    print(f"  accuracy:   {res.accuracy:.4f}")
    print(f"  macro-F1:   {res.macro_f1:.4f}")
    leakage_f1 = res.per_class_f1.get("leakage")
    if leakage_f1 is not None:
        print(f"  leakage F1: {leakage_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
