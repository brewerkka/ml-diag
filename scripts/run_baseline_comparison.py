from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_RESULTS = _REPO_ROOT / "results"

_CORPUS_TAGS: tuple[str, ...] = ("8ds", "5ds", "3ds")


def _safe_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _extract_full_metrics(slice_full: dict) -> tuple[float | None, float | None, float | None]:
    if not isinstance(slice_full, dict):
        return (None, None, None)
    acc = slice_full.get("accuracy")
    mf1 = slice_full.get("macro_f1")
    leakage = slice_full.get("leakage_f1")
    if leakage is None:
        per_class = slice_full.get("per_class_f1") or {}
        if isinstance(per_class, dict):
            leakage = per_class.get("leakage")
    return (
        float(acc) if acc is not None else None,
        float(mf1) if mf1 is not None else None,
        float(leakage) if leakage is not None else None,
    )


def _flat_acc(corpus_tag: str) -> tuple[float | None, float | None, float | None]:
    hybrid_candidates = [
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_full_pipeline.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_conformal.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking_gbm_standalone.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking_gbm.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_di_proxy.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_llm.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}.json",
    ]
    for p in hybrid_candidates:
        d = _safe_load(p)
        if d is None:
            continue
        slc = (((d.get("baselines") or {}).get("flat") or {}).get("slices") or {}).get("full")
        triple = _extract_full_metrics(slc or {})
        if triple[0] is not None:
            return triple
    flat_candidates = [
        _RESULTS / f"flat_baseline_report{'' if corpus_tag == '8ds' else '_' + corpus_tag}.json",
        _RESULTS / f"flat_baseline_report_{corpus_tag}.json",
    ]
    for p in flat_candidates:
        d = _safe_load(p)
        if d is None:
            continue
        full = (d.get("reports") or {}).get("full") or {}
        triple = _extract_full_metrics(full)
        if triple[0] is not None:
            return triple
    return (None, None, None)


def _hybrid_policy(
    corpus_tag: str, policy_name: str
) -> tuple[float | None, float | None, float | None]:
    candidates = [
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_full_pipeline.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_conformal.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking_gbm_standalone.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking_gbm.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_stacking.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_di_proxy.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}_with_llm.json",
        _RESULTS / f"hybrid_evaluation_{corpus_tag}.json",
    ]
    for p in candidates:
        d = _safe_load(p)
        if d is None:
            continue
        if policy_name == "cascade":
            slc = (((d.get("baselines") or {}).get("cascade") or {}).get("slices") or {}).get(
                "full"
            )
        else:
            slc = (((d.get("hybrid") or {}).get(policy_name) or {}).get("slices") or {}).get("full")
        triple = _extract_full_metrics(slc or {})
        if triple[0] is not None:
            return triple
    return (None, None, None)


def _deepfd(corpus_tag: str) -> tuple[float | None, float | None, float | None]:
    p = _RESULTS / f"deepfd_baseline_{corpus_tag}.json"
    d = _safe_load(p)
    if d is None:
        return (None, None, None)
    leakage = (d.get("per_class_f1") or {}).get("leakage")
    return (
        float(d.get("accuracy")) if d.get("accuracy") is not None else None,
        float(d.get("macro_f1")) if d.get("macro_f1") is not None else None,
        float(leakage) if leakage is not None else None,
    )


def _fmt(x: float | None, fmt: str = ".4f") -> str:
    if x is None:
        return "—"
    return f"{x:{fmt}}"


def _build_per_corpus_row(corpus_tag: str) -> dict[str, dict[str, float | None]]:
    return {
        "flat": dict(zip(("acc", "macro_f1", "leakage_f1"), _flat_acc(corpus_tag))),
        "cascade": dict(
            zip(("acc", "macro_f1", "leakage_f1"), _hybrid_policy(corpus_tag, "cascade"))
        ),
        "stacking_gbm": dict(
            zip(("acc", "macro_f1", "leakage_f1"), _hybrid_policy(corpus_tag, "stacking"))
        ),
        "stacking_w_conformal": dict(
            zip(
                ("acc", "macro_f1", "leakage_f1"),
                _hybrid_policy(corpus_tag, "stacking_with_conformal"),
            )
        ),
        "deepfd_inspired": dict(zip(("acc", "macro_f1", "leakage_f1"), _deepfd(corpus_tag))),
    }


def _render_md(table: dict[str, dict[str, dict[str, float | None]]]) -> str:
    md = """# Baseline comparison — structured_diag vs DeepFD-inspired

Headline metrics on the canonical test fold of each corpus
(``StratifiedKFold(5, shuffle=True, random_state=0).first``).

## Method legend

* **flat** — Random Forest multiclass classifier on the engineered feature
  matrix (the project's flat baseline; reference point in §2.3 and §3.3).
* **cascade** — hierarchical 4-stage classifier (composed marginal
  probability, Stage 1–3 chain).
* **stacking_gbm** — Wolpert (1992) OOF stacking with a GBM meta-classifier
  (Stage 57).
* **stacking_w_conformal** — stacking + split-conformal abstain layer
  (Stage 58); accuracy on the FULL slice is identical to ``stacking_gbm``
  because conformal does not change argmax — it only adds an abstain
  signal. Conditional accuracy on the confident subset is reported in
  ``cross_corpus_summary.md``.
* **deepfd_inspired** — DeepFD-style (Cao et al., ICSE 2022) decision-tree
  classifier on the same feature matrix; honesty caveat: surrogate
  replication only — DeepFD's original work uses gradient-derived
  per-batch features unavailable in our corpus.

## Accuracy

| Corpus | flat | cascade | stacking_gbm | stacking_w_conformal | **deepfd_inspired** |
|---|---|---|---|---|---|
"""
    for corpus_tag in _CORPUS_TAGS:
        row = table.get(corpus_tag, {})
        md += (
            f"| {corpus_tag} | "
            f"{_fmt(row.get('flat', {}).get('acc'))} | "
            f"{_fmt(row.get('cascade', {}).get('acc'))} | "
            f"{_fmt(row.get('stacking_gbm', {}).get('acc'))} | "
            f"{_fmt(row.get('stacking_w_conformal', {}).get('acc'))} | "
            f"**{_fmt(row.get('deepfd_inspired', {}).get('acc'))}** |\n"
        )
    md += "\n## Macro-F1\n\n"
    md += (
        "| Corpus | flat | cascade | stacking_gbm | stacking_w_conformal | **deepfd_inspired** |\n"
    )
    md += "|---|---|---|---|---|---|\n"
    for corpus_tag in _CORPUS_TAGS:
        row = table.get(corpus_tag, {})
        md += (
            f"| {corpus_tag} | "
            f"{_fmt(row.get('flat', {}).get('macro_f1'))} | "
            f"{_fmt(row.get('cascade', {}).get('macro_f1'))} | "
            f"{_fmt(row.get('stacking_gbm', {}).get('macro_f1'))} | "
            f"{_fmt(row.get('stacking_w_conformal', {}).get('macro_f1'))} | "
            f"**{_fmt(row.get('deepfd_inspired', {}).get('macro_f1'))}** |\n"
        )
    md += "\n## Leakage F1 — primary class of interest\n\n"
    md += (
        "| Corpus | flat | cascade | stacking_gbm | stacking_w_conformal | **deepfd_inspired** |\n"
    )
    md += "|---|---|---|---|---|---|\n"
    for corpus_tag in _CORPUS_TAGS:
        row = table.get(corpus_tag, {})
        md += (
            f"| {corpus_tag} | "
            f"{_fmt(row.get('flat', {}).get('leakage_f1'))} | "
            f"{_fmt(row.get('cascade', {}).get('leakage_f1'))} | "
            f"{_fmt(row.get('stacking_gbm', {}).get('leakage_f1'))} | "
            f"{_fmt(row.get('stacking_w_conformal', {}).get('leakage_f1'))} | "
            f"**{_fmt(row.get('deepfd_inspired', {}).get('leakage_f1'))}** |\n"
        )
    md += (
        "\n## Interpretation guide\n\n"
        "* If ``deepfd_inspired`` ≤ ``flat``: the literature-anchored "
        "decision-tree classifier on the same feature space underperforms "
        "Random Forest — expected, since RF is an ensemble of trees and "
        "DeepFD's J48 single-tree advantage was its explainability, not "
        'raw accuracy. Comment in §3.3: "single-tree DeepFD-style '
        "classifier confirms our feature matrix is informative; ensemble "
        'improvements from RF onward are orthogonal to feature engineering."\n'
        "* If ``deepfd_inspired`` ≥ ``flat`` on leakage F1: DeepFD's "
        "single-tree pruning recovers a leakage-relevant signal that the "
        "RF averages out. This would be a *publishable observation* about "
        "DT vs RF for ML-fault diagnosis — worth highlighting.\n"
        "* If ``deepfd_inspired`` ≥ ``stacking_gbm``: the additional "
        "complexity of OOF stacking + LLM arbitrator + conformal abstain "
        "doesn't pay off vs a simple decision tree on engineered features. "
        'This would be a candid, defensible finding for the thesis — "complexity is not always justified".\n'
        "* All other patterns (typical case): the structured_diag "
        "pipeline strictly dominates the literature baseline on accuracy "
        "and matches or beats it on per-class F1.\n"
    )
    return md


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    table = {tag: _build_per_corpus_row(tag) for tag in _CORPUS_TAGS}
    md_text = _render_md(table)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md_text, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "comparison": "structured_diag vs deepfd_inspired",
                    "corpus_tags": list(_CORPUS_TAGS),
                    "policies_compared": [
                        "flat",
                        "cascade",
                        "stacking_gbm",
                        "stacking_w_conformal",
                        "deepfd_inspired",
                    ],
                    "test_fold": "StratifiedKFold(5, shuffle=True, random_state=0).first",
                    "results": table,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    print()
    print("Baseline comparison summary:")
    for tag in _CORPUS_TAGS:
        row = table.get(tag, {})
        print(f"  [{tag}]")
        for method in (
            "flat",
            "cascade",
            "stacking_gbm",
            "stacking_w_conformal",
            "deepfd_inspired",
        ):
            r = row.get(method, {})
            acc = r.get("acc")
            mf1 = r.get("macro_f1")
            lk = r.get("leakage_f1")
            acc_s = f"{acc:.4f}" if acc is not None else "—"
            mf1_s = f"{mf1:.4f}" if mf1 is not None else "—"
            lk_s = f"{lk:.4f}" if lk is not None else "—"
            print(f"    {method:24s} acc={acc_s} mf1={mf1_s} leak_f1={lk_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
