from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from structured_diag.benchmark import partition_corpus  # noqa: E402
from structured_diag.diagnosis import (  # noqa: E402
    POLICY_NAMES,
    HybridResolverConfig,
    build_stacking_diagnoses,
    build_stacking_with_conformal_diagnoses,
    cascade_marginal_proba,
    flat_proba_aligned,
    resolve_batch,
)
from structured_diag.diagnosis.conformal_layer import (  # noqa: E402
    calibrate_split_conformal,
    compute_meta_oof_probabilities,
    evaluate_conformal,
)
from structured_diag.diagnosis.oof_predictions import (  # noqa: E402
    STAGE_PROBA_COLS,
    read_oof_parquet,
)
from structured_diag.diagnosis.stacking_resolver import (  # noqa: E402
    train_stacking_meta,
)
from structured_diag.evaluation import (  # noqa: E402
    bootstrap_delta_ci,
    bootstrap_delta_ci_grouped,
    build_evidence,
    classification_report,
)
from structured_diag.features import build_feature_table  # noqa: E402
from structured_diag.labels import HEALTHY, LEAKAGE, PRIMARY_LABELS  # noqa: E402
from structured_diag.models import (  # noqa: E402
    load_cascade,
    slices_from_partition,
    train_flat_baseline,
)
from structured_diag.models.flat_baseline import _split_train_test  # noqa: E402
from structured_diag.models.inference import diagnose_batch  # noqa: E402
from structured_diag.models.model_zoo import default_zoo  # noqa: E402
from structured_diag.utils import setup_logging  # noqa: E402


def _confusion_counters(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    return {
        "leakage_called_healthy": int(((y_true == LEAKAGE) & (y_pred == HEALTHY)).sum()),
        "healthy_called_leakage": int(((y_true == HEALTHY) & (y_pred == LEAKAGE)).sum()),
        "healthy_called_faulty": int(((y_true == HEALTHY) & (y_pred != HEALTHY)).sum()),
    }


def _slice_metrics(
    *,
    y_true: pd.Series,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
) -> dict[str, Any]:
    label_order = list(PRIMARY_LABELS)
    rep = classification_report(
        y_true,
        y_pred,
        y_proba=proba,
        proba_classes=label_order if proba is not None else None,
        label_order=label_order,
    )
    cc = _confusion_counters(y_true.to_numpy(dtype=object), y_pred)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(rep.accuracy),
        "macro_f1": float(rep.macro_f1),
        "weighted_f1": float(rep.weighted_f1),
        "ece": (None if rep.ece is None else float(rep.ece)),
        "leakage_f1": float(rep.per_class_f1.get(LEAKAGE, 0.0)),
        "per_class_f1": {k: float(v) for k, v in rep.per_class_f1.items()},
        **cc,
    }


def _baseline_block(
    *,
    y_te: pd.Series,
    pred_full: np.ndarray,
    proba_full: np.ndarray,
    slices: dict[str, pd.Index],
) -> dict[str, Any]:
    pred_series = pd.Series(pred_full, index=y_te.index)
    proba_df = pd.DataFrame(proba_full, index=y_te.index, columns=list(PRIMARY_LABELS))
    out: dict[str, Any] = {"slices": {}}
    for slice_name, idx in slices.items():
        if len(idx) == 0:
            continue
        sub_y = y_te.loc[idx]
        sub_pred = pred_series.loc[idx].to_numpy(dtype=object)
        sub_proba = proba_df.loc[idx].to_numpy(dtype=float)
        out["slices"][slice_name] = _slice_metrics(
            y_true=sub_y,
            y_pred=sub_pred,
            proba=sub_proba,
        )
    return out


def _hybrid_block(
    *,
    y_te: pd.Series,
    flat_pred: pd.Series,
    cascade_pred: pd.Series,
    hybrid_pred: pd.Series,
    hybrid_paths: pd.Series,
    slices: dict[str, pd.Index],
    flat_proba_full: pd.DataFrame,
    cascade_proba_full: pd.DataFrame,
    alpha: float,
) -> dict[str, Any]:
    agreement_full = flat_pred == cascade_pred
    block: dict[str, Any] = {"slices": {}}
    for slice_name, idx in slices.items():
        if len(idx) == 0:
            continue
        sub_y = y_te.loc[idx]
        sub_flat = flat_pred.loc[idx]
        sub_casc = cascade_pred.loc[idx]
        sub_hyb = hybrid_pred.loc[idx]
        sub_paths = hybrid_paths.loc[idx]
        sub_agree = agreement_full.loc[idx]
        sub_flat_p = flat_proba_full.loc[idx].to_numpy(dtype=float)
        sub_casc_p = cascade_proba_full.loc[idx].to_numpy(dtype=float)
        mixed = alpha * sub_flat_p + (1.0 - alpha) * sub_casc_p
        m = _slice_metrics(
            y_true=sub_y,
            y_pred=sub_hyb.to_numpy(dtype=object),
            proba=mixed,
        )
        n = int(len(sub_y))
        n_agree = int(sub_agree.sum())
        n_disag = n - n_agree
        truth = sub_y.to_numpy(dtype=object)
        flat_arr = sub_flat.to_numpy(dtype=object)
        casc_arr = sub_casc.to_numpy(dtype=object)
        hyb_arr = sub_hyb.to_numpy(dtype=object)
        agree_mask = sub_agree.to_numpy(dtype=bool)
        disagree_mask = ~agree_mask

        def _safe_acc(mask: np.ndarray, pred_arr: np.ndarray) -> float | None:
            if not mask.any():
                return None
            return float((pred_arr[mask] == truth[mask]).mean())

        m.update(
            {
                "agreement_rate": (n_agree / n) if n else 0.0,
                "n_agreement": n_agree,
                "n_disagreement": n_disag,
                "accuracy_on_agreement": _safe_acc(agree_mask, hyb_arr),
                "accuracy_on_disagreement": _safe_acc(disagree_mask, hyb_arr),
                "flat_accuracy_on_disagreement": _safe_acc(disagree_mask, flat_arr),
                "cascade_accuracy_on_disagreement": _safe_acc(disagree_mask, casc_arr),
                "hybrid_accuracy_on_disagreement": _safe_acc(disagree_mask, hyb_arr),
                "resolution_paths": dict(Counter(sub_paths.tolist())),
            }
        )
        block["slices"][slice_name] = m
    return block


def _compute_cascade_stage_probs_test(cascade, X_test: pd.DataFrame) -> pd.DataFrame:
    from structured_diag.diagnosis.oof_predictions import (
        _build_cascade_predictions,
    )

    class _FakeRes:
        def __init__(self, sm):
            self.stage_name = sm.name
            self.model = sm.model
            self.classes = list(sm.classes)
            self.feature_columns = list(sm.feature_columns)

    s1_res = _FakeRes(cascade.stage1)
    s2_res = _FakeRes(cascade.stage2) if cascade.stage2 is not None else None
    s3d_res = _FakeRes(cascade.stage3_data) if cascade.stage3_data is not None else None
    s3o_res = _FakeRes(cascade.stage3_opt) if cascade.stage3_opt is not None else None
    _composed_unused, stage_probs = _build_cascade_predictions(
        stage1_res=s1_res,
        stage2_res=s2_res,
        stage3d_res=s3d_res,
        stage3o_res=s3o_res,
        X_holdout=X_test,
    )
    return stage_probs[list(STAGE_PROBA_COLS)]


def _arbitrator_outputs_for_test(
    *,
    arbitrator_diags: list | None,
    test_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cols = list(PRIMARY_LABELS)
    label_probs = pd.DataFrame(0.0, index=test_index, columns=cols)
    triggered = pd.Series(False, index=test_index)
    confidence = pd.Series(0.0, index=test_index)
    if arbitrator_diags is None:
        return label_probs, triggered, confidence
    if len(arbitrator_diags) != len(test_index):
        raise ValueError(
            f"arbitrator_diags has {len(arbitrator_diags)} entries but "
            f"test_index has {len(test_index)}; resolve_batch should "
            "produce one diagnosis per row in order."
        )
    from structured_diag.diagnosis.arbitrator import _soft_label_probabilities

    for rid, d in zip(test_index, arbitrator_diags):
        if d.resolution_path not in ("llm_arbitrated", "llm_skipped"):
            continue
        soft = _soft_label_probabilities(d.final_label, d.final_confidence)
        for c in cols:
            label_probs.at[rid, c] = float(soft.get(c, 0.0))
        triggered.loc[rid] = True
        confidence.loc[rid] = float(d.final_confidence)
    return label_probs, triggered, confidence


def _build_arbitration_stats(
    *,
    diags: list,
    run_ids: list[str],
    y_true: pd.Series,
    flat_pred: pd.Series,
    cascade_pred: pd.Series,
) -> dict[str, Any]:
    n_calls = 0
    n_cached = 0
    n_template_fallback = 0
    backend_counts: Counter = Counter()
    trigger_counts: Counter = Counter()
    chosen_source_counts: Counter = Counter()
    decisions: list[dict[str, Any]] = []
    for d, rid in zip(diags, run_ids):
        if d.resolution_path not in ("llm_arbitrated", "llm_skipped"):
            continue
        n_calls += 1
        if d.arbitration_cached:
            n_cached += 1
        backend = d.arbitration_backend or "?"
        if backend == "template":
            n_template_fallback += 1
        backend_counts[backend] += 1
        trigger_counts[d.arbitration_trigger or "?"] += 1
        chosen_source_counts[d.arbitration_chosen_source or "?"] += 1
        gt = str(y_true.loc[rid])
        decision_record = {
            "run_id": str(rid),
            "trigger": d.arbitration_trigger,
            "flat_label": d.flat_label,
            "cascade_label": d.cascade_label,
            "chosen_label": d.final_label,
            "chosen_source": d.arbitration_chosen_source,
            "confidence": float(d.final_confidence),
            "ground_truth": gt,
            "correct": (d.final_label == gt),
            "reasoning_excerpt": ((d.arbitration_reasoning or "")[:240]),
            "backend": d.arbitration_backend,
            "cached": d.arbitration_cached,
        }
        decisions.append(decision_record)
    return {
        "n_calls": int(n_calls),
        "n_cached": int(n_cached),
        "n_template_fallback": int(n_template_fallback),
        "backend_used_counts": dict(backend_counts),
        "trigger_breakdown": dict(trigger_counts),
        "chosen_source_counts": dict(chosen_source_counts),
        "decisions": decisions,
    }


def _load_run_to_entry_map(corpus_path: Path) -> dict[str, str]:
    manifest_path = corpus_path / "corpus.manifest.json"
    if not manifest_path.is_file():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in data.get("entries", []) or []:
        eid = str(entry.get("entry_id", ""))
        for rid in entry.get("run_ids", []) or []:
            out[str(rid)] = eid
    return out


def _vs_baseline_bootstrap(
    *,
    y_true: pd.Series,
    pred_baseline: np.ndarray,
    pred_hybrid: np.ndarray,
    n_bootstrap: int,
    seed: int,
    group_ids: Sequence[Any] | None = None,
    n_comparisons: int = 1,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_f1"):
        ci = bootstrap_delta_ci(
            y_true.to_numpy(dtype=object),
            pred_baseline,
            pred_hybrid,
            metric=metric,
            n_bootstrap=n_bootstrap,
            seed=seed,
            label_order=list(PRIMARY_LABELS),
            n_comparisons=n_comparisons,
        )
        block: dict[str, float] = {
            "delta_point": float(ci["delta_point"]),
            "ci_low": float(ci["delta_ci_low"]),
            "ci_high": float(ci["delta_ci_high"]),
            "p_b_better": float(ci["p_b_better"]),
        }
        if "alpha_bonferroni" in ci:
            block["family_wise_correction"] = {
                "alpha_bonferroni": float(ci["alpha_bonferroni"]),
                "n_comparisons": int(ci["n_comparisons"]),
                "p_b_better_bonferroni_significant": bool(ci["p_b_better_bonferroni_significant"]),
            }
        out[metric] = block
        if group_ids is not None:
            ci_g = bootstrap_delta_ci_grouped(
                y_true.to_numpy(dtype=object),
                pred_baseline,
                pred_hybrid,
                group_ids,
                metric=metric,
                n_bootstrap=n_bootstrap,
                seed=seed,
                label_order=list(PRIMARY_LABELS),
                n_comparisons=n_comparisons,
            )
            grouped = {
                "delta_point": float(ci_g["delta_point"]),
                "ci_low": float(ci_g["delta_ci_low"]),
                "ci_high": float(ci_g["delta_ci_high"]),
                "p_b_better": float(ci_g["p_b_better"]),
                "n_groups": int(ci_g["n_groups"]),
            }
            if "alpha_bonferroni" in ci_g:
                grouped["family_wise_correction"] = {
                    "alpha_bonferroni": float(ci_g["alpha_bonferroni"]),
                    "n_comparisons": int(ci_g["n_comparisons"]),
                    "p_b_better_bonferroni_significant": bool(
                        ci_g["p_b_better_bonferroni_significant"]
                    ),
                }
            out[f"{metric}_grouped"] = grouped
    return out


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _render_markdown(
    *,
    corpus_name: str,
    n_test: int,
    flat_model: str,
    cascade_threshold: float,
    alpha: float,
    policies: list[str],
    baselines: dict[str, Any],
    hybrid: dict[str, Any],
) -> str:
    out: list[str] = []
    out.append(f"# Hybrid evaluation — {corpus_name}")
    out.append("")
    out.append(f"- n test = {n_test}")
    out.append(f"- flat model: `{flat_model}`")
    out.append(f"- cascade Stage 1 threshold: {cascade_threshold:.2f}")
    out.append(f"- α (confidence_weighted): {alpha:.2f}")
    out.append(f"- policies: {', '.join(f'`{p}`' for p in policies)}")
    out.append("")
    rows = [("flat", baselines["flat"]), ("cascade", baselines["cascade"])]
    for p in policies:
        rows.append((p, hybrid[p]))
    out.append("## Per-slice metrics")
    out.append("")
    out.append(
        "| contour | slice | n | acc | macro-F1 | ECE | leakage F1 | L→H | H→leak | H→faulty |"
    )
    out.append("|---|---|--:|---:|---:|---:|---:|--:|--:|--:|")
    for label, block in rows:
        for slice_name, m in (block.get("slices") or {}).items():
            out.append(
                f"| `{label}` | `{slice_name}` | {m['n_samples']} "
                f"| {_fmt(m['accuracy'])} | {_fmt(m['macro_f1'])} "
                f"| {_fmt(m['ece'])} | {_fmt(m['leakage_f1'])} "
                f"| {m['leakage_called_healthy']} "
                f"| {m['healthy_called_leakage']} "
                f"| {m['healthy_called_faulty']} |"
            )
    out.append("")
    out.append("## Disagreement subset analysis (slice `full`)")
    out.append("")
    out.append(
        "| policy | agreement rate | n agree | n disag | "
        "flat acc on disag | cascade acc on disag | hybrid acc on disag | "
        "resolution paths |"
    )
    out.append("|---|---:|--:|--:|---:|---:|---:|---|")
    for p in policies:
        m = (hybrid[p].get("slices") or {}).get("full")
        if not m:
            continue
        paths = ", ".join(f"{k}={v}" for k, v in (m.get("resolution_paths") or {}).items())
        out.append(
            f"| `{p}` | {_fmt(m.get('agreement_rate'), 3)} "
            f"| {m.get('n_agreement', 0)} | {m.get('n_disagreement', 0)} "
            f"| {_fmt(m.get('flat_accuracy_on_disagreement'))} "
            f"| {_fmt(m.get('cascade_accuracy_on_disagreement'))} "
            f"| {_fmt(m.get('hybrid_accuracy_on_disagreement'))} "
            f"| {paths} |"
        )
    out.append("")
    if "llm_arbitrate" in hybrid:
        stats = (hybrid["llm_arbitrate"] or {}).get("arbitration_stats") or {}
        decisions = stats.get("decisions") or []
        out.append("## Arbitration trace (Stage 56)")
        out.append("")
        out.append(
            f"- n_calls = **{stats.get('n_calls', 0)}** "
            f"(cached = {stats.get('n_cached', 0)}, "
            f"template_fallback = {stats.get('n_template_fallback', 0)})"
        )
        out.append(
            "- backend usage: "
            + ", ".join(f"`{k}`={v}" for k, v in (stats.get("backend_used_counts") or {}).items())
        )
        out.append(
            "- trigger breakdown: "
            + ", ".join(f"`{k}`={v}" for k, v in (stats.get("trigger_breakdown") or {}).items())
        )
        out.append(
            "- chosen_source: "
            + ", ".join(f"`{k}`={v}" for k, v in (stats.get("chosen_source_counts") or {}).items())
        )
        n_correct = sum(1 for r in decisions if r.get("correct"))
        out.append(
            f"- arbitrator accuracy on consulted rows: "
            f"**{n_correct}/{len(decisions)}** "
            f"({n_correct / max(1, len(decisions)):.3f})"
        )
        out.append("")
        out.append(
            "| run_id | trigger | flat | cascade | LLM choice | source | gt | correct? | reasoning excerpt |"
        )
        out.append("|---|---|---|---|---|---|---|:-:|---|")
        for d in decisions:
            mark = "✓" if d.get("correct") else "✗"
            out.append(
                f"| `{d.get('run_id')}` | {d.get('trigger') or '—'} "
                f"| {d.get('flat_label')} | {d.get('cascade_label')} "
                f"| {d.get('chosen_label')} | {d.get('chosen_source') or '—'} "
                f"| {d.get('ground_truth')} | {mark} "
                f"| {d.get('reasoning_excerpt') or ''} |"
            )
        out.append("")
    if "stacking" in hybrid:
        meta = (hybrid["stacking"] or {}).get("stacking_meta_model") or {}
        out.append("## Stacking meta-classifier (Stage 57)")
        out.append("")
        out.append(
            f"- classifier: `{meta.get('classifier')}` "
            f"({meta.get('n_features')} features, "
            f"{meta.get('n_train_rows')} OOF rows)"
        )
        out.append(
            f"- 5-fold CV macro-F1 on OOF features: **{_fmt(meta.get('cv_score_macro_f1'))}**"
        )
        out.append("")
        out.append("### Top-10 feature importances")
        out.append("")
        out.append("| feature | importance |")
        out.append("|---|---:|")
        for name, score in meta.get("feature_importances_top10") or []:
            out.append(f"| `{name}` | {_fmt(score)} |")
        out.append("")
        vs_arb = (hybrid["stacking"] or {}).get("vs_llm_arbitrate_bootstrap")
        if vs_arb:
            out.append("### Stacking vs LLM arbitrator (paired bootstrap)")
            out.append("")
            for metric in ("accuracy", "macro_f1"):
                v = vs_arb.get(metric) or {}
                out.append(
                    f"- **{metric}** Δ = {_fmt(v.get('delta_point'))} "
                    f"[{_fmt(v.get('ci_low'))}, {_fmt(v.get('ci_high'))}], "
                    f"P(stacking > llm_arbitrate) = {_fmt(v.get('p_b_better'), 3)}"
                )
            out.append("")
    if "stacking_with_conformal" in hybrid:
        cb = (hybrid["stacking_with_conformal"] or {}).get("conformal") or {}
        cal = cb.get("calibrator") or {}
        m = cb.get("metrics") or {}
        out.append("## Conformal abstain layer (Stage 58)")
        out.append("")
        out.append(
            f"- target coverage 1−α = {_fmt(cal.get('target_coverage'), 3)} "
            f"(α = {_fmt(cal.get('alpha'), 3)})"
        )
        out.append(
            f"- empirical coverage = {_fmt(m.get('empirical_coverage'), 3)} "
            f"(gap vs target = {_fmt(m.get('coverage_gap'), 3)})"
        )
        out.append(
            f"- abstain rate = {_fmt(m.get('abstain_rate'), 3)} "
            f"({m.get('n_abstained', 0)}/{m.get('n_test', 0)} test rows)"
        )
        out.append(
            f"- conditional accuracy on confident subset "
            f"(n={m.get('n_confident', 0)}) = "
            f"{_fmt(m.get('conditional_accuracy'), 3)}"
        )
        out.append(f"- average prediction set size = {_fmt(m.get('average_set_size'), 3)}")
        out.append(
            f"- calibrator: q_hat = {_fmt(cal.get('quantile'), 4)}, "
            f"score = `{cal.get('score_method')}`, "
            f"n_calibration = {cal.get('n_calibration')}"
        )
        out.append("")
        out.append("### Per-class breakdown")
        out.append("")
        out.append("| class | n | coverage | abstain rate |")
        out.append("|---|---:|---:|---:|")
        n_per = m.get("per_class_n") or {}
        cov_per = m.get("per_class_coverage") or {}
        ab_per = m.get("per_class_abstain_rate") or {}
        for cls in PRIMARY_LABELS:
            out.append(
                f"| `{cls}` | {n_per.get(cls, 0)} "
                f"| {_fmt(cov_per.get(cls), 3)} "
                f"| {_fmt(ab_per.get(cls), 3)} |"
            )
        out.append("")
        abstained = cb.get("abstained_examples") or []
        out.append(f"### Abstained examples (showing up to 5 of {len(abstained)})")
        out.append("")
        if abstained:
            out.append("| run_id | true | prediction_set | argmax | argmax_p | set_size |")
            out.append("|---|---|---|---|---:|---:|")
            for ex in abstained[:5]:
                ps = "{" + ", ".join(ex.get("prediction_set") or []) + "}"
                out.append(
                    f"| `{ex.get('run_id')}` | {ex.get('true_label')} "
                    f"| {ps} | {ex.get('argmax')} "
                    f"| {_fmt(ex.get('argmax_proba'))} "
                    f"| {ex.get('set_size')} |"
                )
        else:
            out.append("_(no abstained rows)_")
        out.append("")
    out.append("## Bootstrap CIs (slice `full`, paired, n_bootstrap=1000, α=0.05)")
    out.append("")
    out.append(
        "| policy | metric | vs flat Δ (CI) | P(hybrid > flat) | "
        "vs cascade Δ (CI) | P(hybrid > cascade) |"
    )
    out.append("|---|---|---|---:|---|---:|")
    for p in policies:
        vs_flat = hybrid[p].get("vs_flat_bootstrap") or {}
        vs_casc = hybrid[p].get("vs_cascade_bootstrap") or {}
        for metric in ("accuracy", "macro_f1"):
            f = vs_flat.get(metric) or {}
            c = vs_casc.get(metric) or {}
            out.append(
                f"| `{p}` | {metric} "
                f"| {_fmt(f.get('delta_point'))} "
                f"[{_fmt(f.get('ci_low'))}, {_fmt(f.get('ci_high'))}] "
                f"| {_fmt(f.get('p_b_better'), 3)} "
                f"| {_fmt(c.get('delta_point'))} "
                f"[{_fmt(c.get('ci_low'))}, {_fmt(c.get('ci_high'))}] "
                f"| {_fmt(c.get('p_b_better'), 3)} |"
            )
    out.append("")
    return "\n".join(out)


def _parse_policies(s: str) -> list[str]:
    items = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in items if p not in POLICY_NAMES]
    if bad:
        raise argparse.ArgumentTypeError(
            f"Unknown policy/policies: {bad}; expected from {list(POLICY_NAMES)}"
        )
    return items


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--hier-artifacts", required=True, type=Path)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument(
        "--policies",
        type=_parse_policies,
        default=list(POLICY_NAMES),
        help=f"Comma-separated subset of {list(POLICY_NAMES)} (default: all three).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Mixing weight for `confidence_weighted` (α·p_flat + (1-α)·p_cascade); default 0.5.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--no-catboost", action="store_true")
    p.add_argument(
        "--drop-feature-prefix",
        default=None,
        help="Comma-separated feature-name prefixes to drop "
        "from X before training/prediction. Should match "
        "the value passed to run_hierarchical_train.py "
        "for fair Stage 54 / Stage 60 ablation.",
    )
    p.add_argument(
        "--group-bootstrap",
        action="store_true",
        help="Also compute paired bootstrap by entry_id "
        "(group bootstrap) on top of the standard row "
        "bootstrap. Adds `*_grouped` blocks to every "
        "`vs_*_bootstrap` entry without changing the "
        "headline `p_b_better` values. Group ids are "
        "read from corpus.manifest.json.",
    )
    p.add_argument(
        "--bonferroni-n-comparisons",
        type=int,
        default=1,
        help="Family size for Bonferroni correction. When "
        "> 1, every bootstrap block gets an extra "
        "`family_wise_correction` field with α/m and "
        "a strict-significance flag. Default 1 = no "
        "extra field (matches historical reports).",
    )
    p.add_argument(
        "--arbitrator-backend",
        default="auto",
        choices=["auto", "groq", "ollama", "template"],
        help="LLM backend chain for the `llm_arbitrate` policy. "
        "`auto` tries groq → ollama → template.",
    )
    p.add_argument(
        "--arbitrator-low-conf",
        type=float,
        default=0.0,
        help="Trigger arbitration on agreement when "
        "min(flat_conf, cascade_conf) < this threshold. "
        "Default 0.0 = disagreement-only (empirically the "
        "low-conf branch is a net-negative on 8ds: cascade's "
        "soft marginals routinely sit below 0.65 even on "
        "correct calls, so the LLM gets dragged in too "
        "often and breaks more than it fixes).",
    )
    p.add_argument(
        "--arbitrator-cache",
        type=Path,
        default=Path(".cache/arbitrator"),
        help="Persistent cache dir for arbitration responses.",
    )
    p.add_argument(
        "--stacking-classifier",
        default="lr",
        choices=["lr", "gbm"],
        help="Meta-classifier for the `stacking` policy (default: lr — L2 LogisticRegression).",
    )
    p.add_argument(
        "--stacking-oof",
        type=Path,
        default=None,
        help="Path to the OOF predictions parquet (required when `stacking` is in --policies).",
    )
    p.add_argument(
        "--conformal-alpha",
        type=float,
        default=0.05,
        help="1 − target marginal coverage (default 0.05 = 95%%).",
    )
    p.add_argument(
        "--conformal-score-method",
        default="lac",
        choices=["lac", "aps"],
        help="Nonconformity score (default: lac).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    setup_logging(level=args.log_level)
    ftable = build_feature_table(args.corpus)
    X, y = ftable.aligned_xy()
    if args.drop_feature_prefix:
        prefixes = tuple(p.strip() for p in args.drop_feature_prefix.split(",") if p.strip())
        if prefixes:
            dropped = [c for c in X.columns if c.startswith(prefixes)]
            if dropped:
                X = X.drop(columns=dropped)
                print(f"Ablation: dropped {len(dropped)} features matching prefixes {prefixes}.")
    partition = partition_corpus(args.corpus, skip_broken=True)
    pt = partition.table.copy()
    train_idx, test_idx = _split_train_test(X, y, seed=args.seed)
    test_run_ids = X.index[test_idx]
    X_te = X.loc[test_run_ids]
    y_te = y.loc[test_run_ids]
    slices = slices_from_partition(pt, X.index, holdout_index=test_run_ids)
    zoo = default_zoo(include_catboost=not args.no_catboost)
    flat = train_flat_baseline(X, y, seed=args.seed, candidate_models=zoo)
    print(f"Trained flat baseline: {flat.model_name}")
    cascade = load_cascade(args.hier_artifacts)
    print(
        f"Loaded cascade: stages={cascade.stages_available}, "
        f"stage1 threshold={cascade.stage1_healthy_threshold:.3f} "
        f"({cascade.threshold_source})"
    )
    flat_proba_df = flat_proba_aligned(flat, X_te)
    cascade_proba_df = cascade_marginal_proba(cascade, X_te)
    flat_pred = flat_proba_df.idxmax(axis=1)
    cascade_pred = cascade_proba_df.idxmax(axis=1)
    print(
        f"Built proba frames: flat shape={tuple(flat_proba_df.shape)}, "
        f"cascade shape={tuple(cascade_proba_df.shape)}"
    )
    baselines = {
        "flat": _baseline_block(
            y_te=y_te,
            pred_full=flat_pred.to_numpy(dtype=object),
            proba_full=flat_proba_df.to_numpy(dtype=float),
            slices=slices,
        ),
        "cascade": _baseline_block(
            y_te=y_te,
            pred_full=cascade_pred.to_numpy(dtype=object),
            proba_full=cascade_proba_df.to_numpy(dtype=float),
            slices=slices,
        ),
    }
    evidence_by_run: dict[str, Any] | None = None
    if "llm_arbitrate" in args.policies:
        print(f"Building structured evidence for {len(X_te)} test rows…")
        diags_for_evidence = diagnose_batch(cascade, X_te)
        evidence_by_run = {}
        for d in diags_for_evidence:
            row = X_te.loc[d.run_id] if d.run_id in X_te.index else None
            if row is None:
                continue
            evidence_by_run[str(d.run_id)] = build_evidence(
                diagnosis=d,
                feature_row=row,
                cascade=cascade,
            )
        print(f"  evidence ready for {len(evidence_by_run)} runs")
    stacking_meta = None
    stacking_inputs_test: dict[str, Any] | None = None
    arbitrator_diags_for_stacking: list | None = None
    conformal_calibrator = None
    needs_stacking = "stacking" in args.policies or "stacking_with_conformal" in args.policies
    if needs_stacking:
        if args.stacking_oof is None or not Path(args.stacking_oof).is_file():
            print(
                f"ERROR: --stacking-oof points to a missing file: "
                f"{args.stacking_oof}. Run scripts/generate_oof_predictions.py "
                "first.",
                file=sys.stderr,
            )
            return 2
        print(f"Loading OOF predictions from {args.stacking_oof}…")
        oof = read_oof_parquet(args.stacking_oof)
        y_oof = y.loc[oof.index()]
        print(f"  OOF rows: {len(oof.index())} (matched to y_train)")
        print(f"Training stacking meta-classifier ({args.stacking_classifier})…")
        stacking_meta = train_stacking_meta(
            oof,
            y_oof,
            classifier=str(args.stacking_classifier),
            seed=int(args.seed),
        )
        print(f"  CV macro-F1 (5-fold) = {stacking_meta.cv_score_macro_f1:.4f}")
        print("Building stacking inputs on test fold…")
        cascade_stage_probs_test = _compute_cascade_stage_probs_test(cascade, X_te)
        stacking_inputs_test = {
            "cascade_stage_probs": cascade_stage_probs_test,
        }
        if "stacking_with_conformal" in args.policies:
            print(
                "Computing OOF-of-OOF meta probabilities (5-fold inner CV) "
                "for honest split-conformal calibration…"
            )
            meta_oof_proba = compute_meta_oof_probabilities(
                oof=oof,
                y_train=y_oof,
                classifier=str(args.stacking_classifier),
                seed=int(args.seed),
                n_folds=5,
            )
            conformal_calibrator = calibrate_split_conformal(
                proba_oof=meta_oof_proba,
                y_oof=y_oof,
                alpha=float(args.conformal_alpha),
                score_method=str(args.conformal_score_method),  # type: ignore[arg-type]
            )
            print(
                f"  conformal calibrator: q_hat={conformal_calibrator.quantile:.4f}, "
                f"alpha={conformal_calibrator.alpha}, "
                f"n_cal={conformal_calibrator.n_calibration}"
            )
    policy_run_order = list(args.policies)
    if "llm_arbitrate" in policy_run_order:
        for st in ("stacking", "stacking_with_conformal"):
            if st in policy_run_order and (
                policy_run_order.index(st) < policy_run_order.index("llm_arbitrate")
            ):
                policy_run_order.remove("llm_arbitrate")
                policy_run_order.insert(
                    policy_run_order.index(st),
                    "llm_arbitrate",
                )
                break
    hybrid: dict[str, Any] = {}
    for policy in policy_run_order:
        cfg = HybridResolverConfig(
            policy=policy,
            alpha=float(args.alpha),
            arbitrator_backend=str(args.arbitrator_backend),
            arbitrator_low_conf_trigger=float(args.arbitrator_low_conf),
            arbitrator_cache=Path(args.arbitrator_cache),
            stacking_meta_model=(
                stacking_meta if policy in ("stacking", "stacking_with_conformal") else None
            ),
            stacking_oof_path=Path(args.stacking_oof) if args.stacking_oof else None,
            stacking_classifier=str(args.stacking_classifier),
            conformal_calibrator=(
                conformal_calibrator if policy == "stacking_with_conformal" else None
            ),
            conformal_alpha=float(args.conformal_alpha),
        )
        if policy in ("stacking", "stacking_with_conformal"):
            arb_label_probs, arb_triggered, arb_conf = _arbitrator_outputs_for_test(
                arbitrator_diags=arbitrator_diags_for_stacking,
                test_index=X_te.index,
            )
            if policy == "stacking_with_conformal":
                diags = build_stacking_with_conformal_diagnoses(
                    config=cfg,
                    flat_proba=flat_proba_df,
                    cascade_proba=cascade_proba_df,
                    cascade_stage_probs=stacking_inputs_test["cascade_stage_probs"],
                    arbitrator_label_probs=arb_label_probs,
                    arbitrator_triggered=arb_triggered,
                    arbitrator_confidence=arb_conf,
                )
            else:
                diags = build_stacking_diagnoses(
                    config=cfg,
                    flat_proba=flat_proba_df,
                    cascade_proba=cascade_proba_df,
                    cascade_stage_probs=stacking_inputs_test["cascade_stage_probs"],
                    arbitrator_label_probs=arb_label_probs,
                    arbitrator_triggered=arb_triggered,
                    arbitrator_confidence=arb_conf,
                )
        else:
            diags = resolve_batch(
                flat_proba=flat_proba_df,
                cascade_proba=cascade_proba_df,
                config=cfg,
                evidence_by_run=evidence_by_run if policy == "llm_arbitrate" else None,
            )
            if policy == "llm_arbitrate":
                arbitrator_diags_for_stacking = diags
        hyb_pred = pd.Series(
            [d.final_label for d in diags],
            index=X_te.index,
        )
        hyb_paths = pd.Series(
            [d.resolution_path for d in diags],
            index=X_te.index,
        )
        block = _hybrid_block(
            y_te=y_te,
            flat_pred=flat_pred,
            cascade_pred=cascade_pred,
            hybrid_pred=hyb_pred,
            hybrid_paths=hyb_paths,
            slices=slices,
            flat_proba_full=flat_proba_df,
            cascade_proba_full=cascade_proba_df,
            alpha=float(args.alpha),
        )
        full_idx = slices.get("full")
        if full_idx is not None and len(full_idx) > 0:
            group_ids: list[Any] | None = None
            if getattr(args, "group_bootstrap", False):
                run_to_entry = _load_run_to_entry_map(args.corpus)
                if run_to_entry:
                    full_ids = list(full_idx)
                    group_ids = [run_to_entry.get(str(rid), str(rid)) for rid in full_ids]
            block["vs_flat_bootstrap"] = _vs_baseline_bootstrap(
                y_true=y_te.loc[full_idx],
                pred_baseline=flat_pred.loc[full_idx].to_numpy(dtype=object),
                pred_hybrid=hyb_pred.loc[full_idx].to_numpy(dtype=object),
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
                group_ids=group_ids,
                n_comparisons=int(args.bonferroni_n_comparisons),
            )
            block["vs_cascade_bootstrap"] = _vs_baseline_bootstrap(
                y_true=y_te.loc[full_idx],
                pred_baseline=cascade_pred.loc[full_idx].to_numpy(dtype=object),
                pred_hybrid=hyb_pred.loc[full_idx].to_numpy(dtype=object),
                n_bootstrap=int(args.n_bootstrap),
                seed=int(args.seed),
                group_ids=group_ids,
                n_comparisons=int(args.bonferroni_n_comparisons),
            )
        if policy == "llm_arbitrate":
            block["arbitration_stats"] = _build_arbitration_stats(
                diags=diags,
                run_ids=list(X_te.index),
                y_true=y_te,
                flat_pred=flat_pred,
                cascade_pred=cascade_pred,
            )
        if policy == "stacking_with_conformal" and conformal_calibrator is not None:
            block["stacking_meta_model"] = stacking_meta.to_summary_dict()
            conformal_results = []
            for d in diags:
                from structured_diag.diagnosis.conformal_layer import ConformalResult

                conformal_results.append(
                    ConformalResult(
                        run_id=str(d.flat_proba and ""),
                        prediction_set=list(d.conformal_prediction_set or []),
                        set_size=int(d.conformal_set_size or 1),
                        point_prediction=str(d.final_label),
                        point_confidence=float(d.final_confidence),
                        is_abstained=bool(d.conformal_is_abstained),
                        nonconformity=float(d.conformal_nonconformity or 0.0),
                        proba=dict(d.stacking_probabilities or {}),
                    )
                )
            from dataclasses import replace as _replace

            conformal_results = [
                _replace(cr, run_id=str(rid)) for cr, rid in zip(conformal_results, X_te.index)
            ]
            metrics = evaluate_conformal(
                results=conformal_results,
                y_test=y_te,
            )
            metrics["target_coverage"] = float(1.0 - conformal_calibrator.alpha)
            metrics["coverage_gap"] = float(
                metrics["empirical_coverage"] - metrics["target_coverage"]
            )
            abstained_examples = []
            for cr, rid in zip(conformal_results, X_te.index):
                if not cr.is_abstained:
                    continue
                abstained_examples.append(
                    {
                        "run_id": str(rid),
                        "true_label": str(y_te.loc[rid]),
                        "prediction_set": list(cr.prediction_set),
                        "argmax": str(cr.point_prediction),
                        "argmax_proba": float(cr.point_confidence),
                        "set_size": int(cr.set_size),
                        "nonconformity": float(cr.nonconformity),
                    }
                )
            block["conformal"] = {
                "calibrator": conformal_calibrator.to_summary_dict(),
                "metrics": metrics,
                "abstained_examples": abstained_examples,
            }
        if policy == "stacking" and stacking_meta is not None:
            block["stacking_meta_model"] = stacking_meta.to_summary_dict()
            if "llm_arbitrate" in hybrid and full_idx is not None and len(full_idx) > 0:
                llm_diags = arbitrator_diags_for_stacking or []
                llm_pred_series = (
                    pd.Series(
                        [d.final_label for d in llm_diags],
                        index=list(X_te.index)[: len(llm_diags)],
                    )
                    if llm_diags
                    else None
                )
                if llm_pred_series is not None:
                    aligned = llm_pred_series.reindex(full_idx)
                    if not aligned.isna().any():
                        block["vs_llm_arbitrate_bootstrap"] = _vs_baseline_bootstrap(
                            y_true=y_te.loc[full_idx],
                            pred_baseline=aligned.to_numpy(dtype=object),
                            pred_hybrid=hyb_pred.loc[full_idx].to_numpy(dtype=object),
                            n_bootstrap=int(args.n_bootstrap),
                            seed=int(args.seed),
                            group_ids=group_ids,
                            n_comparisons=int(args.bonferroni_n_comparisons),
                        )
        hybrid[policy] = block
    hybrid = {p: hybrid[p] for p in args.policies if p in hybrid}
    md = _render_markdown(
        corpus_name=ftable.corpus_name,
        n_test=int(len(X_te)),
        flat_model=flat.model_name,
        cascade_threshold=float(cascade.stage1_healthy_threshold),
        alpha=float(args.alpha),
        policies=args.policies,
        baselines=baselines,
        hybrid=hybrid,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    if args.out_json is not None:
        payload = {
            "corpus": ftable.corpus_name,
            "test_n": int(len(X_te)),
            "policies": args.policies,
            "alpha": float(args.alpha),
            "flat_model": flat.model_name,
            "cascade_default_threshold": float(cascade.stage1_healthy_threshold),
            "baselines": baselines,
            "hybrid": hybrid,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote JSON     -> {args.out_json}")
    full_flat = baselines["flat"]["slices"].get("full") or {}
    full_casc = baselines["cascade"]["slices"].get("full") or {}
    print()
    full_idx_for_print = slices.get("full")
    print(f"Full-slice summary (n={0 if full_idx_for_print is None else len(full_idx_for_print)}):")
    print(
        f"{'policy':<24} {'acc':>7} {'Δacc(flat)':>11} {'Δacc(casc)':>11} "
        f"{'macro_f1':>9} {'Δmf1(flat)':>11} {'L→H':>4} {'agree':>6}"
    )
    print(
        f"{'flat':<24} {full_flat.get('accuracy', 0):>7.4f} "
        f"{0.0:>+11.4f} {0.0:>+11.4f} "
        f"{full_flat.get('macro_f1', 0):>9.4f} {0.0:>+11.4f} "
        f"{full_flat.get('leakage_called_healthy', 0):>4d} {'—':>6}"
    )
    print(
        f"{'cascade':<24} {full_casc.get('accuracy', 0):>7.4f} "
        f"{full_casc.get('accuracy', 0) - full_flat.get('accuracy', 0):>+11.4f} "
        f"{0.0:>+11.4f} "
        f"{full_casc.get('macro_f1', 0):>9.4f} "
        f"{full_casc.get('macro_f1', 0) - full_flat.get('macro_f1', 0):>+11.4f} "
        f"{full_casc.get('leakage_called_healthy', 0):>4d} {'—':>6}"
    )
    for policy in args.policies:
        m = (hybrid[policy].get("slices") or {}).get("full") or {}
        d_flat = m.get("accuracy", 0) - full_flat.get("accuracy", 0)
        d_casc = m.get("accuracy", 0) - full_casc.get("accuracy", 0)
        d_mf1 = m.get("macro_f1", 0) - full_flat.get("macro_f1", 0)
        print(
            f"{policy:<24} {m.get('accuracy', 0):>7.4f} "
            f"{d_flat:>+11.4f} {d_casc:>+11.4f} "
            f"{m.get('macro_f1', 0):>9.4f} {d_mf1:>+11.4f} "
            f"{m.get('leakage_called_healthy', 0):>4d} "
            f"{m.get('agreement_rate', 0):>6.3f}"
        )
    if "stacking_with_conformal" in args.policies and conformal_calibrator is not None:
        cb = (hybrid.get("stacking_with_conformal") or {}).get("conformal") or {}
        m = cb.get("metrics") or {}
        cal = cb.get("calibrator") or {}
        print()
        print(
            f"Conformal layer: alpha={cal.get('alpha', 0):.3f} "
            f"(target coverage {cal.get('target_coverage', 0):.3f})"
        )
        print(
            f"                 empirical coverage on test fold: "
            f"{m.get('empirical_coverage', 0):.3f}"
        )
        print(
            f"                 abstain rate: "
            f"{m.get('abstain_rate', 0):.3f} "
            f"({m.get('n_abstained', 0)}/{m.get('n_test', 0)})"
        )
        print(
            f"                 conditional accuracy on confident subset: "
            f"{m.get('conditional_accuracy', 0):.3f}"
        )
        print(f"                 average prediction set size: {m.get('average_set_size', 0):.3f}")
    if "stacking" in args.policies and stacking_meta is not None:
        m_st = (hybrid.get("stacking") or {}).get("slices", {}).get("full") or {}
        meta_summary = stacking_meta.to_summary_dict()
        top5 = (meta_summary.get("feature_importances_top10") or [])[:5]
        vs_flat = ((hybrid.get("stacking") or {}).get("vs_flat_bootstrap") or {}).get(
            "accuracy"
        ) or {}
        vs_arb = ((hybrid.get("stacking") or {}).get("vs_llm_arbitrate_bootstrap") or {}).get(
            "accuracy"
        ) or {}
        print()
        print(
            f"Stacking meta-classifier: "
            f"{meta_summary.get('classifier').upper()}"
            + ("(L2,C=1.0)" if meta_summary.get("classifier") == "lr" else "(GB,depth=3)")
            + f", {meta_summary.get('n_features')} features, "
            f"{meta_summary.get('n_train_rows')} OOF rows"
        )
        print(
            f"                          CV macro-F1 (5-fold) = "
            f"{meta_summary.get('cv_score_macro_f1', 0):.4f}"
        )
        print(
            "                          Top 5 features: "
            + ", ".join(f"{name}({score:.3f})" for name, score in top5)
        )
        print(
            f"                          Hybrid full-slice acc with stacking: "
            f"{m_st.get('accuracy', 0):.4f} "
            f"(Δacc vs flat = {m_st.get('accuracy', 0) - full_flat.get('accuracy', 0):+.4f}, "
            f"P_better={vs_flat.get('p_b_better', float('nan')):.3f})"
        )
        if vs_arb:
            print(
                f"                          Δacc vs llm_arbitrate = "
                f"{vs_arb.get('delta_point', 0):+.4f}, "
                f"P_better={vs_arb.get('p_b_better', float('nan')):.3f}"
            )
    if "llm_arbitrate" in args.policies:
        stats = (hybrid.get("llm_arbitrate") or {}).get("arbitration_stats") or {}
        decisions = stats.get("decisions") or []
        n_calls = int(stats.get("n_calls", 0))
        n_cached = int(stats.get("n_cached", 0))
        triggers = stats.get("trigger_breakdown") or {}
        n_disag = int(triggers.get("disagreement", 0))
        n_lc = int(triggers.get("low_conf_agreement", 0))
        disag_decisions = [d for d in decisions if d.get("trigger") == "disagreement"]
        n_disag_correct = sum(1 for d in disag_decisions if d.get("correct"))
        full_idx_for_print2 = slices.get("full")
        n_full = 0 if full_idx_for_print2 is None else len(full_idx_for_print2)
        m_arb = (hybrid["llm_arbitrate"].get("slices") or {}).get("full") or {}
        vs_flat_acc = (hybrid["llm_arbitrate"].get("vs_flat_bootstrap") or {}).get("accuracy") or {}
        print()
        print(
            f"LLM arbitrator: {n_calls} calls "
            f"({n_disag} disagreement + {n_lc} low-conf), {n_cached} cached"
        )
        print(
            f"                accuracy on disagreement subset: "
            f"{n_disag_correct}/{n_disag} = "
            f"{(n_disag_correct / n_disag) if n_disag else float('nan'):.3f}"
        )
        print(
            f"                hybrid full-slice acc with arbitrator: "
            f"{m_arb.get('accuracy', 0):.4f} "
            f"(Δacc vs flat = {m_arb.get('accuracy', 0) - full_flat.get('accuracy', 0):+.4f}, "
            f"P_better={vs_flat_acc.get('p_b_better', float('nan')):.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
