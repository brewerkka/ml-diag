from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RESULTS = Path(__file__).resolve().parent.parent / "results"

_CORPORA = ("8ds", "5ds", "3ds")

_HYBRID_PATHS = {
    "8ds": _RESULTS / "hybrid_evaluation_8ds_with_stacking_gbm.json",
    "5ds": _RESULTS / "hybrid_evaluation_5ds_full_pipeline.json",
    "3ds": _RESULTS / "hybrid_evaluation_3ds_full_pipeline.json",
}

_HYBRID_PATHS_DI_PROXY = {
    "8ds": _RESULTS / "hybrid_evaluation_8ds_with_di_proxy.json",
    "5ds": _RESULTS / "hybrid_evaluation_5ds_with_di_proxy.json",
    "3ds": _RESULTS / "hybrid_evaluation_3ds_with_di_proxy.json",
}

_PER_DATASET_PATHS_DI_PROXY = {
    "8ds": _RESULTS / "per_dataset_breakdown_8ds_with_di_proxy.json",
    "5ds": _RESULTS / "per_dataset_breakdown_5ds_with_di_proxy.json",
    "3ds": _RESULTS / "per_dataset_breakdown_3ds_with_di_proxy.json",
}

_CONFORMAL_PRIMARY_PATH_8DS = _RESULTS / "hybrid_evaluation_8ds_with_conformal.json"

_PER_DATASET_PATHS = {
    "8ds": _RESULTS / "per_dataset_breakdown_8ds.json",
    "5ds": _RESULTS / "per_dataset_breakdown_5ds.json",
    "3ds": _RESULTS / "per_dataset_breakdown_3ds.json",
}

_CONFORMAL_SWEEP_PATHS = {
    "8ds": {
        "0.10": _RESULTS / "hybrid_evaluation_8ds_conformal_alpha0.10.json",
        "0.05": _RESULTS / "hybrid_evaluation_8ds_with_conformal.json",
        "0.02": _RESULTS / "hybrid_evaluation_8ds_conformal_alpha0.02.json",
    },
    "5ds": {
        "0.10": _RESULTS / "hybrid_eval_5ds_conformal_alpha0.10.json",
        "0.05": _RESULTS / "hybrid_eval_5ds_conformal_alpha0.05.json",
        "0.02": _RESULTS / "hybrid_eval_5ds_conformal_alpha0.02.json",
    },
    "3ds": {
        "0.10": _RESULTS / "hybrid_eval_3ds_conformal_alpha0.10.json",
        "0.05": _RESULTS / "hybrid_eval_3ds_conformal_alpha0.05.json",
        "0.02": _RESULTS / "hybrid_eval_3ds_conformal_alpha0.02.json",
    },
}

_3DS_CORE = (
    "sklearn:breast_cancer",
    "openml:credit-g",
    "openml:ionosphere",
)

_POLICIES = (
    "flat",
    "cascade",
    "agreement_or_cascade",
    "agreement_or_flat",
    "confidence_weighted",
    "llm_arbitrate",
    "stacking",
    "stacking_with_conformal",
)


def _safe_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        print(f"WARNING: missing {path}, this column will be empty", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"WARNING: failed to parse {path}: {e}", file=sys.stderr)
        return None


def _full_slice(payload: dict[str, Any] | None, policy: str) -> dict[str, Any] | None:
    if payload is None:
        return None
    if policy in ("flat", "cascade"):
        baselines = payload.get("baselines") or {}
        return ((baselines.get(policy) or {}).get("slices") or {}).get("full")
    hyb = payload.get("hybrid") or {}
    return ((hyb.get(policy) or {}).get("slices") or {}).get("full")


def _conformal_block(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    hyb = payload.get("hybrid") or {}
    return (hyb.get("stacking_with_conformal") or {}).get("conformal") or {}


def _build_headline(
    hybrid_payloads: dict[str, dict[str, Any] | None],
    *,
    secondary_payloads: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    secondary_payloads = secondary_payloads or {}
    headline: dict[str, Any] = {}
    for corpus, payload in hybrid_payloads.items():
        n_test = payload.get("test_n") if payload else None
        row: dict[str, Any] = {"n_test": n_test}
        for policy in _POLICIES:
            sl = _full_slice(payload, policy)
            if sl is None:
                sl = _full_slice(
                    secondary_payloads.get(corpus),
                    policy,
                )
            if sl is None:
                row[f"{policy}_acc"] = None
                row[f"{policy}_macro_f1"] = None
                row[f"{policy}_leakage_f1"] = None
                row[f"{policy}_leakage_called_healthy"] = None
                continue
            row[f"{policy}_acc"] = (
                float(sl.get("accuracy")) if sl.get("accuracy") is not None else None
            )
            row[f"{policy}_macro_f1"] = (
                float(sl.get("macro_f1")) if sl.get("macro_f1") is not None else None
            )
            row[f"{policy}_leakage_f1"] = (
                float(sl.get("leakage_f1")) if sl.get("leakage_f1") is not None else None
            )
            row[f"{policy}_leakage_called_healthy"] = sl.get("leakage_called_healthy")
        headline[corpus] = row
    return headline


def _build_conformal_sweep(
    *,
    override_alpha_05_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    sweep: dict[str, Any] = {}
    for corpus, paths in _CONFORMAL_SWEEP_PATHS.items():
        per_alpha: dict[str, Any] = {}
        for alpha, p in paths.items():
            if (
                override_alpha_05_paths is not None
                and alpha == "0.05"
                and corpus in override_alpha_05_paths
            ):
                p = override_alpha_05_paths[corpus]
            payload = _safe_load(p)
            cb = _conformal_block(payload)
            m = (cb or {}).get("metrics") or {}
            per_alpha[f"alpha_{alpha}"] = {
                "empirical_coverage": m.get("empirical_coverage"),
                "target_coverage": m.get("target_coverage"),
                "abstain_rate": m.get("abstain_rate"),
                "n_abstained": m.get("n_abstained"),
                "n_test": m.get("n_test"),
                "conditional_accuracy": m.get("conditional_accuracy"),
                "average_set_size": m.get("average_set_size"),
            }
        sweep[corpus] = per_alpha
    return sweep


def _build_core_leakage_f1(
    per_dataset_payloads: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {ds: {} for ds in _3DS_CORE}
    for corpus, payload in per_dataset_payloads.items():
        if payload is None:
            for ds in _3DS_CORE:
                out[ds][f"{corpus}_flat"] = None
                out[ds][f"{corpus}_cascade"] = None
            continue
        ds_block = payload.get("datasets") or {}
        for ds in _3DS_CORE:
            entry = ds_block.get(ds)
            if entry is None:
                out[ds][f"{corpus}_flat"] = None
                out[ds][f"{corpus}_cascade"] = None
                continue
            flat = (entry.get("flat") or {}).get("per_class_f1") or {}
            casc = (entry.get("cascade") or {}).get("per_class_f1") or {}
            out[ds][f"{corpus}_flat"] = (
                float(flat.get("leakage")) if flat.get("leakage") is not None else None
            )
            out[ds][f"{corpus}_cascade"] = (
                float(casc.get("leakage")) if casc.get("leakage") is not None else None
            )
    return out


def _check_invariants(
    *,
    headline: dict[str, Any],
    conformal_sweep: dict[str, Any],
    hybrid_payloads: dict[str, dict[str, Any] | None],
) -> dict[str, Any]:
    cov_details: dict[str, Any] = {}
    cov_holds = True
    for c in _CORPORA:
        cov = ((conformal_sweep.get(c) or {}).get("alpha_0.05") or {}).get("empirical_coverage")
        cov_details[c] = cov
        if cov is None or float(cov) < 0.92:
            cov_holds = False
    best_resolver: dict[str, Any] = {}
    best_holds = True
    for c in _CORPORA:
        h = headline.get(c) or {}
        flat_acc = h.get("flat_acc")
        if flat_acc is None:
            best_resolver[c] = None
            best_holds = False
            continue
        best_name = None
        best_acc = -1.0
        for policy in (
            "agreement_or_cascade",
            "agreement_or_flat",
            "confidence_weighted",
            "llm_arbitrate",
            "stacking",
            "stacking_with_conformal",
        ):
            a = h.get(f"{policy}_acc")
            if a is not None and a > best_acc:
                best_acc = a
                best_name = policy
        if best_name is None or best_acc < float(flat_acc) - 5e-4:
            best_holds = False
        best_resolver[c] = {
            "policy": best_name,
            "acc": best_acc,
            "delta_vs_flat": (best_acc - float(flat_acc)) if best_name else None,
        }
    hl_details: dict[str, Any] = {}
    hl_holds = True
    for c in _CORPORA:
        payload = hybrid_payloads.get(c)
        if payload is None:
            hl_details[c] = None
            hl_holds = False
            continue
        cb = (payload.get("baselines") or {}).get("cascade") or {}
        sl = (cb.get("slices") or {}).get("full") or {}
        n = sl.get("n_samples")
        acc = sl.get("accuracy")
        if not n or acc is None:
            hl_details[c] = None
            hl_holds = False
            continue
        n_errors = int(round((1.0 - float(acc)) * int(n)))
        l2h = int(sl.get("leakage_called_healthy") or 0)
        h2l = int(sl.get("healthy_called_leakage") or 0)
        share = (l2h + h2l) / n_errors if n_errors else 0.0
        hl_details[c] = {
            "n_errors": n_errors,
            "leakage_called_healthy": l2h,
            "healthy_called_leakage": h2l,
            "share_of_total_confusion": share,
            "is_main": share >= 0.40,
        }
        if not hl_details[c]["is_main"]:
            hl_holds = False
    return {
        "coverage_holds_across_corpora": bool(cov_holds),
        "coverage_details": {
            **{c: cov_details[c] for c in _CORPORA},
            "all_geq_0.92": bool(cov_holds),
        },
        "best_resolver_beats_flat_across_corpora": bool(best_holds),
        "best_resolver_details": best_resolver,
        "healthy_leakage_main_confusion_across_corpora": bool(hl_holds),
        "healthy_leakage_details": hl_details,
    }


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _render_markdown(payload: dict[str, Any]) -> str:
    out: list[str] = []
    out.append("# Cross-corpus replication summary (Stage 59)")
    out.append("")
    out.append(f"- generated: {payload.get('generated_at')}")
    out.append(f"- corpora compared: {', '.join(payload.get('corpora') or [])}")
    out.append(
        "- common 3-dataset core: "
        + ", ".join(f"`{ds}`" for ds in payload.get("common_3ds_core") or [])
    )
    n_test = payload.get("n_test_per_corpus") or {}
    out.append("- test fold sizes: " + ", ".join(f"{c}={n_test.get(c) or '?'}" for c in _CORPORA))
    out.append("")
    out.append("## 1. Headline metrics — full slice, all policies")
    out.append("")
    for metric, label in (("acc", "accuracy"), ("macro_f1", "macro-F1")):
        out.append(f"### {label}")
        out.append("")
        cols = list(_POLICIES)
        out.append("| corpus | " + " | ".join(cols) + " |")
        out.append("|---|" + "|".join(["---:" for _ in cols]) + "|")
        for c in _CORPORA:
            row = (payload.get("headline_metrics") or {}).get(c) or {}
            cells = " | ".join(_fmt(row.get(f"{p}_{metric}")) for p in cols)
            out.append(f"| `{c}` | {cells} |")
        out.append("")
    out.append("## 2. Conformal α-sweep")
    out.append("")
    out.append("| corpus | α | target | empirical | abstain | conditional_acc | avg_set_size |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in _CORPORA:
        per_alpha = (payload.get("conformal_coverage_sweep") or {}).get(c) or {}
        for a in ("0.10", "0.05", "0.02"):
            block = per_alpha.get(f"alpha_{a}") or {}
            out.append(
                f"| `{c}` | {a} "
                f"| {_fmt(block.get('target_coverage'), 3)} "
                f"| {_fmt(block.get('empirical_coverage'), 3)} "
                f"| {_fmt(block.get('abstain_rate'), 3)} "
                f"| {_fmt(block.get('conditional_accuracy'), 3)} "
                f"| {_fmt(block.get('average_set_size'), 3)} |"
            )
    out.append("")
    inv = payload.get("invariant_checks") or {}
    out.append("## 3. Architectural invariants")
    out.append("")
    out.append(
        f"### Invariant 1 — empirical coverage ≥ 0.92 on each corpus at α=0.05: "
        f"**{'✓' if inv.get('coverage_holds_across_corpora') else '✗'}**"
    )
    out.append("")
    cd = inv.get("coverage_details") or {}
    out.append("| corpus | empirical coverage |")
    out.append("|---|---:|")
    for c in _CORPORA:
        out.append(f"| `{c}` | {_fmt(cd.get(c), 3)} |")
    out.append("")
    out.append(
        f"### Invariant 2 — best resolver beats flat baseline on each corpus: "
        f"**{'✓' if inv.get('best_resolver_beats_flat_across_corpora') else '✗'}**"
    )
    out.append("")
    br = inv.get("best_resolver_details") or {}
    out.append("| corpus | best policy | acc | Δacc vs flat |")
    out.append("|---|---|---:|---:|")
    for c in _CORPORA:
        d = br.get(c) or {}
        out.append(
            f"| `{c}` | `{d.get('policy')}` "
            f"| {_fmt(d.get('acc'))} | {_fmt(d.get('delta_vs_flat'), 4)} |"
        )
    out.append("")
    out.append(
        f"### Invariant 3 — healthy↔leakage is the dominant confusion on each corpus: "
        f"**{'✓' if inv.get('healthy_leakage_main_confusion_across_corpora') else '✗'}**"
    )
    out.append("")
    hl = inv.get("healthy_leakage_details") or {}
    out.append("| corpus | n_errors | leak→healthy | healthy→leak | share | is_main |")
    out.append("|---|---:|---:|---:|---:|:-:|")
    for c in _CORPORA:
        d = hl.get(c) or {}
        out.append(
            f"| `{c}` | {d.get('n_errors')} "
            f"| {d.get('leakage_called_healthy')} "
            f"| {d.get('healthy_called_leakage')} "
            f"| {_fmt(d.get('share_of_total_confusion'), 3)} "
            f"| {'✓' if d.get('is_main') else '✗'} |"
        )
    out.append("")
    out.append("## 4. Per-dataset leakage F1 on shared 3-dataset core")
    out.append("")
    pd_core = payload.get("per_dataset_leakage_f1_on_3ds_core") or {}
    out.append(
        "| dataset | 8ds_flat | 8ds_cascade | 5ds_flat | 5ds_cascade | 3ds_flat | 3ds_cascade |"
    )
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for ds in _3DS_CORE:
        row = pd_core.get(ds) or {}
        out.append(
            f"| `{ds}` "
            f"| {_fmt(row.get('8ds_flat'))} | {_fmt(row.get('8ds_cascade'))} "
            f"| {_fmt(row.get('5ds_flat'))} | {_fmt(row.get('5ds_cascade'))} "
            f"| {_fmt(row.get('3ds_flat'))} | {_fmt(row.get('3ds_cascade'))} |"
        )
    out.append("")
    return "\n".join(out)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-md", required=True, type=Path)
    p.add_argument("--out-json", required=True, type=Path)
    p.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "di_proxy"],
        help="`baseline` (default) reads Stage 59 artefacts; "
        "`di_proxy` reads Stage 60 *_with_di_proxy.json files.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.variant == "di_proxy":
        hybrid_paths = _HYBRID_PATHS_DI_PROXY
        per_dataset_paths = _PER_DATASET_PATHS_DI_PROXY
    else:
        hybrid_paths = _HYBRID_PATHS
        per_dataset_paths = _PER_DATASET_PATHS
    hybrid_payloads = {c: _safe_load(hybrid_paths[c]) for c in _CORPORA}
    per_dataset_payloads = {c: _safe_load(per_dataset_paths[c]) for c in _CORPORA}
    secondary = {
        "8ds": _safe_load(_CONFORMAL_PRIMARY_PATH_8DS),
        "5ds": None,
        "3ds": None,
    }
    headline = _build_headline(hybrid_payloads, secondary_payloads=secondary)
    conformal_sweep = _build_conformal_sweep(
        override_alpha_05_paths=hybrid_paths if args.variant == "di_proxy" else None,
    )
    core_leakage = _build_core_leakage_f1(per_dataset_payloads)
    invariants = _check_invariants(
        headline=headline,
        conformal_sweep=conformal_sweep,
        hybrid_payloads=hybrid_payloads,
    )
    n_test = {}
    for c in _CORPORA:
        payload = hybrid_payloads[c]
        n_test[c] = payload.get("test_n") if payload else None
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpora": list(_CORPORA),
        "common_3ds_core": list(_3DS_CORE),
        "n_test_per_corpus": n_test,
        "headline_metrics": headline,
        "conformal_coverage_sweep": conformal_sweep,
        "per_dataset_leakage_f1_on_3ds_core": core_leakage,
        "invariant_checks": invariants,
    }
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"Wrote markdown -> {args.out_md}")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote JSON     -> {args.out_json}")
    print()
    print(f"Cross-corpus summary ({' / '.join(_CORPORA)}):")
    h = payload["headline_metrics"]
    sweep = payload["conformal_coverage_sweep"]

    def _row(label: str, getter):
        vals = [getter(c) for c in _CORPORA]
        cells = " / ".join(_fmt(v, 3) if v is not None else "—" for v in vals)
        print(f"  {label:<22} {cells}")

    _row(
        "best policy acc:",
        lambda c: (invariants.get("best_resolver_details") or {}).get(c, {}).get("acc"),
    )
    _row(
        "conformal coverage:",
        lambda c: (sweep.get(c, {}).get("alpha_0.05") or {}).get("empirical_coverage"),
    )
    _row("abstain rate:", lambda c: (sweep.get(c, {}).get("alpha_0.05") or {}).get("abstain_rate"))
    _row(
        "conditional acc:",
        lambda c: (sweep.get(c, {}).get("alpha_0.05") or {}).get("conditional_accuracy"),
    )
    print()
    print("Invariants:")
    print(
        f"  {'✓' if invariants['coverage_holds_across_corpora'] else '✗'}"
        " Coverage ≥ 0.92 on all corpora"
    )
    print(
        f"  {'✓' if invariants['best_resolver_beats_flat_across_corpora'] else '✗'}"
        " Best resolver > flat on all corpora"
    )
    print(
        f"  {'✓' if invariants['healthy_leakage_main_confusion_across_corpora'] else '✗'}"
        " healthy↔leakage = main confusion on all corpora"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
