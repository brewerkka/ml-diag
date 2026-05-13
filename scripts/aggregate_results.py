from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return {"_error": f"invalid JSON: {e}"}


def _first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def _summarise_partition(payload: dict) -> dict[str, Any]:
    return {
        "corpus_name": payload.get("corpus_name"),
        "rules": payload.get("rules"),
        "counts": payload.get("counts"),
        "class_balance": payload.get("class_balance"),
        "reason_counts": payload.get("reason_counts"),
    }


def _summarise_scenario_inventory(payload: dict) -> dict[str, Any]:
    return {
        "n_runs": payload.get("n_runs"),
        "n_entries": payload.get("n_entries"),
        "primary_label_counts": payload.get("primary_label_counts"),
        "severity_counts": payload.get("severity_counts"),
        "branch_counts": payload.get("branch_counts"),
        "stage3_leaf_counts": payload.get("stage3_leaf_counts"),
        "multi_label_count": payload.get("multi_label_count"),
        "short_history_count": payload.get("short_history_count"),
        "n_gaps": len(payload.get("gaps") or []),
        "n_blockers": sum(1 for g in (payload.get("gaps") or []) if g.get("severity") == "blocker"),
        "n_warnings": sum(1 for g in (payload.get("gaps") or []) if g.get("severity") == "warning"),
    }


def _classification_brief(rep: dict | None) -> dict[str, Any]:
    if not rep:
        return {}
    return {
        "n_samples": rep.get("n_samples"),
        "accuracy": rep.get("accuracy"),
        "macro_f1": rep.get("macro_f1"),
        "weighted_f1": rep.get("weighted_f1"),
        "ece": rep.get("ece"),
        "per_class_f1": rep.get("per_class_f1"),
    }


def _summarise_flat_baseline(payload: dict) -> dict[str, Any]:
    reports = payload.get("reports") or {}
    return {
        "model": payload.get("model"),
        "feature_source": payload.get("feature_source"),
        "n_features": len(payload.get("feature_columns") or []),
        "classes": payload.get("classes"),
        "slices": {k: _classification_brief(v) for k, v in reports.items()},
    }


def _summarise_comparison(payload: dict) -> dict[str, Any]:
    slices = payload.get("slices") or {}
    out = {
        "flat_model": payload.get("flat_model"),
        "cascade_stages": payload.get("cascade_stages"),
        "feature_source": payload.get("feature_source"),
        "slices": {},
    }
    for name, c in slices.items():
        out["slices"][name] = {
            "n_samples": c.get("n_samples"),
            "deltas": c.get("deltas"),
            "per_class_deltas": c.get("per_class_deltas"),
            "flat": _classification_brief(c.get("flat_report")),
            "hier": _classification_brief(c.get("hier_report")),
            "stage_reports": {
                k: _classification_brief(v) for k, v in (c.get("stage_reports") or {}).items()
            },
            "leakage_healthy_confusion": c.get("leakage_healthy_confusion"),
            "error_propagation": c.get("error_propagation"),
        }
    return out


def _summarise_grouped(payload: dict) -> dict[str, Any]:
    reports = payload.get("reports") or {}
    return {
        "model": payload.get("model"),
        "n_groups": payload.get("n_groups"),
        "n_singletons": payload.get("n_singletons"),
        "feature_source": payload.get("feature_source"),
        "slices": {k: _classification_brief(v) for k, v in reports.items()},
    }


def _summarise_prototype(payload: dict) -> dict[str, Any]:
    out = {
        "feature_source": payload.get("feature_source"),
        "n_proto_features": payload.get("n_proto_features"),
        "include_dtw": payload.get("include_dtw"),
        "variants": [],
    }
    for v in payload.get("variants") or []:
        slices = v.get("reports") or {}
        out["variants"].append(
            {
                "name": v.get("name"),
                "model": v.get("model"),
                "slices": {k: _classification_brief(rep) for k, rep in slices.items()},
            }
        )
    return out


def _summarise_leakage_integrity(payload: dict) -> dict[str, Any]:
    out = {
        "feature_source": payload.get("feature_source"),
        "integrity_columns": payload.get("integrity_columns"),
        "variants": [],
    }
    for v in payload.get("variants") or []:
        slices = v.get("reports") or {}
        out["variants"].append(
            {
                "name": v.get("name"),
                "model": v.get("model"),
                "leakage_healthy_focus": v.get("leakage_healthy_focus"),
                "slices": {k: _classification_brief(rep) for k, rep in slices.items()},
            }
        )
    return out


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _delta(d: Any) -> str:
    if d is None:
        return "—"
    try:
        f = float(d)
    except Exception:
        return str(d)
    if f > 1e-4:
        return f"▲ {f:+.4f}"
    if f < -1e-4:
        return f"▼ {f:+.4f}"
    return "≈ 0"


def render_summary_markdown(agg: dict) -> str:
    out: list[str] = []
    out.append("# Aggregated empirical summary")
    out.append("")
    out.append(f"- generated: {agg['generated_at']}")
    out.append(f"- results dir: `{agg['results_dir']}`")
    out.append("")
    inv = agg.get("scenario_inventory")
    if inv:
        out.append("## Scenario inventory")
        out.append("")
        out.append(f"- runs: **{inv.get('n_runs')}**, entries: **{inv.get('n_entries')}**")
        out.append(f"- multi-label runs: {inv.get('multi_label_count')}")
        out.append(f"- short-history runs: {inv.get('short_history_count')}")
        out.append(
            f"- coverage gaps: {inv.get('n_gaps')} "
            f"({inv.get('n_blockers')} blockers, {inv.get('n_warnings')} warnings)"
        )
        out.append("")
    part = agg.get("partition")
    if part:
        c = part.get("counts") or {}
        out.append("## Partition")
        out.append("")
        out.append(
            f"- core: **{c.get('core')}**, extended: **{c.get('extended')}**, total: **{c.get('total')}**"
        )
        out.append("")
    flat = agg.get("flat_baseline")
    if flat:
        out.append("## Flat baseline (test fold)")
        out.append("")
        out.append(f"- best model: `{flat.get('model')}`")
        out.append(f"- feature source: `{flat.get('feature_source')}`")
        out.append("")
        out.append("| slice | n | accuracy | macro-F1 | weighted-F1 | ECE |")
        out.append("|---|---:|---:|---:|---:|---:|")
        for s, brief in (flat.get("slices") or {}).items():
            out.append(
                f"| `{s}` | {brief.get('n_samples')} | {_fmt(brief.get('accuracy'))} "
                f"| {_fmt(brief.get('macro_f1'))} | {_fmt(brief.get('weighted_f1'))} | {_fmt(brief.get('ece'))} |"
            )
        out.append("")
    cmp_ = agg.get("comparison")
    if cmp_:
        out.append("## Flat vs Hierarchical (test fold)")
        out.append("")
        out.append(f"- flat model: `{cmp_.get('flat_model')}`")
        out.append(f"- cascade stages: {', '.join(cmp_.get('cascade_stages') or [])}")
        out.append("")
        out.append(
            "| slice | n | flat acc | hier acc | Δ acc | flat macro-F1 | hier macro-F1 | Δ macro-F1 |"
        )
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for s, c in (cmp_.get("slices") or {}).items():
            d = c.get("deltas") or {}
            f = c.get("flat") or {}
            h = c.get("hier") or {}
            out.append(
                f"| `{s}` | {c.get('n_samples')} "
                f"| {_fmt(f.get('accuracy'))} | {_fmt(h.get('accuracy'))} | {_delta(d.get('delta_accuracy'))} "
                f"| {_fmt(f.get('macro_f1'))} | {_fmt(h.get('macro_f1'))} | {_delta(d.get('delta_macro_f1'))} |"
            )
        out.append("")
        out.append("### Leakage ↔ healthy confusion (per slice)")
        out.append("")
        out.append("| slice | contour | leakage→healthy | healthy→leakage |")
        out.append("|---|---|---:|---:|")
        for s, c in (cmp_.get("slices") or {}).items():
            lh = c.get("leakage_healthy_confusion") or {}
            for contour, vals in lh.items():
                out.append(
                    f"| `{s}` | {contour} | {vals.get('leakage_called_healthy')} "
                    f"| {vals.get('healthy_called_leakage')} |"
                )
        out.append("")
    grp = agg.get("grouped_baseline")
    if grp:
        out.append("## Grouped baseline (per entry_id)")
        out.append("")
        out.append(
            f"- best model: `{grp.get('model')}`, groups: {grp.get('n_groups')} (singletons: {grp.get('n_singletons')})"
        )
        out.append("")
        out.append("| slice | n | accuracy | macro-F1 | ECE |")
        out.append("|---|---:|---:|---:|---:|")
        for s, brief in (grp.get("slices") or {}).items():
            out.append(
                f"| `{s}` | {brief.get('n_samples')} | {_fmt(brief.get('accuracy'))} "
                f"| {_fmt(brief.get('macro_f1'))} | {_fmt(brief.get('ece'))} |"
            )
        out.append("")
    prot = agg.get("prototype_ablation")
    if prot:
        out.append("## Prototype-distance ablation (full slice, test fold)")
        out.append("")
        out.append("| variant | model | accuracy | macro-F1 |")
        out.append("|---|---|---:|---:|")
        for v in prot.get("variants") or []:
            full = (v.get("slices") or {}).get("full") or {}
            out.append(
                f"| `{v.get('name')}` | `{v.get('model')}` "
                f"| {_fmt(full.get('accuracy'))} | {_fmt(full.get('macro_f1'))} |"
            )
        out.append("")
    li = agg.get("leakage_integrity")
    if li:
        out.append("## Leakage-integrity ablation (full slice, test fold)")
        out.append("")
        out.append(
            "| variant | model | accuracy | macro-F1 | leakage F1 | healthy F1 | leakage→healthy | healthy→leakage |"
        )
        out.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for v in li.get("variants") or []:
            full = (v.get("slices") or {}).get("full") or {}
            f = v.get("leakage_healthy_focus") or {}
            conf = f.get("confusion") or {}
            leak = f.get("leakage") or {}
            healthy = f.get("healthy") or {}
            out.append(
                f"| `{v.get('name')}` | `{v.get('model')}` "
                f"| {_fmt(full.get('accuracy'))} | {_fmt(full.get('macro_f1'))} "
                f"| {_fmt(leak.get('f1'))} | {_fmt(healthy.get('f1'))} "
                f"| {conf.get('leakage_called_healthy', '—')} | {conf.get('healthy_called_leakage', '—')} |"
            )
        out.append("")
    out.append("## Where to look for full reports")
    out.append("")
    for name, path in agg.get("source_files", {}).items():
        out.append(f"- **{name}**: `{path}`")
    out.append("")
    return "\n".join(out)


def aggregate(results_dir: Path) -> dict[str, Any]:
    sources: dict[str, Path] = {}
    out: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results_dir": str(results_dir.resolve()),
    }
    partition_dir = results_dir / "partition"
    if partition_dir.is_dir():
        candidates = sorted(partition_dir.glob("*.json"))
        if candidates:
            sources["partition"] = candidates[0]
            payload = _load_json(candidates[0])
            if payload:
                out["partition"] = _summarise_partition(payload)
    pairs = [
        ("scenario_inventory", "scenario_inventory.json", _summarise_scenario_inventory),
        ("flat_baseline", "flat_baseline_report.json", _summarise_flat_baseline),
        ("comparison", "flat_vs_hierarchical_report.json", _summarise_comparison),
        ("grouped_baseline", "grouped_baseline_report.json", _summarise_grouped),
        ("prototype_ablation", "prototype_ablation.json", _summarise_prototype),
        ("leakage_integrity", "leakage_integrity_report.json", _summarise_leakage_integrity),
    ]
    for key, fname, fn in pairs:
        p = results_dir / fname
        if p.is_file():
            sources[key] = p
            payload = _load_json(p)
            if isinstance(payload, dict):
                out[key] = fn(payload)
    out["source_files"] = {k: str(v) for k, v in sources.items()}
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out-md", type=Path, default=None)
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.results_dir.is_dir():
        print(f"ERROR: results dir not found: {args.results_dir}", file=sys.stderr)
        return 2
    agg = aggregate(args.results_dir)
    found = [
        k
        for k in (
            "partition",
            "scenario_inventory",
            "flat_baseline",
            "comparison",
            "grouped_baseline",
            "prototype_ablation",
            "leakage_integrity",
        )
        if k in agg
    ]
    print(f"Sources found: {found}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON     -> {args.out_json}")
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_summary_markdown(agg), encoding="utf-8")
        print(f"Wrote markdown -> {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
