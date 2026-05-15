from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml_diag.actions.allowlist import (
    Action,
    ActionParameterError,
    get_action,
    validate_parameters,
)
from ml_diag.evaluation.explanation import (
    StructuredEvidence,
    build_evidence,
)
from ml_diag.evaluation.explanation import (
    render_markdown as render_evidence_markdown,
)
from ml_diag.features.run_features import FeatureTable
from ml_diag.labels import HEALTHY, to_stage1
from ml_diag.models.inference import (
    HierarchicalCascade,
    HierarchicalDiagnosis,
    diagnose_one,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

PATCH_OUTCOMES: tuple[str, ...] = (
    "improved",
    "partial",
    "neutral",
    "degraded",
    "observe_only",
)


@dataclass(frozen=True)
class PatchCase:
    case_id: str
    before_run_id: str
    after_run_id: str
    action_name: str
    action_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchOutcome:
    status: str
    rationale: list[str]
    delta_p_healthy: float
    delta_p_faulty_chosen: float
    before_class: str
    after_class: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "rationale": list(self.rationale),
            "delta_p_healthy": float(self.delta_p_healthy),
            "delta_p_faulty_chosen": float(self.delta_p_faulty_chosen),
            "before_class": self.before_class,
            "after_class": self.after_class,
        }


@dataclass(frozen=True)
class PatchReport:
    schema_version: str
    generated_at: str
    case: PatchCase
    action: Action
    parameters: dict[str, Any]
    applicability: dict[str, Any]
    meta_delta: dict[str, Any]
    before_diagnosis: HierarchicalDiagnosis
    after_diagnosis: HierarchicalDiagnosis
    before_evidence: StructuredEvidence
    after_evidence: StructuredEvidence
    outcome: PatchOutcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "case": {
                "case_id": self.case.case_id,
                "before_run_id": self.case.before_run_id,
                "after_run_id": self.case.after_run_id,
                "action_name": self.case.action_name,
                "action_parameters": dict(self.case.action_parameters),
            },
            "action": self.action.to_dict(),
            "parameters": self.parameters,
            "applicability": self.applicability,
            "meta_delta": self.meta_delta,
            "before_diagnosis": self.before_diagnosis.to_dict(),
            "after_diagnosis": self.after_diagnosis.to_dict(),
            "before_evidence": self.before_evidence.to_dict(),
            "after_evidence": self.after_evidence.to_dict(),
            "outcome": self.outcome.to_dict(),
        }


PATCH_REPORT_SCHEMA_VERSION = "1.0"

_MATERIAL_DELTA = 0.05


def _row_or_raise(df: pd.DataFrame, run_id: str, label: str) -> pd.Series:
    if run_id not in df.index:
        raise KeyError(f"{label} run_id {run_id!r} not in feature table.")
    return df.loc[run_id]


def _compute_meta_delta(
    before_meta: dict[str, Any], after_meta: dict[str, Any], keys: Iterable[str]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        out[k] = {
            "before": before_meta.get(k),
            "after": after_meta.get(k),
            "changed": before_meta.get(k) != after_meta.get(k),
        }
    return out


def _faulty_chosen_prob(d: HierarchicalDiagnosis) -> float:
    leaf = d.final_class
    if to_stage1(leaf) == HEALTHY:
        return 0.0
    return float(d.class_probabilities.get(leaf, 0.0))


def _classify_outcome(
    *,
    action: Action,
    before: HierarchicalDiagnosis,
    after: HierarchicalDiagnosis,
) -> PatchOutcome:
    rationale: list[str] = []
    p_healthy_before = float(before.class_probabilities.get(HEALTHY, 0.0))
    p_healthy_after = float(after.class_probabilities.get(HEALTHY, 0.0))
    delta_healthy = p_healthy_after - p_healthy_before
    p_chosen_before = _faulty_chosen_prob(before)
    p_chosen_after = _faulty_chosen_prob(after)
    delta_faulty_chosen = p_chosen_after - p_chosen_before
    before_class = before.final_class
    after_class = after.final_class
    if action.name == "observe_only":
        rationale.append("action is observe_only — interpreting `after` as a replication.")
        if before_class == after_class:
            rationale.append("classes match → consistent diagnosis under replication.")
        else:
            rationale.append(
                f"classes differ (before={before_class}, after={after_class}) → unstable diagnosis."
            )
        return PatchOutcome(
            status="observe_only",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    if to_stage1(before_class) != HEALTHY and after_class == HEALTHY:
        rationale.append(f"before was faulty ({before_class}); after is healthy.")
        return PatchOutcome(
            status="improved",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    if delta_healthy >= 2 * _MATERIAL_DELTA and delta_faulty_chosen <= -_MATERIAL_DELTA:
        rationale.append(
            f"P(healthy) increased by {delta_healthy:+.3f} and "
            f"P({before_class}) dropped by {-delta_faulty_chosen:.3f}."
        )
        return PatchOutcome(
            status="improved",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    if before_class == HEALTHY and after_class != HEALTHY:
        rationale.append(f"before was healthy; after became {after_class}.")
        return PatchOutcome(
            status="degraded",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    if delta_healthy <= -2 * _MATERIAL_DELTA:
        rationale.append(f"P(healthy) decreased by {-delta_healthy:.3f}.")
        return PatchOutcome(
            status="degraded",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    if (
        to_stage1(before_class) != HEALTHY
        and to_stage1(after_class) != HEALTHY
        and (delta_healthy >= _MATERIAL_DELTA or delta_faulty_chosen <= -_MATERIAL_DELTA)
    ):
        rationale.append("still faulty, but materially closer to healthy on the cascade.")
        if before_class != after_class:
            rationale.append(f"faulty class also moved from `{before_class}` to `{after_class}`.")
        return PatchOutcome(
            status="partial",
            rationale=rationale,
            delta_p_healthy=delta_healthy,
            delta_p_faulty_chosen=delta_faulty_chosen,
            before_class=before_class,
            after_class=after_class,
        )
    rationale.append(
        f"Δ P(healthy) = {delta_healthy:+.3f}, Δ P({before_class}) = {delta_faulty_chosen:+.3f} — "
        "below material thresholds."
    )
    return PatchOutcome(
        status="neutral",
        rationale=rationale,
        delta_p_healthy=delta_healthy,
        delta_p_faulty_chosen=delta_faulty_chosen,
        before_class=before_class,
        after_class=after_class,
    )


def evaluate_patch(
    *,
    case: PatchCase,
    cascade: HierarchicalCascade,
    feature_table: FeatureTable,
    full_feature_df: pd.DataFrame | None = None,
    integrity_columns: Iterable[str] | None = None,
    before_meta: dict[str, Any] | None = None,
    after_meta: dict[str, Any] | None = None,
) -> PatchReport:
    action = get_action(case.action_name)
    try:
        params = validate_parameters(action, dict(case.action_parameters))
    except ActionParameterError as e:
        raise ActionParameterError(f"[case {case.case_id}] {e}") from e
    feat_df = full_feature_df if full_feature_df is not None else feature_table.df
    feature_cols = feature_table.feature_columns
    before_row = _row_or_raise(feat_df, case.before_run_id, "before")
    after_row = _row_or_raise(feat_df, case.after_run_id, "after")
    before_x = before_row.reindex(feature_cols).fillna(0.0)
    after_x = after_row.reindex(feature_cols).fillna(0.0)
    before_diag = diagnose_one(cascade, run_id=case.before_run_id, x_row=before_x)
    after_diag = diagnose_one(cascade, run_id=case.after_run_id, x_row=after_x)
    before_evidence = build_evidence(
        diagnosis=before_diag,
        feature_row=feat_df.loc[case.before_run_id],
        cascade=cascade,
        integrity_columns=integrity_columns,
    )
    after_evidence = build_evidence(
        diagnosis=after_diag,
        feature_row=feat_df.loc[case.after_run_id],
        cascade=cascade,
        integrity_columns=integrity_columns,
    )
    before_evidence_dict = before_evidence.to_dict()
    applicable = action.applies_to(before_diag.final_class, before_evidence_dict)
    applicability = {
        "applicable": bool(applicable),
        "checks": [
            f"diagnosis_class_in_target: {before_diag.final_class in action.target_classes}",
            f"applies_to_returned: {applicable}",
        ],
    }
    if not applicable:
        _LOG.warning(
            "[case %s] action %s reports not-applicable for before-class %s — proceeding anyway.",
            case.case_id,
            action.name,
            before_diag.final_class,
        )
    meta_delta: dict[str, Any] = {}
    if before_meta is not None or after_meta is not None:
        meta_delta = _compute_meta_delta(
            before_meta or {}, after_meta or {}, action.meta_delta_keys
        )
    outcome = _classify_outcome(action=action, before=before_diag, after=after_diag)
    return PatchReport(
        schema_version=PATCH_REPORT_SCHEMA_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        case=case,
        action=action,
        parameters=params,
        applicability=applicability,
        meta_delta=meta_delta,
        before_diagnosis=before_diag,
        after_diagnosis=after_diag,
        before_evidence=before_evidence,
        after_evidence=after_evidence,
        outcome=outcome,
    )


def render_patch_markdown(report: PatchReport) -> str:
    out: list[str] = []
    c = report.case
    out.append(f"# Patch evaluation — case `{c.case_id}`")
    out.append("")
    out.append(f"- generated: {report.generated_at}")
    out.append(f"- before run: `{c.before_run_id}`")
    out.append(f"- after run:  `{c.after_run_id}`")
    out.append(f"- action:     `{report.action.name}` — {report.action.description}")
    if report.parameters:
        out.append(f"- parameters: `{report.parameters}`")
    out.append("")
    out.append(f"- **outcome:** `{report.outcome.status}`")
    out.append(f"  - Δ P(healthy) = `{report.outcome.delta_p_healthy:+.4f}`")
    out.append(f"  - Δ P(before-class) = `{report.outcome.delta_p_faulty_chosen:+.4f}`")
    for r in report.outcome.rationale:
        out.append(f"  - {r}")
    out.append("")
    out.append("## Applicability")
    out.append("")
    out.append(f"- applicable: **{report.applicability['applicable']}**")
    for chk in report.applicability["checks"]:
        out.append(f"  - {chk}")
    out.append("")
    if report.meta_delta:
        out.append("## meta.json delta")
        out.append("")
        out.append("| key | before | after | changed |")
        out.append("|---|---|---|---:|")
        for k, v in report.meta_delta.items():
            out.append(
                f"| `{k}` | {v['before']} | {v['after']} | {'yes' if v['changed'] else 'no'} |"
            )
        out.append("")
    out.append("## Before — structured evidence")
    out.append("")
    out.append(render_evidence_markdown(report.before_evidence))
    out.append("")
    out.append("## After — structured evidence")
    out.append("")
    out.append(render_evidence_markdown(report.after_evidence))
    return "\n".join(out)


def write_patch_report(
    report: PatchReport,
    *,
    md_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if md_path is not None:
        p = Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_patch_markdown(report), encoding="utf-8")
        out["md"] = p
    if json_path is not None:
        import json as _json

        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        out["json"] = p
    return out


__all__ = [
    "PATCH_OUTCOMES",
    "PatchCase",
    "PatchOutcome",
    "PatchReport",
    "PATCH_REPORT_SCHEMA_VERSION",
    "evaluate_patch",
    "render_patch_markdown",
    "write_patch_report",
]
