from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_diag.evaluation.explanation import (
    StructuredEvidence,
)
from ml_diag.evaluation.explanation import (
    render_markdown as render_evidence_markdown,
)
from ml_diag.interpretation.recommendations import (
    InterpretationResult,
)
from ml_diag.labels import to_stage1
from ml_diag.models.inference import HierarchicalDiagnosis
from ml_diag.utils.logging import get_logger


def to_stage1_helper(label: str) -> str:
    try:
        return to_stage1(label)
    except Exception:
        return "unknown"


_LOG = get_logger(__name__)

CASE_OUTPUTS_SCHEMA_VERSION = "1.0"

SYSTEM_NAME = "ml_diag"

REQUIRED_FILES: tuple[str, ...] = (
    "diagnosis.json",
    "evidence.json",
    "evidence.md",
    "interpretation.json",
    "interpretation.md",
    "recommendations.json",
    "case_summary.json",
    "case_summary.md",
)

OPTIONAL_FILES: tuple[str, ...] = (
    "patch_summary.json",
    "patch_summary.md",
    "curves.png",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _system_version() -> str:
    try:
        from ml_diag import __version__ as v

        return str(v)
    except Exception:
        return "0.0.0"


def _wrap(
    payload: dict[str, Any],
    *,
    run_id: str,
    model_name: str | None = None,
    schema_kind: str | None = None,
) -> dict[str, Any]:
    head = {
        "schema_version": CASE_OUTPUTS_SCHEMA_VERSION,
        "schema_kind": schema_kind,
        "system_name": SYSTEM_NAME,
        "system_version": _system_version(),
        "created_at": _now(),
        "run_id": run_id,
    }
    if model_name is not None:
        head["model_name"] = model_name
    out: dict[str, Any] = dict(head)
    out["payload"] = payload
    return out


def extract_recommendations_payload(
    interpretation: InterpretationResult,
) -> dict[str, Any]:
    return {
        "interpretation_schema_version": interpretation.schema_version,
        "interpretation_backend": interpretation.backend,
        "final_class": interpretation.final_class,
        "final_confidence": float(interpretation.final_confidence),
        "n_recommendations": len(interpretation.recommendations),
        "recommendations": [r.to_dict() for r in interpretation.recommendations],
        "warnings": list(interpretation.warnings),
        "limitations": list(interpretation.limitations),
    }


def _summary_payload(
    *,
    diagnosis: HierarchicalDiagnosis,
    interpretation: InterpretationResult,
    patch_summary: dict[str, Any] | None,
    files: dict[str, Path],
    extras: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "diagnosis": {
            "final_class": diagnosis.final_class,
            "final_confidence": float(diagnosis.final_confidence),
            "stage1": diagnosis.stage1.to_dict(),
            "stage2": diagnosis.stage2.to_dict() if diagnosis.stage2 else None,
            "stage3": diagnosis.stage3.to_dict() if diagnosis.stage3 else None,
            "alternative_hypotheses": [
                {"class": c, "probability": float(p)} for c, p in diagnosis.alternative_hypotheses
            ],
        },
        "interpretation": {
            "schema_version": interpretation.schema_version,
            "backend": interpretation.backend,
            "summary": interpretation.summary,
            "n_recommendations": len(interpretation.recommendations),
            "n_warnings": len(interpretation.warnings),
        },
        "patch_summary": patch_summary,
        "files": {k: str(v.name) for k, v in files.items()},
        "extras": dict(extras or {}),
    }


def _llm_descriptor(interpretation: InterpretationResult) -> str:
    backend = interpretation.backend
    if backend == "template":
        return "deterministic template renderer (LLM не использовалась)"
    if backend == "groq":
        return "Groq Cloud (Llama 3.3 70B по умолчанию)"
    if backend == "ollama":
        return "Ollama локально (Qwen 2.5 7B Instruct по умолчанию)"
    return backend


def _render_case_summary_md(
    *,
    run_id: str,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    interpretation: InterpretationResult,
    patch_summary: dict[str, Any] | None,
    files: dict[str, Path],
) -> str:
    from ml_diag.evaluation.explanation import classify_evidence_notes

    out: list[str] = []
    out.append(f"# Case `{run_id}`")
    out.append("")
    out.append(f"- generated: {_now()}")
    out.append(
        f"- predicted class: **`{diagnosis.final_class}`** "
        f"(composed P = **{diagnosis.final_confidence:.3f}**)"
    )
    out.append(f"- interpretation: `{interpretation.backend}` — {_llm_descriptor(interpretation)}")
    out.append(f"- recommendations: **{len(interpretation.recommendations)}**")
    if interpretation.warnings:
        out.append(f"- warnings: **{len(interpretation.warnings)}**")
    out.append("")
    out.append("## Stage trace")
    out.append("")
    out.append("| stage | predicted | confidence |")
    out.append("|---|---|---:|")
    for s in (diagnosis.stage1, diagnosis.stage2, diagnosis.stage3):
        if s is None:
            continue
        out.append(f"| `{s.stage_name}` | `{s.predicted}` | {s.confidence:.3f} |")
    out.append("")
    decisive_curve, secondary_curve = classify_evidence_notes(
        list(evidence.curve_evidence.notes), diagnosis.final_class
    )
    decisive_integ, secondary_integ = classify_evidence_notes(
        list(evidence.integrity_evidence.notes), diagnosis.final_class
    )
    decisive_diag, secondary_diag = classify_evidence_notes(
        list(evidence.diagnostic_notes), diagnosis.final_class
    )
    secondary = secondary_curve + secondary_integ + secondary_diag
    out.append("## Decisive evidence")
    out.append("")
    out.append(f"_Признаки и заметки, поддерживающие диагноз `{diagnosis.final_class}`._")
    out.append("")
    if evidence.top_features:
        out.append("**Most-informative features per stage (importance):**")
        out.append("")
        for f in evidence.top_features[:6]:
            v_disp = f"{f.value:.4f}" if isinstance(f.value, float) and f.value == f.value else "—"
            out.append(
                f"- `{f.column}` ({f.source_stage}) — value={v_disp}, importance={f.importance:.4f}"
            )
        out.append("")
    if decisive_curve:
        out.append("**Training-curve evidence:**")
        out.append("")
        for note in decisive_curve:
            out.append(f"- {note}")
        out.append("")
    if decisive_integ:
        out.append("**Data-integrity evidence:**")
        out.append("")
        for note in decisive_integ:
            out.append(f"- {note}")
        out.append("")
    if decisive_diag:
        out.append("**Diagnostic notes:**")
        out.append("")
        for note in decisive_diag:
            out.append(f"- {note}")
        out.append("")
    if not (evidence.top_features or decisive_curve or decisive_integ or decisive_diag):
        out.append(
            "_Узких decisive-признаков для этого диагноза не сформулировано — "
            "решение определяется композицией stage-вероятностей._"
        )
        out.append("")
    if secondary:
        out.append("## Secondary / non-decisive signals")
        out.append("")
        if to_stage1_helper(diagnosis.final_class) == "healthy":
            out.append(
                "_Это fault-like прокси, которые в изолированном виде "
                "выглядели бы как индикаторы ошибки, но **не доминировали** "
                "в композиции каскада. Run всё равно классифицирован как "
                f"`{diagnosis.final_class}`._"
            )
        else:
            out.append(
                "_Альтернативные сигналы, отвергнутые каскадом в пользу "
                f"`{diagnosis.final_class}`._"
            )
        out.append("")
        for note in secondary:
            out.append(f"- {note}")
        out.append("")
    out.append("## Short interpretation")
    out.append("")
    out.append(interpretation.summary)
    out.append("")
    if interpretation.recommendations:
        out.append("## Recommendations")
        out.append("")
        out.append("| # | action | parameters | rationale |")
        out.append("|---:|---|---|---|")
        for r in interpretation.recommendations:
            params_s = ", ".join(f"`{k}={v}`" for k, v in r.parameters.items()) or "—"
            out.append(f"| {r.priority} | `{r.action_name}` | {params_s} | {r.rationale} |")
        out.append("")
    if patch_summary:
        out.append("## Patch")
        out.append("")
        out.append(f"- action: `{patch_summary.get('case', {}).get('action_name', '?')}`")
        out.append(
            f"- before run: `{patch_summary.get('case', {}).get('before_run_id', '?')}` "
            f"→ after run: `{patch_summary.get('case', {}).get('after_run_id', '?')}`"
        )
        outcome = patch_summary.get("outcome") or {}
        out.append(
            f"- outcome: **`{outcome.get('status', '?')}`** "
            f"(ΔP(healthy) = {outcome.get('delta_p_healthy', 0.0):+.4f})"
        )
        out.append("")
    out.append("## Files")
    out.append("")
    for k, p in files.items():
        out.append(f"- `{k}`: [`{p.name}`](./{p.name})")
    out.append("")
    return "\n".join(out)


def _try_render_curves(case_dir: Path, *, run_dir: Path | None) -> Path | None:
    if run_dir is None or not run_dir.is_dir():
        return None
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception:
        return None
    try:
        df = pd.read_csv(history_path)
    except Exception:
        return None
    if df.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    epochs = df.index.values
    ax = axes[0]
    if "train_loss" in df.columns:
        ax.plot(epochs, df["train_loss"], label="train_loss")
    if "val_loss" in df.columns:
        ax.plot(epochs, df["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    if "train_acc" in df.columns:
        ax.plot(epochs, df["train_acc"], label="train_acc")
    if "val_acc" in df.columns:
        ax.plot(epochs, df["val_acc"], label="val_acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = case_dir / "curves.png"
    try:
        fig.savefig(out_path)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return out_path


def write_case_outputs(
    case_dir: str | Path,
    *,
    diagnosis: HierarchicalDiagnosis,
    evidence: StructuredEvidence,
    interpretation: InterpretationResult,
    patch_summary: dict[str, Any] | None = None,
    run_dir: str | Path | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(diagnosis.run_id)
    files: dict[str, Path] = {}
    p = case_dir / "diagnosis.json"
    p.write_text(
        json.dumps(
            _wrap(diagnosis.to_dict(), run_id=run_id, schema_kind="diagnosis"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["diagnosis.json"] = p
    p = case_dir / "evidence.json"
    p.write_text(
        json.dumps(
            _wrap(evidence.to_dict(), run_id=run_id, schema_kind="evidence"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["evidence.json"] = p
    p = case_dir / "evidence.md"
    p.write_text(render_evidence_markdown(evidence), encoding="utf-8")
    files["evidence.md"] = p
    p = case_dir / "interpretation.json"
    p.write_text(
        json.dumps(
            _wrap(
                interpretation.to_dict(),
                run_id=run_id,
                schema_kind="interpretation",
                model_name=f"interpretation_backend:{interpretation.backend}",
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["interpretation.json"] = p
    p = case_dir / "interpretation.md"
    from ml_diag.interpretation import render_markdown as render_interp_md

    p.write_text(render_interp_md(interpretation), encoding="utf-8")
    files["interpretation.md"] = p
    p = case_dir / "recommendations.json"
    p.write_text(
        json.dumps(
            _wrap(
                extract_recommendations_payload(interpretation),
                run_id=run_id,
                schema_kind="recommendations",
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["recommendations.json"] = p
    if patch_summary is not None:
        p = case_dir / "patch_summary.json"
        p.write_text(
            json.dumps(
                _wrap(patch_summary, run_id=run_id, schema_kind="patch_summary"),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        files["patch_summary.json"] = p
        outcome = patch_summary.get("outcome") or {}
        case_block = patch_summary.get("case") or {}
        md = [
            f"# Patch summary — case `{run_id}`",
            "",
            f"- generated: {_now()}",
            f"- before: `{case_block.get('before_run_id', '?')}` "
            f"→ after: `{case_block.get('after_run_id', '?')}`",
            f"- action: `{case_block.get('action_name', '?')}` "
            f"(parameters: `{patch_summary.get('parameters')}`)",
            f"- outcome: **`{outcome.get('status', '?')}`**",
            f"- ΔP(healthy) = `{outcome.get('delta_p_healthy', 0.0):+.4f}`",
            f"- ΔP(before-class) = `{outcome.get('delta_p_faulty_chosen', 0.0):+.4f}`",
            "",
        ]
        for r in outcome.get("rationale") or []:
            md.append(f"- {r}")
        p = case_dir / "patch_summary.md"
        p.write_text("\n".join(md), encoding="utf-8")
        files["patch_summary.md"] = p
    curves = _try_render_curves(case_dir, run_dir=Path(run_dir) if run_dir else None)
    if curves is not None:
        files["curves.png"] = curves
    summary_payload = _summary_payload(
        diagnosis=diagnosis,
        interpretation=interpretation,
        patch_summary=patch_summary,
        files=files,
        extras=dict(extras or {}),
    )
    p = case_dir / "case_summary.json"
    p.write_text(
        json.dumps(
            _wrap(summary_payload, run_id=run_id, schema_kind="case_summary"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    files["case_summary.json"] = p
    p = case_dir / "case_summary.md"
    p.write_text(
        _render_case_summary_md(
            run_id=run_id,
            diagnosis=diagnosis,
            evidence=evidence,
            interpretation=interpretation,
            patch_summary=patch_summary,
            files=files,
        ),
        encoding="utf-8",
    )
    files["case_summary.md"] = p
    _LOG.info("Wrote case outputs (%d files) -> %s", len(files), case_dir)
    return files


def validate_case_dir(case_dir: str | Path) -> tuple[bool, list[str]]:
    case_dir = Path(case_dir)
    errors: list[str] = []
    if not case_dir.is_dir():
        return False, [f"case dir does not exist: {case_dir}"]
    seen_run_ids: set[str] = set()
    for name in REQUIRED_FILES:
        p = case_dir / name
        if not p.is_file():
            errors.append(f"missing required file: {name}")
            continue
        if name.endswith(".json"):
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                errors.append(f"{name}: invalid JSON ({e})")
                continue
            for key in ("schema_version", "system_name", "created_at", "run_id", "payload"):
                if key not in payload:
                    errors.append(f"{name}: missing required header key {key!r}")
            sv = payload.get("schema_version")
            if sv != CASE_OUTPUTS_SCHEMA_VERSION:
                errors.append(
                    f"{name}: schema_version is {sv!r}; expected {CASE_OUTPUTS_SCHEMA_VERSION!r}"
                )
            rid = payload.get("run_id")
            if isinstance(rid, str):
                seen_run_ids.add(rid)
    if len(seen_run_ids) > 1:
        errors.append(f"inconsistent run_id across JSONs: {sorted(seen_run_ids)}")
    return (not errors), errors


__all__ = [
    "CASE_OUTPUTS_SCHEMA_VERSION",
    "OPTIONAL_FILES",
    "REQUIRED_FILES",
    "SYSTEM_NAME",
    "extract_recommendations_payload",
    "validate_case_dir",
    "write_case_outputs",
]
