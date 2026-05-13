from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from structured_diag.evaluation.metrics import ClassificationReport


def _fmt_float(x: float | None, digits: int = 4) -> str:
    if x is None:
        return "—"
    return f"{x:.{digits}f}"


def report_to_markdown(report: ClassificationReport, *, heading: str = "Report") -> str:
    lines: list[str] = []
    lines.append(f"### {heading}")
    lines.append("")
    lines.append(f"- samples: **{report.n_samples}**")
    lines.append(f"- accuracy: **{_fmt_float(report.accuracy)}**")
    lines.append(f"- macro F1: **{_fmt_float(report.macro_f1)}**")
    lines.append(f"- weighted F1: **{_fmt_float(report.weighted_f1)}**")
    lines.append(f"- ECE (top-1, 10 bins): **{_fmt_float(report.ece)}**")
    lines.append("")
    lines.append("**Per-class metrics**")
    lines.append("")
    lines.append("| class | precision | recall | F1 | support |")
    lines.append("|---|---:|---:|---:|---:|")
    for cls in report.confusion_labels:
        lines.append(
            f"| {cls} "
            f"| {_fmt_float(report.per_class_precision.get(cls, 0.0))} "
            f"| {_fmt_float(report.per_class_recall.get(cls, 0.0))} "
            f"| {_fmt_float(report.per_class_f1.get(cls, 0.0))} "
            f"| {report.per_class_support.get(cls, 0)} |"
        )
    lines.append("")
    lines.append("**Confusion matrix** (rows = true, cols = predicted)")
    lines.append("")
    header = "| | " + " | ".join(report.confusion_labels) + " |"
    sep = "|---|" + "|".join(["---:"] * len(report.confusion_labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for cls, row in zip(report.confusion_labels, report.confusion_matrix):
        lines.append(f"| **{cls}** | " + " | ".join(str(v) for v in row) + " |")
    lines.append("")
    return "\n".join(lines)


def render_flat_baseline_markdown(
    *,
    corpus_name: str,
    feature_source: str,
    model_name: str,
    reports: Mapping[str, ClassificationReport],
    extras: Mapping[str, Any] | None = None,
) -> str:
    out: list[str] = []
    out.append("# Flat baseline report")
    out.append("")
    out.append(f"- corpus: **{corpus_name}**")
    out.append(f"- feature source: `{feature_source}`")
    out.append(f"- model: `{model_name}`")
    out.append(f"- generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    if extras:
        for k, v in extras.items():
            out.append(f"- {k}: {v}")
    out.append("")
    out.append(
        "This report corresponds to the **flat multiclass** formulation "
        "reproduced inside `structured_diag`. It exists so the structured "
        "(hierarchical) contour can be compared on the same features and "
        "the same core/extended slices."
    )
    out.append("")
    for slice_name, rep in reports.items():
        out.append("---")
        out.append("")
        out.append(report_to_markdown(rep, heading=f"Slice: `{slice_name}`"))
    return "\n".join(out)


def reports_to_json(reports: Mapping[str, ClassificationReport]) -> dict[str, Any]:
    return {name: rep.to_dict() for name, rep in reports.items()}


def write_report(
    md_path: str | Path,
    json_path: str | Path | None,
    *,
    md_text: str,
    json_payload: dict[str, Any] | None = None,
) -> None:
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_text, encoding="utf-8")
    if json_path is not None and json_payload is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")


__all__ = [
    "report_to_markdown",
    "render_flat_baseline_markdown",
    "reports_to_json",
    "write_report",
]
