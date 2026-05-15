from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ml_diag.data import CorpusManifest, load_manifest, load_runs_table
from ml_diag.labels import (
    PRIMARY_LABELS,
    STAGE2_LABELS,
    STAGE3_LABELS_BY_BRANCH,
    to_stage1,
    to_stage2,
)
from ml_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

INVENTORY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CoverageGap:
    kind: str
    label: str
    detail: str
    severity: str
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "detail": self.detail,
            "severity": self.severity,
            "extras": dict(self.extras),
        }


@dataclass(frozen=True)
class ScenarioInventory:
    schema_version: str
    generated_at: str
    corpus_name: str
    corpus_path: Path
    n_runs: int
    n_entries: int
    primary_label_counts: dict[str, int]
    severity_counts: dict[str, int]
    dataset_counts: dict[str, int]
    label_x_severity: dict[str, dict[str, int]]
    label_x_dataset: dict[str, dict[str, int]]
    label_x_severity_x_dataset: dict[str, dict[str, dict[str, int]]]
    multi_label_count: int
    short_history_count: int
    branch_counts: dict[str, int]
    stage3_leaf_counts: dict[str, dict[str, int]]
    gaps: list[CoverageGap]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "corpus_name": self.corpus_name,
            "corpus_path": str(self.corpus_path),
            "n_runs": self.n_runs,
            "n_entries": self.n_entries,
            "primary_label_counts": dict(self.primary_label_counts),
            "severity_counts": dict(self.severity_counts),
            "dataset_counts": dict(self.dataset_counts),
            "label_x_severity": _nested_dict(self.label_x_severity),
            "label_x_dataset": _nested_dict(self.label_x_dataset),
            "label_x_severity_x_dataset": {
                lab: {sev: dict(ds) for sev, ds in by_sev.items()}
                for lab, by_sev in self.label_x_severity_x_dataset.items()
            },
            "multi_label_count": self.multi_label_count,
            "short_history_count": self.short_history_count,
            "branch_counts": dict(self.branch_counts),
            "stage3_leaf_counts": _nested_dict(self.stage3_leaf_counts),
            "gaps": [g.to_dict() for g in self.gaps],
        }

    def is_taxonomy_complete(self) -> bool:
        return not any(g.kind == "taxonomy" for g in self.gaps)

    def is_branch_complete(self) -> bool:
        return not any(g.kind == "branch" for g in self.gaps)

    def has_blockers(self) -> bool:
        return any(g.severity == "blocker" for g in self.gaps)


def _nested_dict(d: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {k: dict(v) for k, v in d.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _crosstab_2d(rows: pd.DataFrame, x: str, y: str) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(dict)
    for (a, b), n in rows.groupby([x, y]).size().items():
        out[str(a)][str(b)] = int(n)
    return dict(out)


def _crosstab_3d(rows: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for (lab, sev, ds), n in rows.groupby(["primary_label", "severity", "dataset"]).size().items():
        out[str(lab)][str(sev)][str(ds)] = int(n)
    return {lab: {sev: dict(ds) for sev, ds in v.items()} for lab, v in out.items()}


def _build_gaps(
    *,
    primary_counts: dict[str, int],
    branch_counts: dict[str, int],
    stage3_leaf_counts: dict[str, dict[str, int]],
    label_x_severity: dict[str, dict[str, int]],
    min_leaf_support: int,
) -> list[CoverageGap]:
    gaps: list[CoverageGap] = []
    for label in PRIMARY_LABELS:
        if primary_counts.get(label, 0) == 0:
            gaps.append(
                CoverageGap(
                    kind="taxonomy",
                    label=label,
                    detail=(
                        f"primary class `{label}` is not represented at all — "
                        "the failure taxonomy is incomplete on this corpus."
                    ),
                    severity="blocker",
                )
            )
    for branch in STAGE2_LABELS:
        if branch_counts.get(branch, 0) == 0:
            gaps.append(
                CoverageGap(
                    kind="branch",
                    label=branch,
                    detail=(
                        f"Stage-2 branch `{branch}` has zero faulty runs — "
                        "Stage-2 / Stage-3 models for this side cannot be trained."
                    ),
                    severity="blocker",
                )
            )
    for branch, vocab in STAGE3_LABELS_BY_BRANCH.items():
        leaf_counts = stage3_leaf_counts.get(branch, {})
        for leaf in vocab:
            n = leaf_counts.get(leaf, 0)
            if n < min_leaf_support:
                gaps.append(
                    CoverageGap(
                        kind="stage3_leaf",
                        label=leaf,
                        detail=(
                            f"leaf class `{leaf}` (branch `{branch}`) has only "
                            f"{n} runs (< {min_leaf_support}) — Stage-3 model "
                            "for this branch will be unstable."
                        ),
                        severity="warning" if n > 0 else "blocker",
                        extras={"branch": branch, "support": n, "min_required": min_leaf_support},
                    )
                )
    for label in PRIMARY_LABELS:
        if to_stage1(label) == "healthy":
            continue
        sev_dist = label_x_severity.get(label, {})
        if (sev_dist.get("moderate", 0) + sev_dist.get("severe", 0)) == 0:
            gaps.append(
                CoverageGap(
                    kind="severity_dataset",
                    label=label,
                    detail=(
                        f"class `{label}` has no moderate/severe runs — "
                        "it cannot enter the core benchmark slice."
                    ),
                    severity="warning",
                    extras={"severity_distribution": dict(sev_dist)},
                )
            )
    return gaps


def build_inventory(
    corpus: str | Path | CorpusManifest,
    *,
    min_history_epochs: int = 5,
    min_leaf_support: int = 5,
    skip_broken: bool = True,
) -> ScenarioInventory:
    if isinstance(corpus, CorpusManifest):
        manifest = corpus
    else:
        manifest = load_manifest(corpus)
    n_entries = len(manifest.raw.get("entries") or []) or len(
        {em.get("entry_id") for em in manifest.entry_metadata.values()}
    )
    df = load_runs_table(manifest, skip_broken=skip_broken).copy()
    df["primary_label"] = df["primary_label"].fillna("<unlabeled>").astype(str)
    df["severity"] = df["severity"].fillna("<unknown>").astype(str)
    df["dataset"] = df["dataset"].fillna("<unknown>").astype(str)
    primary_counts = dict(Counter(df["primary_label"].tolist()))
    severity_counts = dict(Counter(df["severity"].tolist()))
    dataset_counts = dict(Counter(df["dataset"].tolist()))
    label_x_severity = _crosstab_2d(df, "primary_label", "severity")
    label_x_dataset = _crosstab_2d(df, "primary_label", "dataset")
    label_x_severity_x_dataset = _crosstab_3d(df)
    multi_label = int(df["is_multi_label"].sum()) if "is_multi_label" in df.columns else 0
    short_history = (
        int((df["n_history_rows"].astype(int) < min_history_epochs).sum())
        if "n_history_rows" in df.columns
        else 0
    )
    faulty_mask = df["primary_label"].apply(
        lambda x: to_stage1(x) != "healthy" if x in PRIMARY_LABELS else False
    )
    branch_counts: dict[str, int] = {b: 0 for b in STAGE2_LABELS}
    for label, n in df.loc[faulty_mask, "primary_label"].value_counts().items():
        if label in PRIMARY_LABELS:
            branch_counts[to_stage2(label)] = branch_counts.get(to_stage2(label), 0) + int(n)
    stage3_leaf_counts: dict[str, dict[str, int]] = {
        b: {l: 0 for l in v} for b, v in STAGE3_LABELS_BY_BRANCH.items()
    }
    for label, n in df.loc[faulty_mask, "primary_label"].value_counts().items():
        if label not in PRIMARY_LABELS:
            continue
        branch = to_stage2(label)
        if label in STAGE3_LABELS_BY_BRANCH[branch]:
            stage3_leaf_counts[branch][label] = stage3_leaf_counts[branch].get(label, 0) + int(n)
    gaps = _build_gaps(
        primary_counts=primary_counts,
        branch_counts=branch_counts,
        stage3_leaf_counts=stage3_leaf_counts,
        label_x_severity=label_x_severity,
        min_leaf_support=min_leaf_support,
    )
    inv = ScenarioInventory(
        schema_version=INVENTORY_SCHEMA_VERSION,
        generated_at=_now(),
        corpus_name=manifest.name,
        corpus_path=manifest.corpus_path,
        n_runs=int(len(df)),
        n_entries=int(n_entries),
        primary_label_counts=primary_counts,
        severity_counts=severity_counts,
        dataset_counts=dataset_counts,
        label_x_severity=label_x_severity,
        label_x_dataset=label_x_dataset,
        label_x_severity_x_dataset=label_x_severity_x_dataset,
        multi_label_count=multi_label,
        short_history_count=short_history,
        branch_counts=branch_counts,
        stage3_leaf_counts=stage3_leaf_counts,
        gaps=gaps,
    )
    _LOG.info(
        "Scenario inventory for %s: %d runs / %d entries, %d gaps (%d blockers, %d warnings).",
        manifest.name,
        inv.n_runs,
        inv.n_entries,
        len(gaps),
        sum(1 for g in gaps if g.severity == "blocker"),
        sum(1 for g in gaps if g.severity == "warning"),
    )
    return inv


def validate_inventory(inv: ScenarioInventory, *, strict: bool = False) -> tuple[bool, list[str]]:
    errs: list[str] = []
    for g in inv.gaps:
        if g.severity == "blocker" or (strict and g.severity == "warning"):
            errs.append(f"[{g.severity}] {g.kind}/{g.label}: {g.detail}")
    return (not errs), errs


def _render_2d(table: dict[str, dict[str, int]], *, row_name: str, col_name: str) -> str:
    if not table:
        return f"_no data for {row_name} × {col_name}_"
    cols = sorted({c for r in table.values() for c in r.keys()})
    out: list[str] = []
    out.append(f"| {row_name} \\\\ {col_name} | " + " | ".join(cols) + " | total |")
    out.append("|" + "---|" * (len(cols) + 2))
    for r in sorted(table):
        cells = [str(table[r].get(c, 0)) for c in cols]
        total = sum(table[r].get(c, 0) for c in cols)
        out.append(f"| `{r}` | " + " | ".join(cells) + f" | **{total}** |")
    col_totals = {c: sum(t.get(c, 0) for t in table.values()) for c in cols}
    out.append(
        "| **total** | "
        + " | ".join(str(col_totals[c]) for c in cols)
        + f" | **{sum(col_totals.values())}** |"
    )
    return "\n".join(out)


def render_markdown(inv: ScenarioInventory) -> str:
    out: list[str] = []
    out.append(f"# Scenario inventory — `{inv.corpus_name}`")
    out.append("")
    out.append(f"- generated: {inv.generated_at}")
    out.append(f"- corpus path: `{inv.corpus_path}`")
    out.append(f"- runs: **{inv.n_runs}**, entries: **{inv.n_entries}**")
    out.append(
        f"- multi-label runs: **{inv.multi_label_count}**, "
        f"short-history runs: **{inv.short_history_count}**"
    )
    out.append("")
    out.append("## Taxonomy coverage")
    out.append("")
    out.append("| primary class | runs |")
    out.append("|---|---:|")
    for label in PRIMARY_LABELS:
        out.append(f"| `{label}` | {inv.primary_label_counts.get(label, 0)} |")
    out.append("")
    out.append("## Stage-2 branch coverage")
    out.append("")
    out.append("| branch | faulty runs |")
    out.append("|---|---:|")
    for b in STAGE2_LABELS:
        out.append(f"| `{b}` | {inv.branch_counts.get(b, 0)} |")
    out.append("")
    out.append("## Stage-3 leaf coverage")
    out.append("")
    for branch, leaf_counts in inv.stage3_leaf_counts.items():
        out.append(f"**Branch `{branch}`**")
        out.append("")
        out.append("| leaf | runs |")
        out.append("|---|---:|")
        for leaf, n in leaf_counts.items():
            out.append(f"| `{leaf}` | {n} |")
        out.append("")
    out.append("## Severity distribution")
    out.append("")
    out.append("| severity | runs |")
    out.append("|---|---:|")
    for sev in sorted(inv.severity_counts, key=lambda s: -inv.severity_counts[s]):
        out.append(f"| `{sev}` | {inv.severity_counts[sev]} |")
    out.append("")
    out.append("## Dataset distribution")
    out.append("")
    out.append("| dataset | runs |")
    out.append("|---|---:|")
    for ds in sorted(inv.dataset_counts, key=lambda s: -inv.dataset_counts[s]):
        out.append(f"| `{ds}` | {inv.dataset_counts[ds]} |")
    out.append("")
    out.append("## Cross-tabs")
    out.append("")
    out.append("### `primary_label × severity`")
    out.append("")
    out.append(_render_2d(inv.label_x_severity, row_name="label", col_name="severity"))
    out.append("")
    out.append("### `primary_label × dataset`")
    out.append("")
    out.append(_render_2d(inv.label_x_dataset, row_name="label", col_name="dataset"))
    out.append("")
    if inv.gaps:
        out.append("## Coverage gaps")
        out.append("")
        out.append("| severity | kind | label | detail |")
        out.append("|---|---|---|---|")
        for g in inv.gaps:
            out.append(f"| `{g.severity}` | `{g.kind}` | `{g.label}` | {g.detail} |")
        out.append("")
    else:
        out.append("## Coverage gaps")
        out.append("")
        out.append("_No gaps detected — taxonomy and branch coverage are complete._")
        out.append("")
    out.append("## Summary")
    out.append("")
    n_blockers = sum(1 for g in inv.gaps if g.severity == "blocker")
    n_warnings = sum(1 for g in inv.gaps if g.severity == "warning")
    out.append(f"- taxonomy complete: **{inv.is_taxonomy_complete()}**")
    out.append(f"- branch coverage complete: **{inv.is_branch_complete()}**")
    out.append(f"- blocker gaps: **{n_blockers}**")
    out.append(f"- warning gaps: **{n_warnings}**")
    out.append("")
    return "\n".join(out)


def write_inventory(
    inv: ScenarioInventory,
    *,
    md_path: str | Path | None = None,
    json_path: str | Path | None = None,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if md_path is not None:
        p = Path(md_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(render_markdown(inv), encoding="utf-8")
        out["md"] = p
    if json_path is not None:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(inv.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        out["json"] = p
    return out


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "CoverageGap",
    "ScenarioInventory",
    "build_inventory",
    "render_markdown",
    "validate_inventory",
    "write_inventory",
]
