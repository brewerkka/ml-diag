from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from structured_diag.data import CorpusManifest, load_manifest, load_runs_table
from structured_diag.utils.logging import get_logger

_LOG = get_logger(__name__)

DEFAULT_CORE_SEVERITIES: tuple[str, ...] = ("moderate", "severe")

DEFAULT_EXTENDED_SEVERITIES: tuple[str, ...] = ("mild",)

DEFAULT_MIN_HISTORY_EPOCHS: int = 5

DEFAULT_NEAR_SATURATED_DATASETS: tuple[str, ...] = ()


@dataclass(frozen=True)
class PartitionRules:
    core_severities: tuple[str, ...] = DEFAULT_CORE_SEVERITIES
    extended_severities: tuple[str, ...] = DEFAULT_EXTENDED_SEVERITIES
    require_single_label_for_core: bool = True
    min_history_epochs: int = DEFAULT_MIN_HISTORY_EPOCHS
    near_saturated_datasets: tuple[str, ...] = DEFAULT_NEAR_SATURATED_DATASETS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["core_severities"] = list(self.core_severities)
        d["extended_severities"] = list(self.extended_severities)
        d["near_saturated_datasets"] = list(self.near_saturated_datasets)
        return d


@dataclass(frozen=True)
class PartitionResult:
    corpus_name: str
    corpus_path: Path
    rules: PartitionRules
    table: pd.DataFrame

    @property
    def core_ids(self) -> list[str]:
        return self.table.loc[self.table["slice"] == "core", "run_id"].tolist()

    @property
    def extended_ids(self) -> list[str]:
        return self.table.loc[self.table["slice"] == "extended", "run_id"].tolist()

    @property
    def n_core(self) -> int:
        return int((self.table["slice"] == "core").sum())

    @property
    def n_extended(self) -> int:
        return int((self.table["slice"] == "extended").sum())

    def class_balance(self, slice_name: str) -> dict[str, int]:
        sub = self.table[self.table["slice"] == slice_name]
        return dict(Counter(sub["primary_label"].tolist()))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "corpus_name": self.corpus_name,
            "corpus_path": str(self.corpus_path),
            "rules": self.rules.to_dict(),
            "counts": {
                "total": int(len(self.table)),
                "core": self.n_core,
                "extended": self.n_extended,
            },
            "class_balance": {
                "core": self.class_balance("core"),
                "extended": self.class_balance("extended"),
            },
            "reason_counts": _count_reasons(self.table),
            "core_run_ids": self.core_ids,
            "extended_run_ids": self.extended_ids,
        }

    def save_summary(self, out_path: str | Path) -> Path:
        out_path = Path(out_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.to_summary_dict(), indent=2), encoding="utf-8")
        _LOG.info("Wrote partition summary to %s", out_path)
        return out_path


def _count_reasons(table: pd.DataFrame) -> dict[str, int]:
    c: Counter[str] = Counter()
    for reasons in table["reasons"]:
        for r in reasons:
            c[r] += 1
    return dict(c.most_common())


def _rule_multi_label(row: pd.Series, rules: PartitionRules) -> list[str]:
    if rules.require_single_label_for_core and bool(row.get("is_multi_label", False)):
        return ["multi_label"]
    return []


def _rule_severity_mild(row: pd.Series, rules: PartitionRules) -> list[str]:
    sev = row.get("severity")
    if isinstance(sev, str) and sev.lower() in rules.extended_severities:
        return ["severity_mild"]
    return []


def _rule_severity_unknown(row: pd.Series, rules: PartitionRules) -> list[str]:
    sev = row.get("severity")
    label = row.get("primary_label")
    if sev is None or (isinstance(sev, float) and pd.isna(sev)) or sev == "":
        return ["severity_unknown"]
    if not isinstance(sev, str):
        return ["severity_unknown"]
    sev_l = sev.lower()
    if sev_l == "none":
        return [] if label == "healthy" else ["severity_none_on_faulty"]
    if sev_l not in (*rules.core_severities, *rules.extended_severities):
        return ["severity_out_of_vocabulary"]
    return []


def _rule_short_history(row: pd.Series, rules: PartitionRules) -> list[str]:
    n = row.get("n_history_rows")
    try:
        n_int = int(n)
    except (TypeError, ValueError):
        return ["history_unparseable"]
    if n_int < rules.min_history_epochs:
        return [f"short_history_lt_{rules.min_history_epochs}"]
    return []


def _rule_trivial_mild_on_saturated(row: pd.Series, rules: PartitionRules) -> list[str]:
    sev = row.get("severity")
    ds = row.get("dataset")
    if (
        isinstance(sev, str)
        and sev.lower() == "mild"
        and isinstance(ds, str)
        and ds in rules.near_saturated_datasets
    ):
        return ["mild_on_near_saturated"]
    return []


def _rule_unlabeled(row: pd.Series, rules: PartitionRules) -> list[str]:
    label = row.get("primary_label")
    if label is None or (isinstance(label, float) and pd.isna(label)) or label == "":
        return ["unlabeled"]
    return []


_EXTENDED_RULES: Sequence = (
    _rule_unlabeled,
    _rule_multi_label,
    _rule_severity_mild,
    _rule_severity_unknown,
    _rule_short_history,
    _rule_trivial_mild_on_saturated,
)


def assign_slice(row: pd.Series, rules: PartitionRules) -> tuple[str, list[str]]:
    reasons: list[str] = []
    for rule in _EXTENDED_RULES:
        reasons.extend(rule(row, rules))
    if reasons:
        return "extended", reasons
    label = row.get("primary_label")
    sev = row.get("severity")
    sev_l = sev.lower() if isinstance(sev, str) else None
    if label == "healthy" and sev_l in (None, "none", *rules.core_severities):
        return "core", []
    if sev_l in rules.core_severities:
        return "core", []
    return "extended", ["severity_not_in_core_whitelist"]


def partition_corpus(
    corpus: str | Path | CorpusManifest,
    rules: PartitionRules | None = None,
    *,
    skip_broken: bool = False,
) -> PartitionResult:
    rules = rules or PartitionRules()
    if isinstance(corpus, CorpusManifest):
        manifest = corpus
    else:
        manifest = load_manifest(corpus)
    table = load_runs_table(manifest, skip_broken=skip_broken).copy()
    slices: list[str] = []
    reasons_col: list[list[str]] = []
    for _, row in table.iterrows():
        slc, reasons = assign_slice(row, rules)
        slices.append(slc)
        reasons_col.append(reasons)
    table["slice"] = slices
    table["reasons"] = reasons_col
    n_core = int((table["slice"] == "core").sum())
    n_ext = int((table["slice"] == "extended").sum())
    _LOG.info(
        "Partitioned %s: core=%d, extended=%d (total=%d).",
        manifest.name,
        n_core,
        n_ext,
        len(table),
    )
    return PartitionResult(
        corpus_name=manifest.name,
        corpus_path=manifest.corpus_path,
        rules=rules,
        table=table,
    )


def rules_from_mapping(payload: dict[str, Any] | None) -> PartitionRules:
    if not payload:
        return PartitionRules()

    def _seq(x: Iterable[Any] | None, default: tuple[str, ...]) -> tuple[str, ...]:
        if x is None:
            return default
        return tuple(str(v) for v in x)

    return PartitionRules(
        core_severities=_seq(payload.get("core_severities"), DEFAULT_CORE_SEVERITIES),
        extended_severities=_seq(payload.get("extended_severities"), DEFAULT_EXTENDED_SEVERITIES),
        require_single_label_for_core=bool(payload.get("require_single_label_for_core", True)),
        min_history_epochs=int(payload.get("min_history_epochs", DEFAULT_MIN_HISTORY_EPOCHS)),
        near_saturated_datasets=_seq(
            payload.get("near_saturated_datasets"), DEFAULT_NEAR_SATURATED_DATASETS
        ),
    )


__all__ = [
    "PartitionRules",
    "PartitionResult",
    "assign_slice",
    "partition_corpus",
    "rules_from_mapping",
]


def _benchmark_init_export() -> None:  # pragma: no cover - documentation aid
    pass
