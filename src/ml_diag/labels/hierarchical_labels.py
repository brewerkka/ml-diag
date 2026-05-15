from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

HEALTHY: Final[str] = "healthy"

FAULTY: Final[str] = "faulty"

DATA_RELATED: Final[str] = "data_related"

OPT_GEN_RELATED: Final[str] = "optimization_or_generalization_related"

OVERFITTING: Final[str] = "overfitting"

UNDERFITTING: Final[str] = "underfitting"

LEAKAGE: Final[str] = "leakage"

LABEL_NOISE: Final[str] = "label_noise"

INSTABILITY: Final[str] = "instability"

PRIMARY_LABELS: Final[tuple[str, ...]] = (
    HEALTHY,
    OVERFITTING,
    UNDERFITTING,
    LEAKAGE,
    LABEL_NOISE,
    INSTABILITY,
)

STAGE1_LABELS: Final[tuple[str, ...]] = (HEALTHY, FAULTY)

STAGE2_LABELS: Final[tuple[str, ...]] = (DATA_RELATED, OPT_GEN_RELATED)

STAGE3_LABELS_BY_BRANCH: Final[Mapping[str, tuple[str, ...]]] = {
    DATA_RELATED: (LEAKAGE, LABEL_NOISE),
    OPT_GEN_RELATED: (OVERFITTING, UNDERFITTING, INSTABILITY),
}

_STAGE1_MAP: Final[Mapping[str, str]] = {
    HEALTHY: HEALTHY,
    OVERFITTING: FAULTY,
    UNDERFITTING: FAULTY,
    LEAKAGE: FAULTY,
    LABEL_NOISE: FAULTY,
    INSTABILITY: FAULTY,
}

_STAGE2_MAP: Final[Mapping[str, str]] = {
    LEAKAGE: DATA_RELATED,
    LABEL_NOISE: DATA_RELATED,
    OVERFITTING: OPT_GEN_RELATED,
    UNDERFITTING: OPT_GEN_RELATED,
    INSTABILITY: OPT_GEN_RELATED,
}

_STAGE3_MAP: Final[Mapping[str, str]] = {
    LEAKAGE: LEAKAGE,
    LABEL_NOISE: LABEL_NOISE,
    OVERFITTING: OVERFITTING,
    UNDERFITTING: UNDERFITTING,
    INSTABILITY: INSTABILITY,
}


class UnknownPrimaryLabel(ValueError):
    pass


@dataclass(frozen=True)
class HierarchicalLabel:
    primary: str
    stage1: str
    stage2: str | None
    stage3: str | None

    def is_healthy(self) -> bool:
        return self.stage1 == HEALTHY

    def stage2_branch(self) -> str | None:
        return self.stage2


def _check_known(label: str) -> None:
    if label not in PRIMARY_LABELS:
        raise UnknownPrimaryLabel(
            f"Unknown primary label {label!r}; expected one of {PRIMARY_LABELS}."
        )


def to_stage1(primary: str) -> str:
    _check_known(primary)
    return _STAGE1_MAP[primary]


def to_stage2(primary: str) -> str | None:
    _check_known(primary)
    if primary == HEALTHY:
        return None
    return _STAGE2_MAP[primary]


def to_stage3(primary: str) -> str | None:
    _check_known(primary)
    if primary == HEALTHY:
        return None
    return _STAGE3_MAP[primary]


def to_hierarchical(primary: str) -> HierarchicalLabel:
    return HierarchicalLabel(
        primary=primary,
        stage1=to_stage1(primary),
        stage2=to_stage2(primary),
        stage3=to_stage3(primary),
    )


def stage3_vocab(branch: str) -> tuple[str, ...]:
    if branch not in STAGE3_LABELS_BY_BRANCH:
        raise ValueError(
            f"Unknown Stage-2 branch {branch!r}; expected one of {tuple(STAGE3_LABELS_BY_BRANCH)}."
        )
    return STAGE3_LABELS_BY_BRANCH[branch]


def validate_schema() -> None:
    assert set(PRIMARY_LABELS) == set(_STAGE1_MAP), (
        f"Stage-1 map missing keys: {set(PRIMARY_LABELS) - set(_STAGE1_MAP)}"
    )
    faulty = set(PRIMARY_LABELS) - {HEALTHY}
    assert set(_STAGE2_MAP) == faulty, (
        f"Stage-2 map mismatch: faulty={faulty}, map={set(_STAGE2_MAP)}"
    )
    assert set(_STAGE2_MAP.values()) == set(STAGE2_LABELS), (
        f"Stage-2 values must equal {STAGE2_LABELS}, got {set(_STAGE2_MAP.values())}"
    )
    assert set(_STAGE3_MAP) == faulty, (
        f"Stage-3 map mismatch: faulty={faulty}, map={set(_STAGE3_MAP)}"
    )
    for branch, vocab in STAGE3_LABELS_BY_BRANCH.items():
        in_branch = {p for p, s in _STAGE2_MAP.items() if s == branch}
        assert set(vocab) == in_branch, (
            f"Stage-3 vocab for branch {branch!r} = {set(vocab)} "
            f"does not match labels mapped into it = {in_branch}"
        )


def label_distribution(primaries: Iterable[str]) -> dict[str, dict[str, int]]:
    stage1: dict[str, int] = {label: 0 for label in STAGE1_LABELS}
    stage2: dict[str, int] = {label: 0 for label in STAGE2_LABELS}
    stage3: dict[str, dict[str, int]] = {
        branch: {leaf: 0 for leaf in vocab} for branch, vocab in STAGE3_LABELS_BY_BRANCH.items()
    }
    for primary in primaries:
        if primary is None:
            continue
        try:
            h = to_hierarchical(str(primary))
        except UnknownPrimaryLabel:
            continue
        stage1[h.stage1] = stage1.get(h.stage1, 0) + 1
        if h.stage2 is not None:
            stage2[h.stage2] = stage2.get(h.stage2, 0) + 1
        if h.stage2 is not None and h.stage3 is not None:
            stage3[h.stage2][h.stage3] = stage3[h.stage2].get(h.stage3, 0) + 1
    return {
        "stage1": stage1,
        "stage2": stage2,
        "stage3": {                           
            branch: dict(counts) for branch, counts in stage3.items()
        },
    }


__all__ = [
    "PRIMARY_LABELS",
    "STAGE1_LABELS",
    "STAGE2_LABELS",
    "STAGE3_LABELS_BY_BRANCH",
    "HEALTHY",
    "FAULTY",
    "DATA_RELATED",
    "OPT_GEN_RELATED",
    "OVERFITTING",
    "UNDERFITTING",
    "LEAKAGE",
    "LABEL_NOISE",
    "INSTABILITY",
    "HierarchicalLabel",
    "UnknownPrimaryLabel",
    "to_stage1",
    "to_stage2",
    "to_stage3",
    "to_hierarchical",
    "stage3_vocab",
    "validate_schema",
    "label_distribution",
]
