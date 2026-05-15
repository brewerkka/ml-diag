from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ml_diag.labels import (
    INSTABILITY,
    LABEL_NOISE,
    LEAKAGE,
    OVERFITTING,
    PRIMARY_LABELS,
    UNDERFITTING,
)


class ActionUnknownError(KeyError):
    pass


class ActionApplicabilityError(ValueError):
    pass


class ActionParameterError(ValueError):
    pass


@dataclass(frozen=True)
class ActionParameter:
    name: str
    type: str
    description: str
    default: Any = None
    min: float | int | None = None
    max: float | int | None = None
    choices: tuple[Any, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "default": self.default,
            "min": self.min,
            "max": self.max,
            "choices": list(self.choices) if self.choices else None,
        }


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    target_classes: tuple[str, ...]
    parameters: tuple[ActionParameter, ...]
    applies_to: Callable[[str, dict[str, Any]], bool]
    meta_delta_keys: tuple[str, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_classes": list(self.target_classes),
            "parameters": [p.to_dict() for p in self.parameters],
            "meta_delta_keys": list(self.meta_delta_keys),
            "notes": list(self.notes),
        }


def _ev_get(evidence: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = evidence
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _applies_overfitting(diag_class: str, evidence: dict[str, Any]) -> bool:
    if diag_class == OVERFITTING:
        return True
    gap = _ev_get(evidence, "curve_evidence.final_acc_gap")
    if isinstance(gap, (int, float)) and gap > 0.10:
        return True
    return False


def _applies_underfitting(diag_class: str, evidence: dict[str, Any]) -> bool:
    if diag_class == UNDERFITTING:
        return True
    train_loss = _ev_get(evidence, "curve_evidence.final_train_loss")
    return isinstance(train_loss, (int, float)) and train_loss > 1.0


def _applies_instability(diag_class: str, evidence: dict[str, Any]) -> bool:
    if diag_class == INSTABILITY:
        return True
    diverged = _ev_get(evidence, "curve_evidence.diverged")
    return bool(diverged)


def _applies_leakage(diag_class: str, evidence: dict[str, Any]) -> bool:
    if diag_class == LEAKAGE:
        return True
    overlap = _ev_get(evidence, "integrity_evidence.columns.di_train_val_overlap")
    return isinstance(overlap, (int, float)) and overlap > 0.01


def _applies_label_noise(diag_class: str, evidence: dict[str, Any]) -> bool:
    if diag_class == LABEL_NOISE:
        return True
    declared = _ev_get(evidence, "integrity_evidence.columns.di_label_noise_rate_declared")
    return isinstance(declared, (int, float)) and declared > 0.05


def _applies_observe(diag_class: str, evidence: dict[str, Any]) -> bool:
    return True


_REDUCE_LR = Action(
    name="reduce_lr",
    description="Reduce the learning rate by a factor; intended to stabilise diverging or oscillating runs.",
    target_classes=(INSTABILITY, OVERFITTING),
    parameters=(
        ActionParameter(
            "factor",
            "float",
            "Multiplicative LR reduction factor.",
            default=0.1,
            min=0.0001,
            max=1.0,
        ),
    ),
    applies_to=_applies_instability,
    meta_delta_keys=("lr", "learning_rate"),
    notes=("New_lr = old_lr * factor.",),
)

_INCREASE_CAPACITY = Action(
    name="increase_capacity",
    description="Increase model capacity (width / depth / parameter count) to address underfitting.",
    target_classes=(UNDERFITTING,),
    parameters=(
        ActionParameter(
            "scale", "float", "Multiplicative capacity scale.", default=2.0, min=1.0, max=8.0
        ),
    ),
    applies_to=_applies_underfitting,
    meta_delta_keys=("n_params", "hidden_size", "depth", "width"),
)

_ADD_REGULARIZATION = Action(
    name="add_regularization",
    description="Add or strengthen weight decay / dropout to combat overfitting.",
    target_classes=(OVERFITTING,),
    parameters=(
        ActionParameter(
            "weight_decay", "float", "Target weight decay value.", default=1e-4, min=0.0, max=1.0
        ),
        ActionParameter(
            "dropout", "float", "Optional dropout rate.", default=None, min=0.0, max=0.95
        ),
    ),
    applies_to=_applies_overfitting,
    meta_delta_keys=("weight_decay", "dropout"),
)

_EARLY_STOP = Action(
    name="early_stop",
    description="Stop training at the best validation epoch instead of the final epoch.",
    target_classes=(OVERFITTING,),
    parameters=(
        ActionParameter("patience", "int", "Validation-epoch patience.", default=5, min=1, max=100),
    ),
    applies_to=_applies_overfitting,
    meta_delta_keys=("early_stop", "patience", "best_epoch"),
)

_CLEAN_LABEL_NOISE = Action(
    name="clean_label_noise",
    description="Re-label or filter the training set to reduce noisy labels.",
    target_classes=(LABEL_NOISE,),
    parameters=(
        ActionParameter(
            "removed_fraction",
            "float",
            "Fraction of suspect labels removed.",
            default=0.1,
            min=0.0,
            max=0.5,
        ),
    ),
    applies_to=_applies_label_noise,
    meta_delta_keys=("label_noise_rate", "n_train", "removed_fraction"),
)

_FIX_SPLIT = Action(
    name="fix_split",
    description="Rebuild train/val/test split to remove leakage / overlap.",
    target_classes=(LEAKAGE,),
    parameters=(
        ActionParameter("split_seed", "int", "New deterministic split seed.", default=0),
        ActionParameter("dedup", "bool", "Deduplicate before re-splitting.", default=True),
    ),
    applies_to=_applies_leakage,
    meta_delta_keys=("split_seed", "train_val_overlap", "duplicate_index_overlap"),
)

_RETRAIN_WITH_SEED = Action(
    name="retrain_with_seed",
    description="Re-train with a different random seed to estimate variance / confirm an instability hypothesis.",
    target_classes=tuple(PRIMARY_LABELS),
    parameters=(ActionParameter("seed", "int", "New training seed.", default=1),),
    applies_to=_applies_observe,
    meta_delta_keys=("seed", "training_seed"),
)

_OBSERVE_ONLY = Action(
    name="observe_only",
    description="Take no action — collect more runs to disambiguate.",
    target_classes=tuple(PRIMARY_LABELS),
    parameters=(),
    applies_to=_applies_observe,
    meta_delta_keys=(),
    notes=("Always applicable; serves as the default 'no-op' recommendation.",),
)

ACTIONS: dict[str, Action] = {
    a.name: a
    for a in (
        _REDUCE_LR,
        _INCREASE_CAPACITY,
        _ADD_REGULARIZATION,
        _EARLY_STOP,
        _CLEAN_LABEL_NOISE,
        _FIX_SPLIT,
        _RETRAIN_WITH_SEED,
        _OBSERVE_ONLY,
    )
}


def list_actions() -> list[Action]:
    return list(ACTIONS.values())


def get_action(name: str) -> Action:
    if name not in ACTIONS:
        raise ActionUnknownError(f"Unknown action {name!r}. Allowed: {sorted(ACTIONS)}")
    return ACTIONS[name]


def applicable_actions(diagnosis_class: str, evidence: dict[str, Any]) -> list[Action]:
    return [a for a in ACTIONS.values() if a.applies_to(diagnosis_class, evidence)]


def recommend_actions(
    diagnosis_class: str,
    evidence: dict[str, Any],
    *,
    max_recommendations: int = 3,
) -> list[Action]:
    applicable = applicable_actions(diagnosis_class, evidence)

    def _score(a: Action) -> tuple[int, int]:
        target_match = 0 if diagnosis_class in a.target_classes else 1
        try:
            idx = list(ACTIONS).index(a.name)
        except ValueError:
            idx = 999
        return target_match, idx

    applicable.sort(key=_score)
    out = applicable[:max_recommendations]
    if not out:
        out = [ACTIONS["observe_only"]]
    return out


def validate_parameters(action: Action, params: dict[str, Any]) -> dict[str, Any]:
    known = {p.name: p for p in action.parameters}
    extra = set(params) - set(known)
    if extra:
        raise ActionParameterError(f"Action {action.name!r}: unknown parameter(s): {sorted(extra)}")
    out: dict[str, Any] = {}
    for spec in action.parameters:
        if spec.name in params:
            v = params[spec.name]
        else:
            v = spec.default
        if v is None:
            if spec.default is None:
                continue
        try:
            if spec.type == "float":
                v = float(v)
            elif spec.type == "int":
                v = int(v)
            elif spec.type == "bool":
                if isinstance(v, str):
                    v = v.lower() in ("true", "1", "yes", "y")
                else:
                    v = bool(v)
            elif spec.type == "str":
                v = str(v)
        except (TypeError, ValueError) as e:
            raise ActionParameterError(
                f"Action {action.name!r}: parameter {spec.name!r} must be {spec.type}, got {v!r}: {e}"
            ) from e
        if spec.choices is not None and v not in spec.choices:
            raise ActionParameterError(
                f"Action {action.name!r}: parameter {spec.name!r} must be in {spec.choices}, got {v!r}."
            )
        if isinstance(v, (int, float)):
            if spec.min is not None and v < spec.min:
                raise ActionParameterError(
                    f"Action {action.name!r}: {spec.name}={v} < min {spec.min}."
                )
            if spec.max is not None and v > spec.max:
                raise ActionParameterError(
                    f"Action {action.name!r}: {spec.name}={v} > max {spec.max}."
                )
        out[spec.name] = v
    return out


__all__ = [
    "ACTIONS",
    "Action",
    "ActionApplicabilityError",
    "ActionParameter",
    "ActionParameterError",
    "ActionUnknownError",
    "applicable_actions",
    "get_action",
    "list_actions",
    "recommend_actions",
    "validate_parameters",
]
