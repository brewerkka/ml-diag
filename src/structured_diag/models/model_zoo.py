from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[int], Pipeline | object]
    needs_scaling: bool


def _logreg(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=None,
                ),
            ),
        ]
    )


def _random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _gradient_boosting(seed: int) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=seed,
    )


def _maybe_catboost(seed: int):  # type: ignore[no-untyped-def]
    try:
        from catboost import CatBoostClassifier  # type: ignore[import-not-found]
    except Exception:
        return None
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=seed,
        verbose=False,
        auto_class_weights="Balanced",
        allow_writing_files=False,
    )


def default_zoo(*, include_catboost: bool = True) -> Sequence[ModelSpec]:
    specs = [
        ModelSpec("logreg", _logreg, needs_scaling=True),
        ModelSpec("random_forest", _random_forest, needs_scaling=False),
        ModelSpec("gradient_boosting", _gradient_boosting, needs_scaling=False),
    ]
    if include_catboost:
        specs.append(ModelSpec("catboost", _maybe_catboost, needs_scaling=False))
    return specs


__all__ = ["ModelSpec", "default_zoo"]
