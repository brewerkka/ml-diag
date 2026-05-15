from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ml_diag.data import load_run, manifest_for_single_run
from ml_diag.evaluation import build_evidence, write_case_outputs
from ml_diag.features import build_data_integrity_features, build_feature_table
from ml_diag.interpretation import InterpretationConfig, interpret
from ml_diag.models import diagnose_one, load_cascade


def default_artifacts_dir() -> Path:
    """Путь к предобученному каскаду, вшитому в пакет."""
    return Path(__file__).resolve().parent / "_artifacts" / "cascade_default"


@dataclass(frozen=True)
class Diagnosis:
    """Результат диагностики одного запуска. Возвращается функцией :func:`diagnose`."""

    run_id: str
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]
    class_probabilities: dict[str, float]
    summary: str
    explanation: str
    symptoms: list[str]
    recommendations: list[dict[str, Any]]
    warnings: list[str] = field(default_factory=list)
    backend: str = "template"
    evidence: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "label": self.label,
            "confidence": float(self.confidence),
            "alternatives": [[c, float(p)] for c, p in self.alternatives],
            "class_probabilities": {k: float(v) for k, v in self.class_probabilities.items()},
            "summary": self.summary,
            "explanation": self.explanation,
            "symptoms": list(self.symptoms),
            "recommendations": [dict(r) for r in self.recommendations],
            "warnings": list(self.warnings),
            "backend": self.backend,
        }

    def save(self, out_dir: str | Path) -> dict[str, Path]:
        """Записать полный отчёт (JSON + Markdown + curves.png) в ``out_dir``."""
        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        diagnosis_obj = self.raw["diagnosis"]
        interpretation_obj = self.raw["interpretation"]
        evidence_obj = self.raw["evidence"]
        return write_case_outputs(
            out_dir,
            diagnosis=diagnosis_obj,
            evidence=evidence_obj,
            interpretation=interpretation_obj,
            patch_summary=None,
            run_dir=self.raw.get("run_dir"),
            extras={
                "interpretation_backend": self.backend,
                "input_mode": "high_level_api",
            },
        )

    def __repr__(self) -> str:
        top_alt = ", ".join(f"{c}={p:.2f}" for c, p in self.alternatives[:3])
        return (
            f"Diagnosis(run_id={self.run_id!r}, label={self.label!r}, "
            f"confidence={self.confidence:.3f}, alternatives=[{top_alt}])"
        )


_REQUIRED_META_FOR_DIAGNOSE = ("run_id",)


def _materialize_run_dir(
    *,
    meta: Mapping[str, Any],
    history: pd.DataFrame | str | Path,
    tmp_root: Path,
) -> Path:
    for key in _REQUIRED_META_FOR_DIAGNOSE:
        if key not in meta:
            raise ValueError(f"В meta отсутствует обязательный ключ: {key!r}")
    run_id = str(meta["run_id"])
    rd = tmp_root / run_id
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "meta.json").write_text(json.dumps(dict(meta), ensure_ascii=False, indent=2))
    if isinstance(history, pd.DataFrame):
        history.to_csv(rd / "history.csv", index=False)
    else:
        src = Path(history).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Файл history не найден: {src}")
        shutil.copyfile(src, rd / "history.csv")
    return rd


def diagnose(
    run_dir: str | Path | None = None,
    *,
    meta: Mapping[str, Any] | None = None,
    history: pd.DataFrame | str | Path | None = None,
    artifacts: str | Path | None = None,
    backend: str = "template",
    model: str | None = None,
    max_recommendations: int = 3,
    language: str = "ru",
    include_integrity: bool = True,
) -> Diagnosis:
    """Диагностика одного запуска обучения.

    Принимает либо ``run_dir`` (папку с ``meta.json`` + ``history.csv``),
    либо ``meta`` (dict) + ``history`` (DataFrame или путь к CSV).
    Возвращает :class:`Diagnosis` с диагнозом, уверенностью, рекомендациями
    и методом ``.save(out_dir)`` для записи отчёта на диск.

    backend: ``"template"`` (по умолчанию, без LLM), ``"auto"``
    (LLM-цепочка с fallback на template), ``"groq"``, ``"ollama"``.
    language: ``"ru"`` или ``"en"``.
    """
    if run_dir is not None and (meta is not None or history is not None):
        raise ValueError("Передайте либо run_dir, либо meta+history — не оба одновременно.")
    if run_dir is None and (meta is None or history is None):
        raise ValueError("Передайте run_dir либо одновременно meta= и history=.")

    artifacts_path = Path(artifacts).expanduser().resolve() if artifacts else default_artifacts_dir()
    if not artifacts_path.is_dir():
        raise FileNotFoundError(
            f"Папка с артефактами каскада не найдена: {artifacts_path}. "
            "Либо передайте свой artifacts=..., либо переустановите ml_diag "
            "из PyPI (пакет идёт с предобученным каскадом по умолчанию)."
        )

    tmp_context: tempfile.TemporaryDirectory | None = None
    try:
        if run_dir is not None:
            resolved_run_dir = Path(run_dir).expanduser().resolve()
            if not resolved_run_dir.is_dir():
                raise FileNotFoundError(f"run_dir не существует: {resolved_run_dir}")
        else:
            tmp_context = tempfile.TemporaryDirectory(prefix="ml_diag_run_")
            resolved_run_dir = _materialize_run_dir(
                meta=meta, history=history, tmp_root=Path(tmp_context.name)
            )

        cascade = load_cascade(artifacts_path)
        rec = load_run(resolved_run_dir)
        resolved_run_id = rec.run_id
        manifest = manifest_for_single_run(resolved_run_dir)

        base = build_feature_table(manifest)
        integrity_columns = None
        full_df = base.df
        if include_integrity:
            try:
                di = build_data_integrity_features(manifest, base_table=base)
                integrity_columns = di.integrity_columns
                full_df = di.df
            except Exception:
                integrity_columns = None
                full_df = base.df

        if resolved_run_id not in full_df.index:
            raise RuntimeError(
                f"run_id {resolved_run_id!r} не прошёл извлечение признаков "
                f"(возможно, history слишком короткая или повреждена)."
            )

        x_row = full_df.loc[resolved_run_id, base.feature_columns]
        diagnosis_obj = diagnose_one(cascade, run_id=resolved_run_id, x_row=x_row)
        full_row = full_df.loc[resolved_run_id]
        evidence_obj = build_evidence(
            diagnosis=diagnosis_obj,
            feature_row=full_row,
            cascade=cascade,
            integrity_columns=integrity_columns,
        )
        cfg = InterpretationConfig(
            backend=backend,
            model=model,
            max_recommendations=max_recommendations,
            language=language,
            cache_dir=None,
        )
        interpretation_obj = interpret(
            diagnosis=diagnosis_obj,
            evidence=evidence_obj,
            config=cfg,
        )

        return Diagnosis(
            run_id=resolved_run_id,
            label=diagnosis_obj.final_class,
            confidence=float(diagnosis_obj.final_confidence),
            alternatives=[(c, float(p)) for c, p in diagnosis_obj.alternative_hypotheses],
            class_probabilities={
                k: float(v) for k, v in diagnosis_obj.class_probabilities.items()
            },
            summary=interpretation_obj.summary,
            explanation=interpretation_obj.explanation,
            symptoms=list(interpretation_obj.symptoms),
            recommendations=[r.to_dict() for r in interpretation_obj.recommendations],
            warnings=list(interpretation_obj.warnings),
            backend=interpretation_obj.backend,
            evidence=evidence_obj.to_dict() if hasattr(evidence_obj, "to_dict") else {},
            raw={
                "diagnosis": diagnosis_obj,
                "evidence": evidence_obj,
                "interpretation": interpretation_obj,
                "cascade": cascade,
                "run_dir": resolved_run_dir if run_dir is not None else None,
            },
        )
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()


__all__ = ["Diagnosis", "diagnose", "default_artifacts_dir"]
