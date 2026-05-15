from __future__ import annotations

import io
import json
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_SRC = _REPO_ROOT / "src"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from ml_diag.evaluation import validate_case_dir
from ml_diag.evaluation.case_outputs import CASE_OUTPUTS_SCHEMA_VERSION
from ml_diag.ui_helpers import (
    CaseResult,
    diagnose_case,
    list_corpus_run_ids,
)

_T: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "ml_diag",
        "page_subtitle": "_A thin visual wrapper over the existing diagnostic pipeline._",
        "language": "Language",
        "config_header": "Configuration",
        "input_mode": "Input mode",
        "input_mode_help": (
            "Upload files: drop your training run's history.csv (and "
            "optionally meta.json) right here.\n"
            "External run: a directory on disk with meta.json + history.csv "
            "produced by ml_diag.logging_sdk.RunLogger.\n"
            "Corpus run: a run_id from an ml_diag-style corpus."
        ),
        "mode_upload": "Upload files",
        "mode_external": "External run",
        "mode_corpus": "Corpus run",
        "drop_files": "**Drop your training files**",
        "use_zip": "Upload one ZIP with the run folder",
        "use_zip_help": (
            "If enabled, upload a single ZIP that contains meta.json and "
            "history.csv (optionally inside a subfolder)."
        ),
        "zip_label": "Run folder (.zip)",
        "zip_help": "ZIP with meta.json + history.csv at the top level or one folder deep.",
        "history_label": "history.csv  (required)",
        "history_help": "Per-epoch metrics with columns like epoch, train_loss, val_loss, train_acc, val_acc.",
        "meta_label": "meta.json  (optional)",
        "meta_help": "Run-level metadata. If omitted, fill in the form below.",
        "form_expander": "Run metadata (used when meta.json is omitted)",
        "form_dataset": "Dataset",
        "form_dataset_ph": "e.g. sklearn:wine, my_table_v3",
        "form_model": "Model",
        "form_model_ph": "e.g. mlp_h32_h16",
        "form_framework": "Framework",
        "form_framework_ph": "e.g. pytorch, tensorflow",
        "form_optimizer": "Optimizer",
        "form_optimizer_ph": "e.g. adam, sgd",
        "form_lr": "Learning rate",
        "form_lr_ph": "e.g. 0.001 (leave empty if unknown)",
        "form_bs": "Batch size",
        "form_bs_ph": "e.g. 64 (leave empty if unknown)",
        "form_epochs": "Epochs planned",
        "form_epochs_ph": "leave empty if unknown",
        "form_status": "Status",
        "form_notes": "Notes",
        "form_notes_ph": "Any free-form context.",
        "layout_expander": "Expected file layout",
        "ext_run_dir": "Run directory",
        "ext_run_dir_help": "Path to a directory containing meta.json + history.csv.",
        "corpus_path": "Corpus path",
        "corpus_path_help": "Path to a directory containing corpus.manifest.json.",
        "run_id": "Run ID",
        "run_id_help_no_enum": "(corpus path could not be enumerated)",
        "corpus_size": "Corpus has **{n}** runs.",
        "max_recs": "Max recommendations",
        "no_integrity": "Disable integrity feature layer",
        "no_integrity_help": "Useful only for debugging; leave unchecked for normal use.",
        "write_outputs": "Write case directory to disk",
        "write_outputs_help": "When checked, writes results/cases/<run_id>/ with the canonical layout.",
        "run_btn": "Run diagnosis",
        "welcome": (
            "Configure inputs in the sidebar and click **Run diagnosis**.\n\n"
            "* **Upload files** mode lets you drop your `history.csv` (and "
            "optionally `meta.json`) right here — no filesystem path required.\n"
            "* **External run** mode reads a folder on disk that was logged "
            "by `ml_diag.logging_sdk.RunLogger`.\n"
            "* **Corpus run** mode reads a run from an ml_diag-style corpus."
        ),
        "interp_text_lang_note": (
            "_The language toggle drives both the UI chrome and the "
            "interpretation text (Summary / Explanation / Recommendations)._"
        ),
        "tab_overview": "Overview",
        "tab_curves": "Curves",
        "tab_diagnosis": "Diagnosis",
        "tab_evidence": "Evidence",
        "tab_interpretation": "Interpretation",
        "tab_files": "Files",
        "ovr_run_overview": "Run overview",
        "ovr_pipeline_meta": "**Pipeline metadata**",
        "ovr_backend_caption_llm": "Interpretation backend that actually fired: **{name} (LLM-generated text)**",
        "ovr_backend_caption_template": "Interpretation backend that actually fired: **template (deterministic fallback — no LLM call)**",
        "ovr_notes": "**Notes**",
        "ovr_tags": "**Tags**: ",
        "curves_header": "Training curves",
        "curves_no_history": "No history available for this run.",
        "curves_loss": "**Loss**",
        "curves_accuracy": "**Accuracy**",
        "curves_aux": "**Auxiliary**",
        "curves_history_raw": "**History (raw rows)**",
        "diag_header": "Hierarchical diagnosis",
        "diag_final_class": "Final class",
        "diag_composed_conf": "Composed confidence",
        "diag_class_probs": "**Class probabilities (composed marginal)**",
        "diag_top_alts": "**Top alternatives (excluding final class)**",
        "diag_none": "None.",
        "diag_stage_trace": "**Stage trace**",
        "ev_header": "Structured evidence",
        "ev_top_features": "**Top contributing features (per stage)**",
        "ev_top_features_unavail": "Top-feature info unavailable for this stage's model.",
        "ev_curve_evidence": "**Curve evidence**",
        "ev_curve_notes": "**Curve notes**",
        "ev_integrity_evidence": "**Integrity evidence (`di_*` columns)**",
        "ev_integrity_unavail": "No integrity columns recorded for this run.",
        "ev_integrity_notes": "**Integrity notes**",
        "ev_rejected": "**Rejected hypotheses (ruled out by an earlier stage)**",
        "interp_header": "Interpretation",
        "interp_summary": "**Summary**",
        "interp_explanation": "**Explanation**",
        "interp_symptoms": "**Symptoms / supporting evidence**",
        "interp_recs": "**Recommendations** (allowlisted actions only)",
        "interp_conf_notes": "**Confidence notes**",
        "interp_warnings": "**Warnings**",
        "interp_limitations": "**Limitations**",
        "interp_patch": "### Patch verification",
        "interp_patch_outcome": "Outcome",
        "files_header": "Case files",
        "files_not_written": (
            "Outputs were not written to disk (the **Write case directory** "
            "checkbox is off). Re-run with that option enabled to download files."
        ),
        "files_case_dir": "Case directory: `{path}`",
        "files_schema_ok": "Schema-valid (schema_version=`{ver}`).",
        "files_validation_issues": "{n} validation issue(s):",
        "files_downloads": "**Downloads**",
        "files_whole_zip": "**Whole case folder**",
        "files_zip_btn": "⬇ download case folder as .zip",
        "files_preview": "Preview case_summary.md",
        "err_artifacts_missing": (
            "Cascade artifacts directory does not exist: `{path}`. Train the cascade first."
        ),
        "err_run_dir_missing": "Run directory does not exist: `{path}`.",
        "err_corpus_missing": "Corpus directory does not exist: `{path}`.",
        "err_pick_run_id": "Pick a run_id from the corpus.",
        "err_drop_zip": "Drop a ZIP file with the run folder.",
        "err_drop_history": "Drop a `history.csv` file (it is required).",
        "err_upload_rejected": "Upload rejected: {msg}",
        "err_upload_failed": "Upload failed: {msg}",
        "err_diag_failed": "Diagnosis failed: {msg}",
        "err_file_not_found": "File not found: {msg}",
        "err_run_not_in_table": "Run not found in feature table: {msg}",
        "info_uploaded_at": "Uploaded run materialised at `{path}`.",
        "spinner_diagnosing": "Running diagnosis…",
    },
    "ru": {
        "page_title": "ml_diag",
        "page_subtitle": "_Тонкая визуальная обёртка над диагностическим пайплайном._",
        "language": "Язык",
        "config_header": "Параметры",
        "input_mode": "Источник запуска",
        "input_mode_help": (
            "Upload files: загрузите history.csv (и при желании meta.json) "
            "вашего обучающего запуска прямо сюда.\n"
            "External run: путь к папке с meta.json + history.csv, "
            "записанной ml_diag.logging_sdk.RunLogger.\n"
            "Corpus run: run_id из benchmark-корпуса ml_diag."
        ),
        "mode_upload": "Загрузить файлы",
        "mode_external": "Внешний запуск",
        "mode_corpus": "Корпусный запуск",
        "drop_files": "**Загрузите файлы обучения**",
        "use_zip": "Загрузить ZIP с папкой запуска",
        "use_zip_help": (
            "Если включено — загружается один ZIP с meta.json и history.csv "
            "(на верхнем уровне или внутри одной вложенной папки)."
        ),
        "zip_label": "Папка запуска (.zip)",
        "zip_help": "ZIP с meta.json + history.csv на верхнем уровне или внутри одной папки.",
        "history_label": "history.csv  (обязательно)",
        "history_help": "Поэпохные метрики со столбцами вроде epoch, train_loss, val_loss, train_acc, val_acc.",
        "meta_label": "meta.json  (опционально)",
        "meta_help": "Метаданные запуска. Если файла нет — заполните форму ниже.",
        "form_expander": "Метаданные запуска (если meta.json не загружен)",
        "form_dataset": "Датасет",
        "form_dataset_ph": "напр. sklearn:wine, my_table_v3",
        "form_model": "Модель",
        "form_model_ph": "напр. mlp_h32_h16",
        "form_framework": "Фреймворк",
        "form_framework_ph": "напр. pytorch, tensorflow",
        "form_optimizer": "Оптимизатор",
        "form_optimizer_ph": "напр. adam, sgd",
        "form_lr": "Learning rate",
        "form_lr_ph": "напр. 0.001 (можно оставить пустым)",
        "form_bs": "Batch size",
        "form_bs_ph": "напр. 64 (можно оставить пустым)",
        "form_epochs": "Запланировано эпох",
        "form_epochs_ph": "можно оставить пустым",
        "form_status": "Статус",
        "form_notes": "Заметки",
        "form_notes_ph": "Произвольный контекст.",
        "layout_expander": "Ожидаемая структура файлов",
        "ext_run_dir": "Папка запуска",
        "ext_run_dir_help": "Путь к папке с meta.json + history.csv.",
        "corpus_path": "Путь к корпусу",
        "corpus_path_help": "Путь к папке с corpus.manifest.json.",
        "run_id": "Run ID",
        "run_id_help_no_enum": "(корпус не удалось перечислить)",
        "corpus_size": "В корпусе **{n}** запусков.",
        "max_recs": "Макс. число рекомендаций",
        "no_integrity": "Отключить слой data-integrity признаков",
        "no_integrity_help": "Только для отладки; для обычной работы оставьте выключенным.",
        "write_outputs": "Записывать case-папку на диск",
        "write_outputs_help": "Если включено — пишется results/cases/<run_id>/ в каноническом формате.",
        "run_btn": "Запустить диагностику",
        "welcome": (
            "Настройте параметры в сайдбаре и нажмите **Запустить диагностику**.\n\n"
            "* **Загрузить файлы** — дропните `history.csv` (и опционально "
            "`meta.json`) прямо сюда, без указания пути на диске.\n"
            "* **Внешний запуск** — путь к папке, записанной "
            "`ml_diag.logging_sdk.RunLogger`.\n"
            "* **Корпусный запуск** — выбор run_id из benchmark-корпуса ml_diag."
        ),
        "interp_text_lang_note": (
            "_Переключатель языка влияет и на интерфейс, и на текст "
            "интерпретации (Summary / Explanation / Recommendations)._"
        ),
        "tab_overview": "Обзор",
        "tab_curves": "Кривые",
        "tab_diagnosis": "Диагноз",
        "tab_evidence": "Evidence",
        "tab_interpretation": "Интерпретация",
        "tab_files": "Файлы",
        "ovr_run_overview": "Обзор запуска",
        "ovr_pipeline_meta": "**Метаданные пайплайна**",
        "ovr_backend_caption_llm": "Сработавший бэкенд интерпретации: **{name} (текст сгенерирован LLM)**",
        "ovr_backend_caption_template": "Сработавший бэкенд интерпретации: **template (детерминированный fallback, LLM не вызывалась)**",
        "ovr_notes": "**Заметки**",
        "ovr_tags": "**Метки**: ",
        "curves_header": "Кривые обучения",
        "curves_no_history": "Для этого запуска нет истории обучения.",
        "curves_loss": "**Loss**",
        "curves_accuracy": "**Accuracy**",
        "curves_aux": "**Вспомогательные**",
        "curves_history_raw": "**История (исходные строки)**",
        "diag_header": "Иерархический диагноз",
        "diag_final_class": "Итоговый класс",
        "diag_composed_conf": "Композиционная уверенность",
        "diag_class_probs": "**Вероятности классов (композиционная маргинальная)**",
        "diag_top_alts": "**Топ альтернатив (исключая итоговый класс)**",
        "diag_none": "Нет.",
        "diag_stage_trace": "**Stage trace**",
        "ev_header": "Структурированные доказательства",
        "ev_top_features": "**Топ-признаки по стадиям**",
        "ev_top_features_unavail": "Информация о top-признаках недоступна для модели этой стадии.",
        "ev_curve_evidence": "**Свидетельства из кривых**",
        "ev_curve_notes": "**Заметки по кривым**",
        "ev_integrity_evidence": "**Data-integrity свидетельства (`di_*` столбцы)**",
        "ev_integrity_unavail": "Для этого запуска integrity-столбцы не записаны.",
        "ev_integrity_notes": "**Заметки по data-integrity**",
        "ev_rejected": "**Отвергнутые гипотезы (исключены ранней стадией)**",
        "interp_header": "Интерпретация",
        "interp_summary": "**Резюме**",
        "interp_explanation": "**Объяснение**",
        "interp_symptoms": "**Симптомы / подтверждающие сигналы**",
        "interp_recs": "**Рекомендации** (только из allowlist)",
        "interp_conf_notes": "**Заметки об уверенности**",
        "interp_warnings": "**Предупреждения**",
        "interp_limitations": "**Ограничения**",
        "interp_patch": "### Верификация патча",
        "interp_patch_outcome": "Исход",
        "files_header": "Файлы кейса",
        "files_not_written": (
            "Файлы не записаны на диск (галочка **Записывать case-папку на диск** "
            "выключена). Включите её и запустите снова, чтобы скачать файлы."
        ),
        "files_case_dir": "Папка кейса: `{path}`",
        "files_schema_ok": "Схема валидна (schema_version=`{ver}`).",
        "files_validation_issues": "Найдено {n} проблем(ы) валидации:",
        "files_downloads": "**Файлы для скачивания**",
        "files_whole_zip": "**Вся папка кейса**",
        "files_zip_btn": "⬇ скачать папку кейса в .zip",
        "files_preview": "Превью case_summary.md",
        "err_artifacts_missing": (
            "Папка с артефактами каскада не существует: `{path}`. Сначала обучите каскад."
        ),
        "err_run_dir_missing": "Папка запуска не существует: `{path}`.",
        "err_corpus_missing": "Папка корпуса не существует: `{path}`.",
        "err_pick_run_id": "Выберите run_id из корпуса.",
        "err_drop_zip": "Загрузите ZIP-файл с папкой запуска.",
        "err_drop_history": "Загрузите `history.csv` (он обязателен).",
        "err_upload_rejected": "Загрузка отклонена: {msg}",
        "err_upload_failed": "Загрузка не удалась: {msg}",
        "err_diag_failed": "Диагностика провалилась: {msg}",
        "err_file_not_found": "Файл не найден: {msg}",
        "err_run_not_in_table": "Запуск не найден в таблице признаков: {msg}",
        "info_uploaded_at": "Загруженный запуск материализован в `{path}`.",
        "spinner_diagnosing": "Запускаю диагностику…",
    },
}


def _t(key: str, **fmt) -> str:
    lang = st.session_state.get("ui_lang", "en")
    table = _T.get(lang) or _T["en"]
    s = table.get(key) or _T["en"].get(key) or key
    return s.format(**fmt) if fmt else s


_EXPECTED_HISTORY_COLS = (
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "lr",
    "grad_norm",
    "weight_norm",
)


def _uploads_root() -> Path:
    root = st.session_state.get("_uploads_root")
    if root is None:
        root = Path(tempfile.gettempdir()) / "ml_diag_ui_uploads"
        root.mkdir(parents=True, exist_ok=True)
        st.session_state["_uploads_root"] = root
    return root


def _new_upload_dir() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    sub = _uploads_root() / f"upload_{stamp}_{uuid.uuid4().hex[:6]}"
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def _write_meta_form(
    *,
    user_meta: dict | None,
    fallback_run_id: str,
    dataset: str,
    model: str,
    framework: str,
    optimizer: str,
    learning_rate: float | None,
    batch_size: int | None,
    epochs_planned: int | None,
    status: str,
    notes: str,
) -> dict:
    out: dict = dict(user_meta) if user_meta else {}
    out.setdefault("run_id", out.get("run_id") or fallback_run_id)
    out.setdefault("task", out.get("task") or "tabular")
    if dataset:
        out["dataset_name"] = dataset
    if model:
        out["model_name"] = model
    if framework:
        out["framework"] = framework
    if optimizer:
        out["optimizer"] = optimizer
    if learning_rate is not None:
        out["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        out["batch_size"] = int(batch_size)
    if epochs_planned is not None:
        out["epochs_planned"] = int(epochs_planned)
    if status:
        out["status"] = status
    if notes:
        out["notes"] = notes
    return out


def _materialise_uploaded_run(
    *,
    history_csv_bytes: bytes,
    meta_json_bytes: bytes | None,
    form_meta: dict | None,
) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    rd = _new_upload_dir()
    try:
        df = pd.read_csv(io.BytesIO(history_csv_bytes))
    except Exception as e:
        raise ValueError(f"Could not parse history.csv: {type(e).__name__}: {e}") from e
    if df.empty:
        raise ValueError("history.csv has no rows.")
    present = {c.lower() for c in df.columns}
    expected_present = [c for c in _EXPECTED_HISTORY_COLS if c in present]
    if not expected_present:
        warnings.append(
            "history.csv has none of the expected behavioural columns "
            f"({', '.join(_EXPECTED_HISTORY_COLS)}). The pipeline will run "
            "but its diagnosis is unlikely to be informative."
        )
    elif "val_loss" not in present and "val_acc" not in present:
        warnings.append(
            "history.csv has no validation columns (val_loss / val_acc). "
            "Many leakage / overfitting signals will not be available."
        )
    df.to_csv(rd / "history.csv", index=False)
    user_meta: dict | None = None
    if meta_json_bytes is not None:
        try:
            user_meta = json.loads(meta_json_bytes.decode("utf-8"))
            if not isinstance(user_meta, dict):
                raise ValueError("meta.json must be a JSON object.")
        except Exception as e:
            raise ValueError(f"Could not parse meta.json: {type(e).__name__}: {e}") from e
    if user_meta is None and form_meta is None:
        warnings.append(
            "No meta.json provided and no metadata form filled in; "
            "synthesised a minimal meta.json from the run directory name."
        )
        meta = {"run_id": rd.name, "task": "tabular"}
    else:
        meta = _write_meta_form(
            user_meta=user_meta,
            fallback_run_id=rd.name,
            dataset=(form_meta or {}).get("dataset", ""),
            model=(form_meta or {}).get("model", ""),
            framework=(form_meta or {}).get("framework", ""),
            optimizer=(form_meta or {}).get("optimizer", ""),
            learning_rate=(form_meta or {}).get("learning_rate"),
            batch_size=(form_meta or {}).get("batch_size"),
            epochs_planned=(form_meta or {}).get("epochs_planned"),
            status=(form_meta or {}).get("status", ""),
            notes=(form_meta or {}).get("notes", ""),
        )
    (rd / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return rd, warnings


def _materialise_zip_run(zip_bytes: bytes) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    rd = _new_upload_dir()
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for member in zf.namelist():
                norm = Path(member).as_posix()
                if norm.startswith("/") or ".." in Path(norm).parts:
                    raise ValueError(f"Refusing unsafe ZIP entry: {member!r}")
                zf.extract(member, rd)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Not a valid ZIP file: {e}") from e
    candidates = list(rd.rglob("meta.json"))
    if not candidates:
        raise ValueError(
            "ZIP does not contain a meta.json. Expected a folder with meta.json + history.csv."
        )
    inner = candidates[0].parent
    if not (inner / "history.csv").is_file():
        raise ValueError(
            f"Found meta.json under {inner.relative_to(rd)} but no history.csv next to it."
        )
    if inner != rd:
        warnings.append(
            f"Using `{inner.relative_to(rd)}` from inside the ZIP as the run directory."
        )
    return inner, warnings


st.set_page_config(
    page_title="ml_diag",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "ui_lang" not in st.session_state:
    st.session_state["ui_lang"] = "en"

with st.sidebar:
    lang_choice = st.radio(
        "🌐",
        options=["en", "ru"],
        index=0 if st.session_state["ui_lang"] == "en" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="_lang_radio",
    )
    if lang_choice != st.session_state["ui_lang"]:
        st.session_state["ui_lang"] = lang_choice
        st.rerun()

st.markdown(f"## {_t('page_title')}\n{_t('page_subtitle')}")


def _safe_list_run_ids(corpus_path: str) -> list[str]:
    if not corpus_path:
        return []
    try:
        return list_corpus_run_ids(corpus_path)
    except Exception as e:
        st.sidebar.error(f"Could not read corpus: {e}")
        return []


with st.sidebar:
    st.header(_t("config_header"))
    _mode_options = ("upload", "external", "corpus")
    _mode_labels = {
        "upload": _t("mode_upload"),
        "external": _t("mode_external"),
        "corpus": _t("mode_corpus"),
    }
    mode = st.radio(
        _t("input_mode"),
        _mode_options,
        index=0,
        format_func=lambda m: _mode_labels[m],
        help=_t("input_mode_help"),
    )
    run_dir_input: str | None = None
    corpus_input: str | None = None
    run_id_input: str | None = None
    upload_history_file = None
    upload_meta_file = None
    upload_zip_file = None
    upload_form: dict[str, object] = {}
    upload_use_zip = False
    if mode == "upload":
        st.markdown(_t("drop_files"))
        upload_use_zip = st.toggle(
            _t("use_zip"),
            value=False,
            help=_t("use_zip_help"),
        )
        if upload_use_zip:
            upload_zip_file = st.file_uploader(
                _t("zip_label"),
                type=["zip"],
                accept_multiple_files=False,
                help=_t("zip_help"),
            )
        else:
            upload_history_file = st.file_uploader(
                _t("history_label"),
                type=["csv"],
                accept_multiple_files=False,
                help=_t("history_help"),
            )
            upload_meta_file = st.file_uploader(
                _t("meta_label"),
                type=["json"],
                accept_multiple_files=False,
                help=_t("meta_help"),
            )
            with st.expander(_t("form_expander"), expanded=False):
                upload_form["dataset"] = st.text_input(
                    _t("form_dataset"),
                    value="",
                    placeholder=_t("form_dataset_ph"),
                )
                upload_form["model"] = st.text_input(
                    _t("form_model"),
                    value="",
                    placeholder=_t("form_model_ph"),
                )
                upload_form["framework"] = st.text_input(
                    _t("form_framework"),
                    value="",
                    placeholder=_t("form_framework_ph"),
                )
                upload_form["optimizer"] = st.text_input(
                    _t("form_optimizer"),
                    value="",
                    placeholder=_t("form_optimizer_ph"),
                )
                lr_str = st.text_input(
                    _t("form_lr"),
                    value="",
                    placeholder=_t("form_lr_ph"),
                )
                upload_form["learning_rate"] = float(lr_str) if lr_str.strip() else None
                bs_str = st.text_input(
                    _t("form_bs"),
                    value="",
                    placeholder=_t("form_bs_ph"),
                )
                upload_form["batch_size"] = int(bs_str) if bs_str.strip() else None
                ep_str = st.text_input(
                    _t("form_epochs"),
                    value="",
                    placeholder=_t("form_epochs_ph"),
                )
                upload_form["epochs_planned"] = int(ep_str) if ep_str.strip() else None
                upload_form["status"] = st.selectbox(
                    _t("form_status"),
                    ("", "completed", "failed", "diverged", "early_stopped"),
                    index=0,
                )
                upload_form["notes"] = st.text_area(
                    _t("form_notes"),
                    value="",
                    placeholder=_t("form_notes_ph"),
                    height=70,
                )
            with st.expander(_t("layout_expander"), expanded=False):
                st.markdown(
                    "**history.csv** — one row per epoch:\n\n"
                    "```\n"
                    "epoch,train_loss,val_loss,train_acc,val_acc,lr,grad_norm\n"
                    "1,1.144,1.104,0.339,0.333,0.001,0.460\n"
                    "...\n"
                    "```\n\n"
                    "**meta.json** (optional):\n\n"
                    "```json\n"
                    "{\n"
                    '  "run_id": "exp_2026_05_01",\n'
                    '  "task": "tabular",\n'
                    '  "dataset_name": "my_table_v3",\n'
                    '  "model_name": "mlp_h32_h16",\n'
                    '  "framework": "pytorch",\n'
                    '  "optimizer": "adam",\n'
                    '  "learning_rate": 0.001,\n'
                    '  "batch_size": 64,\n'
                    '  "status": "completed"\n'
                    "}\n"
                    "```"
                )
    elif mode == "external":
        run_dir_input = st.text_input(
            _t("ext_run_dir"),
            value="demo_uploads/demo_overfitting",
            help=_t("ext_run_dir_help"),
        )
    else:
        corpus_input = st.text_input(
            _t("corpus_path"),
            value="data/corpus/real_8ds_n5_multi",
            help=_t("corpus_path_help"),
        )
        run_ids = _safe_list_run_ids(corpus_input)
        if run_ids:
            default_idx = 0
            prev = st.session_state.get("last_run_id")
            if prev in run_ids:
                default_idx = run_ids.index(prev)
            run_id_input = st.selectbox(
                _t("run_id"),
                run_ids,
                index=default_idx,
                key="run_id_selectbox",
            )
            st.session_state["last_run_id"] = run_id_input
            st.caption(_t("corpus_size", n=len(run_ids)))
        else:
            run_id_input = st.text_input(
                _t("run_id"),
                value="",
                help=_t("run_id_help_no_enum"),
            )
    artifacts_input = "results/hierarchical/real_8ds_n5_multi"
    backend_input = "auto"
    max_recs_input = st.slider(_t("max_recs"), 1, 5, 3)
    no_integrity = st.checkbox(
        _t("no_integrity"),
        value=False,
        help=_t("no_integrity_help"),
    )
    write_outputs = st.checkbox(
        _t("write_outputs"),
        value=True,
        help=_t("write_outputs_help"),
    )
    run_btn = st.button(_t("run_btn"), type="primary", use_container_width=True)


def _run_diagnosis() -> CaseResult | None:
    artifacts = artifacts_input.strip()
    if not artifacts:
        st.error("Please specify the hierarchical artifacts directory.")
        return None
    if not Path(artifacts).is_dir():
        st.error(_t("err_artifacts_missing", path=artifacts))
        return None
    kwargs: dict = {
        "artifacts_dir": artifacts,
        "backend": backend_input,
        "max_recommendations": int(max_recs_input),
        "language": st.session_state.get("ui_lang", "en"),
        "no_integrity": bool(no_integrity),
        "write_outputs": bool(write_outputs),
    }
    if mode == "upload":
        try:
            if upload_use_zip:
                if upload_zip_file is None:
                    st.error(_t("err_drop_zip"))
                    return None
                run_dir_path, warnings = _materialise_zip_run(upload_zip_file.getvalue())
            else:
                if upload_history_file is None:
                    st.error(_t("err_drop_history"))
                    return None
                meta_bytes = upload_meta_file.getvalue() if upload_meta_file is not None else None
                meaningful_form = any(v not in (None, "", 0) for v in upload_form.values())
                run_dir_path, warnings = _materialise_uploaded_run(
                    history_csv_bytes=upload_history_file.getvalue(),
                    meta_json_bytes=meta_bytes,
                    form_meta=upload_form if meaningful_form else None,
                )
        except ValueError as e:
            st.error(_t("err_upload_rejected", msg=str(e)))
            return None
        except Exception as e:                
            st.error(_t("err_upload_failed", msg=f"{type(e).__name__}: {e}"))
            return None
        for w in warnings:
            st.warning(w)
        st.caption(_t("info_uploaded_at", path=run_dir_path))
        kwargs["run_dir"] = str(run_dir_path)
    elif mode == "external":
        if not run_dir_input or not Path(run_dir_input.strip()).is_dir():
            st.error(_t("err_run_dir_missing", path=run_dir_input))
            return None
        kwargs["run_dir"] = run_dir_input.strip()
    else:
        if not corpus_input or not Path(corpus_input.strip()).is_dir():
            st.error(_t("err_corpus_missing", path=corpus_input))
            return None
        if not run_id_input:
            st.error(_t("err_pick_run_id"))
            return None
        kwargs["corpus"] = corpus_input.strip()
        kwargs["run_id"] = run_id_input.strip()
    with st.spinner(_t("spinner_diagnosing")):
        try:
            return diagnose_case(**kwargs)
        except FileNotFoundError as e:
            st.error(_t("err_file_not_found", msg=str(e)))
        except KeyError as e:
            st.error(_t("err_run_not_in_table", msg=str(e)))
        except Exception as e:                
            st.error(_t("err_diag_failed", msg=f"{type(e).__name__}: {e}"))
    return None


if run_btn:
    res = _run_diagnosis()
    if res is not None:
        st.session_state["case_result"] = res

result: CaseResult | None = st.session_state.get("case_result")

if result is None:
    st.info(_t("welcome"))
    st.caption(_t("interp_text_lang_note"))
    st.stop()

tab_overview, tab_curves, tab_diag, tab_evidence, tab_interp, tab_files = st.tabs(
    [
        _t("tab_overview"),
        _t("tab_curves"),
        _t("tab_diagnosis"),
        _t("tab_evidence"),
        _t("tab_interpretation"),
        _t("tab_files"),
    ]
)

with tab_overview:
    st.subheader(_t("ovr_run_overview"))
    meta = result.meta or {}
    cols = st.columns(3)
    cols[0].metric("run_id", str(meta.get("run_id") or result.diagnosis.run_id))
    cols[1].metric("dataset", str(meta.get("dataset_name") or meta.get("dataset") or "—"))
    cols[2].metric("model", str(meta.get("model_name") or "—"))
    cols = st.columns(3)
    cols[0].metric("framework", str(meta.get("framework") or "—"))
    cols[1].metric("optimizer", str(meta.get("optimizer") or "—"))
    cols[2].metric("learning_rate", str(meta.get("learning_rate") or "—"))
    cols = st.columns(3)
    cols[0].metric("epochs planned", str(meta.get("epochs_planned") or "—"))
    cols[1].metric("epochs logged", str(meta.get("n_epochs_logged") or len(result.history)))
    cols[2].metric("status", str(meta.get("status") or "—"))
    st.markdown("---")
    st.markdown(_t("ovr_pipeline_meta"))
    backend_used = result.interpretation.backend
    if backend_used == "template":
        st.caption(_t("ovr_backend_caption_template"))
    else:
        st.caption(_t("ovr_backend_caption_llm", name=backend_used))
    st.json(
        {
            "feature_source": result.feature_source,
            "integrity_columns_available": result.integrity_columns_available,
            "interpretation_backend": backend_used,
            "case_dir": str(result.case_dir) if result.case_dir else None,
        }
    )
    if meta.get("notes"):
        st.markdown(_t("ovr_notes"))
        st.write(meta["notes"])
    if meta.get("tags"):
        st.markdown(_t("ovr_tags") + ", ".join(f"`{t}`" for t in meta["tags"]))

with tab_curves:
    st.subheader(_t("curves_header"))
    h = result.history
    if h.empty:
        st.warning(_t("curves_no_history"))
    else:
        loss_cols = [c for c in ("train_loss", "val_loss") if c in h.columns]
        if loss_cols:
            st.markdown(_t("curves_loss"))
            st.line_chart(h[loss_cols])
        acc_cols = [c for c in ("train_acc", "val_acc") if c in h.columns]
        if acc_cols:
            st.markdown(_t("curves_accuracy"))
            st.line_chart(h[acc_cols])
        opt_cols = [
            c for c in ("lr", "grad_norm", "weight_norm", "step_time_sec") if c in h.columns
        ]
        if opt_cols:
            st.markdown(_t("curves_aux"))
            for c in opt_cols:
                st.markdown(f"_{c}_")
                st.line_chart(h[[c]])
        st.markdown("---")
        st.markdown(_t("curves_history_raw"))
        st.dataframe(h, use_container_width=True, hide_index=True)

with tab_diag:
    diag = result.diagnosis
    st.subheader(_t("diag_header"))
    cols = st.columns(2)
    cols[0].metric(_t("diag_final_class"), diag.final_class)
    cols[1].metric(_t("diag_composed_conf"), f"{diag.final_confidence:.3f}")
    st.markdown(_t("diag_class_probs"))
    probs_df = pd.DataFrame({"probability": diag.class_probabilities}).sort_values(
        "probability", ascending=False
    )
    st.bar_chart(probs_df)
    st.dataframe(probs_df, use_container_width=True)
    st.markdown(_t("diag_top_alts"))
    if diag.alternative_hypotheses:
        st.dataframe(
            pd.DataFrame(
                [{"class": c, "probability": float(p)} for c, p in diag.alternative_hypotheses]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(_t("diag_none"))
    st.markdown("---")
    st.markdown(_t("diag_stage_trace"))
    rows: list[dict] = []
    for stage in (diag.stage1, diag.stage2, diag.stage3):
        if stage is None:
            continue
        rows.append(
            {
                "stage": stage.stage_name,
                "predicted": stage.predicted,
                "confidence": float(stage.confidence),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab_evidence:
    ev = result.evidence
    st.subheader(_t("ev_header"))
    st.caption(f"schema_version=`{ev.schema_version}`, generated at `{ev.generated_at}`")
    st.markdown(_t("ev_top_features"))
    if ev.top_features:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "stage": f.source_stage,
                        "feature": f.column,
                        "value": (
                            None
                            if f.value is None
                            or (isinstance(f.value, float) and f.value != f.value)
                            else float(f.value)
                        ),
                        "importance": float(f.importance),
                    }
                    for f in ev.top_features
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(_t("ev_top_features_unavail"))
    st.markdown("---")
    st.markdown(_t("ev_curve_evidence"))
    ce = ev.curve_evidence
    ce_dict = {
        "n_epochs": ce.n_epochs,
        "final_train_loss": ce.final_train_loss,
        "final_val_loss": ce.final_val_loss,
        "val_loss_min": ce.val_loss_min,
        "val_loss_argmin_frac": ce.val_loss_argmin_frac,
        "final_acc_gap": ce.final_acc_gap,
        "max_acc_gap": ce.max_acc_gap,
        "diverged": ce.diverged,
    }
    st.json(ce_dict)
    if ce.notes:
        st.markdown(_t("ev_curve_notes"))
        for n in ce.notes:
            st.markdown(f"- {n}")
    st.markdown("---")
    st.markdown(_t("ev_integrity_evidence"))
    ie = ev.integrity_evidence
    if ie.columns:
        st.json(ie.columns)
    else:
        st.caption(_t("ev_integrity_unavail"))
    if ie.notes:
        st.markdown(_t("ev_integrity_notes"))
        for n in ie.notes:
            st.markdown(f"- {n}")
    st.markdown("---")
    st.markdown(_t("ev_rejected"))
    if ev.rejected_hypotheses:
        st.dataframe(
            pd.DataFrame(
                [{"class": c, "probability": float(p)} for c, p in ev.rejected_hypotheses]
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(_t("diag_none"))

with tab_interp:
    interp = result.interpretation
    st.subheader(_t("interp_header"))
    st.caption(f"backend=`{interp.backend}`, schema=`{interp.schema_version}`")
    st.markdown(_t("interp_summary"))
    st.write(interp.summary or "—")
    st.markdown(_t("interp_explanation"))
    st.write(interp.explanation or "—")
    st.markdown("---")
    st.markdown(_t("interp_symptoms"))
    if interp.symptoms:
        for s in interp.symptoms:
            st.markdown(f"- {s}")
    else:
        st.caption("—")
    st.markdown("---")
    st.markdown(_t("interp_recs"))
    if interp.recommendations:
        rec_rows = []
        for r in interp.recommendations:
            rec_rows.append(
                {
                    "priority": int(r.priority),
                    "action": r.action_name,
                    "parameters": ", ".join(f"{k}={v}" for k, v in r.parameters.items()) or "—",
                    "rationale": r.rationale,
                    "target_classes": ", ".join(r.target_classes) or "any",
                }
            )
        st.dataframe(
            pd.DataFrame(rec_rows).sort_values("priority"),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(_t("diag_none"))
    if interp.confidence_notes:
        st.markdown(_t("interp_conf_notes"))
        for n in interp.confidence_notes:
            st.markdown(f"- {n}")
    if interp.warnings:
        st.markdown(_t("interp_warnings"))
        for w in interp.warnings:
            st.markdown(f"- {w}")
    if interp.limitations:
        st.markdown(_t("interp_limitations"))
        for l in interp.limitations:
            st.markdown(f"- {l}")
    if result.patch_summary:
        st.markdown("---")
        st.markdown(_t("interp_patch"))
        outcome = result.patch_summary.get("outcome") or {}
        case_block = result.patch_summary.get("case") or {}
        st.metric(_t("interp_patch_outcome"), str(outcome.get("status", "?")))
        st.json(
            {
                "before_run_id": case_block.get("before_run_id"),
                "after_run_id": case_block.get("after_run_id"),
                "action_name": case_block.get("action_name"),
                "delta_p_healthy": outcome.get("delta_p_healthy"),
                "delta_p_faulty_chosen": outcome.get("delta_p_faulty_chosen"),
            }
        )

with tab_files:
    st.subheader(_t("files_header"))
    if not result.case_dir:
        st.info(_t("files_not_written"))
    else:
        st.caption(_t("files_case_dir", path=result.case_dir))
        ok, errs = validate_case_dir(result.case_dir)
        if ok:
            st.success(_t("files_schema_ok", ver=CASE_OUTPUTS_SCHEMA_VERSION))
        else:
            st.warning(_t("files_validation_issues", n=len(errs)))
            for e in errs[:10]:
                st.markdown(f"- {e}")
        st.markdown(_t("files_downloads"))
        for short, fname in (
            ("case_summary.md", "case_summary.md"),
            ("interpretation.md", "interpretation.md"),
            ("diagnosis.json", "diagnosis.json"),
            ("evidence.json", "evidence.json"),
            ("recommendations.json", "recommendations.json"),
            ("interpretation.json", "interpretation.json"),
            ("case_summary.json", "case_summary.json"),
        ):
            p = result.case_dir / fname
            if not p.is_file():
                continue
            mime = "application/json" if fname.endswith(".json") else "text/markdown"
            try:
                content = p.read_text(encoding="utf-8")
            except Exception as e:
                st.markdown(f"- `{fname}` (could not read: {e})")
                continue
            st.download_button(
                label=f"⬇ {short}",
                data=content,
                file_name=f"{result.diagnosis.run_id}__{fname}",
                mime=mime,
                use_container_width=False,
            )
        st.markdown("---")
        st.markdown(_t("files_whole_zip"))
        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(Path(result.case_dir).iterdir()):
                    if p.is_file():
                        zf.write(p, arcname=p.name)
            buf.seek(0)
            st.download_button(
                label=_t("files_zip_btn"),
                data=buf.getvalue(),
                file_name=f"{result.diagnosis.run_id}_case.zip",
                mime="application/zip",
            )
        except Exception as e:
            st.warning(f"Could not zip case folder: {e}")
        summary_path = result.case_dir / "case_summary.md"
        if summary_path.is_file():
            with st.expander(_t("files_preview"), expanded=False):
                st.markdown(summary_path.read_text(encoding="utf-8"))
