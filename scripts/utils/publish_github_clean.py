"""Собрать чистый snapshot репозитория для публикации на GitHub.

Запускается из корня проекта. Создаёт `_publish_ml_diag/` рядом с проектом
с минимальным набором файлов, необходимых для запуска и публикации.

Принципы:
- Никаких .docx / .bak / draft-файлов
- Никаких .venv / __pycache__ / IDE-кешей
- Никаких CLAUDE.md и других внутренних project-memory
- Никаких thesis-specific артефактов и сборщиков
- Большие data/results корпуса исключены — данные регенерируются скриптами
- Сохраняем README, LICENSE, CITATION, .gitignore, CI workflow, pyproject
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEST = REPO_ROOT / "_publish_ml_diag"

# === что копируем (allow-list для top-level) ===
TOP_LEVEL_FILES = [
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
    ".env.example",
]

TOP_LEVEL_DIRS = [
    ".github",
    "src",
    "tests",
    "examples",
    "ui",
    "configs",
    "scripts",
    "demo_uploads",
]

# === что исключаем при копировании (deny-list, проверяется по basename/частям пути) ===
DENY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".DS_Store",
    "_archive",
    "ml_diag",   # legacy parallel package, excluded from pyproject build
}

DENY_SCRIPTS = {
    # thesis-specific сборщики и фигуро-генераторы — не нужны для запуска проекта
    "thesis_build",
    "thesis_build_legacy",
    "regenerate_v3_figures.py",
    "generate_v3_figures.py",
    "generate_v3_cross_corpus.py",
    "generate_v3_tier0_figures.py",
    "generate_v2_figures.py",
    "generate_thesis_figures.py",
    "fill_final_report.py",
    "augment_reports_bonferroni.py",
    "render_explanation_example.py",
    "render_interpretation_examples.py",
}

DENY_SUFFIXES = (
    ".docx",
    ".bak",
)


def _should_copy(src: Path) -> bool:
    """Решение: копировать ли этот путь."""
    parts = set(src.parts)
    if parts & DENY_NAMES:
        return False
    name = src.name
    if name in DENY_SCRIPTS:
        return False
    # bak-файлы вида foo.docx.bak.20260513-144122
    if ".bak." in name:
        return False
    if name.endswith(".docx") or name.endswith(".pyc"):
        return False
    if name.startswith("~$"):
        return False
    return True


def _copy_tree(src: Path, dst: Path) -> tuple[int, int]:
    """Рекурсивно копирует src → dst, применяя _should_copy на каждом узле.
    Возвращает (n_files, n_skipped)."""
    n_files = 0
    n_skipped = 0
    if src.is_file():
        if not _should_copy(src):
            return 0, 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return 1, 0
    if not src.is_dir():
        return 0, 0
    if not _should_copy(src):
        return 0, 1
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        sub_n, sub_s = _copy_tree(child, dst / child.name)
        n_files += sub_n
        n_skipped += sub_s
    return n_files, n_skipped


def main() -> None:
    if DEST.exists():
        print(f"Destination {DEST} already exists.")
        print(f"  Please remove or rename it manually before re-running.")
        return
    DEST.mkdir(parents=True)
    print(f"Building clean snapshot at {DEST}\n")

    total_files = 0
    total_skipped = 0

    # 1. Top-level files
    for fname in TOP_LEVEL_FILES:
        src = REPO_ROOT / fname
        if src.is_file():
            shutil.copy2(src, DEST / fname)
            total_files += 1
            print(f"  + {fname}")
        else:
            print(f"  ! missing: {fname}")

    # 2. Top-level dirs
    for dname in TOP_LEVEL_DIRS:
        src = REPO_ROOT / dname
        if not src.is_dir():
            print(f"  ! missing dir: {dname}")
            continue
        n, s = _copy_tree(src, DEST / dname)
        total_files += n
        total_skipped += s
        print(f"  + {dname}/  copied={n}  skipped={s}")

    # 3. Создаём пустую data/ + README с инструкцией где взять корпус
    data_dir = DEST / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "README.md").write_text(
        "# data/\n\n"
        "Корпуса экспериментов (real_3ds_n3_multi, real_5ds_n5_multi, real_8ds_n5_multi) "
        "не включены в репозиторий из-за размера (~45 MB).\n\n"
        "Способы получить:\n"
        "1. Сгенерировать локально через `scripts/run_scenario_inventory.py` "
        "   с конфигурацией из `configs/default.yaml`.\n"
        "2. Скачать с релиз-страницы проекта (если опубликована).\n"
        "3. Связаться с автором.\n",
        encoding="utf-8",
    )
    print("  + data/README.md  (stub)")

    # 4. Создаём пустую results/ + README
    results_dir = DEST / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / "README.md").write_text(
        "# results/\n\n"
        "Артефакты экспериментов (JSON-метрики, MD-отчёты, обученные модели) "
        "не включены в репозиторий — они регенерируются скриптами из `scripts/`.\n\n"
        "Запуск минимального headline pipeline на основном корпусе:\n\n"
        "```bash\n"
        "python scripts/run_hierarchical_train.py \\\n"
        "    --corpus  data/corpus/real_8ds_n5_multi \\\n"
        "    --out-dir results/hierarchical/real_8ds_n5_multi \\\n"
        "    --no-calibrate\n\n"
        "python scripts/run_hybrid_evaluation.py \\\n"
        "    --corpus              data/corpus/real_8ds_n5_multi \\\n"
        "    --hier-artifacts      results/hierarchical/real_8ds_n5_multi \\\n"
        "    --out-md              results/hybrid_evaluation_8ds.md \\\n"
        "    --out-json            results/hybrid_evaluation_8ds.json \\\n"
        "    --policies            stacking,stacking_with_conformal \\\n"
        "    --stacking-classifier gbm \\\n"
        "    --stacking-oof        results/oof_predictions_8ds.parquet \\\n"
        "    --conformal-alpha     0.05\n"
        "```\n",
        encoding="utf-8",
    )
    print("  + results/README.md  (stub)")

    print(f"\n=== Done ===")
    print(f"  total files copied:  {total_files}")
    print(f"  total skipped:       {total_skipped}")
    print(f"  destination:         {DEST}")


if __name__ == "__main__":
    main()
