# ml_diag

`ml_diag` — библиотека автоматизированной диагностики ошибок обучения ML-моделей по метаданным экспериментов и кривым обучения.

## Быстрый старт

```bash
pip install ml_diag
```

```python
from ml_diag import diagnose

result = diagnose("runs/my_exp")        # папка с meta.json + history.csv
print(result.label)                     # 'overfitting'
print(result.confidence)                # 0.84
for rec in result.recommendations:
    print(rec["action_name"], "—", rec["rationale"])

result.save("results/cases/my_exp")     # полный отчет на диск
```

Пакет идет с **предобученным каскадом из коробки** — никаких отдельных артефактов скачивать не нужно.

Альтернативный вариант — передать `meta` и `history` напрямую:

```python
import pandas as pd
from ml_diag import diagnose

history = pd.DataFrame({
    "epoch": range(20),
    "train_loss": [...],
    "val_loss":   [...],
    "train_acc":  [...],
    "val_acc":    [...],
})
meta = {"run_id": "exp_001", "dataset_name": "cifar10", "model_name": "resnet18",
        "framework": "pytorch", "optimizer": "adam", "learning_rate": 1e-3,
        "batch_size": 64, "epochs_planned": 20, "seed": 42}

result = diagnose(meta=meta, history=history)
```

Или собрать `run_dir` через `RunLogger` прямо в цикле обучения:

```python
from ml_diag import RunLogger, diagnose

with RunLogger(output_dir="runs/exp_001", meta={...}) as log:
    for epoch in range(num_epochs):
        log.log_epoch(epoch=epoch, train_loss=..., val_loss=...,
                      train_acc=..., val_acc=..., lr=...)
    log.finalize(status="completed")

result = diagnose("runs/exp_001")
```

## Что возвращает `diagnose()`

Объект `Diagnosis` со следующими полями:

| Поле | Тип | Что внутри |
|---|---|---|
| `label` | `str` | Один из 6 классов диагностики |
| `confidence` | `float` | Вероятность в `[0, 1]` |
| `alternatives` | `list[(str, float)]` | Топ-K альтернативных гипотез |
| `class_probabilities` | `dict[str, float]` | Полное распределение вероятностей |
| `summary` | `str` | Короткое резюме |
| `explanation` | `str` | Развернутый разбор |
| `symptoms` | `list[str]` | Конкретные наблюдения из evidence |
| `recommendations` | `list[dict]` | Корректирующие действия из allowlist |
| `warnings` | `list[str]` | Например, «LLM-бэкенд недоступен» |
| `evidence` | `dict` | Структурированное свидетельство |

Методы: `result.to_dict()`, `result.save(out_dir)` — записывает 9 файлов отчета (`diagnosis.json`, `evidence.md`, `interpretation.md`, `curves.png`, ...).

## Диагностические классы

- `healthy`
- `overfitting`
- `underfitting`
- `leakage`
- `label_noise`
- `instability`

## Основные возможности

- извлечение признаков из `meta.json` и `history.csv`;
- плоский (flat) бейзлайн для прямой многоклассовой классификации;
- иерархический каскад Stage 1 / Stage 2 / Stage 3;
- LLM-арбитраж на подмножестве несогласия моделей;
- ограничение выхода LLM-арбитра (hard-snap к одному из двух кандидатов);
- интерпретация результата на основе structured evidence;
- белый список (allowlist) корректирующих действий;
- слой абстенции на основе split-conformal калибровки;
- CLI для обучения, оценки и диагностики;
- Streamlit-демо для визуальной проверки одного запуска;
- поддержка диагностики внешних логированных запусков.

## Структура проекта

| Компонент | Назначение |
|---|---|
| `scenarios/` | инвентаризация и проверка сценариев |
| `models/` | плоский бейзлайн и иерархический каскад |
| `diagnosis/` | гибридные резолверы, стэкинг, арбитраж |
| `evaluation/` | метрики, сравнение моделей, отчеты |
| `interpretation/` | шаблонная и LLM-интерпретация |
| `actions/` | белый список корректирующих действий |
| `logging_sdk/` | логирование внешних запусков |
| `ui/` | Streamlit-демо |
| `scripts/` | CLI-команды для запуска пайплайна |

## Установка

Из PyPI:

```bash
pip install ml_diag
```
Для разработки (включая CLI-скрипты, тесты, линтеры) — клонировать репозиторий:

```bash
git clone https://github.com/brewerkka/ml-diag.git
cd ml-diag
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui,llm]"
```

## LLM-интерпретация

По умолчанию `diagnose()` использует детерминированный template-бэкенд — никаких LLM-вызовов, ключей и сетевых обращений. Текст рекомендаций собирается из шаблонов на основе диагноза и evidence.

Если хотите более развернутую интерпретацию через LLM, доступны два бэкенда:

**Groq Cloud** (бесплатный тариф доступен):

```bash
pip install "ml_diag[llm]"
export GROQ_API_KEY=gsk_...                 # получить на console.groq.com
```

```python
result = diagnose(run_dir, backend="auto")   # пробует groq - ollama - template
# или
result = diagnose(run_dir, backend="groq")
```

**Ollama**:

```bash
ollama serve                                  # запустить сервер
ollama pull qwen2.5:7b-instruct
```

```python
result = diagnose(run_dir, backend="ollama")
```

Если LLM-бэкенд недоступен (нет ключа, нет SDK, нет сети) — система **молча откатывается на template** и помещает причину в `result.warnings`.

## Формат входных данных

Один запуск должен содержать:

```text
run_dir/
    meta.json
    history.csv
```

`meta.json` содержит параметры запуска и служебную информацию.  
`history.csv` содержит значения метрик и функции потерь по эпохам.

## CLI и Streamlit-демо

CLI-скрипты (`scripts/`) и Streamlit-интерфейс (`ui/app.py`) не входят в PyPI-дистрибутив пакета — они доступны только при установке из git-репозитория. Все примеры команд ниже подразумевают именно такую установку.

## Диагностика одного запуска

```bash
python scripts/run_full_case.py \
    --run-dir runs/exp_001 \
    --artifacts results/hierarchical/real_8ds_n5_multi \
    --backend template
```

Результат сохраняется в:

```text
results/cases/<run_id>/
    diagnosis.json
    evidence.json
    evidence.md
    interpretation.json
    interpretation.md
    recommendations.json
    case_summary.json
    case_summary.md
```

## Обучение каскада на корпусе

```bash
NAME=real_8ds_n5_multi
CORPUS=data/corpus/$NAME

python scripts/run_hierarchical_train.py \
    --corpus $CORPUS \
    --out-dir results/hierarchical/$NAME
```

## Оценочный pipeline

```bash
bash scripts/run_all.sh

python scripts/aggregate_results.py \
    --results-dir results \
    --out-md results/aggregate_summary.md \
    --out-json results/aggregate_summary.json
```

## Streamlit-демо

```bash
streamlit run ui/app.py
```

Интерфейс поддерживает два режима:

- диагностика запуска из корпуса;
- диагностика внешнего запуска в формате `meta.json` + `history.csv`.

## Статус

- реализован плоский бейзлайн;
- реализован иерархический каскад;
- реализован гибридный резолвер;
- реализован LLM-арбитраж с hard-snap;
- реализован слой интерпретации;
- реализован белый список корректирующих действий;
- реализовано Streamlit-демо;
- реализован SDK для логирования внешних запусков;
- реализованы CLI-скрипты для обучения, оценки и диагностики.

## Примечание

Проект реализован в рамках выпускной квалификационной работы Милены Пивоваровой, НИУ ВШЭ.
