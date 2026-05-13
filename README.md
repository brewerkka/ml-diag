# ml_diag

`ml_diag` — исследовательский прототип системы автоматизированной диагностики ошибок обучения ML-моделей по метаданным экспериментов и кривым обучения.

Система принимает логированный запуск или корпус запусков, классифицирует его в один из шести классов диагностики, формирует structured evidence, генерирует интерпретацию результата и возвращает допустимые корректирующие действия из закрытого allowlist.

## Диагностические классы

- `healthy`
- `overfitting`
- `underfitting`
- `leakage`
- `label_noise`
- `instability`

## Основные возможности

- извлечение признаков из `meta.json` и `history.csv`;
- flat baseline для прямой многоклассовой классификации;
- иерархический каскад Stage 1 / Stage 2 / Stage 3;
- LLM-арбитраж на disagreement-подмножестве;
- hard-snap-ограничение выхода LLM-арбитра;
- evidence-based интерпретация результата;
- allowlist корректирующих действий;
- split-conformal слой абстенции;
- CLI для обучения, оценки и диагностики;
- Streamlit-демо для визуальной проверки одного запуска;
- поддержка диагностики внешних логированных запусков.

## Структура проекта

| Компонент | Назначение |
|---|---|
| `scenarios/` | инвентаризация и проверка сценариев |
| `models/` | flat baseline и иерархический каскад |
| `diagnosis/` | гибридные резолверы, стэкинг, арбитраж |
| `evaluation/` | метрики, сравнение моделей, отчёты |
| `interpretation/` | template- и LLM-интерпретация |
| `actions/` | allowlist корректирующих действий |
| `logging_sdk/` | логирование внешних запусков |
| `ui/` | Streamlit-демо |
| `scripts/` | CLI-команды для запуска пайплайна |
| `docs/` | документация по архитектуре и экспериментам |

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Для Streamlit-интерфейса:

```bash
pip install -e ".[ui]"
```

Для LLM-интерпретации через Anthropic:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
```

Если LLM-backend недоступен, система автоматически использует template-backend.

## Формат входных данных

Один запуск должен содержать:

```text
run_dir/
    meta.json
    history.csv
```

`meta.json` содержит параметры запуска и служебную информацию.  
`history.csv` содержит значения метрик и функции потерь по эпохам.

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

## Логирование внешнего запуска

```python
from structured_diag.logging_sdk import RunLogger

with RunLogger(
    output_dir="runs/exp_001",
    meta={
        "run_id": "exp_001",
        "dataset_name": "my_dataset",
        "model_name": "resnet18",
        "framework": "pytorch",
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epochs_planned": 20,
        "seed": 42,
    },
) as logger:
    for epoch in range(num_epochs):
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            lr=current_lr,
        )

    logger.finalize(
        status="completed",
        final_metrics={"best_val_acc": best_val_acc},
    )
```

После этого запуск можно диагностировать стандартной командой `run_full_case.py`.

## Инварианты проекта

- LLM не является самостоятельным диагностом.
- LLM-арбитр выбирает только между `flat_label` и `cascade_label`.
- Корректирующие действия выбираются только из allowlist.
- При недоступности LLM используется template-backend.
- Корпус не изменяется во время диагностики.
- Все основные артефакты сохраняются в версионированном формате.
- OOF-протокол проверяется runtime-assertion на отсутствие leakage.

## Документация

| Документ | Содержание |
|---|---|
| `docs/architecture.md` | архитектура системы |
| `docs/workflows.md` | основные workflow |
| `docs/failure_taxonomy.md` | таксономия классов |
| `docs/diagnostic_features.md` | признаки диагностики |
| `docs/experimental_protocol.md` | экспериментальный протокол |
| `docs/llm_interpretation.md` | LLM-интерпретация |
| `docs/external_run_logging.md` | логирование внешних запусков |
| `docs/streamlit_demo.md` | Streamlit-интерфейс |
| `docs/limitations.md` | ограничения подхода |

## Статус

- реализован flat baseline;
- реализован иерархический каскад;
- реализован гибридный резолвер;
- реализован LLM-арбитраж с hard-snap;
- реализован интерпретационный слой;
- реализован allowlist корректирующих действий;
- реализован Streamlit-демо;
- реализован SDK для логирования внешних запусков;
- реализованы CLI-скрипты для обучения, оценки и диагностики.

## Примечание

Проект реализован в рамках выпускной квалификационной работы Милены Пивоваровой, НИУ ВШЭ.
