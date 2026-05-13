# results/

Артефакты экспериментов (JSON-метрики, MD-отчёты, обученные модели) не включены в репозиторий — они регенерируются скриптами из `scripts/`.

Запуск минимального headline pipeline на основном корпусе:

```bash
python scripts/run_hierarchical_train.py \
    --corpus  data/corpus/real_8ds_n5_multi \
    --out-dir results/hierarchical/real_8ds_n5_multi \
    --no-calibrate

python scripts/run_hybrid_evaluation.py \
    --corpus              data/corpus/real_8ds_n5_multi \
    --hier-artifacts      results/hierarchical/real_8ds_n5_multi \
    --out-md              results/hybrid_evaluation_8ds.md \
    --out-json            results/hybrid_evaluation_8ds.json \
    --policies            stacking,stacking_with_conformal \
    --stacking-classifier gbm \
    --stacking-oof        results/oof_predictions_8ds.parquet \
    --conformal-alpha     0.05
```
