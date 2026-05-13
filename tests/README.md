# Tests

Smoke and integration tests for `structured_diag`.

## Running

```bash
# From project root, with .venv activated:
pip install pytest
pytest tests/ -v

# Or via stdlib without pytest:
PYTHONPATH=src python -c "
import importlib.util, traceback
spec = importlib.util.spec_from_file_location('test_smoke', 'tests/test_smoke.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
for name in dir(mod):
    if name.startswith('test_'):
        try: getattr(mod, name)(); print(f'PASS {name}')
        except Exception as e: print(f'FAIL {name}: {e}')
"
```

## Coverage

`test_smoke.py` exercises 7 functional areas with 15 tests (no real corpus required):

| Area | Tests | What it validates |
|------|-------|-------------------|
| Imports | 2 | Public modules + diagnosis submodules import without circular-dependency errors |
| Label vocabulary | 1 | `PRIMARY_LABELS`, `STAGE1_LABELS`, `STAGE2_LABELS`, `to_stage1`, `to_stage2` are mutually consistent |
| Metrics | 4 | `classification_report` (perfect + per-class), `bootstrap_metric_ci`, `bootstrap_delta_ci` (zero-when-identical invariant) |
| Conformal layer | 2 | `calibrate_split_conformal` produces valid quantile; empirical coverage ≥ 1−α−slack on synthetic data |
| Stacking featurizer | 2 | `featurize` returns 33-column DataFrame with documented schema; preserves index |
| Schema alignment | 2 | `align_features_to_schema` drops extras / adds missing / is idempotent |
| Hybrid resolver | 1 | `agreement_or_flat` policy returns common class when flat and cascade agree |
| Evidence layer | 1 | `classify_evidence_notes` is callable on minimal input |

## Test design notes

- **No real corpus loading.** Tests run on synthetic data only — they exercise library logic, not the full pipeline. End-to-end validation lives in `scripts/run_*.py` artefacts.
- **No external services.** Tests don't call Groq or Ollama; LLM arbitration is exercised with `backend="template"` if needed.
- **Deterministic.** All RNG instances use fixed seeds.

## What's NOT covered (by design)

- Per-stage cascade training (requires 100+ rows of curve data)
- Full pipeline end-to-end (covered by `scripts/run_hybrid_evaluation.py` artefacts in `results/`)
- LLM responses (covered by persistent cache + manual review of `arbitration_stats`)
- Cross-corpus replication (covered by `scripts/run_cross_corpus_summary.py` artefact)

For production deployment, additional tests would be needed for: per-stage train/inference round-trips, OOF generation against real corpora, conformal coverage under distribution shift, LLM cache persistence across processes.
