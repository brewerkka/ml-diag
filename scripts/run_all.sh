#!/usr/bin/env bash

set -euo pipefail

CORPUS="${CORPUS:-data/corpus/real_8ds_n5_multi}"
NAME="$(basename "${CORPUS}")"
SEED="${SEED:-0}"
SKIP_DONE="${SKIP_DONE:-0}"

BANNER() {
    echo
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

skip_if_exists() {
    [[ "${SKIP_DONE}" == "1" && -f "$1" ]]
}

mkdir -p results results/partition results/hierarchical \
         results/interpretation results/cases results/interpretation_examples

BANNER "1/11  scenario inventory"
if skip_if_exists results/scenario_inventory.json; then
    echo "  [skip] results/scenario_inventory.json already exists"
else
    python scripts/run_scenario_inventory.py \
        --corpus    "${CORPUS}" \
        --out-md    results/scenario_inventory.md \
        --out-json  results/scenario_inventory.json
fi

BANNER "2/11  partition summary"
if skip_if_exists "results/partition/${NAME}.json"; then
    echo "  [skip] results/partition/${NAME}.json already exists"
else
    python scripts/run_partition_summary.py \
        --corpus "${CORPUS}" \
        --out    "results/partition/${NAME}.json"
fi

BANNER "3/11  flat baseline"
if skip_if_exists results/flat_baseline_report.json; then
    echo "  [skip] results/flat_baseline_report.json already exists"
else
    python scripts/run_flat_baseline.py \
        --corpus     "${CORPUS}" \
        --out-md     results/flat_baseline_report.md \
        --out-json   results/flat_baseline_report.json \
        --model-out  results/flat_baseline.joblib \
        --seed       "${SEED}"
fi

BANNER "4/11  hierarchical training"
if skip_if_exists "results/hierarchical/${NAME}/hierarchical_manifest.json"; then
    echo "  [skip] hierarchical artifacts already present"
else
    python scripts/run_hierarchical_train.py \
        --corpus  "${CORPUS}" \
        --out-dir "results/hierarchical/${NAME}" \
        --seed    "${SEED}"
fi

BANNER "5/11  comparison flat vs hierarchical"
if skip_if_exists results/flat_vs_hierarchical_report.json; then
    echo "  [skip] results/flat_vs_hierarchical_report.json already exists"
else
    python scripts/run_comparison.py \
        --corpus         "${CORPUS}" \
        --hier-artifacts "results/hierarchical/${NAME}" \
        --flat-model     results/flat_baseline.joblib \
        --out-md         results/flat_vs_hierarchical_report.md \
        --out-json       results/flat_vs_hierarchical_report.json \
        --seed           "${SEED}"
fi

BANNER "6/11  grouped baseline (per entry_id)"
if skip_if_exists results/grouped_baseline_report.json; then
    echo "  [skip] results/grouped_baseline_report.json already exists"
else
    python scripts/run_grouped_baseline.py \
        --corpus   "${CORPUS}" \
        --out-md   results/grouped_baseline_report.md \
        --out-json results/grouped_baseline_report.json \
        --seed     "${SEED}"
fi

BANNER "7/11  prototype ablation"
if skip_if_exists results/prototype_ablation.json; then
    echo "  [skip] results/prototype_ablation.json already exists"
else
    python scripts/run_prototype_features_experiment.py \
        --corpus   "${CORPUS}" \
        --out-md   results/prototype_ablation.md \
        --out-json results/prototype_ablation.json \
        --seed     "${SEED}"
fi

BANNER "8/11  leakage integrity ablation"
if skip_if_exists results/leakage_integrity_report.json; then
    echo "  [skip] results/leakage_integrity_report.json already exists"
else
    python scripts/run_leakage_integrity_experiment.py \
        --corpus   "${CORPUS}" \
        --out-md   results/leakage_integrity_report.md \
        --out-json results/leakage_integrity_report.json \
        --seed     "${SEED}"
fi

BANNER "9/11  LLM interpretation (template, 5 runs)"
python scripts/run_llm_interpretation.py \
    --corpus    "${CORPUS}" \
    --artifacts "results/hierarchical/${NAME}" \
    --n         5 \
    --backend   template \
    --out-dir   results/interpretation/

BANNER "10/11 reference interpretation examples (synthetic)"
python scripts/render_interpretation_examples.py \
    --out-dir results/interpretation_examples/

BANNER "11/11 aggregate + fill final report"
python scripts/aggregate_results.py \
    --results-dir results \
    --out-md      results/aggregate_summary.md \
    --out-json    results/aggregate_summary.json

if [ -f scripts/fill_final_report.py ]; then
    python scripts/fill_final_report.py \
        --aggregate results/aggregate_summary.json \
        --out       results/final_comparative_report.md
fi

BANNER "done"
echo "  results/ tree:"
ls -1 results/ | head -40
