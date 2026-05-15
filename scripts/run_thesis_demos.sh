#!/usr/bin/env bash

set -euo pipefail

CORPUS="${CORPUS:-data/corpus/real_8ds_n5_multi}"
NAME="$(basename "${CORPUS}")"
ART="results/hierarchical/${NAME}"

if [ ! -f "${ART}/stage1_healthy_vs_faulty.joblib" ]; then
    echo "ERROR: cascade artifacts missing at ${ART}." >&2
    echo "Run scripts/run_all.sh first." >&2
    exit 2
fi

run_case() {
    local label="$1"
    local rid="$2"
    shift 2
    echo
    echo "================================================================"
    echo "  Case: ${label}  run_id=${rid}"
    echo "================================================================"
    python scripts/run_full_case.py \
        --corpus    "${CORPUS}" \
        --artifacts "${ART}" \
        --run-id    "${rid}" \
        --backend   template \
        "$@"
}

run_case easy_overfitting   1776944962_22b0bcd6
run_case leakage_severe     1776944966_0c57ab0a
run_case label_noise_mild   1776944966_4cd25e15

echo
echo "All three thesis demonstrators are in results/cases/."
ls -1 results/cases/
