#!/usr/bin/env bash
# scripts/run_thesis_demos.sh — produce 3 showcase full-case directories.
#
# Pre-requisites:
#   - the canonical workflow has been run on real_8ds_n5_multi (run_all.sh).
#   - results/hierarchical/real_8ds_n5_multi/ contains the 4 stage joblibs.
#
# The three showcase cases were picked from the manifest as representative:
#
#   easy_overfitting   — single-fault, severity=severe, sklearn:breast_cancer
#                        (canonical "clear" diagnosis)
#   leakage_severe     — single-fault leakage, severity=severe
#                        (canonical "hybrid" diagnosis where data-integrity
#                         features matter)
#   label_noise_mild   — single-fault label_noise, severity=mild
#                        (canonical "extended slice" diagnosis: should be
#                         routed to extended by partition rules)

set -euo pipefail

CORPUS="${CORPUS:-data/corpus/real_8ds_n5_multi}"
NAME="$(basename "${CORPUS}")"
ART="results/hierarchical/${NAME}"

if [ ! -f "${ART}/stage1_healthy_vs_faulty.joblib" ]; then
    echo "ERROR: cascade artifacts missing at ${ART}." >&2
    echo "Run scripts/run_all.sh first." >&2
    exit 2
fi

# Run a case with optional arguments forwarded.
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
