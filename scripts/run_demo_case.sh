#!/usr/bin/env bash
# scripts/run_demo_case.sh — one-line demonstrator of the canonical workflow.
#
# Usage:
#     bash scripts/run_demo_case.sh <run_id>
#     bash scripts/run_demo_case.sh <run_id> anthropic   # uses Anthropic backend if available
#
# Environment overrides:
#     CORPUS=path/to/corpus     # default: data/corpus/real_8ds_n5_multi
#     OUT=path/to/out_dir       # default: results/cases/<run_id>
#
# What it does:
#     diagnosis → evidence → interpretation → recommendations → case folder.
# Pre-requisite:
#     hierarchical artifacts must already exist at
#     results/hierarchical/$(basename $CORPUS)/. Produce them via
#     `python scripts/run_hierarchical_train.py`.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/run_demo_case.sh <run_id> [backend=template|anthropic]" >&2
    exit 2
fi

RUN_ID="$1"
BACKEND="${2:-template}"
CORPUS="${CORPUS:-data/corpus/real_8ds_n5_multi}"
NAME="$(basename "${CORPUS}")"
ART="results/hierarchical/${NAME}"
OUT="${OUT:-results/cases/${RUN_ID}}"

if [ ! -f "${ART}/stage1_healthy_vs_faulty.joblib" ]; then
    echo "ERROR: cascade artifacts missing at ${ART}." >&2
    echo "Train them first: python scripts/run_hierarchical_train.py --corpus ${CORPUS} --out-dir ${ART}" >&2
    exit 3
fi

python scripts/run_full_case.py \
    --corpus    "${CORPUS}" \
    --artifacts "${ART}" \
    --run-id    "${RUN_ID}" \
    --backend   "${BACKEND}" \
    --out-dir   "${OUT}"

echo
echo "Case ready: ${OUT}/case_summary.md"
