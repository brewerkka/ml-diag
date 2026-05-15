#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

CORPORA=(
    "8ds:data/corpus/real_8ds_n5_multi"
    "5ds:data/corpus/real_5ds_n5_multi"
    "3ds:data/corpus/real_3ds_n3_multi"
)
SEEDS=(1 2 3 4)

ARBITRATOR_BACKEND="${ARBITRATOR_BACKEND:-auto}"
echo "[config] ARBITRATOR_BACKEND=${ARBITRATOR_BACKEND}"

echo "=== Stage 70 multi-seed orchestrator ==="
echo "Corpora: ${CORPORA[@]}"
echo "Seeds:   ${SEEDS[@]}"
echo ""

for entry in "${CORPORA[@]}"; do
    TAG="${entry%%:*}"
    CORPUS_DIR="${entry##*:}"
    for SEED in "${SEEDS[@]}"; do
        OUT="results/oof_predictions_${TAG}_seed${SEED}.parquet"
        if [[ -f "${OUT}" ]]; then
            echo "[skip] ${OUT} already exists."
            continue
        fi
        echo "[run] corpus=${TAG} seed=${SEED} -> ${OUT}"
        python scripts/generate_oof_predictions.py \
            --corpus "${CORPUS_DIR}" \
            --out    "${OUT}" \
            --arbitrator-backend "${ARBITRATOR_BACKEND}" \
            --seed   "${SEED}"
    done
done

echo ""
echo "=== Aggregating multi-seed disagreement pool ==="
python scripts/run_multiseed_disagreement.py \
    --out-parquet results/multiseed_disagreement_aggregate.parquet \
    --out-md      results/multiseed_disagreement_aggregate.md \
    --out-json    results/multiseed_disagreement_aggregate.json

echo ""
echo "=== Stage 70 complete ==="
