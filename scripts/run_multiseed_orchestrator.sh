#!/usr/bin/env bash
# Stage 70 — multi-seed OOF orchestrator
#
# Runs ``generate_oof_predictions.py`` with seeds in {1, 2, 3, 4} across
# all three corpora, then aggregates the resulting parquets via
# ``run_multiseed_disagreement.py``. The existing seed=0 parquets are
# kept untouched and are picked up by the aggregator automatically.
#
# Expected wall-clock time:
#   * Groq available:  ~10 min per (corpus, seed) pair → ~2 h total.
#   * Groq exhausted, retries thrash:  ~12 min per pair → ~2.5 h total.
#   * Template-only via ARBITRATOR_BACKEND=template:  ~4 min per pair → ~50 min total.
# Set ``ARBITRATOR_BACKEND=template`` to skip Groq entirely (3× faster when
# free-tier quota is exhausted).
#
# Expected output: aggregated disagreement pool with n ≈ 800+
# (170 from seed=0 + ~170 per extra seed = 850).
#
# Usage:
#   source .venv/bin/activate
#   bash scripts/run_multiseed_orchestrator.sh

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

# Allow callers to override the arbitrator backend (auto / template / groq).
# When Groq TPD is exhausted, ``ARBITRATOR_BACKEND=template`` makes each
# disagreement ~7× faster than the default ``auto`` chain.
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
