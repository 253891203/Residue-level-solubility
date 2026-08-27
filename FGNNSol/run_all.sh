#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

EMBEDDING_DIR="${EMBEDDING_DIR:-cache/esm2_650m_embeddings}"
SKIP_EMBED="${SKIP_EMBED:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

python 00_fetch_official_fasta.py
python 01_check_and_prepare_data.py

if [[ "${SKIP_EMBED}" != "1" ]]; then
  python 02_embed_esm2_650m.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --resume True
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  python 03_train_transformer_regression.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --output_dir outputs \
    --seeds 2024 2025 2026 2027 2028 \
    --max_epochs 80 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --early_stopping False
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  python 04_evaluate_fixed_test_sets.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --output_dir outputs
fi
