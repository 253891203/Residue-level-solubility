#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; this script moves to the PROTSOLM project root.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"

# Data paths
TRAIN_CSV="${TRAIN_CSV:-data_PDBSol/train.csv}"
VALID_CSV="${VALID_CSV:-data_PDBSol/valid.csv}"
TEST_CSV="${TEST_CSV:-data_PDBSol/test.csv}"

# Output paths
EMBEDDING_DIR="${EMBEDDING_DIR:-outputs/embeddings}"
OUT_DIR="${OUT_DIR:-outputs}"

# ESM2 embedding parameters
MODEL_NAME="${MODEL_NAME:-esm2_t33_650M_UR50D}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-1}"
EMBED_DEVICE="${EMBED_DEVICE:-cuda}"

# Transformer training parameters
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
VERIFY_CHUNK_SIZE="${VERIFY_CHUNK_SIZE:-8}"
TOKEN_POOL_SIZE="${TOKEN_POOL_SIZE:-1}"

# Control switches
SKIP_EMBED="${SKIP_EMBED:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
RUN_VERIFY="${RUN_VERIFY:-0}"

echo "===== Step 1/4: Check PDBSol CSV files ====="
python scripts/01_check_pdbsol_csv.py \
  --train_csv "${TRAIN_CSV}" \
  --valid_csv "${VALID_CSV}" \
  --test_csv "${TEST_CSV}"

if [[ "${SKIP_EMBED}" != "1" ]]; then
  echo "===== Step 2/4: Generate ESM2-650M embeddings ====="
  python scripts/02_embed_pdbsol_esm2.py \
    --train_csv "${TRAIN_CSV}" \
    --valid_csv "${VALID_CSV}" \
    --test_csv "${TEST_CSV}" \
    --out_dir "${EMBEDDING_DIR}" \
    --model_name "${MODEL_NAME}" \
    --batch_size "${EMBED_BATCH_SIZE}" \
    --device "${EMBED_DEVICE}"
else
  echo "===== Step 2/4: Skip embedding generation because SKIP_EMBED=1 ====="
fi

if [[ "${RUN_VERIFY}" == "1" ]]; then
  echo "===== Step 3/4: Verify saved embedding files before training ====="
  python scripts/02b_verify_embeddings.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --chunk_size "${VERIFY_CHUNK_SIZE}"
else
  echo "===== Step 3/4: Skip embedding verification because RUN_VERIFY=0 ====="
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  echo "===== Step 4/4: Train Transformer and evaluate test set ====="
  python scripts/03_train_transformer.py \
    --embedding_dir "${EMBEDDING_DIR}" \
    --out_dir "${OUT_DIR}" \
    --epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${TRAIN_BATCH_SIZE}" \
    --lr "${LR}" \
    --device "${TRAIN_DEVICE}" \
    --num_workers "${NUM_WORKERS}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --token_pool_size "${TOKEN_POOL_SIZE}"
else
  echo "===== Step 4/4: Skip training because SKIP_TRAIN=1 ====="
fi

echo "Pipeline finished."
