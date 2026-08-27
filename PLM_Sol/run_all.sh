#!/usr/bin/env bash
set -euo pipefail

python 01_generate_embeddings.py
python 02_train_transformer.py
python 03_evaluate.py \
  --checkpoint outputs/checkpoints/best_model.pt \
  --output-dir outputs/results/evaluation
