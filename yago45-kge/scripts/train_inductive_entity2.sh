#!/usr/bin/env bash
# Full strict-inductive two-pass recipe. Resume with the identical script + --resume.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/scripts/inductive.sh" train \
  --dataset "${ROOT_DIR}/../data/yago45_full/inductive.npz" \
  --run-dir "${ROOT_DIR}/../data/yago45_full/inductive_full_entity2_h128_mp3" \
  --entity-passes 2 --batch-size 512 --negatives 8 --sampling-version 3 \
  --hidden-dim 128 --dim 200 --base-dim 32 --fanouts 8 4 2 --residual \
  --learning-rate 0.001 --temperature 0.1 \
  --negative-policy mixed --validation-policy both --validation-split temporal_valid \
  --feature-mode full --sampler gestaltdb-csr --embedding-device cpu \
  --shallow-optimizer sparse-adagrad --shallow-learning-rate 0.05 \
  --keep-checkpoints 1 --checkpoint-every 5000 --eval-every 5000 \
  --eval-triples 300 --export-checkpoint latest --require-gpu "$@"
