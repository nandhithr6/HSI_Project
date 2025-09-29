#!/bin/bash

# Improved training script for better convergence
# Addresses class imbalance, learning rate, and training duration

set -euo pipefail

echo "Starting improved HSI training optimized for 4x RTX 3080 Ti (12GB VRAM each)..."

# Adjust these paths as needed
DATA_DIR=${DATA_DIR:-/ssd_scratch/placenta/Placenta}
RUN_NAME=${RUN_NAME:-placenta_v2_4x3080ti}

TS=$(date +%Y%m%d-%H%M%S)
EXP_NAME="${RUN_NAME}_${TS}"

python -m src.training.train \
  --data-dir "$DATA_DIR" \
  --epochs 50 \
  --batch-size 4 \
  --num-workers 8 \
  --run-name "$EXP_NAME" \
  --merge-icg-to-base \
  --use-hcmff \
  --both-fusions \
  --hcmff-tokens 128 \
  --crop-size 512 \
  --auc-max-pixels 50000 \
  --force-all-gpus \
  --lr 1e-3 \
  --weight-decay 5e-5 \
  --alpha 0.75 \
  --gamma 3.0 \
  --lambda-u 0.3 \
  --warmup-epochs 10 \
  --fast-mode \
  --channels-last \
  --grad-checkpoint \
  --accum-steps 2 \
  --ema-decay 0.999 \
  --verbose-model-once \
  --workflow-logs \
  --workflow-interval 10

echo "Training completed. Results saved under organized run folder with weights and logs: $EXP_NAME"