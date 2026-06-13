#!/usr/bin/env bash
set -euo pipefail

# Edit values directly if needed.
POINTALIGN_ROOT=/path/to/PointQA_Eval/trainer/PointAlign
DATASET_ROOT=/path/to/dataset_root
CUDA_VISIBLE_DEVICES=0

export WANDB_DISABLED=true
export WANDB_MODE=disabled

cd "$POINTALIGN_ROOT"

echo "[1/2] Prepare PointAlign-format data"
python3 ./scripts/prepare_training_data.py \
  --dataset-root "$DATASET_ROOT" \
  --pointalign-root "$POINTALIGN_ROOT" \
  --pointnum 8192

echo "[2/2] Start PointAlign training (test config)"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" python3 train.py --cfg-path ./finetune_pointalign.yaml
