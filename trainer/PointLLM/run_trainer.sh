#!/usr/bin/env bash
set -euo pipefail

# Data and model paths
POINTLLM_ROOT=/path/to/PointQA_Eval/trainer/PointLLM
DATASET_ROOT=/path/to/dataset_root

# Base LLM initialization checkpoint directory (Stage-1 starting point)
MODEL_NAME_OR_PATH=/path/to/model/PointLLM_7B_v1.1_init
# Point cloud backbone initialization checkpoint (PointBERT)
POINT_BACKBONE_CKPT=/path/to/model/PointLLM_7B_v1.1_init/point_bert_v1.2.pt

# Training inputs and annotations
DATA_PATH=/path/to/PointQA_Eval/trainer/PointLLM/data/pointllm_train_data
ANNO_STAGE1=/path/to/PointQA_Eval/trainer/PointLLM/data/anno_data/pointllm_train_stage1.json
ANNO_STAGE2=/path/to/PointQA_Eval/trainer/PointLLM/data/anno_data/pointllm_train_stage2.json

# Output directories: change run_name and output paths for different experiments to avoid overwriting
OUTPUT_STAGE1=/path/to/PointQA_Eval/trainer/PointLLM/outputs/pointllm_train_stage1
OUTPUT_STAGE2=/path/to/PointQA_Eval/trainer/PointLLM/outputs/pointllm_train_stage2

# ===== Runtime =====
# Number of GPU processes (usually equals the number of GPUs in use). Reduce first if you hit OOM.
NPROC_PER_NODE=2
export CUDA_VISIBLE_DEVICES=0,1

MASTER_PORT=$((RANDOM % (65535 - 49152 + 1) + 49152))
# Options: none / wandb / tensorboard
REPORT_TO=none
# Whether to enable FSDP for Stage-2 on multi-GPU runs: true / false / auto
USE_FSDP_STAGE2=auto

FSDP_ARGS=()
if [[ "$USE_FSDP_STAGE2" == "true" || ("$USE_FSDP_STAGE2" == "auto" && "$NPROC_PER_NODE" -gt 1) ]]; then
  FSDP_ARGS=(
    --fsdp "full_shard auto_wrap"
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
  )
fi

cd "$POINTLLM_ROOT"

if [[ -f "$ANNO_STAGE1" && -f "$ANNO_STAGE2" ]]; then
  echo "[1/3] Found existing annotation JSON files, skip data preparation"
else
  echo "[1/3] Prepare PointLLM-format data"
  python3 ./scripts/prepare_training_data.py \
    --dataset-root "$DATASET_ROOT" \
    --pointllm-root "$POINTLLM_ROOT" \
    --pointnum 8192
fi

if [[ ! -d "$MODEL_NAME_OR_PATH" ]]; then
  echo "Missing model init checkpoint dir: $MODEL_NAME_OR_PATH"
  exit 1
fi

mkdir -p "$OUTPUT_STAGE1" "$OUTPUT_STAGE2"

# 1) --num_train_epochs
# 2) --per_device_train_batch_size / --gradient_accumulation_steps (together determine the effective batch size)
# 3) --learning_rate (Stage-1 is usually larger)
# 4) --fix_llm / --fix_pointnet (freezing strategy)
# 5) --bf16 / --gradient_checkpointing (memory and speed tradeoff)
# 6) --evaluation_strategy / --save_strategy (whether to run evaluation and save checkpoints)
echo "[2/3] Stage-1 training"
PYTHONPATH="$POINTLLM_ROOT:${PYTHONPATH:-}" \
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" pointllm/train/train_mem.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --data_path "$DATA_PATH" \
  --anno_path "$ANNO_STAGE1" \
  --output_dir "$OUTPUT_STAGE1" \
  --version v1 \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 2400 \
  --save_total_limit 1 \
  --learning_rate 2e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --bf16 True \
  --fix_llm True \
  --fix_pointnet True \
  --gradient_checkpointing True \
  --report_to "$REPORT_TO" \
  --run_name "pointllm_train_stage1" \
  --point_backbone_ckpt "$POINT_BACKBONE_CKPT" \
  --use_color True

# 1) --per_device_train_batch_size (Stage-2 is more likely to OOM)
# 2) --learning_rate (Stage-2 is usually smaller)
# 3) --fix_llm (True = train only the projector; False = finetune the LLM)
# 4) USE_FSDP_STAGE2 / NPROC_PER_NODE (multi-GPU parallel strategy)
# 5) --run_name / --output_dir (experiment management to avoid overwriting)
echo "[3/3] Stage-2 training"
PYTHONPATH="$POINTLLM_ROOT:${PYTHONPATH:-}" \
torchrun --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" pointllm/train/train_mem.py \
  --model_name_or_path "$OUTPUT_STAGE1" \
  --data_path "$DATA_PATH" \
  --anno_path "$ANNO_STAGE2" \
  --output_dir "$OUTPUT_STAGE2" \
  --version v1 \
  --model_max_length 2048 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --eval_steps 100 \
  --save_strategy "no" \
  --save_steps 2400 \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --bf16 True \
  --fix_llm False \
  --fix_pointnet True \
  --report_to "$REPORT_TO" \
  --run_name "pointllm_train_stage2" \
  --gradient_checkpointing True \
  --stage_2 True \
  --conversation_types "single_round" \
  --use_color True \
  "${FSDP_ARGS[@]}"
