#!/bin/bash
set -e

# Expose physical GPUs here; --devices uses logical ids after this remapping.
# Example for two GPUs: CUDA_VISIBLE_DEVICES=6,7 and --devices cuda:0,cuda:1.
export CUDA_VISIBLE_DEVICES=2,3
export HF_ENDPOINT=https://hf-mirror.com
export SENTENCE_TRANSFORMERS_HOME=/home/wangxingjian/model/sentence_transformers

# ShapeLLM dynamic eval
# python /home/wangxingjian/AnyPoint/run_dynamic_eval.py \
#   --metadata /home/wangxingjian/data/texverse/metadata.jsonl \
#   --pcd-dir /home/wangxingjian/data/texverse/points_npy \
#   --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model shapellm \
#   --test-ckpt "${SHAPELLM_CKPT}" \
#   --recon-path "${SHAPELLM_RECON}" \
#   --EVA-path "${SHAPELLM_EVA}" \
#   --output ./output/shapellm_dyn \
#   --budget 50 \
#   --batch-size 10 \
#   --pool-size 200 \
#   --lambda-explore 0.2

# PointLLM dynamic eval
# python /home/wangxingjian/AnyPoint/run_dynamic_eval.py \
#   --metadata /home/wangxingjian/data/texverse/metadata.jsonl \
#   --pcd-dir /home/wangxingjian/data/texverse/points_npy \
#   --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model pointllm \
#   --checkpoint /home/wangxingjian/model/PointLLM_7B_v1.2 \
#   --output ./output/pointllm_dyn \
#   --budget 50 \
#   --batch-size 10 \
#   --pool-size 200 \
#   --lambda-explore 0.2

# ShapeLLM compare random vs dynamic
# python3 compare_eval_strategies.py \
#   --metadata /home/wangxingjian/data/texverse/metadata.jsonl \
#   --pcd-dir /home/wangxingjian/data/texverse/points_npy \
#   --background_dir /home/wangxingjian/data/texverse/background \
#   --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model shapellm \
#   --test-ckpt /home/wangxingjian/model/ShapeLLM_7B_general_v1.0 \
#   --recon-path /home/wangxingjian/PointQA_Eval/checkpoints/recon/large.pth \
#   --EVA-path /home/wangxingjian/model/eva_large_patch14_336.in22k_ft_in22k_in1k/model.safetensors \
#   --output /home/wangxingjian/AnyPoint/output/compare_shapellm \
#   --devices cuda:0,cuda:1 \
#   --budget 100 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /home/wangxingjian/AnyPoint/output/pointllm_dyn \
#   --lambda-explore 0.2

python3 compare_eval_strategies.py \
  --metadata /home/wangxingjian/data/texverse/metadata.jsonl \
  --pcd-dir /home/wangxingjian/data/texverse/points_npy \
  --background_dir /home/wangxingjian/data/texverse/background \
  --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
  --model pointllm \
  --checkpoint /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output /home/wangxingjian/AnyPoint/output/compare_pointllm \
  --devices cuda:0,cuda:1 \
  --budget 100 \
  --batch-size 10 \
  --pool-size 1000 \
  --pool-cache-dir /home/wangxingjian/AnyPoint/output/pointllm_dyn/task_pool_cache \
  --lambda-explore 0.2
