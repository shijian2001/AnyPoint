#!/bin/bash
set -e

# Run from this script's own directory so the relative compare_eval_strategies.py
# reference resolves regardless of the caller's working directory.
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=4,5
export HF_ENDPOINT=https://hf-mirror.com
export SENTENCE_TRANSFORMERS_HOME=/home/wangxingjian/model

# ----------------------------------------------------------------------------
# Available --strategies values (always run alongside the implicit "random" baseline):
#   utility-guided (UTILITY_STRATEGIES):
#     dynamic        : U = A - λ·R                          (subtractive)
#     dynamic_mul    : U = A · (1 - λ·R)                    (heuristic mul gating)
#     dynamic_geo    : U = A^(1-λ) · (1-R)^λ, λ∈[0,1]      (weighted geo mean, exp form)
#     dynamic_geo_log: log U = (1-λ)·log A + λ·log(1-R)     (= geo, log-space — argmax-equivalent, more stable)
#     affinity_only  : U = A
#     novelty_only   : U = log(1 - R)
#   external baselines (BASELINE_STRATEGIES):
#     acd_style, autobencher_style, sea_style
#
# Shortcuts: 'all' (default), 'utility', 'baselines', or comma list e.g.
#   --strategies dynamic_geo,dynamic,dynamic_mul
# ----------------------------------------------------------------------------


# ShapeLLM
# python3 compare_eval_strategies.py \
#   --metadata /data/texverse/metadata.jsonl \
#   --pcd-dir /data/texverse/points_npy \
#   --background-dir /data/texverse/background \
#   --layouts /AnyPoint/outputs_gpt_oss/layouts.json \
#   --model shapellm \
#   --test-ckpt /model/ShapeLLM_7B_general_v1.0 \
#   --recon-path /PointQA_Eval/checkpoints/recon/large.pth \
#   --EVA-path /model/eva_large_patch14_336.in22k_ft_in22k_in1k/model.safetensors \
#   --output /AnyPoint/output/compare_shapellm \
#   --devices cuda:0 \
#   --budget 100 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /AnyPoint/output/pointllm_dyn/task_pool_cache \
#   --strategies all \
#   --lambda-explore 0.2

# PointLLM
python3 compare_eval_strategies.py \
  --metadata /home/wangxingjian/data/point_cloud/texverse_metadata_000-000.jsonl \
  --pcd-dir /home/wangxingjian/data/point_cloud/npys_2k/000-000 \
  --background-dir /home/wangxingjian/data/point_cloud/background \
  --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
  --model pointllm \
  --checkpoint /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output /home/wangxingjian/AnyPoint/output/pointllm \
  --devices cuda:0,cuda:1 \
  --budget 200 \
  --batch-size 10 \
  --pool-size 1000 \
  --pool-cache-dir /home/wangxingjian/AnyPoint/output/task_pool_cache \
  --strategies dynamic_geo,dynamic_geo_log \
  --lambda-explore 0.2

# OneLLM
# python3 compare_eval_strategies.py \
#   --metadata /home/wangxingjian/data/point_cloud/texverse_metadata_000-000.jsonl \
#   --pcd-dir /home/wangxingjian/data/point_cloud/npys_2k/000-000 \
#   --background-dir /home/wangxingjian/data/point_cloud/background \
#   --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model onellm \
#   --checkpoint /home/wangxingjian/model/OneLLM-7B/consolidated.00-of-01.pth \
#   --clip-pretrained-path /home/wangxingjian/model/vit_large_patch14_clip_224/open_clip_pytorch_model.bin \
#   --point-format xyzrgb \
#   --offline true \
#   --output /home/wangxingjian/AnyPoint/output/onellm \
#   --devices cuda:0,cuda:1 \
#   --budget 200 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /home/wangxingjian/AnyPoint/output/task_pool_cache \
#   --strategies all \
#   --lambda-explore 0.2

# MiniGPT-3D
# --batch-size mirrors the official MiniGPT-3D eval (batch=15).
# python3 compare_eval_strategies.py \
#   --metadata /data/texverse/metadata.jsonl \
#   --pcd-dir /data/texverse/points_npy \
#   --background-dir /data/texverse/background \
#   --layouts /AnyPoint/outputs_gpt_oss/layouts.json \
#   --model minigpt3d \
#   --cfg-path /PointQA_Eval/models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml \
#   --output /AnyPoint/output/compare_minigpt3d \
#   --devices cuda:0 \
#   --budget 100 \
#   --batch-size 15 \
#   --pool-size 1000 \
#   --pool-cache-dir /AnyPoint/output/pointllm_dyn/task_pool_cache \
#   --strategies all \
#   --lambda-explore 0.2

# PointAlign
# python3 compare_eval_strategies.py \
#   --metadata /data/texverse/metadata.jsonl \
#   --pcd-dir /data/texverse/points_npy \
#   --background-dir /data/texverse/background \
#   --layouts /AnyPoint/outputs_gpt_oss/layouts.json \
#   --model pointalign \
#   --cfg-path /PointQA_Eval/models/dependence/pointalign/eval_configs/benchmark_evaluation_paper.yaml \
#   --weights-root /model/pointalign \
#   --llama-model-path /model/pointalign/Phi_2 \
#   --bert-base-uncased-path /model/pointalign/bert-base-uncased \
#   --pc-encoder-path /model/pointalign/pc_encoder/point_model.pth \
#   --pretrain-ckpt /model/pointalign/pointalign/pretrain.pth \
#   --finetune-ckpt /model/pointalign/pointalign/finetune.pth \
#   --qformer-pretrained-path /model/pointalign/blip2_pretrained_flant5xxl.pth \
#   --output /AnyPoint/output/compare_pointalign \
#   --devices cuda:0 \
#   --budget 100 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /AnyPoint/output/pointllm_dyn/task_pool_cache \
#   --strategies all \
#   --lambda-explore 0.2

# GreenPLM
# python3 compare_eval_strategies.py \
#   --metadata /home/wangxingjian/data/point_cloud/texverse_metadata_000-000.jsonl \
#   --pcd-dir /home/wangxingjian/data/point_cloud/npys_2k/000-000 \
#   --background-dir /home/wangxingjian/data/point_cloud/background \
#   --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model greenplm \
#   --model-path /home/wangxingjian/model/greenplm/lava-vicuna_2024_4_Phi-3-mini-4k-instruct \
#   --lora-path /home/wangxingjian/model/greenplm/release/paper/weight/stage_3 \
#   --pretrain-mm-mlp-adapter /home/wangxingjian/model/greenplm/release/paper/weight/stage_3/non_lora_trainables.bin \
#   --pc-ckpt-path /home/wangxingjian/model/greenplm/pretrained_weight/Uni3D_PC_encoder/modelzoo/uni3d-small/model.pt \
#   --pc-encoder-type small \
#   --get-pc-tokens-way OM_Pooling \
#   --output /home/wangxingjian/AnyPoint/output/greenplm \
#   --devices cuda:0 \
#   --budget 200 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /home/wangxingjian/AnyPoint/output/task_pool_cache \
#   --strategies all \
#   --lambda-explore 0.2
