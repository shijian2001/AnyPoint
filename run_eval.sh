#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=4,5
export HF_ENDPOINT=https://hf-mirror.com
export SENTENCE_TRANSFORMERS_HOME=/root/weishuai/model


# ShapeLLM
# python3 /AnyPoint/compare_eval_strategies.py \
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
#   --lambda-explore 0.2

# PointLLM
python3 compare_eval_strategies.py \
  --metadata /root/weishuai/data/point_cloud/texverse_metadata_000-000.jsonl \
  --pcd-dir /root/weishuai/data/point_cloud/npys_2k/000-000 \
  --background-dir /root/weishuai/data/point_cloud/background \
  --layouts /root/weishuai/AnyPoint/outputs_gpt_oss/layouts.json \
  --model pointllm \
  --checkpoint /root/weishuai/model/PointLLM_7B_v1.2 \
  --output /root/weishuai/AnyPoint/output/pointllm \
  --devices cuda:0,cuda:1 \
  --budget 200 \
  --batch-size 10 \
  --pool-size 1000 \
  --pool-cache-dir /root/weishuai/AnyPoint/output \
  --lambda-explore 0.2

# OneLLM
# Single-GPU (batched inference inside generate via MetaModel.generate left-pad loop):
# python3 compare_eval_strategies.py \
#   --metadata /root/weishuai/data/point_cloud/texverse_metadata_000-000.jsonl \
#   --pcd-dir /root/weishuai/data/point_cloud/npys_2k/000-000 \
#   --background-dir /root/weishuai/data/point_cloud/background \
#   --layouts /root/weishuai/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model onellm \
#   --checkpoint /root/weishuai/model/OneLLM-7B/consolidated.00-of-01.pth \
#   --clip-pretrained-path /root/weishuai/model/vit_large_patch14_clip_224/open_clip_pytorch_model.bin \
#   --point-format xyzrgb \
#   --offline true \
#   --output /root/weishuai/AnyPoint/output \
#   --devices cuda:0\
#   --budget 200 \
#   --batch-size 50 \
#   --pool-size 1000 \
#   --pool-cache-dir /root/weishuai/AnyPoint/output \
#   --lambda-explore 0.2

# Multi-GPU (fairscale tensor-parallel, like csuhan/OneLLM/demos/cli.py):
# torchrun --nproc_per_node=2 --master_port=23862 compare_eval_strategies.py \
#   --metadata /root/weishuai/data/point_cloud/texverse_metadata_000-000.jsonl \
#   --pcd-dir /root/weishuai/data/point_cloud/npys_2k/000-000 \
#   --background-dir /root/weishuai/data/point_cloud/background \
#   --layouts /root/weishuai/AnyPoint/outputs_gpt_oss/layouts.json \
#   --model onellm \
#   --checkpoint /root/weishuai/model/OneLLM-7B/consolidated.00-of-01.pth \
#   --clip-pretrained-path /root/weishuai/model/vit_large_patch14_clip_224/open_clip_pytorch_model.bin \
#   --point-format xyzrgb \
#   --offline true \
#   --output /root/weishuai/AnyPoint/output \
#   --devices cuda:0,cuda:1 \
#   --budget 100 \
#   --batch-size 4 \
#   --pool-size 1000 \
#   --pool-cache-dir /root/weishuai/AnyPoint/output \
#   --lambda-explore 0.2

# MiniGPT-3D
# --batch-size mirrors the official MiniGPT-3D eval (batch=15).
# python3 /AnyPoint/compare_eval_strategies.py \
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
#   --lambda-explore 0.2

# PointAlign
# python3 /AnyPoint/compare_eval_strategies.py \
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
#   --lambda-explore 0.2

# GreenPLM
# python3 /AnyPoint/compare_eval_strategies.py \
#   --metadata /data/texverse/metadata.jsonl \
#   --pcd-dir /data/texverse/points_npy \
#   --background-dir /data/texverse/background \
#   --layouts /AnyPoint/outputs_gpt_oss/layouts.json \
#   --model greenplm \
#   --model-path /PointQA_Eval/cankao/GreenPLM/lava-vicuna_2024_4_Phi-3-mini-4k-instruct \
#   --lora-path /PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3 \
#   --pretrain-mm-mlp-adapter /PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3/non_lora_trainables.bin \
#   --pc-ckpt-path /PointQA_Eval/cankao/GreenPLM/pretrained_weight/Uni3D_PC_encoder/modelzoo/uni3d-small/model.pt \
#   --pc-encoder-type small \
#   --get-pc-tokens-way OM_Pooling \
#   --output /AnyPoint/output/compare_greenplm \
#   --devices cuda:0 \
#   --budget 100 \
#   --batch-size 10 \
#   --pool-size 1000 \
#   --pool-cache-dir /AnyPoint/output/pointllm_dyn/task_pool_cache \
#   --lambda-explore 0.2
