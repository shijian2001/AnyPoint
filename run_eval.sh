#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export SENTENCE_TRANSFORMERS_HOME=/home/wangxingjian/model/sentence_transformers

python /home/wangxingjian/AnyPoint/run_dynamic_eval.py \
  --metadata /home/wangxingjian/data/texverse/metadata.jsonl \
  --pcd-dir /home/wangxingjian/data/texverse/points_npy \
  --layouts /home/wangxingjian/AnyPoint/outputs_gpt_oss/layouts.json \
  --model pointllm \
  --checkpoint /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output ./output/pointllm_dyn \
  --device cuda:0 \
  --budget 10 \
  --batch-size 10 \
  --pool-size 100 \
  --lambda-explore 0.2
