export CUDA_VISIBLE_DEVICES=0
export HF_ENDPOINT=https://hf-mirror.com
export SENTENCE_TRANSFORMERS_HOME=/home/wangxingjian/model/sentence_transformers


# ShapeLLM
# python main.py \
#   --model_name shapellm \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --test_ckpt /home/wangxingjian/model/ShapeLLM-7B-General \
#   --llava_model_base /home/wangxingjian/model/llava-v1.5-7b \
#   --output_dir ./eval_results/shapellm \
#   --device cuda

# PointLLM
python3 main.py \
  --model_name pointllm \
  --tasks_file ./what_distance_farthest/tasks.jsonl \
  --point_cloud_dir ./what_distance_farthest/pcd \
  --checkpoint_path /home/wangxingjian/model/PointLLM_7B_v1.2 \
  --output_dir ./eval_results/pointllm \
  --device cuda

# OneLLM
# python main.py \
#   --model_name onellm \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --checkpoint_path /home/wangxingjian/model/OneLLM-7B/consolidated.00-of-01.pth \
#   --clip_pretrained_path /home/wangxingjian/model/vit_large_patch14_clip_224/open_clip_pytorch_model.bin \
#   --point_format xyzrgb \
#   --offline True \
#   --output_dir ./eval_results/onellm \
#   --device cuda


# MiniGPT-3D
# python /home/wangxingjian/PointQA_Eval/main.py \
#   --model_name minigpt3d \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --cfg_path /home/wangxingjian/PointQA_Eval/models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml \
#   --output_dir ./eval_results/minigpt3d \
#   --device cuda

# PointAlign
# python3 /home/wangxingjian/PointQA_Eval/main.py \
#   --model_name pointalign \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --cfg_path /home/wangxingjian/PointQA_Eval/models/dependence/pointalign/eval_configs/benchmark_evaluation_paper.yaml \
#   --weights_root /home/wangxingjian/model/pointalign \
#   --output_dir ./eval_results/pointalign \
#   --device cuda:0 \
#   --llama_model_path /home/wangxingjian/model/pointalign/Phi_2 \
#   --bert_base_uncased_path /home/wangxingjian/model/pointalign/bert-base-uncased \
#   --pc_encoder_path /home/wangxingjian/model/pointalign/pc_encoder/point_model.pth \
#   --pretrain_ckpt /home/wangxingjian/model/pointalign/pointalign/pretrain.pth \
#   --finetune_ckpt /home/wangxingjian/model/pointalign/pointalign/finetune.pth \
#   --qformer_pretrained_path /home/wangxingjian/model/pointalign/blip2_pretrained_flant5xxl.pth

# greenplm
# python main.py \
#   --model_name greenplm \
#   --tasks_file ./what_distance_farthest/tasks.jsonl \
#   --point_cloud_dir ./what_distance_farthest/pcd \
#   --model_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/lava-vicuna_2024_4_Phi-3-mini-4k-instruct \
#   --lora_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3 \
#   --pretrain_mm_mlp_adapter /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/release/paper/weight/stage_3/non_lora_trainables.bin \
#   --pc_ckpt_path /home/wangxingjian/PointQA_Eval/cankao/GreenPLM/pretrained_weight/Uni3D_PC_encoder/modelzoo/uni3d-small/model.pt \
#   --pc_encoder_type small \
#   --get_pc_tokens_way OM_Pooling \
#   --output_dir ./eval_results/greenplm \
#   --device cuda
