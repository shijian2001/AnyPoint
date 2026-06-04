# PointQA_Eval

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Every environment below is managed via `uv sync` from inside the corresponding `envs/<model>/` directory. The general flow is:

```bash
cd AnyPoint/envs/<model>
uv sync
source .venv/bin/activate
```

## 2. Supported Models

### 2.1 PointLLM

#### 2.1.1 Virtual Environment

```bash
cd AnyPoint/envs/pointllm
uv sync
source .venv/bin/activate
```

#### 2.1.2 Download Checkpoints

```bash
hf download RunsenXu/PointLLM_7B_v1.2 --local-dir /path/PointLLM_7B_v1.2
```

### 2.2 ShapeLLM

#### 2.2.1 Virtual Environment

```bash
cd AnyPoint/envs/shapellm
uv sync
source .venv/bin/activate
```

#### 2.2.2 Install Pointnet2_PyTorch

NOTE: The Torch version must match your CUDA toolkit version. See this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174) before building.

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib
uv pip install -e . --no-build-isolation
```

#### 2.2.3 Download Weights

```bash
hf download qizekun/ReConV2 zeroshot/large/best_modelnet40_overall.pth --local-dir /path/ReconV2_large
hf download timm/eva_large_patch14_336.in22k_ft_in22k_in1k --local-dir /path/eva_large_patch14_336.in22k_ft_in22k_in1k
hf download qizekun/ShapeLLM_7B_general_v1.0 --local-dir /path/ShapeLLM_7B_general_v1.0
```

### 2.3 GreenPLM

#### 2.3.1 Virtual Environment

```bash
cd AnyPoint/envs/greenplm
uv sync
source .venv/bin/activate
```

#### 2.3.2 Pointnet2_PyTorch

GreenPLM depends on `pointnet2_ops` built from a local clone of [erikwijmans/Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch). Before `uv sync`, **manually update `envs/greenplm/pyproject.toml`** so that the local path matches your machine, e.g.:

```toml
[tool.uv.sources]
pointnet2-ops = { path = "/your/abs/path/Pointnet2_PyTorch/pointnet2_ops_lib", editable = true }
```

Then clone the repo if you do not have it yet, re-run `uv sync` after the path is correct. If the build fails, **check this [issue](https://github.com/erikwijmans/Pointnet2_PyTorch/issues/174)** for CUDA / Torch compatibility caveats.:

```bash
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /your/abs/path/Pointnet2_PyTorch
```



### 2.4 MiniGPT3D

#### 2.4.1 Virtual Environment

```bash
cd "AnyPoint/envs/pointalign&MiniGPT3D"
uv sync
source .venv/bin/activate
```

(The PointAlign env is shared with MiniGPT3D.)

#### 2.4.2 Update the Model Configuration

1. Check this [issue](https://github.com/TangYuan96/MiniGPT-3D/issues/6), and move [MiniGPT-3D/modeling_phi.py](https://github.com/TangYuan96/MiniGPT-3D/blob/main/modeling_phi.py) to `transformers/models/phi/modeling_phi.py`.
2. Update the model paths in [benchmark_evaluation_paper.yaml](models/dependence/minigpt3d/eval_configs/benchmark_evaluation_paper.yaml).

#### 2.4.3 Modify Local Model Paths

Edit the following files and replace the tokenizer path with your local Phi-2 directory:

- [conversation.py](models/dependence/minigpt3d/minigpt4/conversation/conversation.py) line 20
- [base_model.py](models/dependence/minigpt3d/minigpt4/models/base_model.py) line 55

```python
tokenizer = AutoTokenizer.from_pretrained("model/MiniGPT-3D/params_weight/Phi_2")
```

### 2.5 OneLLM

#### 2.5.1 Virtual Environment

```bash
cd AnyPoint/envs/onellm
uv sync
source .venv/bin/activate
```

#### 2.5.2 Install pointnet2

```bash
git clone https://github.com/csuhan/OneLLM.git
cd OneLLM/model/lib/pointnet2
python setup.py install
```

#### 2.5.3 Download Checkpoints

```bash
hf download timm/vit_large_patch14_clip_224.openai --local-dir /model/vit_large_patch14_clip_224
hf download csuhan/OneLLM-7B --local-dir /model/OneLLM-7B
```

### 2.6 PointAlign

#### 2.6.1 Virtual Environment

PointAlign shares its environment with MiniGPT3D:

```bash
cd "AnyPoint/envs/pointalign&MiniGPT3D"
uv sync
source .venv/bin/activate
```

#### 2.6.2 Download Checkpoints

```bash
hf download ShijianW01/PointAlign_weight --local-dir /path
hf download Vision-CAIR/minigpt4 blip2_pretrained_flant5xxl.pth --local-dir /path --repo-type=space
```

#### 2.6.3 Update the Model Configuration

PointAlign uses the same underlying framework as MiniGPT3D. Before running evaluation, make sure to:
- update the model paths in [benchmark_evaluation_paper.yaml](models/dependence/pointalign/eval_configs/benchmark_evaluation_paper.yaml)
- move [modeling_phi.py](models/dependence/pointalign/minigpt4/models/modeling_phi.py) to `transformers/models/phi/modeling_phi.py`

## Quick Start

Before running evaluation, update the model paths in the script to match your local environment.

```bash
bash run_eval.sh
```
