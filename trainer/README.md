# PointQA Trainer

## 1. PointLLM

### 1.1 Install Packages

```bash
cd envs/pointllm
uv sync
source .venv/bin/activate
```

### 1.2 Download checkpoints

```bash
hf download RunsenXu/PointLLM_7B_v1.1_init --local-dir /path/PointLLM_7B_v1.1_init
```

### 1.3 Update the Model Configuration

Modify the path in [run_trainer.sh](/trainer/PointLLM/run_trainer.sh)

### 1.4 Run Trainer

```bash
cd trainer/PointLLM
bash run_trainer.sh
```
## 2. PointAlign

### 2.1 Install Packages

```bash
cd pointalign&MiniGPT3D
uv sync
source .venv/bin/activate
```

### 2.2 Update the Model Configuration

Modify the model path in [finetune_pointalign.yaml](/trainer/PointAlign/finetune_pointalign.yaml) and [run_trainer.sh](/trainer/PointAlign/run_trainer.sh)

### 2.3 Run Trainer

```bash
cd trainer/PointAlign
bash run_trainer.sh
```