# Dynamic Evaluation

基于信息效用检索的动态评测模块，在有限预算下高效发现 3D 模型的认知缺陷与能力边界。

## 算法

### 核心思想
在预算 B 限制下，平衡**利用**（挖掘已知错误模式）与**探索**（测试未知区域），最大化发现模型错误。

### 效用函数
```
U(t) = max_{e∈E} ⟨v_t, v_e⟩ - λ·max_{h∈H} ⟨v_t, v_h⟩
```
- **E**: 错误集（模型失败的任务）
- **H**: 历史集（所有已测试任务）  
- **λ**: 探索权重，λ ∈ [0,1]

### 算法流程

**输入**: 预算 B, 批次大小 K, 候选池大小 N

**1. Cold Start**
- 随机生成并评测 K 个任务
- 初始化历史集 H 和错误集 E

**2. 迭代循环** (while 已评测 < B)
```
a. 生成 N 个候选任务 (低成本)
b. 计算每个任务的效用 U(t)
c. 选择 Top-K 高效用任务
d. 评测选中任务 (高成本: 渲染 + 推理)
e. 更新 H 和 E
```

## 使用

### 基础用法

```python
from point_qa_generator.generator import PointQAGenerator
from models.point_qa_model import PointQAModel
from dynamic_evaluation import DynamicEvaluator, EvalConfig

# 初始化
qa_gen = PointQAGenerator(
    metadata_file="data/metadata.jsonl",
    pcd_dir="data/point_clouds",
    layouts_file="data/layouts.json",
    seed=42
)

model = PointQAModel(
    model_name="your_model",
    checkpoint_path="checkpoints/model.pth"
)

# 配置
config = EvalConfig(
    budget=100,           # B: 总预算
    batch_size=10,        # K: 每轮评测数
    pool_size=1000,       # N: 候选池大小 (N >> K)
    lambda_explore=0.2,   # λ: 探索权重
    seed=42
)

# 运行
evaluator = DynamicEvaluator(qa_gen, model, config)
results = evaluator.run("output/")
```

### 命令行

在 AnyPoint 根目录下运行：

```bash
python run_dynamic_eval.py \
    --metadata data/metadata.jsonl \
    --pcd-dir data/point_clouds \
    --layouts data/layouts.json \
    --model your_model \
    --checkpoint checkpoints/model.pth \
    --output results/eval \
    --budget 100 \
    --batch-size 10 \
    --pool-size 1000 \
    --lambda-explore 0.2
```

## 参数说明

| 参数 | 符号 | 推荐值 | 说明 |
|-----|------|--------|------|
| `budget` | B | 100-1000 | 总评测预算 |
| `batch_size` | K | 10-20 | 每轮评测任务数 |
| `pool_size` | N | 1000-10000 | 候选池大小，N >> K |
| `lambda_explore` | λ | 0.2-0.5 | 探索权重 |

**λ 选择**:
- λ=0: 纯利用（aggressive 挖掘错误）
- λ=0.2: 默认（轻微偏向利用）
- λ=0.5: 平衡
- λ=1: 纯探索（广覆盖）

## 输出

评测完成后生成两部分输出：

### 1. 完整评测结果

`output_dir/results.json`:
```json
{
  "config": {
    "budget": 100,
    "batch_size": 10,
    "pool_size": 1000,
    "lambda_explore": 0.2
  },
  "stats": {
    "total": 100,
    "errors": 25,
    "error_rate": 0.25,
    "error_indices": [3, 7, 12, 15, ...]
  },
  "results": [
    {
      "task_id": 0,
      "question": "...",
      "answer": "...",
      "model_answer": "...",
      "is_correct": true,
      "utility": null,
      "category": null,
      "options": null,
      "layout_description": null
    },
    {
      "task_id": 3,
      "question": "What is the color of the chair?",
      "answer": "blue",
      "model_answer": "red",
      "is_correct": false,
      "utility": 0.85,
      "category": "what_attribute_",
      "options": ["blue", "red", "green", "yellow"],
      "layout_description": "The chair is at the center. The table is on the chair."
    },
    ...
  ]
}
```

### 2. 错误任务数据集（hard_data）

`output_dir/hard_data/` - 标准生成器格式，可直接用于重测或训练：

```
hard_data/
├── tasks.jsonl           # JSONL格式任务列表
├── tasks_info.json       # 数据集元信息
└── pcd/                  # 点云文件
    ├── 000000.npy
    ├── 000001.npy
    └── ...
```

**tasks.jsonl** (每行一个任务，与生成器格式完全一致):
```json
{"question_id": 0, "point": "000000.npy", "category": "what_attribute_", "question": "What is the color of the chair?", "options": ["blue", "red", "green", "yellow"], "answer": "blue"}
{"question_id": 1, "point": "000001.npy", "category": "where_distance_", "question": "Where is the closest object?", "options": ["left", "right", "front", "back"], "answer": "left"}
```

**tasks_info.json**:
```json
{
  "task_plan": {
    "generator_type": "mixed",
    "num_options": 4,
    "seed": 42
  },
  "generation_stats": {
    "num_tasks_requested": 25,
    "num_tasks_generated": 25,
    "output_directory": "output_dir/hard_data"
  }
}
```

**映射关系**:
- `hard_data/tasks.jsonl` 中 `question_id: 0` 对应 `results.json` 中 `task_id: 3` (error_indices[0])
- `hard_data/tasks.jsonl` 中 `question_id: 1` 对应 `results.json` 中 `task_id: 7` (error_indices[1])
- 通过 `error_indices` 可追溯每个错误任务在完整评测中的位置

## 模块

- `config.py`: 配置和数据结构
- `embedder.py`: 三组件任务编码 (layout + question + answer)
- `utility.py`: 信息效用计算
- `task_pool.py`: 多样化任务生成
- `evaluator.py`: 主评测引擎

