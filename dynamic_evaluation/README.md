# Dynamic Evaluation v2

## 1. 简介

Dynamic Evaluation v2 的目标是在有限查询预算 `B` 下，更高效地发现被测 3D 多模态模型的错误样本、薄弱任务类型和能力边界。

相比 v1，v2 不再把候选池理解为“每一轮重新生成的一批任务”，而是先由 AnyPoint 一次性合成一个固定候选池，并缓存到磁盘。后续动态策略和随机基线都从同一个固定池中抽样，因此两种策略面对的候选空间一致，比较更公平，也更容易复现实验。

v2 的核心思想仍然是利用已知错误，但去冗余方式发生了变化：

- v1 惩罚候选任务与全部历史任务 `H` 的相似度。
- v2 只惩罚候选任务与已答对任务集合 `C` 的相似度。

也就是说，v2 鼓励继续挖掘“像错题”的区域，同时主动避开模型已经答对的安全区域。

## 2. Problem Formulation

给定被测模型 `M`、查询预算 `B`、批次大小 `K` 和固定候选池大小 `N`，动态评测的目标是在最多评测 `B` 个任务的条件下，最大化发现错误任务数量。

### 2.1 组件定义

- **生成器 `G`**：AnyPoint 任务生成系统，用于根据布局、物体元数据和题型模板生成候选 3D QA 任务。
- **被测模型 `M`**：需要评测的 3D QA 模型，例如 PointLLM、ShapeLLM 等。
- **评测任务 `t`**：一个多选 3D QA 样本：

```text
t = (l, q, o, a, p)
```

其中：

- `l`：实例化后的 3D 场景布局文本描述。
- `q`：问题文本。
- `o`：多选选项。
- `a`：Ground Truth 答案。
- `p`：可物化为 `.npy` 点云的场景数据。

### 2.2 任务向量

v2 使用三组件编码，而不是把 layout、question、answer 简单拼接成一个字符串后编码。

```text
v_layout   = Embed(l)
v_question = Embed(q)
v_answer   = Embed(a)
v_t        = Normalize([v_layout || v_question || v_answer])
```

其中 `||` 表示向量拼接。当前实现使用 `sentence-transformers/all-mpnet-base-v2` 编码三个文本组件，并对拼接后的向量重新归一化。

### 2.3 状态集合

在任意评测步 `i`，系统维护以下集合：

- **历史集 `H_i`**：所有已评测任务。
- **正确集 `C_i`**：模型回答正确的任务集合，`C_i ⊆ H_i`。
- **错误集 `E_i`**：模型回答错误的任务集合，`E_i ⊆ H_i`。
- **剩余候选池 `R_i`**：固定候选池中尚未被评测的任务。

其中：

```text
H_i = C_i ∪ E_i
C_i ∩ E_i = ∅
```

## 3. v2 核心机制：错误亲和度 × 正确区域新颖性

v2 的信息效用函数为：

```text
U(t) = Affinity(t, E_i) · Novelty(t, C_i)
```

展开为：

```text
U(t) = max_sim(v_t, E_i) · (1 - λ · max_sim(v_t, C_i))
```

其中：

```text
max_sim(v_t, S) = max_{s ∈ S} cos(v_t, v_s)
```

若集合为空，则对应 `max_sim` 为 0。

### 3.1 Affinity：错误亲和度

```text
Affinity(t, E_i) = max_sim(v_t, E_i)
```

该项衡量候选任务与已知错误任务的最大相似度。

直观含义：如果一个候选任务在布局、问题和答案语义上接近已知错题，它更可能处在模型的薄弱区域，因此应优先评测。

### 3.2 Novelty：正确区域新颖性

```text
Novelty(t, C_i) = 1 - λ · max_sim(v_t, C_i)
```

该项惩罚候选任务与已答对任务的相似度。

直观含义：如果模型已经在某一类任务上答对了很多次，继续评测相似样本的边际收益较低。v2 会降低这些候选任务的分数，把预算转向更可能出错的区域。

### 3.3 λ 的作用

`λ` 是正确区域去冗余权重，当前默认值为 `0.2`。

- `λ = 0`：只利用错误亲和度，不惩罚正确区域。
- `0 < λ < 1`：在追踪错误模式的同时，降低已知正确区域的重复采样概率。
- `λ = 1`：最强去冗余，候选任务越像已答对任务，效用越低。

当前实现会把相似度裁剪到 `[0, 1]`，因此效用分数也保持在稳定范围内。

## 4. 固定候选池与缓存机制

v2 的 `pool_size = N` 表示一次性预生成的固定候选池总大小，而不是每轮重新生成的候选数量。

候选池由多个任务生成计划混合构成，包括：

- 距离相关：closest / farthest
- 属性相关：what / list / count attribute
- 数量相关：count object
- 频次相关：most / least frequent
- 尺寸相关：largest / smallest
- 位置相关：where distance / where size

生成后，系统将候选任务写入：

```text
task_pool_cache/task_pool_manifest.json
```

缓存中保存：

- `pool_size`
- `seed`
- 任务生成计划 `task_plans`
- 数据源签名 `source_signature`
- 候选任务的问题、选项、答案和 metadata
- `cache_format_version`

如果输入数据、生成计划或配置发生变化，缓存签名会不匹配，系统会自动重新生成候选池。

## 5. 算法流程

### 输入

- 总预算 `B`
- 批次大小 `K`
- 固定候选池大小 `N`
- 去冗余权重 `λ`
- 随机种子 `seed`

### 5.1 Pool Preparation

1. 使用 AnyPoint 生成固定候选池 `P`，大小最多为 `N`。
2. 将任务池缓存到 `task_pool_cache/task_pool_manifest.json`。
3. 初始化剩余候选池：

```text
R_0 = P
```

### 5.2 Cold Start

1. 从 `R_0` 中随机抽取 `K` 个任务。
2. 物化点云并调用模型 `M` 推理。
3. 根据模型是否答对，初始化：

```text
H, C, E
```

Cold Start 的 utility 记为 `null`，因为这一步尚未使用动态效用排序。

### 5.3 Dynamic Loop

当已评测数量 `< B` 时，重复以下步骤：

1. **候选重排**
   - 对当前剩余池 `R_i` 中所有任务编码，得到任务向量。
   - 使用当前 `C_i` 和 `E_i` 计算每个候选任务的 `U(t)`。

2. **Top-K 选择**
   - 选择效用最高的 `K` 个任务。
   - 将它们从剩余候选池中移除，避免重复评测。

3. **高成本评测**
   - 若候选任务尚未携带点云，则调用 AnyPoint 物化点云。
   - 保存 `.npy` 点云。
   - 调用被测模型进行多选 QA 推理。

4. **状态更新**
   - 答对任务加入 `C`。
   - 答错任务加入 `E`。
   - 重新编码 `C` 和 `E`，用于下一轮动态排序。

### 5.4 伪代码

```text
Input: budget B, batch size K, fixed pool size N, λ

P = BuildOrLoadFixedPool(N)
R = P
H, C, E = ∅, ∅, ∅

S = RandomSample(R, K)
Evaluate(S)
Update(H, C, E)
R = R \ S

while |H| < B:
    for t in R:
        v_t = Encode(layout(t), question(t), answer(t))
        affinity = max_sim(v_t, E)
        redundancy = max_sim(v_t, C)
        U(t) = affinity * (1 - λ * redundancy)

    S = TopK(R, U, min(K, B - |H|))
    Evaluate(S)
    Update(H, C, E)
    R = R \ S

return results, hard_data
```

## 6. Random vs Dynamic 对比实验

v2 提供 `compare_eval_strategies.py`，用于在同一个固定任务池上比较随机采样和动态采样。

### 6.1 Random Baseline

随机策略从完整固定池中一次性随机抽取 `B` 个任务，然后按 batch 评测。它不使用错误反馈，也不计算 utility。

### 6.2 Dynamic Strategy

动态策略先随机 cold start 一个 batch，然后在每一轮根据 `U(t)` 对剩余池重排，选择 top-K 任务继续评测。

### 6.3 公平性

两种策略使用同一个 `base_items` 候选池副本：

- 候选任务空间相同。
- 预算 `B` 相同。
- batch size 相同。
- seed 相同。
- 被测模型和点云物化逻辑相同。

因此输出中的差异主要来自采样策略，而不是候选任务生成差异。

## 7. 输出格式

### 7.1 单策略动态评测

`run_dynamic_eval.py` 输出：

```text
output_dir/
├── results.json
├── eval_point_clouds/
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── task_pool_cache/
│   └── task_pool_manifest.json
└── hard_data/
    ├── tasks.jsonl
    ├── tasks_info.json
    └── pcd/
        ├── 000000.npy
        └── ...
```

`results.json` 包含：

```json
{
  "config": {
    "budget": 100,
    "batch_size": 10,
    "pool_size": 1000,
    "lambda_explore": 0.2,
    "seed": 42
  },
  "stats": {
    "total": 100,
    "errors": 25,
    "error_rate": 0.25,
    "error_indices": [3, 7, 12]
  },
  "results": [
    {
      "task_id": 0,
      "question": "...",
      "answer": "...",
      "model_raw_output": "...",
      "model_answer": "...",
      "is_correct": true,
      "utility": null,
      "category": "what_attribute",
      "options": ["A", "B", "C", "D"],
      "layout_description": "..."
    }
  ]
}
```

### 7.2 Hard Data

`hard_data/` 只保存模型答错的任务，可直接作为 hard benchmark、回归测试集或后续训练数据。

```text
hard_data/
├── tasks.jsonl
├── tasks_info.json
└── pcd/
```

`tasks.jsonl` 每行一个错误任务：

```json
{"question_id": 0, "point": "000000.npy", "category": "where_distance_farthest", "question": "...", "options": ["A", "B", "C", "D"], "answer": "A"}
```

`results.json` 中的 `error_indices` 可追溯 hard task 在完整评测序列中的原始 `task_id`。

### 7.3 Random vs Dynamic 输出

`compare_eval_strategies.py` 输出：

```text
output_dir/
├── random/
│   ├── results.json
│   ├── eval_point_clouds/
│   └── hard_data/
├── dynamic/
│   ├── results.json
│   ├── eval_point_clouds/
│   └── hard_data/
├── task_pool_cache/
│   └── task_pool_manifest.json
└── compare_summary.json
```

`compare_summary.json` 记录：

```json
{
  "random": {
    "total": 100,
    "errors": 18,
    "error_rate": 0.18,
    "error_indices": [1, 5, 9]
  },
  "dynamic": {
    "total": 100,
    "errors": 25,
    "error_rate": 0.25,
    "error_indices": [3, 7, 12]
  },
  "delta": {
    "errors": 7,
    "error_rate": 0.07
  }
}
```

## 8. 使用方式

### 8.1 运行动态评测

```bash
python run_dynamic_eval.py \
  --metadata /path/to/metadata.jsonl \
  --pcd-dir /path/to/points_npy \
  --layouts /path/to/layouts.json \
  --model pointllm \
  --checkpoint /path/to/PointLLM_7B_v1.2 \
  --output ./output/pointllm_dyn \
  --budget 100 \
  --batch-size 10 \
  --pool-size 1000 \
  --lambda-explore 0.2 \
  --seed 42
```

### 8.2 比较随机采样和动态采样

```bash
python compare_eval_strategies.py \
  --metadata /path/to/metadata.jsonl \
  --pcd-dir /path/to/points_npy \
  --background_dir /path/to/background \
  --layouts /path/to/layouts.json \
  --model pointllm \
  --checkpoint /path/to/PointLLM_7B_v1.2 \
  --output ./output/compare_pointllm \
  --devices cuda:0 \
  --budget 100 \
  --batch-size 10 \
  --pool-size 1000 \
  --lambda-explore 0.2
```

如果已有候选池缓存，可以复用：

```bash
python compare_eval_strategies.py \
  ... \
  --pool-cache-dir ./output/pointllm_dyn/task_pool_cache
```

## 9. Key Insight

v2 的高 utility 样本满足两个条件：

1. **像错题**：与历史错误任务相似，说明它可能落在模型薄弱区域。
2. **不像已答对题**：与历史正确任务不太相似，说明它不是模型已经稳定掌握的区域。

因此，v2 的动态策略不只是追着错误样本做局部重复，而是在固定候选池中持续寻找“靠近错误区域、远离正确区域”的任务。这种机制能更快地发现模型在距离、尺寸、属性、频次和空间关系等维度上的失败模式，并把答错样本沉淀为可复用的 `hard_data`。

## 10. v1 到 v2 的主要变化

| 项目 | v1 | v2 |
| --- | --- | --- |
| 候选池 | 每轮生成新的候选池 `C_i` | 启动时生成固定池 `P`，每轮重排剩余池 `R_i` |
| pool_size 含义 | 每轮候选数量 | 固定候选池总大小 |
| 效用函数 | `max_sim(t, E) - λ max_sim(t, H)` | `max_sim(t, E) * (1 - λ max_sim(t, C))` |
| 去冗余对象 | 全部历史任务 `H` | 已答对任务 `C` |
| 任务向量 | 拼接文本后编码 | layout / question / answer 分别编码后拼接 |
| 对比实验 | 随机与动态候选空间可能不同 | 随机和动态共享同一固定候选池 |
| 可复现性 | 依赖每轮生成过程 | 任务池带签名缓存，可复用、可追溯 |
| 输出 | 动态结果与 hard data | 增加 random/dynamic 对比和 `compare_summary.json` |
