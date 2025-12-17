# Layout Generator 使用说明

## 功能概述

Layout Generator 是一个基于 LLM 的 3D 场景布局生成器，核心设计思想：

```
LLM 生成语义 DSL → 抽象为可复用模板 → Solver 映射到几何坐标
```

- **LLM**：提供语义多样性，生成物体关系描述
- **Solver**：提供精确坐标，通过程序性扰动添加多样性

## 模块结构

```
layout_generator/
├── schema.py       # 数据结构：DSL、Template、Layout
├── constants.py    # 常量：尺寸映射、空间关系
├── validator.py    # DSL 验证与解析
├── solver.py       # 约束求解器
├── generator.py    # 主生成流程
└── api/            # LLM API 封装
```

## 快速开始

### 1. 配置 API Keys

在 `AnyPoint/configs/keys.yaml` 中配置：

```yaml
keys:
  - YOUR_API_KEY_1
  - YOUR_API_KEY_2
```

### 2. 基本使用

```python
import asyncio
import yaml
from layout_generator import LayoutGenerator, sample_object_names

async def main():
    # 加载 API keys
    api_keys = yaml.safe_load(open("configs/keys.yaml"))["keys"]
    
    # 初始化生成器
    generator = LayoutGenerator(
        model_name="qwen2.5-72b-instruct",  # LLM 模型名
        api_keys=api_keys,
        max_concurrent_per_key=100,          # 每个 key 最大并发
        max_retries=5,                       # 重试次数
        solver_threads=4,                    # 求解器线程数
        seed=42                              # 随机种子（可选）
    )
    
    # 单次生成
    objects = ["table", "chair", "book", "lamp", "cup"]
    template, layouts = await generator.generate_single(
        object_names=objects,
        num_layouts=5  # 生成5个布局变体
    )
    
    # 输出结果
    print(template.to_dict())
    for layout in layouts:
        print(layout.to_dict())

asyncio.run(main())
```

### 3. 批量生成

```python
# 准备多组物体列表
object_lists = [
    ["sofa", "table", "TV", "plant"],
    ["bed", "lamp", "desk", "chair", "book"],
    ["refrigerator", "oven", "pot", "pan"],
]

# 批量生成
templates, layouts = await generator.generate_batch(
    object_lists=object_lists,
    layouts_per_template=3  # 每个模板生成3个变体
)
```

### 4. 随机采样物体

```python
from layout_generator import sample_object_names

# 从可用物体中随机采样 2-9 个
available = ["table", "chair", "sofa", "lamp", "book", "cup", "plant", "TV"]
sampled = sample_object_names(
    available_objects=available,
    count=5,      # 指定数量（可选，默认随机2-9）
    seed=42       # 随机种子（可选）
)
```

## 算法原理

### 整体流程

```
物体列表 → [LLM] → DSL → [抽象] → 模板 → [Solver] → 布局坐标
```

1. **LLM 生成 DSL**：根据物体列表，生成语义化的空间关系描述
2. **抽象为模板**：将物体名替换为占位符（obj_0, obj_1...），提高复用性
3. **Solver 求解**：基于约束的几何布局生成

### Solver 算法

**核心思想**：拓扑排序 + 约束驱动的拒绝采样

```python
# 伪代码
for each object in topological_order:
    for attempt in range(max_attempts):
        # 1. 采样候选位置（根据空间关系约束）
        position = sample_from_constraints(object.relations)
        
        # 2. 验证有效性（边界内 + 无碰撞）
        if is_valid(position):
            place_object(position)
            break
```

**关键步骤**：

1. **拓扑排序**：根据依赖关系确定摆放顺序
   - 被引用的物体先摆放（如 "chair in front of table" → table 先放）
   - 使用 Kahn 算法处理依赖图

2. **约束分解**：将 3D 约束分解为独立维度
   - 垂直约束（Y 轴）：`on`, `above`, `below`
   - 水平约束（XZ 平面）：`in front of`, `beside`, `near`...

3. **位置采样**：
   - **"on" 关系**：精确接触，`y = ref.y + ref.half_y + obj.half_y`
   - **方向关系**：在指定方向上采样距离（基于物体尺寸）
   - **径向关系**：随机角度采样

4. **碰撞检测**：AABB 相交测试
   - 使用分离轴定理（SAT）快速判断
   - 特殊处理垂直堆叠（允许接触）

5. **拒绝采样**：不满足约束则重新采样，最多尝试 1000 次

## 数据格式

### DSL（LLM 输出）

```json
{
  "description": "A table with a chair in front and a book on top",
  "objects": [
    {"name": "table", "size": "largest", "rotation": 0},
    {"name": "chair", "size": "medium", "rotation": 180},
    {"name": "book", "size": "smallest", "rotation": 45}
  ],
  "relations": [
    {"subject": "chair", "relation": "in front of", "reference": "table"},
    {"subject": "book", "relation": "on", "reference": "table"}
  ]
}
```

### Template（抽象模板）

```json
{
  "id": 0,
  "count": 3,
  "description": "A [obj_0] with a [obj_1] in front and a [obj_2] on top",
  "objects": [
    {"name": "obj_0", "size": "largest", "rotation": 0},
    {"name": "obj_1", "size": "medium", "rotation": 180},
    {"name": "obj_2", "size": "smallest", "rotation": 45}
  ],
  "relations": [...]
}
```

### Layout（最终输出）

```json
{
  "id": 0,
  "description": "A [obj_0] with a [obj_1] in front and a [obj_2] on top",
  "objects": [
    {
      "name": "obj_0", 
      "position": [0.0, 1.1, 0.0],  // AABB 中心位置 (x, y, z)
      "rotation": 0,                 // 绕 Y 轴旋转角度（度）
      "size": [2.2, 2.2, 2.2]       // AABB half-extents (x, y, z)
    },
    {
      "name": "obj_1", 
      "position": [0.0, 0.7, 2.5], 
      "rotation": 180, 
      "size": [1.4, 1.4, 1.4]
    },
    {
      "name": "obj_2", 
      "position": [0.1, 2.4, 0.1], 
      "rotation": 45, 
      "size": [0.4, 0.4, 0.4]
    }
  ],
  "relations": [
    {
      "subject": "obj_1",
      "relation": "in front of",
      "reference": "obj_0"
    },
    {
      "subject": "obj_2",
      "relation": "on",
      "reference": "obj_0"
    }
  ]
}
```

**字段说明**：
- `position`: AABB 中心坐标 (x, y, z)
- `rotation`: 绕 Y 轴旋转角度（度）
- `size`: AABB half-extents（半尺寸），完整尺寸 = `size * 2`
- `relations`: 空间关系约束，用于 QA 生成等下游任务

## 约束配置

### 尺寸映射（AABB Half-Extents）

在 `constants.py` 的 `SIZE_RANGES` 中定义，当前配置为**立方体**（三个维度相同）：

| 类别 | X 范围 (宽) | Y 范围 (高) | Z 范围 (深) |
|------|------------|------------|------------|
| largest | 2.0 - 2.5 | 2.0 - 2.5 | 2.0 - 2.5 |
| large | 1.5 - 2.0 | 1.5 - 2.0 | 1.5 - 2.0 |
| medium | 1.0 - 1.5 | 1.0 - 1.5 | 1.0 - 1.5 |
| small | 0.6 - 1.0 | 0.6 - 1.0 | 0.6 - 1.0 |
| smallest | 0.3 - 0.6 | 0.3 - 0.6 | 0.3 - 0.6 |

> **注**：值为 AABB half-extents（半长），完整尺寸是 `2 * size`

### 支持的空间关系

**水平关系**：`in front of`, `behind`, `to the left of`, `to the right of`, `beside`, `next to`, `near`, `far from`

**垂直关系**：`on`（接触）, `above`, `below`, `under`

**其他**：`surrounding`, `at the center of`

## 注意事项

1. **物体数量**：每次生成 2-9 个物体
2. **尺寸约束**：DSL 必须包含至少一个 `largest` 和一个 `smallest`
3. **求解失败**：部分 DSL 可能无法求解（约束冲突），会自动跳过
4. **并发控制**：`max_concurrent_per_key` 控制 API 并发，根据 API 限制调整

