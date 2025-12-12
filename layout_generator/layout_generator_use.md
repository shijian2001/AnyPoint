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
  "description": "...",
  "objects": [
    {"name": "obj_0", "position": [0.0, 1.1, 0.0], "rotation": 0, "size": 2.2},
    {"name": "obj_1", "position": [0.0, 0.7, 2.5], "rotation": 180, "size": 1.4},
    {"name": "obj_2", "position": [0.1, 2.4, 0.1], "rotation": 45, "size": 0.4}
  ]
}
```

## 约束配置

### 尺寸映射

| 类别 | 缩放范围 |
|------|----------|
| largest | 2.0 - 2.5 |
| large | 1.5 - 2.0 |
| medium | 1.0 - 1.5 |
| small | 0.6 - 1.0 |
| smallest | 0.3 - 0.6 |

> 注：所有物体已归一化为单位球内，尺寸为相对比例

### 支持的空间关系

**水平关系**：`in front of`, `behind`, `to the left of`, `to the right of`, `beside`, `next to`, `near`, `far from`

**垂直关系**：`on`（接触）, `above`, `below`, `under`

**其他**：`surrounding`, `at the center of`

## 注意事项

1. **物体数量**：每次生成 2-9 个物体
2. **尺寸约束**：DSL 必须包含至少一个 `largest` 和一个 `smallest`
3. **求解失败**：部分 DSL 可能无法求解（约束冲突），会自动跳过
4. **并发控制**：`max_concurrent_per_key` 控制 API 并发，根据 API 限制调整

