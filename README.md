# AnyPoint: Programmatically Scaling Point Cloud Instruction Data Generation

A fast, scalable data engine for programmatically synthesizing 3D point cloud instruction datasets with guaranteed ground truth and diverse point cloud scenes.

## Installation

```bash
git clone https://github.com/shijian2001/AnyPoint.git
cd AnyPoint
conda create -n anypoint python==3.11
conda activate anypoint
pip install -r requirements.txt
```

## Quick Start

### Generate Mixed Dataset (Recommended)

```python
from point_qa_generator import PointQAGenerator, TaskPlan

generator = PointQAGenerator(
    metadata_file="/path/to/metadata.jsonl",
    pcd_dir="/path/to/point_clouds",
    layouts_file="/path/to/layouts.jsonl",
    background_dir="/path/to/background",
    seed=42
)

# All 17 types, equally distributed, random 4-6 options
task_plan = TaskPlan(
    generator_type=list(generator.generators.keys()),
    num_options=(4, 6),
    seed=42
)
generator.generate(task_plan, num_tasks=1000, output_dir="./output")
```

### Single Type

```python
task_plan = TaskPlan(
    generator_type="what_distance",
    num_options=4,
    seed=42,
    generator_config={"distance_type": "closest"}
)
generator.generate(task_plan, num_tasks=100, output_dir="./output")
```

### Weighted Types

```python
task_plan = TaskPlan(
    generator_type={
        "what_distance": 0.3,
        "what_relation": 0.4,
        "multi_hop_relation": 0.3
    },
    num_options=(4, 6),
    seed=42
)
generator.generate(task_plan, num_tasks=500, output_dir="./output")
```

### CLI Script

```bash
# All 17 types, 1000 tasks
python run_point_qa_gen.py --num-tasks 1000 --output ./output

# Specific types
python run_point_qa_gen.py --num-tasks 100 --types what_distance what_relation --output ./output

# Weighted types
python run_point_qa_gen.py --num-tasks 500 --types what_distance:0.4 what_relation:0.3 multi_hop_relation:0.3 --output ./output

# Fixed 4 options with specific config
python run_point_qa_gen.py --num-tasks 100 --types what_distance --num-options 4 --config distance_type=closest --output ./output
```

## Available Generators (17 Types)

### 1. Distance-Based (4 types)

| Generator | Question Pattern |
|-----------|-----------------|
| `what_distance` | "What is the object closest/farthest from the {ref}?" |
| `where_distance` | "Where is the object closest/farthest from the {ref}?" |
| `list_attribute_distance` | "List all {attr}s of the object closest/farthest from {ref}." |
| `count_attribute_distance` | "How many {attr}s does the object closest/farthest from {ref} have?" |

Config: `{"distance_type": "closest"}` or `{"distance_type": "farthest"}`

### 2. Attribute-Based (3 types)

| Generator | Question Pattern |
|-----------|-----------------|
| `what_attribute` | "What is the {attr} of the {component} in the {object}?" |
| `list_attribute` | "List all {attr}s in the components of the {object}." |
| `count_attribute` | "How many {attr}s are in the components of the {object}?" |

Config: none required (attribute sampled randomly from material/color/shape/texture)

### 3. Number/Frequency-Based (4 types)

| Generator | Question Pattern |
|-----------|-----------------|
| `count_object` | "How many {object} are in the scene?" |
| `frequent_object` | "What is the most/least frequent object in the scene?" |
| `list_attribute_frequent` | "List all {attr}s of the most/least frequent object." |
| `count_attribute_frequent` | "How many {attr}s does the most/least frequent object have?" |

Config: `{"frequency_type": "most"}` or `{"frequency_type": "least"}`

### 4. Size-Based (4 types)

| Generator | Question Pattern |
|-----------|-----------------|
| `what_size` | "What is the largest/smallest object in the scene?" |
| `list_attribute_size` | "List all {attr}s of the largest/smallest object." |
| `count_attribute_size` | "How many {attr}s does the largest/smallest object have?" |
| `where_size` | "Where is the largest/smallest object relative to {ref}?" |

Config: `{"size_type": "largest"}` or `{"size_type": "smallest"}`
For `where_size`: also `{"reference_mode": "with_reference"}` or `{"reference_mode": "reference_to_target"}`

### 5. Relation-Based (2 types)

| Generator | Question Pattern |
|-----------|-----------------|
| `what_relation` | "What is the object that is {relation} the {ref}?" |
| `multi_hop_relation` | "What is the object {rel2} the object {rel1} the {anchor}?" |

Config: none required (relation sampled from layout)

## Configuration

### TaskPlan Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `generator_type` | `str`, `List[str]`, or `Dict[str, float]` | Generator type(s). Dict values are weights (must sum to 1.0). |
| `num_options` | `int` or `(int, int)` | Fixed option count (2-6) or random range. |
| `seed` | `int` | Random seed for reproducibility. |
| `generator_config` | `Dict` | Generator-specific config. If empty, a random variant is selected. |

### Key Design Decisions

- **Distractors from scene**: For What-type questions, all distractors come from objects in the same scene (no out-of-scene shortcuts).
- **Minimal edit distance**: For List-type questions, distractors differ from the correct answer by exactly 1 element.
- **6 answerable directions**: Where-type questions only use directions that the coordinate system can compute (left/right/above/below/front/behind).
- **Answer position balance**: Correct answer position is precisely balanced across all option slots.
- **Global dedup**: No duplicate questions in output.
- **Background diversity**: Randomly selected from background pool or no background (uniform probability).

## Output Structure

```
output_directory/
├── pcd/                    # Generated point cloud scenes
│   ├── 000000.npy         # Scene point cloud (N, 6): xyz + rgb
│   ├── 000001.npy
│   ├── ...
│   └── metadata.json      # Scene metadata (layout, objects per scene)
├── tasks.jsonl            # Question-answer pairs
└── tasks_info.json        # Generation config and statistics
```

### `tasks.jsonl`
```json
{
  "question_id": 0,
  "point": "000000.npy",
  "category": "what_distance_closest",
  "question": "Which object is nearest to the chair?",
  "options": ["table", "lamp", "book", "sofa"],
  "answer": "table"
}
```

### `tasks_info.json`
```json
{
  "task_plan": {
    "generator_type": ["what_distance", "what_relation", ...],
    "num_options": [4, 6],
    "seed": 42,
    "generator_config": {}
  },
  "generation_stats": {
    "num_tasks_requested": 1000,
    "num_tasks_generated": 1000,
    "output_directory": "./output",
    "category_distribution": {"what_distance_closest": 60, ...}
  }
}
```

## Point Cloud Visualization

```python
from visualizer import PointCloudVisualizer, ColorScheme

viz = PointCloudVisualizer()
viz.add_point_cloud("./output/pcd/000000.npy", "Scene")
viz.visualize(ColorScheme.ORIGINAL)
```

## Performance

- **Generation speed**: ~5-8 tasks/s (parallel, 17 generator types)
- **Bottleneck**: Point cloud IO (~130ms per .npy file on cold read)
- **Parallelism**: ThreadPoolExecutor across generator types (up to 32 workers)
- **Deferred loading**: Point cloud files are only loaded after all QA logic passes

## Architecture

```
point_qa_generator/
├── base.py           # TaskPlan, Task, BasePointQAGenerator
├── generator.py      # PointQAGenerator (main interface)
├── metadata.py       # Object metadata loading
├── scene_builder.py  # Point cloud scene assembly
├── templates.py      # Question template library
├── utils.py          # Constants, spatial functions
├── distance.py       # Distance-based generators
├── attribute.py      # Attribute-based generators
├── number.py         # Count/frequency generators
├── size.py           # Size-based generators
└── relation.py       # Relation and multi-hop generators
```
