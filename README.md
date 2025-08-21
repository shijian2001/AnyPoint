# AnyPoint: Programmatically Scaling Point Cloud Question-Answer Generation

A Python library for programmatically generating question-answer pairs from 3D point cloud scenes with rich object metadata.

## Installation

```bash
git clone https://github.com/shijian2001/AnyPoint.git
cd AnyPoint
conda create -n anypoint python==3.11
conda activate anypoint
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from point_qa_generator import PointQAGenerator, TaskPlan

# Initialize generator
generator = PointQAGenerator(
    jsonl_file="/path/to/metadata.jsonl",  # Object metadata
    pcd_dir="/path/to/point_clouds",       # Point cloud directory
    seed=42
)

# Create task plan
task_plan = TaskPlan(
    generator_type="what_distance",
    num_options=4,
    num_scene_distractors=2,
    seed=42,
    generator_config={"distance_type": "closest"}
)

# Generate tasks
generator.generate(task_plan, num_tasks=100, output_dir="./output")
```

### Point Cloud Visualization

```python
from visualizer import PointCloudVisualizer, ColorScheme

viz = PointCloudVisualizer()
viz.add_point_cloud("./data/scene.npy", "My Scene")
viz.visualize(ColorScheme.HEIGHT)  # Visualize by height
```

## Available Generators

### 1. Distance-Based Generators

#### `what_distance`
**Question Type**: "What is the object that is closest/farthest from the [reference_object]?"

```python
task_plan = TaskPlan(
    generator_type="what_distance",
    num_options=4,
    num_scene_distractors=3,
    generator_config={"distance_type": "closest"}  # or "farthest"
)
```

#### `where_distance`  
**Question Type**: "Where is the object that is closest/farthest from the [reference_object]?"

```python
task_plan = TaskPlan(
    generator_type="where_distance",
    num_options=4,
    num_scene_distractors=2,
    generator_config={"distance_type": "farthest"}
)
```

#### `list_attribute_distance`
**Question Type**: "List all [attribute]s in the components of the object closest/farthest from [reference_object]."

```python
task_plan = TaskPlan(
    generator_type="list_attribute_distance",
    num_options=4,
    num_scene_distractors=2,
    generator_config={"distance_type": "closest"}
)
```

#### `count_attribute_distance`
**Question Type**: "How many [attribute]s are in the components of the object closest/farthest from [reference_object]?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute_distance",
    num_options=4,
    num_scene_distractors=2,
    generator_config={"distance_type": "farthest"}
)
```

### 2. Attribute-Based Generators

#### `what_attribute`
**Question Type**: "What is the [attribute] of the [component] in the [object]?"

```python
task_plan = TaskPlan(
    generator_type="what_attribute",
    num_options=4,
    num_scene_distractors=2
)
```

#### `list_attribute`
**Question Type**: "List all [attribute]s in the components of the [object]."

```python
task_plan = TaskPlan(
    generator_type="list_attribute",
    num_options=4,
    num_scene_distractors=3
)
```

#### `count_attribute`
**Question Type**: "How many [attribute]s are in the components of the [object]?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute",
    num_options=4,
    num_scene_distractors=1
)
```

### 3. Number-Based Generators

> ⚠️ **Important**: Number generators ignore `num_scene_distractors` and use internal object count logic.

#### `count_object`
**Question Type**: "How many [object] in the scene?"

```python
task_plan = TaskPlan(
    generator_type="count_object",
    num_options=4,
    # num_scene_distractors is ignored
)
```

#### `frequent_object`
**Question Type**: "What is the most/least frequent object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="frequent_object",
    num_options=4,
    generator_config={"frequency_type": "most"}  # or "least"
)
```

#### `list_attribute_frequent`
**Question Type**: "List all [attribute]s in the components of the most/least frequent object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="list_attribute_frequent",
    num_options=4,
    generator_config={"frequency_type": "least"}
)
```

#### `count_attribute_frequent`
**Question Type**: "How many [attribute]s are in the components of the most/least frequent object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute_frequent",
    num_options=4,
    generator_config={"frequency_type": "most"}
)
```

### 4. Size-Based Generators

#### `what_size`
**Question Type**: "What is the largest/smallest object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="what_size",
    num_options=4,
    num_scene_distractors=2,
    generator_config={"size_type": "largest"}  # or "smallest"
)
```

#### `list_attribute_size`
**Question Type**: "List all [attribute]s in the components of the largest/smallest object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="list_attribute_size",
    num_options=4,
    num_scene_distractors=3,
    generator_config={"size_type": "smallest"}
)
```

#### `count_attribute_size`
**Question Type**: "How many [attribute]s are in the components of the largest/smallest object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute_size",
    num_options=4,
    num_scene_distractors=2,
    generator_config={"size_type": "largest"}
)
```

#### `where_size`
**Question Type**: "Where is the largest/smallest object in the scene?" (with configurable reference modes)

```python
task_plan = TaskPlan(
    generator_type="where_size",
    num_options=4,
    num_scene_distractors=2,
    generator_config={
        "size_type": "largest",
        "reference_mode": "with_reference"  # "no_reference", "reference_to_target"
    }
)
```

## Configuration Parameters

### TaskPlan Parameters

- **`generator_type`** (str): Type of generator to use
- **`num_options`** (int, 2-6): Number of multiple choice options
- **`num_scene_distractors`** (int, 0-7): Number of distractor objects in scene
- **`seed`** (int): Random seed for reproducibility
- **`generator_config`** (dict): Generator-specific configuration

### Generator-Specific Configurations

| Generator Type | Required Config | Options |
|---------------|----------------|---------|
| Distance-based | `distance_type` | `"closest"`, `"farthest"` |
| Number-based | `frequency_type` | `"most"`, `"least"` |
| Size-based | `size_type` | `"largest"`, `"smallest"` |
| Size-based (`where_size`) | `reference_mode` | `"with_reference"`, `"no_reference"`, `"reference_to_target"` |

## Important Notes

### 🔴 Critical Warnings

1. **Number Generators**: All number-based generators (`count_object`, `frequent_object`, `list_attribute_frequent`, `count_attribute_frequent`) **ignore** the `num_scene_distractors` parameter. They use internal logic to create scenes with varying object counts (3-9 objects) to ensure meaningful frequency-based questions.

2. **Grid System Limitation**: `num_scene_distractors` is limited to 0-7 due to the 3×3 grid positioning system.

3. **Metadata Requirements**: Objects must have component-level attributes (material, color, shape, texture) for attribute-based generators to work properly.

### 📁 Output Structure

```
output_directory/
├── pcd/                    # Generated point cloud files
│   ├── 000000.npy
│   ├── 000001.npy
│   └── ...
├── tasks.jsonl            # Question-answer pairs
└── tasks_info.json        # Generation metadata and statistics
```