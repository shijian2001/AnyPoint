# AnyPoint: Programmatically Scaling Point Cloud Instruction Data Generation

A fast, scalable data engine for programmatically synthesizing 3D point cloud instruction datasets with guaranteed ground truth and diverse point cloud scenes
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

# Initialize generator with layout system
generator = PointQAGenerator(
    metadata_file="/path/to/metadata.jsonl",  # Object metadata
    pcd_dir="/path/to/point_clouds",          # Point cloud directory
    layouts_file="/path/to/layouts.json",     # Layout definitions
    seed=42
)

# Create task plan
task_plan = TaskPlan(
    generator_type="what_distance",
    num_options=4,  # Number of multiple choice options
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
    generator_config={"distance_type": "closest"}  # or "farthest"
)
```

#### `where_distance`  
**Question Type**: "Where is the object that is closest/farthest from the [reference_object]?"  
**Answer Options**: Spatial relations from layout generator (e.g., "in front of", "beside", "above")

```python
task_plan = TaskPlan(
    generator_type="where_distance",
    num_options=4,
    generator_config={"distance_type": "farthest"}
)
```

#### `list_attribute_distance`
**Question Type**: "List all [attribute]s in the components of the object closest/farthest from [reference_object]."

```python
task_plan = TaskPlan(
    generator_type="list_attribute_distance",
    num_options=4,
    generator_config={"distance_type": "closest"}
)
```

#### `count_attribute_distance`
**Question Type**: "How many [attribute]s are in the components of the object closest/farthest from [reference_object]?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute_distance",
    num_options=4,
    generator_config={"distance_type": "farthest"}
)
```

### 2. Attribute-Based Generators

#### `what_attribute`
**Question Type**: "What is the [attribute] of the [component] in the [object]?"

```python
task_plan = TaskPlan(
    generator_type="what_attribute",
    num_options=4
)
```

#### `list_attribute`
**Question Type**: "List all [attribute]s in the components of the [object]."

```python
task_plan = TaskPlan(
    generator_type="list_attribute",
    num_options=4
)
```

#### `count_attribute`
**Question Type**: "How many [attribute]s are in the components of the [object]?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute",
    num_options=4
)
```

### 3. Number-Based Generators

#### `count_object`
**Question Type**: "How many [object] in the scene?"

```python
task_plan = TaskPlan(
    generator_type="count_object",
    num_options=4
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
    generator_config={"size_type": "largest"}  # or "smallest"
)
```

#### `list_attribute_size`
**Question Type**: "List all [attribute]s in the components of the largest/smallest object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="list_attribute_size",
    num_options=4,
    generator_config={"size_type": "smallest"}
)
```

#### `count_attribute_size`
**Question Type**: "How many [attribute]s are in the components of the largest/smallest object in the scene?"

```python
task_plan = TaskPlan(
    generator_type="count_attribute_size",
    num_options=4,
    generator_config={"size_type": "largest"}
)
```

#### `where_size`
**Question Type**: "Where is the largest/smallest object in the scene?"  
**Answer Options**: Spatial relations from layout generator (e.g., "in front of", "beside", "above")

```python
task_plan = TaskPlan(
    generator_type="where_size",
    num_options=4,
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

1. **Layout System**: The system uses pre-generated layouts that define object positions, sizes, and spatial relations. Each layout contains 2-9 objects with guaranteed size variations (one largest, one smallest).

2. **Metadata Requirements**: Objects must have component-level attributes (material, color, shape, texture) for attribute-based generators to work properly.

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

## Future Generators (Based on Layout Relations)

The layout system provides rich semantic relations between objects, enabling more advanced question types:

- **Relation-based What**: "What is the object **on** the table?"
- **Relation-based Where**: "Where is the chair **relative to** the table?"
- **Complex Reasoning**: "What is the object that is on the largest object?"
- **Multi-hop Questions**: "What is beside the object in front of the lamp?"

See `layout_generator/constants.py::VALID_RELATIONS` for available spatial relations.