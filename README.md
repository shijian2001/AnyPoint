# AnyPoint: Programmatically Scaling Point Cloud Instruction Data Generation

A fast, scalable data engine for programmatically synthesizing 3D point cloud instruction datasets with guaranteed ground truth and diverse point cloud scenes.

## Installation

```bash
git clone https://github.com/shijian2001/AnyPoint.git
cd AnyPoint
conda create -n anypoint python==3.11
conda activate anypoint
pip install -e .   # editable install; makes the packages importable from anywhere
```

## Quick Start

### Generate Mixed Dataset (Recommended)

```python
from point_qa_generator import PointQAGenerator, TaskPlan

generator = PointQAGenerator(
    metadata_file="<data>/metadata.jsonl",
    pcd_dir="<data>/point_clouds",
    layouts_file="<data>/layouts.jsonl",
    background_dir="<data>/background",
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

### CLI Scripts

All scripts live under `scripts/`, grouped by stage:
`data_prep/` · `generate/` · `eval/` · `vis/`.

```bash
# Small / single-process generation (debugging, small batches)
python scripts/generate/run_point_qa_gen.py --num-tasks 1000 --output ./output
python scripts/generate/run_point_qa_gen.py --num-tasks 100 --types what_distance what_relation --output ./output
python scripts/generate/run_point_qa_gen.py --num-tasks 500 --types what_distance:0.4 what_relation:0.3 multi_hop_relation:0.3 --output ./output

# Large-scale parallel generation (sharded, resumable, balanced across 17 generators)
python scripts/generate/run_point_qa_gen_parallel.py \
    --num-tasks 2000000 --output <dataset_dir> \
    --workers 120 --shard-size 40 --cache-mb 4096

# Sample a subset (by per-generator quota / weights / hierarchical groups)
python scripts/generate/sample_qa_subset.py --dataset <dataset_dir> --total 10000 --output train_10k.jsonl
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
- **Dedup**: deduplicated on (question, options, answer). The same question text
  with a different scene/answer is kept — it is a valid sample since the model
  must read the point cloud to answer.
- **Background diversity**: Randomly selected from background pool or no background (uniform probability).

## Output Structure

**Single-process** (`run_point_qa_gen.py`) — one flat output directory:

```
output_directory/
├── pcd/
│   ├── 00000000.npy        # Scene point cloud (N, 6): xyz + rgb
│   └── ...
├── metadata.jsonl          # Per-scene reproduction metadata
├── tasks.jsonl             # Question-answer pairs
└── tasks_info.json         # Generation config and statistics
```

**Parallel** (`run_point_qa_gen_parallel.py`) — organized by generator, with a
top-level `index.jsonl` that is the single entry point for training/eval:

```
output_directory/
├── index.jsonl             # ALL tasks, global question_id, point = relative path
└── <generator>/            # 17 dirs, e.g. what_distance/
    └── shard_00000000/
        ├── pcd/*.npy
        ├── tasks.jsonl
        └── metadata.jsonl  # generator_type, layout_id, background_id, objects ...
```

Training/eval only need `index.jsonl`; load a point cloud with
`os.path.join(dataset_root, record["point"])`. The `shard_*` layout is an
internal sharding detail and never needs to be referenced directly.

### `index.jsonl` / `tasks.jsonl` record
```json
{
  "question_id": 0,
  "scene_id": 0,
  "point": "what_distance/shard_00000000/pcd/00000000.npy",
  "category": "what_distance_closest",
  "question": "Which object is nearest to the chair?",
  "options": ["table", "lamp", "book", "sofa"],
  "answer": "table",
  "generator": "what_distance"
}
```

### `metadata.jsonl` (reproduction)
Each scene stores everything needed to rebuild its point cloud bit-for-bit:
`generator_type`, `generator_config`, `layout_id`, `background_id`, and the
placeholder→object_id mapping.

## Point Cloud Visualization

On a machine with a display (Open3D interactive window):

```python
from visualizer import PointCloudVisualizer, ColorScheme

viz = PointCloudVisualizer()
viz.add_point_cloud("./output/pcd/000000.npy", "Scene")
viz.visualize(ColorScheme.ORIGINAL)
```

On a headless server (no X11/EGL) — render to an interactive HTML or a static PNG:

```bash
python scripts/vis/vis_scene_headless.py scene.npy              # -> interactive .html
python scripts/vis/vis_scene_headless.py scene.npy --backend mpl  # -> static .png
```

## Performance

Measured on a 2M-task run (`run_point_qa_gen_parallel.py`, 120 workers):

- **Throughput**: ~9k+ tasks/min; 2M completed in a few hours.
- **Parallelism**: one process per shard; generators are interleaved (round-robin)
  so all 17 run concurrently and the slow size/frequent types don't tail.
- **Thread pinning**: BLAS threads pinned to 1 per process (`OMP_NUM_THREADS=1`)
  to avoid oversubscription across many workers.
- **Bottleneck**: reading object `.npy` from the network FS (~130ms cold). Mitigated
  by a per-worker byte-budgeted LRU cache (`--cache-mb`) since the same ~20k
  objects recur across all scenes.
- **Downsampling**: `numpy.random.default_rng` for the per-scene point sampling
  (~100x faster than legacy `RandomState.choice` at large N).
- **Resumable**: each shard writes a `_DONE` marker; rerun the same command to
  skip completed shards.

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

Runnable scripts are grouped by stage under `scripts/`:

```
scripts/
├── data_prep/   # convert_glb_to_npy.py, run_layout_gen.py
├── generate/    # run_point_qa_gen_parallel.py, run_point_qa_gen.py, sample_qa_subset.py
├── eval/        # compare_eval_strategies.py, run_dynamic_eval.py, run_eval.sh
└── vis/         # vis_layout.py, vis_scene_headless.py
```

Other packages: `layout_generator/` (LLM-driven layout synthesis),
`dynamic_evaluation/` (evaluation harness), `visualizer/` (Open3D viewers),
`models/` (baseline model wrappers).
