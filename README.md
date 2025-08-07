## Install
```
git clone https://github.com/shijian2001/AnyPoint.git
cd AnyPoint
conda create -n anypoint python==3.11
pip insall -r requirements.txt
```
## Visualizer
You can easily visualize a point cloud via:
```python
import numpy as np
from visualizer import PointCloudVisualizer, ColorScheme

viz = PointCloudVisualizer()
viz.add_point_cloud("./data/000000.npy", "My Cloud")
viz.visualize(ColorScheme.ORIGINAL) ## Keep original color
```
## Point QA Generator
You can programmatically generate point-based qa like:
```python
from point_qa_generator import PointQAGenerator, TaskPlan

json_file = "/path/to/pointllm_metadata.json"
pcd_dir = "/path/to/point_clouds"
generator = PointQAGenerator(json_file, pcd_dir, seed=42)

task_plan_what = TaskPlan(
    generator_type="what_distance", ## Now supporting what_distance and where_distance
    num_options=4,
    num_scene_distractors=2, ## distractors in the scene
    seed=42,
    generator_config={"distance_type": "farthest"}
)

generator.generate(task_plan_what, 100, "./output/what_distance_farthest")
```
Then you can find generated QAs and meta-info in `output/what_distance_farthest` dir:
```
output/what_distance_farthest/
--- pcd
  --- 000000.npy
  --- 000001.npy
  --- ....
--- tasks.jsonl
--- task_info.json
```