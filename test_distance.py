from point_qa_generator import PointQAGenerator, TaskPlan

json_file = "/path/to/pointllm_metadata.json"
pcd_dir = "/path/to/point_clouds"
generator = PointQAGenerator(json_file, pcd_dir, seed=42)

task_plan_what = TaskPlan(
    generator_type="what_distance",
    num_options=4,
    num_scene_distractors=2,
    seed=42,
    generator_config={"distance_type": "farthest"}
)

generator.generate(task_plan_what, 100, "./output/what_distance_farthest")