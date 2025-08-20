import os
import json
from typing import Dict
import numpy as np
from .base import TaskPlan
from .metadata import PointCloudMetadata
from .distance import (WhatDistanceGenerator, WhereDistanceGenerator,
                       ListAttributeDistanceGenerator, CountAttributeDistanceGenerator)
from .attribute import (WhatAttributeGenerator, ListAttributeGenerator,
                        CountAttributeGenerator)


class PointQAGenerator:
    """Main interface for Point QA generation."""

    def __init__(self, jsonl_file: str, pcd_dir: str, seed: int = 42):
        """
        Initialize Point QA Generator.

        Args:
            jsonl_file: Path to object metadata JSONL file
            pcd_dir: Directory containing point cloud .npy files
            seed: Random seed
        """
        self.metadata = PointCloudMetadata(jsonl_file, pcd_dir, seed)
        self.generators = {
            # Distance-based generators
            "what_distance": WhatDistanceGenerator(self.metadata, seed),
            "where_distance": WhereDistanceGenerator(self.metadata, seed),
            "list_attribute_distance": ListAttributeDistanceGenerator(self.metadata, seed),
            "count_attribute_distance": CountAttributeDistanceGenerator(self.metadata, seed),

            # Attribute-based generators
            "what_attribute": WhatAttributeGenerator(self.metadata, seed),
            "list_attribute": ListAttributeGenerator(self.metadata, seed),
            "count_attribute": CountAttributeGenerator(self.metadata, seed)
        }

    def generate(self, task_plan: TaskPlan, num_tasks: int, output_dir: str) -> Dict[str, any]:
        """
        Generate QA tasks and save to output directory.

        Args:
            task_plan: Task configuration
            num_tasks: Number of tasks to generate
            output_dir: Output directory path

        Returns:
            Dictionary containing generation statistics
        """
        if task_plan.generator_type not in self.generators:
            raise ValueError(f"Unknown generator type: {task_plan.generator_type}")

        generator = self.generators[task_plan.generator_type]
        possible_tasks = generator.count_possible_tasks(task_plan)

        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        os.makedirs(output_dir, exist_ok=True)
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        # Generate tasks
        task_results = generator.generate_tasks(task_plan, num_tasks)

        task_records = []
        for i, (task, point_cloud) in enumerate(task_results):
            pcd_path = os.path.join(pcd_dir, task.point)
            np.save(pcd_path, point_cloud)

            task_record = {
                "question_id": i,
                "point": task.point,
                "category": f"{task_plan.generator_type}_{task_plan.generator_config.get('distance_type', '')}",
                "question": task.question,
                "options": task.options,
                "answer": task.answer,
                "answer_id": task.answer_id
            }
            task_records.append(task_record)

        tasks_file = os.path.join(output_dir, "tasks.jsonl")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for record in task_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        task_info = {
            "task_plan": {
                "generator_type": task_plan.generator_type,
                "num_options": task_plan.num_options,
                "num_scene_distractors": task_plan.num_scene_distractors,
                "seed": task_plan.seed,
                "generator_config": task_plan.generator_config
            },
            "generation_stats": {
                "num_tasks_requested": num_tasks,
                "num_tasks_generated": len(task_records),
                "possible_tasks": possible_tasks,
                "output_directory": output_dir
            }
        }

        task_info_file = os.path.join(output_dir, "tasks_info.json")
        with open(task_info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(task_records)} tasks:")
        print(f"  Tasks file: {tasks_file}")
        print(f"  Point clouds: {pcd_dir}")
        print(f"  Task info: {task_info_file}")

        return task_info