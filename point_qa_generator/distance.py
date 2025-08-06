import numpy as np
from typing import List, Dict, Any, Tuple
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import (get_relative_distance_level, get_max_distance_level,
                    get_farther_grids, get_closer_grids, GRID_POSITIONS)


class DistanceGenerator(BasePointQAGenerator):
    """Base class for distance-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate distance generator configuration."""
        if 'distance_type' not in config:
            raise ValueError("distance_type must be specified in generator_config")
        if config['distance_type'] not in ['closest', 'farthest']:
            raise ValueError("distance_type must be 'closest' or 'farthest'")

    def _get_distance_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['distance_type']

    def _is_valid_distance_combination(self, ref_grid: int, target_grid: int,
                                       distance_type: str) -> bool:
        """Check if grid combination is valid for distance type."""
        if target_grid == ref_grid:
            return False

        distance_level = get_relative_distance_level(ref_grid, target_grid)
        max_level = get_max_distance_level(ref_grid)

        if distance_type == "closest" and distance_level >= max_level:
            return False
        if distance_type == "farthest" and distance_level <= 0:
            return False

        return True

    def _get_scene_distractor_grids(self, task_plan: TaskPlan, ref_grid: int,
                                    target_grid: int) -> List[int]:
        """Get grid positions for scene distractor objects."""
        distance_type = self._get_distance_type(task_plan)
        used_grids = {ref_grid, target_grid}

        if distance_type == "farthest":
            preferred_grids = get_closer_grids(ref_grid, target_grid)
        else:
            preferred_grids = get_farther_grids(ref_grid, target_grid)

        available_grids = [g for g in preferred_grids if g not in used_grids]
        if not available_grids:
            available_grids = [g for g in range(9) if g not in used_grids]

        num_needed = min(task_plan.num_scene_distractors, len(available_grids))
        return self.rng.choice(available_grids, size=num_needed, replace=False).tolist() if num_needed > 0 else []

    def _generate_scene_distractors(self, task_plan: TaskPlan, target_obj: Dict, ref_obj: Dict) -> List[Dict]:
        """Generate scene distractor objects."""
        used_objects = {target_obj["object_id"], ref_obj["object_id"]}
        available_objects = [obj for obj in self.metadata.objects
                             if obj["object_id"] not in used_objects]

        num_distractors = min(task_plan.num_scene_distractors, len(available_objects))
        return self.rng.choice(available_objects, size=num_distractors,
                               replace=False).tolist() if num_distractors > 0 else []


class WhatDistanceGenerator(DistanceGenerator):
    """Generator for 'What object is closest/farthest?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        valid_combinations = 0
        for target_grid in range(9):
            for ref_grid in range(9):
                if self._is_valid_distance_combination(ref_grid, target_grid, distance_type):
                    valid_combinations += 1

        return valid_combinations * len(self.metadata.objects) * (len(self.metadata.objects) - 1)

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-distance tasks."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        while len(tasks) < num_tasks:
            target_obj, ref_obj = self.rng.choice(self.metadata.objects, size=2, replace=False)
            target_grid, ref_grid = self.rng.randint(0, 9, size=2)

            if not self._is_valid_distance_combination(ref_grid, target_grid, distance_type):
                continue

            combo_key = (target_obj["object_id"], ref_obj["object_id"], target_grid, ref_grid)
            if combo_key in seen_combinations:
                continue
            seen_combinations.add(combo_key)

            # Generate scene
            scene_distractor_objects = self._generate_scene_distractors(task_plan, target_obj, ref_obj)
            scene_distractor_grids = self._get_scene_distractor_grids(task_plan, ref_grid, target_grid)

            actual_num_distractors = min(len(scene_distractor_objects), len(scene_distractor_grids))
            scene_distractor_objects = scene_distractor_objects[:actual_num_distractors]
            scene_distractor_grids = scene_distractor_grids[:actual_num_distractors]

            all_objects = [target_obj, ref_obj] + scene_distractor_objects
            all_grids = [target_grid, ref_grid] + scene_distractor_grids
            angles = [self.metadata.sample_angle() for _ in all_objects]

            point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles)

            # Generate QA
            question = f"What is the object that is {distance_type} from the {ref_obj['object_name']}?"
            correct_answer = target_obj["object_name"]

            scene_object_names = [obj["object_name"] for obj in all_objects if obj != target_obj]
            random_object_names = [obj["object_name"] for obj in self.rng.choice(
                [obj for obj in self.metadata.objects if obj not in all_objects],
                size=min(10, len(self.metadata.objects) - len(all_objects)),
                replace=False
            )] if len(self.metadata.objects) > len(all_objects) else []

            candidates = list(dict.fromkeys(scene_object_names + random_object_names))  # Remove duplicates
            options, answer_id = self._compose_options(correct_answer, candidates, task_plan.num_options)

            task = Task(
                point=f"{len(tasks):06d}.npy",
                question=question,
                options=options,
                answer=correct_answer,
                answer_id=answer_id
            )

            tasks.append((task, point_cloud))

        return tasks


class WhereDistanceGenerator(DistanceGenerator):
    """Generator for 'Where is the closest/farthest object?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        valid_combinations = 0
        for target_grid in range(9):
            for ref_grid in range(9):
                if self._is_valid_distance_combination(ref_grid, target_grid, distance_type):
                    valid_combinations += 1

        return valid_combinations * len(self.metadata.objects) ** 2

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate where-distance tasks."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        while len(tasks) < num_tasks:
            target_obj, ref_obj = self.rng.choice(self.metadata.objects, size=2, replace=False)
            target_grid, ref_grid = self.rng.randint(0, 9, size=2)

            if not self._is_valid_distance_combination(ref_grid, target_grid, distance_type):
                continue

            combo_key = (target_obj["object_id"], ref_obj["object_id"], target_grid, ref_grid)
            if combo_key in seen_combinations:
                continue
            seen_combinations.add(combo_key)

            # Generate scene (same as WhatDistanceGenerator)
            scene_distractor_objects = self._generate_scene_distractors(task_plan, target_obj, ref_obj)
            scene_distractor_grids = self._get_scene_distractor_grids(task_plan, ref_grid, target_grid)

            actual_num_distractors = min(len(scene_distractor_objects), len(scene_distractor_grids))
            scene_distractor_objects = scene_distractor_objects[:actual_num_distractors]
            scene_distractor_grids = scene_distractor_grids[:actual_num_distractors]

            all_objects = [target_obj, ref_obj] + scene_distractor_objects
            all_grids = [target_grid, ref_grid] + scene_distractor_grids
            angles = [self.metadata.sample_angle() for _ in all_objects]

            point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles)

            question = f"Where is the object that is {distance_type} from the {ref_obj['object_name']} in the scene?"
            correct_answer = GRID_POSITIONS[target_grid]
            candidates = [GRID_POSITIONS[g] for g in range(9) if g != target_grid]

            options, answer_id = self._compose_options(correct_answer, candidates, task_plan.num_options)

            task = Task(
                point=f"{len(tasks):06d}.npy",
                question=question,
                options=options,
                answer=correct_answer,
                answer_id=answer_id
            )

            tasks.append((task, point_cloud))

        return tasks