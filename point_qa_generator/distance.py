import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES, ANSWERABLE_DIRECTIONS
from .templates import sample_template


class DistanceGenerator(BasePointQAGenerator):
    """Base class for distance-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        if 'distance_type' not in config:
            raise ValueError("distance_type must be specified in generator_config")
        if config['distance_type'] not in ['closest', 'farthest']:
            raise ValueError("distance_type must be 'closest' or 'farthest'")

    def _get_distance_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['distance_type']


class WhatDistanceGenerator(DistanceGenerator):
    """Generator for 'What is the object closest/farthest from {ref}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        min_objs = task_plan.num_options + 1

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating what-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("WhatDistanceGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=min_objs)

                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])

                distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    pos = np.array(obj_spec["position"])
                    dist = np.linalg.norm(pos - ref_pos)
                    distances.append((i, obj_spec["name"], dist))

                if not distances:
                    continue

                if distance_type == "closest":
                    target_idx, target_placeholder, _ = min(distances, key=lambda x: x[2])
                else:
                    target_idx, target_placeholder, _ = max(distances, key=lambda x: x[2])

                target_obj = object_mapping[target_placeholder]

                combo_key = (target_obj["object_id"], ref_obj["object_id"])
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(
                    self.rng, "what_distance", distance_type=distance_type
                ).format(ref=ref_obj['object_name'])
                correct_answer = target_obj["object_name"]

                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] != target_placeholder
                       and obj["name"] != ref_placeholder
                ]

                if len(scene_candidates) < task_plan.num_options - 1:
                    continue

                options = self._compose_options(
                    correct_answer, scene_candidates, task_plan.num_options
                )

                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
                        "generator_type": task_plan.generator_type,
                        "generator_config": task_plan.generator_config,
                        "layout_id": layout.get("id"),
                        "layout_description": layout.get("description"),
                        "objects": [
                            {
                                "name": actual_obj["object_name"],
                                "object_id": actual_obj["object_id"],
                                "placeholder": placeholder
                            }
                            for placeholder, actual_obj in object_mapping.items()
                        ]
                    }
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks


class WhereDistanceGenerator(DistanceGenerator):
    """Generator for 'Where is the object closest/farthest from {ref}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        from .utils import calculate_relation_from_positions

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating where-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("WhereDistanceGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)

                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])

                distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    pos = np.array(obj_spec["position"])
                    dist = np.linalg.norm(pos - ref_pos)
                    distances.append((i, obj_spec["name"], pos, dist))

                if not distances:
                    continue

                if distance_type == "closest":
                    target_idx, target_placeholder, target_pos, _ = min(distances, key=lambda x: x[3])
                else:
                    target_idx, target_placeholder, target_pos, _ = max(distances, key=lambda x: x[3])

                target_obj = object_mapping[target_placeholder]

                combo_key = (target_obj["object_id"], ref_obj["object_id"])
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                correct_answer = calculate_relation_from_positions(target_pos, ref_pos)

                question = sample_template(
                    self.rng, "where_distance", distance_type=distance_type
                ).format(ref=ref_obj['object_name'])

                candidates = [rel for rel in ANSWERABLE_DIRECTIONS if rel != correct_answer]

                options = self._compose_options(
                    correct_answer, candidates, task_plan.num_options
                )

                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
                        "generator_type": task_plan.generator_type,
                        "generator_config": task_plan.generator_config,
                        "layout_id": layout.get("id"),
                        "layout_description": layout.get("description"),
                        "objects": [
                            {
                                "name": actual_obj["object_name"],
                                "object_id": actual_obj["object_id"],
                                "placeholder": placeholder
                            }
                            for placeholder, actual_obj in object_mapping.items()
                        ]
                    }
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks


class ListAttributeDistanceGenerator(DistanceGenerator):
    """Generator for 'List all {attr}s of the object closest/farthest from {ref}.' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("ListAttributeDistanceGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)

                attribute = self.rng.choice(ATTRIBUTES)

                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])

                valid_distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    obj = object_mapping[obj_spec["name"]]
                    if self.metadata.has_components_with_attribute(obj, attribute):
                        pos = np.array(obj_spec["position"])
                        dist = np.linalg.norm(pos - ref_pos)
                        valid_distances.append((i, obj_spec["name"], obj, dist))

                if not valid_distances:
                    continue

                if distance_type == "closest":
                    _, _, target_obj, _ = min(valid_distances, key=lambda x: x[3])
                else:
                    _, _, target_obj, _ = max(valid_distances, key=lambda x: x[3])

                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue

                combo_key = (target_obj["object_id"], ref_obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                correct_answer = ", ".join(sorted(attr_values))

                question = sample_template(
                    self.rng, "list_attribute_distance", distance_type=distance_type
                ).format(attr=attribute, ref=ref_obj['object_name'])

                all_values = self.metadata.get_attribute_values(attribute)
                candidates = self._compose_list_distractors(
                    attr_values, all_values, task_plan.num_options - 1
                )

                if len(candidates) < task_plan.num_options - 1:
                    continue

                options = self._compose_options(
                    correct_answer, candidates, task_plan.num_options
                )

                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
                        "generator_type": task_plan.generator_type,
                        "generator_config": task_plan.generator_config,
                        "layout_id": layout.get("id"),
                        "layout_description": layout.get("description"),
                        "objects": [
                            {
                                "name": actual_obj["object_name"],
                                "object_id": actual_obj["object_id"],
                                "placeholder": placeholder
                            }
                            for placeholder, actual_obj in object_mapping.items()
                        ]
                    }
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks


class CountAttributeDistanceGenerator(DistanceGenerator):
    """Generator for 'How many {attr}s of the object closest/farthest from {ref}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("CountAttributeDistanceGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                attribute = self.rng.choice(ATTRIBUTES)

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)

                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])

                valid_distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    obj = object_mapping[obj_spec["name"]]
                    if self.metadata.has_components_with_attribute(obj, attribute):
                        pos = np.array(obj_spec["position"])
                        dist = np.linalg.norm(pos - ref_pos)
                        valid_distances.append((i, obj_spec["name"], obj, dist))

                if not valid_distances:
                    continue

                if distance_type == "closest":
                    _, _, target_obj, _ = min(valid_distances, key=lambda x: x[3])
                else:
                    _, _, target_obj, _ = max(valid_distances, key=lambda x: x[3])

                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue

                correct_count = len(attr_values)
                combo_key = (target_obj["object_id"], ref_obj["object_id"], attribute, correct_count)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                correct_answer = str(correct_count)

                question = sample_template(
                    self.rng, "count_attribute_distance", distance_type=distance_type
                ).format(attr=attribute, ref=ref_obj['object_name'])

                scene_counts = []
                for obj_spec in layout["objects"]:
                    obj = object_mapping[obj_spec["name"]]
                    comps = self.metadata.get_object_components_with_attribute(obj, attribute)
                    if comps:
                        scene_counts.append(len(set(c[attribute] for c in comps)))

                candidates = self._compose_count_distractors(
                    correct_count, scene_counts, task_plan.num_options - 1
                )

                if len(candidates) < task_plan.num_options - 1:
                    continue

                options = self._compose_options(correct_answer, candidates, task_plan.num_options)

                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
                        "generator_type": task_plan.generator_type,
                        "generator_config": task_plan.generator_config,
                        "layout_id": layout.get("id"),
                        "layout_description": layout.get("description"),
                        "objects": [
                            {
                                "name": actual_obj["object_name"],
                                "object_id": actual_obj["object_id"],
                                "placeholder": placeholder
                            }
                            for placeholder, actual_obj in object_mapping.items()
                        ]
                    }
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks
