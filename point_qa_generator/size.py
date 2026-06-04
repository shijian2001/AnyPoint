import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES, ANSWERABLE_DIRECTIONS
from .templates import sample_template


class SizeGenerator(BasePointQAGenerator):
    """Base class for size-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        if 'size_type' not in config:
            raise ValueError("size_type must be specified in generator_config")
        if config['size_type'] not in ['largest', 'smallest']:
            raise ValueError("size_type must be 'largest' or 'smallest'")

    def _get_size_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['size_type']

    def _calculate_volume(self, size: List[float]) -> float:
        return size[0] * size[1] * size[2] * 8


class WhatSizeGenerator(SizeGenerator):
    """Generator for 'What is the largest/smallest object?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        min_objs = task_plan.num_options + 1

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating what-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure(self.__class__.__name__, num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=min_objs)

                volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    volume = self._calculate_volume(obj_spec["size"])
                    volumes.append((i, obj_spec["name"], volume))

                if size_type == "largest":
                    target_idx, target_placeholder, _ = max(volumes, key=lambda x: x[2])
                else:
                    target_idx, target_placeholder, _ = min(volumes, key=lambda x: x[2])

                target_obj = object_mapping[target_placeholder]

                combo_key = (target_obj["object_id"], size_type)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(
                    self.rng, "what_size", size_type=size_type
                )
                correct_answer = target_obj["object_name"]

                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] != target_placeholder
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
                        "background_id": self._last_background_id,
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


class ListAttributeSizeGenerator(SizeGenerator):
    """Generator for 'List all {attr}s of the largest/smallest object.' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure(self.__class__.__name__, num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)
                attribute = self.rng.choice(ATTRIBUTES)

                valid_volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    obj = object_mapping[obj_spec["name"]]
                    if self.metadata.has_components_with_attribute(obj, attribute):
                        volume = self._calculate_volume(obj_spec["size"])
                        valid_volumes.append((i, obj_spec["name"], obj, volume))

                if not valid_volumes:
                    continue

                if size_type == "largest":
                    _, _, target_obj, _ = max(valid_volumes, key=lambda x: x[3])
                else:
                    _, _, target_obj, _ = min(valid_volumes, key=lambda x: x[3])

                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue

                correct_answer = ", ".join(sorted(attr_values))
                combo_key = (target_obj["object_id"], size_type, attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(
                    self.rng, "list_attribute_size", size_type=size_type
                ).format(attr=attribute)

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
                        "background_id": self._last_background_id,
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


class CountAttributeSizeGenerator(SizeGenerator):
    """Generator for 'How many {attr}s of the largest/smallest object?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure(self.__class__.__name__, num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)
                attribute = self.rng.choice(ATTRIBUTES)

                valid_volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    obj = object_mapping[obj_spec["name"]]
                    if self.metadata.has_components_with_attribute(obj, attribute):
                        volume = self._calculate_volume(obj_spec["size"])
                        valid_volumes.append((i, obj_spec["name"], obj, volume))

                if not valid_volumes:
                    continue

                if size_type == "largest":
                    _, _, target_obj, _ = max(valid_volumes, key=lambda x: x[3])
                else:
                    _, _, target_obj, _ = min(valid_volumes, key=lambda x: x[3])

                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue

                correct_count = len(attr_values)
                combo_key = (target_obj["object_id"], size_type, attribute, correct_count)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(
                    self.rng, "count_attribute_size", size_type=size_type
                ).format(attr=attribute)
                correct_answer = str(correct_count)

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
                        "background_id": self._last_background_id,
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


class WhereSizeGenerator(SizeGenerator):
    """Generator for position-based size questions."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        super().validate_generator_config(config)
        if 'reference_mode' in config:
            if config['reference_mode'] not in ['with_reference', 'reference_to_target']:
                raise ValueError("reference_mode must be 'with_reference' or 'reference_to_target'")

    def _get_reference_mode(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config.get('reference_mode', 'with_reference')

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        reference_mode = self._get_reference_mode(task_plan)

        from .utils import calculate_relation_from_positions

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating where-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure(self.__class__.__name__, num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=3)

                volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    volume = self._calculate_volume(obj_spec["size"])
                    pos = np.array(obj_spec["position"])
                    volumes.append((i, obj_spec["name"], pos, volume))

                if size_type == "largest":
                    target_idx, target_placeholder, target_pos, _ = max(volumes, key=lambda x: x[3])
                else:
                    target_idx, target_placeholder, target_pos, _ = min(volumes, key=lambda x: x[3])

                target_obj = object_mapping[target_placeholder]

                ref_candidates = [(i, name, pos) for i, name, pos, _ in volumes if i != target_idx]
                if not ref_candidates:
                    continue
                ref_choice_idx = self.rng.randint(len(ref_candidates))
                ref_idx, ref_placeholder, ref_pos = ref_candidates[ref_choice_idx]
                ref_obj = object_mapping[ref_placeholder]

                combo_key = (target_obj["object_id"], ref_obj["object_id"], size_type)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                correct_answer = calculate_relation_from_positions(target_pos, ref_pos)

                question = sample_template(
                    self.rng, "where_size",
                    size_type=size_type, reference_mode=reference_mode
                ).format(ref=ref_obj['object_name'])

                if reference_mode == 'reference_to_target':
                    correct_answer = calculate_relation_from_positions(ref_pos, target_pos)

                candidates = [rel for rel in ANSWERABLE_DIRECTIONS if rel != correct_answer]

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
                        "background_id": self._last_background_id,
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
