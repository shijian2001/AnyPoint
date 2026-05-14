import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES
from .templates import sample_template


class AttributeGenerator(BasePointQAGenerator):
    """Base class for attribute-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        pass

    def _get_valid_objects_for_attribute(self, attribute: str) -> List[Dict]:
        return [obj for obj in self.metadata.objects
                if self.metadata.has_components_with_attribute(obj, attribute)]

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        count = 0
        for attribute in ATTRIBUTES:
            count += len(self._get_valid_objects_for_attribute(attribute))
        return count


class WhatAttributeGenerator(AttributeGenerator):
    """Generator for 'What is the {attr} of the {component} in the {object}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating what-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("WhatAttributeGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)
                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)

                if not components_with_attr:
                    continue

                component = self.rng.choice(components_with_attr)

                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])

                remaining_objects = [o for o in self.metadata.objects
                                    if o['object_id'] != target_obj['object_id']]

                object_mapping = {layout["objects"][0]["name"]: target_obj}
                if num_objects > 1:
                    other_objs = self.rng.choice(remaining_objects, size=num_objects - 1, replace=False)
                    for i, obj in enumerate(other_objs, 1):
                        object_mapping[layout["objects"][i]["name"]] = obj

                combo_key = (target_obj["object_id"], component["name"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(self.rng, "what_attribute").format(
                    attr=attribute, comp=component['name'], obj=target_obj['object_name']
                )
                correct_answer = component[attribute]

                all_values = self.metadata.get_attribute_values(attribute)
                candidates = [v for v in all_values if v != correct_answer]

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


class ListAttributeGenerator(AttributeGenerator):
    """Generator for 'List all {attr}s in the components of the {object}.' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating list-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("ListAttributeGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)

                combo_key = (target_obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])

                remaining_objects = [o for o in self.metadata.objects
                                    if o['object_id'] != target_obj['object_id']]

                object_mapping = {layout["objects"][0]["name"]: target_obj}
                if num_objects > 1:
                    other_objs = self.rng.choice(remaining_objects, size=num_objects - 1, replace=False)
                    for i, obj in enumerate(other_objs, 1):
                        object_mapping[layout["objects"][i]["name"]] = obj

                question = sample_template(self.rng, "list_attribute").format(
                    attr=attribute, obj=target_obj['object_name']
                )
                correct_answer = ", ".join(sorted(attribute_values))

                all_values = self.metadata.get_attribute_values(attribute)
                candidates = self._compose_list_distractors(
                    attribute_values, all_values, task_plan.num_options - 1
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


class CountAttributeGenerator(AttributeGenerator):
    """Generator for 'How many {attr}s are in the components of the {object}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating count-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("CountAttributeGenerator", num_tasks, len(tasks), attempts)
                attempts += 1
                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)

                combo_key = (target_obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])

                remaining_objects = [o for o in self.metadata.objects
                                    if o['object_id'] != target_obj['object_id']]

                object_mapping = {layout["objects"][0]["name"]: target_obj}
                if num_objects > 1:
                    other_objs = self.rng.choice(remaining_objects, size=num_objects - 1, replace=False)
                    for i, obj in enumerate(other_objs, 1):
                        object_mapping[layout["objects"][i]["name"]] = obj

                question = sample_template(self.rng, "count_attribute").format(
                    attr=attribute, obj=target_obj['object_name']
                )
                correct_count = len(attribute_values)
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
