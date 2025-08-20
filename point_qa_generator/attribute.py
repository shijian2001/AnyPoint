import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task


class AttributeGenerator(BasePointQAGenerator):
    """Base class for attribute-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate attribute generator configuration."""
        # No specific config needed for attribute generators
        pass

    def _get_valid_objects_for_attribute(self, attribute: str) -> List[Dict]:
        """Get objects that have components with the specified attribute."""
        return [obj for obj in self.metadata.objects
                if self.metadata.has_components_with_attribute(obj, attribute)]


class WhatAttributeGenerator(AttributeGenerator):
    """Generator for 'What is the {attribute} of the {component} in the {object}?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ["material", "color", "shape", "texture"]:
            valid_objects = self._get_valid_objects_for_attribute(attribute)
            for obj in valid_objects:
                components_with_attr = self.metadata.get_object_components_with_attribute(obj, attribute)
                count += len(components_with_attr)
        return count

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-attribute tasks."""
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc="Generating what-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample attribute and object
                attribute = self.rng.choice(["material", "color", "shape", "texture"])
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                obj = self.rng.choice(valid_objects)
                components_with_attr = self.metadata.get_object_components_with_attribute(obj, attribute)

                if not components_with_attr:
                    continue

                component = self.rng.choice(components_with_attr)

                combo_key = (obj["object_id"], component["name"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                # Generate scene with single object
                grid = self.rng.randint(0, 9)
                angle = self.metadata.sample_angle()
                point_cloud = self._create_point_cloud_scene([obj], [grid], [angle])

                # Generate QA
                question = f"What is the {attribute} of the {component['name']} in the {obj['object_name']}?"
                correct_answer = component[attribute]

                # Generate candidate answers
                all_values = self.metadata.get_attribute_values(attribute)
                candidates = [v for v in all_values if v != correct_answer]

                options, answer_id = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    answer_id=answer_id
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks


class ListAttributeGenerator(AttributeGenerator):
    """Generator for 'List all {attribute}s in the components of the {object}.' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ["material", "color", "shape", "texture"]:
            valid_objects = self._get_valid_objects_for_attribute(attribute)
            count += len(valid_objects)
        return count

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate list-attribute tasks."""
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc="Generating list-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample attribute and object
                attribute = self.rng.choice(["material", "color", "shape", "texture"])
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                obj = self.rng.choice(valid_objects)

                combo_key = (obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                # Get all attribute values for this object
                components_with_attr = self.metadata.get_object_components_with_attribute(obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                # Generate scene with single object
                grid = self.rng.randint(0, 9)
                angle = self.metadata.sample_angle()
                point_cloud = self._create_point_cloud_scene([obj], [grid], [angle])

                # Generate QA
                question = f"List all {attribute}s in the components of the {obj['object_name']}."
                correct_answer = ", ".join(sorted(attribute_values))

                # Generate candidate answers
                all_values = self.metadata.get_attribute_values(attribute)
                candidates = set()

                # Generate some incorrect combinations
                for _ in range(min(20, task_plan.num_options * 3)):
                    num_items = self.rng.randint(1, min(4, len(all_values)))
                    sample_values = self.rng.choice(all_values, size=num_items, replace=False)
                    candidate = ", ".join(sorted(sample_values))
                    if candidate != correct_answer:
                        candidates.add(candidate)

                candidates = list(candidates)

                options, answer_id = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    answer_id=answer_id
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks


class CountAttributeGenerator(AttributeGenerator):
    """Generator for 'How many {attribute}s in the components of the {object}?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ["material", "color", "shape", "texture"]:
            valid_objects = self._get_valid_objects_for_attribute(attribute)
            count += len(valid_objects)
        return count

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate count-attribute tasks."""
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc="Generating count-attribute tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample attribute and object
                attribute = self.rng.choice(["material", "color", "shape", "texture"])
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                obj = self.rng.choice(valid_objects)

                combo_key = (obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                # Count unique attribute values for this object
                components_with_attr = self.metadata.get_object_components_with_attribute(obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                # Generate scene with single object
                grid = self.rng.randint(0, 9)
                angle = self.metadata.sample_angle()
                point_cloud = self._create_point_cloud_scene([obj], [grid], [angle])

                # Generate QA
                question = f"How many {attribute}s are in the components of the {obj['object_name']}?"
                correct_count = len(attribute_values)
                correct_answer = str(correct_count)

                # Generate candidate answers
                max_possible = len(self.metadata.get_attribute_values(attribute))
                candidates = []
                for i in range(1, min(max_possible + 1, 10)):
                    if i != correct_count:
                        candidates.append(str(i))

                options, answer_id = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    answer_id=answer_id
                )

                tasks.append((task, point_cloud))
                pbar.update(1)

        return tasks