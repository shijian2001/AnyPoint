import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES


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

    def _generate_scene_distractors(self, task_plan: TaskPlan, target_obj: Dict) -> List[Dict]:
        """Generate scene distractor objects for attribute questions."""
        if task_plan.num_scene_distractors == 0:
            return []

        used_objects = {target_obj["object_id"]}
        available_objects = [obj for obj in self.metadata.objects
                             if obj["object_id"] not in used_objects]

        num_distractors = min(task_plan.num_scene_distractors, len(available_objects))
        return self.rng.choice(available_objects, size=num_distractors,
                               replace=False).tolist() if num_distractors > 0 else []

    def _create_scene(self, task_plan: TaskPlan, target_obj: Dict) -> Tuple[np.ndarray, List[Dict]]:
        """Create scene with target object and optional distractors."""
        distractor_objects = self._generate_scene_distractors(task_plan, target_obj)

        all_objects = [target_obj] + distractor_objects

        num_objects = len(all_objects)
        available_grids = list(range(9))
        selected_grids = self.rng.choice(available_grids, size=num_objects, replace=False).tolist()
        angles = [self.metadata.sample_angle() for _ in all_objects]

        point_cloud = self._create_point_cloud_scene(all_objects, selected_grids, angles)

        return point_cloud, distractor_objects


class WhatAttributeGenerator(AttributeGenerator):
    """Generator for 'What is the {attribute} of the {component} in the {object}?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ATTRIBUTES:
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
                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)
                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)

                if not components_with_attr:
                    continue

                component = self.rng.choice(components_with_attr)

                combo_key = (target_obj["object_id"], component["name"], attribute, task_plan.num_scene_distractors)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                point_cloud, distractor_objects = self._create_scene(task_plan, target_obj)

                question = f"What is the {attribute} of the {component['name']} in the {target_obj['object_name']}?"
                correct_answer = component[attribute]

                # Generate candidates - prioritize scene distractors
                candidates = set()

                # From scene distractors
                for distractor in distractor_objects:
                    distractor_components = self.metadata.get_object_components_with_attribute(distractor, attribute)
                    for comp in distractor_components:
                        if comp[attribute] != correct_answer:
                            candidates.add(comp[attribute])

                # If not enough, supplement from global pool
                needed = task_plan.num_options - 1
                if len(candidates) < needed:
                    all_values = self.metadata.get_attribute_values(attribute)
                    for v in all_values:
                        if v != correct_answer and v not in candidates:
                            candidates.add(v)
                            if len(candidates) >= needed:
                                break

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


class ListAttributeGenerator(AttributeGenerator):
    """Generator for 'List all {attribute}s in the components of the {object}.' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ATTRIBUTES:
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
                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)

                combo_key = (target_obj["object_id"], attribute, task_plan.num_scene_distractors)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                point_cloud, distractor_objects = self._create_scene(task_plan, target_obj)

                question = f"List all {attribute}s in the components of the {target_obj['object_name']}."
                correct_answer = ", ".join(sorted(attribute_values))

                candidates = set()

                # From scene distractors (use complete mistake strategy)
                for distractor in distractor_objects:
                    distractor_components = self.metadata.get_object_components_with_attribute(distractor, attribute)
                    distractor_values = set()
                    for comp in distractor_components:
                        distractor_values.add(comp[attribute])

                    if distractor_values:
                        distractor_answer = ", ".join(sorted(distractor_values))
                        if distractor_answer != correct_answer:
                            candidates.add(distractor_answer)

                # If not enough, supplement from global pool
                needed = task_plan.num_options - 1
                if len(candidates) < needed:
                    all_values = self.metadata.get_attribute_values(attribute)
                    correct_values_set = set(correct_answer.split(", "))
                    
                    for _ in range((needed - len(candidates)) * 3):
                        # Use same number of attributes as correct answer
                        num_items = len(correct_values_set)
                        if num_items > len(all_values):
                            continue
                            
                        sample_values = self.rng.choice(all_values, size=num_items, replace=False)
                        candidate_values_set = set(sample_values)
                        
                        # Ensure candidate is semantically different from correct answer
                        if candidate_values_set != correct_values_set:
                            candidate = ", ".join(sorted(sample_values))
                            candidates.add(candidate)

                        if len(candidates) >= needed:
                            break

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
    """Generator for 'How many {attribute}s are in the components of the {object}?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        count = 0
        for attribute in ATTRIBUTES:
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
                attribute = self.rng.choice(ATTRIBUTES)
                valid_objects = self._get_valid_objects_for_attribute(attribute)

                if not valid_objects:
                    continue

                target_obj = self.rng.choice(valid_objects)

                combo_key = (target_obj["object_id"], attribute, task_plan.num_scene_distractors)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])

                if not attribute_values:
                    continue

                point_cloud, distractor_objects = self._create_scene(task_plan, target_obj)

                question = f"How many {attribute}s are in the components of the {target_obj['object_name']}?"
                correct_count = len(attribute_values)
                correct_answer = str(correct_count)

                # Generate candidates - prioritize scene distractors' counts
                candidates = set()

                # From scene distractors
                for distractor in distractor_objects:
                    distractor_components = self.metadata.get_object_components_with_attribute(distractor, attribute)
                    distractor_values = set()
                    for comp in distractor_components:
                        distractor_values.add(comp[attribute])

                    distractor_count = len(distractor_values)
                    if distractor_count != correct_count:
                        candidates.add(str(distractor_count))

                # If not enough, generate numbers around correct answer
                needed = task_plan.num_options - 1
                if len(candidates) < needed:
                    used_numbers = {correct_count} | {int(c) for c in candidates}

                    offset = 1
                    while len(candidates) < needed:
                        # Try positive offset
                        if correct_count + offset not in used_numbers:
                            candidates.add(str(correct_count + offset))
                            used_numbers.add(correct_count + offset)

                        if len(candidates) >= needed:
                            break

                        # Try negative offset (if >= 0)
                        if correct_count - offset >= 0 and correct_count - offset not in used_numbers:
                            candidates.add(str(correct_count - offset))
                            used_numbers.add(correct_count - offset)

                        if len(candidates) >= needed:
                            break

                        offset += 1

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