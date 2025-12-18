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


class WhatAttributeGenerator(AttributeGenerator):
    """Generator for 'What is the {attribute} of the {component} in the {object}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-attribute tasks."""
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
                
                # Sample layout and create proper object mapping
                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])
                
                # Ensure target_obj is first, then sample others without replacement
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
                
                # Create scene from layout
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                question = f"What is the {attribute} of the {component['name']} in the {target_obj['object_name']}?"
                correct_answer = component[attribute]

                # Generate candidates from global pool
                all_values = self.metadata.get_attribute_values(attribute)
                candidates = [v for v in all_values if v != correct_answer]

                options = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
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
    """Generator for 'List all {attribute}s in the components of the {object}.' questions."""

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

                # Sample layout and create proper object mapping
                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])
                
                # Ensure target_obj is first, then sample others without replacement
                remaining_objects = [o for o in self.metadata.objects 
                                    if o['object_id'] != target_obj['object_id']]
                
                object_mapping = {layout["objects"][0]["name"]: target_obj}
                if num_objects > 1:
                    other_objs = self.rng.choice(remaining_objects, size=num_objects - 1, replace=False)
                    for i, obj in enumerate(other_objs, 1):
                        object_mapping[layout["objects"][i]["name"]] = obj
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                question = f"List all {attribute}s in the components of the {target_obj['object_name']}."
                correct_answer = ", ".join(sorted(attribute_values))

                # Generate candidates from global pool
                needed = task_plan.num_options - 1
                candidates = set()
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

                options = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
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
    """Generator for 'How many {attribute}s are in the components of the {object}?' questions."""

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

                # Sample layout and create proper object mapping
                layout = self.rng.choice(self.layouts)
                num_objects = len(layout["objects"])
                
                # Ensure target_obj is first, then sample others without replacement
                remaining_objects = [o for o in self.metadata.objects 
                                    if o['object_id'] != target_obj['object_id']]
                
                object_mapping = {layout["objects"][0]["name"]: target_obj}
                if num_objects > 1:
                    other_objs = self.rng.choice(remaining_objects, size=num_objects - 1, replace=False)
                    for i, obj in enumerate(other_objs, 1):
                        object_mapping[layout["objects"][i]["name"]] = obj
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)

                question = f"How many {attribute}s are in the components of the {target_obj['object_name']}?"
                correct_count = len(attribute_values)
                correct_answer = str(correct_count)

                # Generate numeric candidates around correct answer
                needed = task_plan.num_options - 1
                candidates = []
                used_numbers = {correct_count}
                offset = 1
                
                while len(candidates) < needed:
                    # Try positive offset
                    if correct_count + offset not in used_numbers:
                        candidates.append(str(correct_count + offset))
                        used_numbers.add(correct_count + offset)
                    
                    if len(candidates) >= needed:
                        break
                    
                    # Try negative offset (if >= 0)
                    if correct_count - offset >= 0 and correct_count - offset not in used_numbers:
                        candidates.append(str(correct_count - offset))
                        used_numbers.add(correct_count - offset)
                    
                    offset += 1

                options = self._compose_options(correct_answer, candidates, task_plan.num_options)

                task = Task(
                    point=f"{len(tasks):06d}.npy",
                    question=question,
                    options=options,
                    answer=correct_answer,
                    metadata={
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