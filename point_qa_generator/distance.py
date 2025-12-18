import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES


class DistanceGenerator(BasePointQAGenerator):
    """Base class for distance-related generators.
    
    Uses Euclidean distance calculation from layout positions.
    All generators use layout-based scene generation.
    """

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate distance generator configuration."""
        if 'distance_type' not in config:
            raise ValueError("distance_type must be specified in generator_config")
        if config['distance_type'] not in ['closest', 'farthest']:
            raise ValueError("distance_type must be 'closest' or 'farthest'")

    def _get_distance_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['distance_type']


class WhatDistanceGenerator(DistanceGenerator):
    """Generator for 'What is the object that is closest/farthest from the {reference_object}?' questions."""

    def __init__(self, metadata, seed: int = 42, layouts = None):
        super().__init__(metadata, seed, layouts)
        
        # Use 'special' layouts (3-9 objects) for sufficient scene distractors
        if self.layouts_by_type:
            special_layouts = self.layouts_by_type.get('special', [])
            if special_layouts:
                self.layouts = special_layouts
            else:
                raise ValueError("No layouts with 3-9 objects available for WhatDistanceGenerator")

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-distance tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating what-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample from 'special' layouts (3-9 objects)
                layout, object_mapping = self._sample_layout_and_map_objects()
                
                # Pick reference object randomly
                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])
                
                # Calculate distances to all other objects
                distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    pos = np.array(obj_spec["position"])
                    dist = np.linalg.norm(pos - ref_pos)
                    distances.append((i, obj_spec["name"], dist))
                
                if not distances:
                    continue
                
                # Find closest or farthest
                if distance_type == "closest":
                    target_idx, target_placeholder, _ = min(distances, key=lambda x: x[2])
                else:  # farthest
                    target_idx, target_placeholder, _ = max(distances, key=lambda x: x[2])
                
                target_obj = object_mapping[target_placeholder]
                
                # Check if seen
                combo_key = (target_obj["object_id"], ref_obj["object_id"])
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene point cloud
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate question and answer
                question = f"What is the object that is {distance_type} from the {ref_obj['object_name']}?"
                correct_answer = target_obj["object_name"]
                
                # Generate candidates: prioritize scene objects, supplement from global
                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] != target_placeholder and 
                       obj["name"] != ref_placeholder
                ]
                
                num_needed = task_plan.num_options - 1
                if len(scene_candidates) < num_needed:
                    # Supplement from global pool
                    used_names = set([correct_answer] + scene_candidates)
                    available = [obj["object_name"] for obj in self.metadata.objects 
                                if obj["object_name"] not in used_names]
                    num_to_add = num_needed - len(scene_candidates)
                    if available and num_to_add > 0:
                        additional = self.rng.choice(available, size=min(num_to_add, len(available)), replace=False)
                        scene_candidates.extend(additional)
                
                # Ensure we have enough candidates
                if len(scene_candidates) < num_needed:
                    continue
                
                options = self._compose_options(
                    correct_answer, scene_candidates, task_plan.num_options
                )
                
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


class WhereDistanceGenerator(DistanceGenerator):
    """Generator for 'Where is the object that is closest/farthest from the {reference_object}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate where-distance tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)
        
        # Import here to avoid circular dependency
        from .utils import calculate_relation_from_positions
        from layout_generator.constants import VALID_RELATIONS

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating where-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample layout with at least 2 objects
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                
                # Pick reference object
                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])
                
                # Calculate distances to find target
                distances = []
                for i, obj_spec in enumerate(layout["objects"]):
                    if i == ref_idx:
                        continue
                    pos = np.array(obj_spec["position"])
                    dist = np.linalg.norm(pos - ref_pos)
                    distances.append((i, obj_spec["name"], pos, dist))
                
                if not distances:
                    continue
                
                # Find closest or farthest
                if distance_type == "closest":
                    target_idx, target_placeholder, target_pos, _ = min(distances, key=lambda x: x[3])
                else:  # farthest
                    target_idx, target_placeholder, target_pos, _ = max(distances, key=lambda x: x[3])
                
                target_obj = object_mapping[target_placeholder]
                
                # Check if seen
                combo_key = (target_obj["object_id"], ref_obj["object_id"])
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene point cloud
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Calculate spatial relation
                correct_answer = calculate_relation_from_positions(target_pos, ref_pos)
                
                # Generate question
                question = f"Where is the object that is {distance_type} from the {ref_obj['object_name']}?"
                
                # Candidates from VALID_RELATIONS
                candidates = [rel for rel in VALID_RELATIONS if rel != correct_answer]
                
                options = self._compose_options(
                    correct_answer, candidates, task_plan.num_options
                )
                
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


class ListAttributeDistanceGenerator(DistanceGenerator):
    """Generator for 'List all {attribute}s in the components of the object closest/farthest from {reference_object}.' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate list-attribute-distance tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample layout
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                
                # Sample attribute
                attribute = self.rng.choice(ATTRIBUTES)
                
                # Pick reference object
                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])
                
                # Find closest/farthest object with the attribute
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
                
                # Select target based on distance type
                if distance_type == "closest":
                    target_idx, target_placeholder, target_obj, _ = min(valid_distances, key=lambda x: x[3])
                else:
                    target_idx, target_placeholder, target_obj, _ = max(valid_distances, key=lambda x: x[3])
                
                # Get attribute values
                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue
                
                combo_key = (target_obj["object_id"], ref_obj["object_id"], attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate answer
                correct_answer = ", ".join(sorted(attr_values))
                question = f"List all {attribute}s in the components of the object {distance_type} from the {ref_obj['object_name']}."
                
                # Generate candidates from global pool
                all_values = self.metadata.get_attribute_values(attribute)
                correct_set = set(correct_answer.split(", "))
                candidates = set()
                
                for _ in range(task_plan.num_options * 3):
                    if len(candidates) >= task_plan.num_options - 1:
                        break
                    sample_values = self.rng.choice(all_values, size=len(correct_set), replace=False)
                    if set(sample_values) != correct_set:
                        candidates.add(", ".join(sorted(sample_values)))
                
                if len(candidates) < task_plan.num_options - 1:
                    continue
                
                options = self._compose_options(
                    correct_answer, list(candidates), task_plan.num_options
                )
                
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


class CountAttributeDistanceGenerator(DistanceGenerator):
    """Generator for 'How many {attribute}s in the components of the object closest/farthest from {reference_object}?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate count-attribute-distance tasks."""
        self.validate_generator_config(task_plan.generator_config)
        distance_type = self._get_distance_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{distance_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                attribute = self.rng.choice(ATTRIBUTES)

                valid_targets = [obj for obj in self.metadata.objects
                                 if self.metadata.has_components_with_attribute(obj, attribute)]
                if not valid_targets:
                    continue

                # Sample layout
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                
                # Pick reference object
                ref_idx = self.rng.randint(len(layout["objects"]))
                ref_placeholder = layout["objects"][ref_idx]["name"]
                ref_obj = object_mapping[ref_placeholder]
                ref_pos = np.array(layout["objects"][ref_idx]["position"])
                
                # Find closest/farthest with attribute
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
                
                # Count attributes
                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue
                
                correct_count = len(attr_values)
                combo_key = (target_obj["object_id"], ref_obj["object_id"], attribute, correct_count)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate answer
                correct_answer = str(correct_count)
                question = f"How many {attribute}s are in the components of the object {distance_type} from the {ref_obj['object_name']}?"
                
                # Generate numeric candidates
                candidates = []
                used = {correct_count}
                offset = 1
                while len(candidates) < task_plan.num_options - 1:
                    if correct_count + offset not in used:
                        candidates.append(str(correct_count + offset))
                        used.add(correct_count + offset)
                    if len(candidates) >= task_plan.num_options - 1:
                        break
                    if correct_count - offset >= 0 and correct_count - offset not in used:
                        candidates.append(str(correct_count - offset))
                        used.add(correct_count - offset)
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