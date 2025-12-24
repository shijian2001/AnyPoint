import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import ATTRIBUTES


class SizeGenerator(BasePointQAGenerator):
    """Base class for size-related generators.
    
    Uses layout's 'size' field (half-extents) for volume-based comparisons.
    Layout system guarantees one largest and one smallest object per scene.
    """

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate size generator configuration."""
        if 'size_type' not in config:
            raise ValueError("size_type must be specified in generator_config")
        if config['size_type'] not in ['largest', 'smallest']:
            raise ValueError("size_type must be 'largest' or 'smallest'")

    def _get_size_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['size_type']
    
    def _calculate_volume(self, size: List[float]) -> float:
        """Calculate volume from AABB half-extents."""
        return size[0] * size[1] * size[2] * 8  # Full AABB volume


class WhatSizeGenerator(SizeGenerator):
    """Generator for 'What is the (largest/smallest) object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-size tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating what-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                # Sample layout
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                
                # Calculate volumes for all objects
                volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    size = obj_spec["size"]
                    volume = self._calculate_volume(size)
                    volumes.append((i, obj_spec["name"], volume))
                
                # Find largest or smallest
                if size_type == "largest":
                    target_idx, target_placeholder, _ = max(volumes, key=lambda x: x[2])
                else:  # smallest
                    target_idx, target_placeholder, _ = min(volumes, key=lambda x: x[2])
                
                target_obj = object_mapping[target_placeholder]
                
                combo_key = (target_obj["object_id"], size_type)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate question and answer
                question = f"What is the {size_type} object in the scene?"
                correct_answer = target_obj["object_name"]
                
                # Candidates: scene objects + global supplement
                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] != target_placeholder
                ]
                
                num_needed = task_plan.num_options - 1
                if len(scene_candidates) < num_needed:
                    used_names = set([correct_answer] + scene_candidates)
                    available = [obj["object_name"] for obj in self.metadata.objects 
                                if obj["object_name"] not in used_names]
                    num_to_add = num_needed - len(scene_candidates)
                    if available and num_to_add > 0:
                        additional = self.rng.choice(available, size=min(num_to_add, len(available)), replace=False)
                        scene_candidates.extend(additional)
                
                options = self._compose_options(
                    correct_answer, scene_candidates, task_plan.num_options
                )
                
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


class ListAttributeSizeGenerator(SizeGenerator):
    """Generator for 'List all {attribute}s in the components of the (largest/smallest) object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate list-attribute-size tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                attribute = self.rng.choice(ATTRIBUTES)
                
                # Find largest/smallest with attribute
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
                
                # Get attributes
                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue
                
                correct_answer = ", ".join(sorted(attr_values))
                combo_key = (target_obj["object_id"], size_type, attribute)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate question
                question = f"List all {attribute}s in the components of the {size_type} object in the scene?"
                
                # Generate candidates from global pool
                all_values = self.metadata.get_attribute_values(attribute)
                correct_set = set(correct_answer.split(", "))
                candidates = set()
                
                for _ in range(task_plan.num_options * 3):
                    if len(candidates) >= task_plan.num_options - 1:
                        break
                    sample = self.rng.choice(all_values, size=len(correct_set), replace=False)
                    if set(sample) != correct_set:
                        candidates.add(", ".join(sorted(sample)))
                
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


class CountAttributeSizeGenerator(SizeGenerator):
    """Generator for 'How many {attribute}s in the components of the (largest/smallest) object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate count-attribute-size tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                attribute = self.rng.choice(ATTRIBUTES)
                
                # Find largest/smallest with attribute
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
                
                # Count attributes
                components = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attr_values = set(comp[attribute] for comp in components)
                if not attr_values:
                    continue
                
                correct_count = len(attr_values)
                combo_key = (target_obj["object_id"], size_type, attribute, correct_count)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Generate question and answer
                question = f"How many {attribute}s are in the components of the {size_type} object in the scene?"
                correct_answer = str(correct_count)
                
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
                
                options = self._compose_options(
                    correct_answer, candidates, task_plan.num_options
                )
                
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


class WhereSizeGenerator(SizeGenerator):
    """Generator for position-based size questions with configurable reference modes."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate where-size generator configuration."""
        super().validate_generator_config(config)
        
        if 'reference_mode' in config:
            if config['reference_mode'] not in ['with_reference', 'reference_to_target', 'no_reference']:
                raise ValueError("reference_mode must be 'with_reference', 'reference_to_target', or 'no_reference'")

    def _get_reference_mode(self, task_plan: TaskPlan) -> str:
        """Get reference mode, default to 'with_reference'."""
        return task_plan.generator_config.get('reference_mode', 'with_reference')

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate where-size tasks using layout system."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        reference_mode = self._get_reference_mode(task_plan)
        
        # Import here to avoid circular dependency
        from .utils import calculate_relation_from_positions
        from layout_generator.constants import VALID_RELATIONS
        
        tasks = []
        seen_combinations = set()

        desc = f"Generating where-{size_type} tasks"
        with tqdm(total=num_tasks, desc=desc) as pbar:
            while len(tasks) < num_tasks:
                # Sample layout
                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=2)
                
                # Calculate volumes for all objects
                volumes = []
                for i, obj_spec in enumerate(layout["objects"]):
                    volume = self._calculate_volume(obj_spec["size"])
                    pos = np.array(obj_spec["position"])
                    volumes.append((i, obj_spec["name"], pos, volume))
                
                # Find largest or smallest
                if size_type == "largest":
                    target_idx, target_placeholder, target_pos, _ = max(volumes, key=lambda x: x[3])
                else:
                    target_idx, target_placeholder, target_pos, _ = min(volumes, key=lambda x: x[3])
                
                target_obj = object_mapping[target_placeholder]
                
                # Pick reference object (exclude target)
                ref_candidates = [(i, name, pos) for i, name, pos, _ in volumes if i != target_idx]
                if not ref_candidates:
                    continue
                ref_idx, ref_placeholder, ref_pos = self.rng.choice(ref_candidates)
                ref_obj = object_mapping[ref_placeholder]
                
                combo_key = (target_obj["object_id"], ref_obj["object_id"], size_type)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                # Create scene
                point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                
                # Calculate spatial relation
                correct_answer = calculate_relation_from_positions(target_pos, ref_pos)
                
                # Generate question based on reference mode
                if reference_mode == 'no_reference':
                    question = f"Where is the {size_type} object in the scene?"
                elif reference_mode == 'with_reference':
                    question = f"Where is the {size_type} object in the scene with respect to the {ref_obj['object_name']}?"
                else:  # reference_to_target
                    question = f"Where is the {ref_obj['object_name']} with respect to the {size_type} object in the scene?"
                    # Swap positions for reference_to_target mode
                    correct_answer = calculate_relation_from_positions(ref_pos, target_pos)
                
                # Candidates from VALID_RELATIONS
                candidates = [rel for rel in VALID_RELATIONS if rel != correct_answer]
                
                options = self._compose_options(correct_answer, candidates, task_plan.num_options)
                
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