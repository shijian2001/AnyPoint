import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import Counter
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import NUMBER_GENERATOR_CONFIGS, ATTRIBUTES


class NumberGenerator(BasePointQAGenerator):
    """Base class for number-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate number generator configuration."""
        if 'frequency_type' not in config:
            raise ValueError("frequency_type must be specified in generator_config")
        if config['frequency_type'] not in ['least', 'most']:
            raise ValueError("frequency_type must be 'least' or 'most'")

    def _get_frequency_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['frequency_type']

    def _generate_layout_with_object_counts(self, task_plan: TaskPlan) -> Tuple[Dict, Dict[str, Dict], Dict[str, int]]:
        """Generate layout-driven object counts with unique most/least frequencies.
        
        Uses layout system with many-to-one mapping: multiple placeholders map to same object.
        Layout-driven: First sample a layout, then use its object count.
        
        Returns:
            Tuple of (layout, object_mapping, object_name_to_count)
        """
        # Get layouts suitable for Number generators (3-9 objects)
        if self.layouts_by_type:
            usable_layouts = self.layouts_by_type.get('special', [])
        else:
            # Fallback if no classification
            usable_layouts = [l for l in self.layouts if 3 <= len(l["objects"]) <= 9]
        
        if not usable_layouts:
            raise ValueError("No usable layouts for Number generators (need 3-9 objects)")
        
        # Sample layout first, then use its object count
        layout = self.rng.choice(usable_layouts)
        total_objects = len(layout["objects"])
        
        # Choose configuration based on layout's object count
        config_options = NUMBER_GENERATOR_CONFIGS[total_objects]
        chosen_idx = self.rng.randint(len(config_options))
        num_types, object_counts = config_options[chosen_idx]
        object_counts = object_counts.copy()
        
        # Sample unique object types (without replacement)
        all_scene_objects = self.rng.choice(self.metadata.objects, size=num_types, replace=False).tolist()
        self.rng.shuffle(object_counts)
        
        # Create object mapping: multiple placeholders -> same object
        object_mapping = {}
        placeholder_idx = 0
        for obj, count in zip(all_scene_objects, object_counts):
            for _ in range(count):
                placeholder_name = layout["objects"][placeholder_idx]["name"]
                object_mapping[placeholder_name] = obj
                placeholder_idx += 1
        
        # Create count mapping
        object_name_to_count = {}
        for obj, count in zip(all_scene_objects, object_counts):
            object_name_to_count[obj["object_name"]] = count
        
        return layout, object_mapping, object_name_to_count

    def _get_target_object_by_frequency(self, object_name_to_count: Dict[str, int], 
                                       frequency_type: str) -> Tuple[str, int]:
        """Get target object by frequency type."""
        if frequency_type == "most":
            target_name = max(object_name_to_count.keys(), key=lambda k: object_name_to_count[k])
        else:  # least
            target_name = min(object_name_to_count.keys(), key=lambda k: object_name_to_count[k])
        
        return target_name, object_name_to_count[target_name]


class CountObjectGenerator(NumberGenerator):
    """Generator for 'How many {object} in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc="Generating count-object tasks") as pbar:
            while len(tasks) < num_tasks:
                try:
                    # Generate layout with object counts (multiple placeholders -> same object)
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)
                    
                    # Choose target object to ask about
                    target_obj_name = self.rng.choice(list(object_name_to_count.keys()))
                    target_count = object_name_to_count[target_obj_name]
                    
                    # Check uniqueness
                    combo_key = (target_obj_name, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)
                    
                    # Create scene using layout system
                    point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                    
                    question = f"How many {target_obj_name} in the scene?"
                    correct_answer = str(target_count)
                    
                    candidates = set()
                    
                    # Use other object counts
                    for obj_name, count in object_name_to_count.items():
                        if obj_name != target_obj_name:
                            candidates.add(str(count))
                    
                    # Add nearby numbers
                    for offset in [-2, -1, 1, 2]:
                        candidate_count = target_count + offset
                        if candidate_count >= 0:
                            candidates.add(str(candidate_count))
                    
                    candidates = list(candidates)
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
                    
                except (ValueError, IndexError):
                    continue

        return tasks


class FrequentObjectGenerator(NumberGenerator):
    """Generator for 'What is the (least/most) frequent object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating {frequency_type}-frequent-object tasks") as pbar:
            while len(tasks) < num_tasks:
                try:
                    # Generate layout with object counts
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)
                    
                    # Ensure different counts exist
                    unique_counts = set(object_name_to_count.values())
                    if len(unique_counts) < 2:
                        continue
                    
                    target_name, target_count = self._get_target_object_by_frequency(
                        object_name_to_count, frequency_type)
                    
                    # Check uniqueness
                    combo_key = (frequency_type, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)
                    
                    # Create scene using layout system
                    point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                    
                    question = f"What is the {frequency_type} frequent object in the scene?"
                    correct_answer = target_name
                    
                    # Generate candidates - prioritize scene objects, supplement globally if needed
                    scene_object_names_wo_answer = [obj_name for obj_name in object_name_to_count.keys() 
                                                   if obj_name != target_name]
                    
                    num_distractors = task_plan.num_options - 1
                    if len(scene_object_names_wo_answer) >= num_distractors:
                        candidates = scene_object_names_wo_answer
                    else:
                        num_needed_from_global = num_distractors - len(scene_object_names_wo_answer)
                        scene_objects = [obj for obj in self.metadata.objects 
                                       if obj["object_name"] in object_name_to_count]
                        available_global_objects = [obj for obj in self.metadata.objects 
                                                  if obj not in scene_objects]
                        
                        if len(available_global_objects) >= num_needed_from_global:
                            random_object_names = [obj["object_name"] for obj in self.rng.choice(
                                available_global_objects, size=num_needed_from_global, replace=False)]
                        else:
                            random_object_names = [obj["object_name"] for obj in available_global_objects]
                        
                        candidates = list(set(scene_object_names_wo_answer + random_object_names))
                    
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
                    
                except (ValueError, IndexError):
                    continue

        return tasks


class ListAttributeFrequentGenerator(NumberGenerator):
    """Generator for 'List all {attribute}s in the components of the (least/most) frequent object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{frequency_type}-frequent tasks") as pbar:
            while len(tasks) < num_tasks:
                try:
                    attribute = self.rng.choice(ATTRIBUTES)
                    
                    # Generate layout with object counts
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)
                    
                    target_name, target_count = self._get_target_object_by_frequency(
                        object_name_to_count, frequency_type)
                    
                    target_obj = None
                    for obj in self.metadata.objects:
                        if obj["object_name"] == target_name:
                            target_obj = obj
                            break
                    
                    if not target_obj or not self.metadata.has_components_with_attribute(target_obj, attribute):
                        continue
                    
                    unique_counts = set(object_name_to_count.values())
                    if len(unique_counts) < 2:
                        continue
                    
                    combo_key = (frequency_type, attribute, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)
                    
                    # Create scene using layout system
                    point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                    
                    # Get attribute values
                    components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                    attribute_values = set()
                    for component in components_with_attr:
                        attribute_values.add(component[attribute])
                    
                    if not attribute_values:
                        continue
                    
                    question = f"List all {attribute}s in the components of the {frequency_type} frequent object in the scene?"
                    correct_answer = ", ".join(sorted(attribute_values))
                    
                    candidates = set()
                    
                    # Get unique objects from object_mapping
                    unique_objects_in_scene = list({obj["object_name"]: obj for obj in object_mapping.values()}.values())
                    for obj in unique_objects_in_scene:
                        if obj["object_name"] != target_name:
                            obj_components = self.metadata.get_object_components_with_attribute(obj, attribute)
                            obj_values = set()
                            for comp in obj_components:
                                obj_values.add(comp[attribute])
                            
                            if obj_values:
                                obj_answer = ", ".join(sorted(obj_values))
                                if obj_answer != correct_answer:
                                    candidates.add(obj_answer)
                    
                    needed = task_plan.num_options - 1
                    if len(candidates) < needed:
                        all_values = self.metadata.get_attribute_values(attribute)
                        correct_values_set = set(correct_answer.split(", "))
                        
                        for _ in range(min(20, task_plan.num_options * 3)):
                            num_items = len(correct_values_set)
                            if num_items > len(all_values):
                                continue
                                
                            sample_values = self.rng.choice(all_values, size=num_items, replace=False)
                            candidate_values_set = set(sample_values)
                            
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
                    
                except (ValueError, IndexError):
                    continue

        return tasks


class CountAttributeFrequentGenerator(NumberGenerator):
    """Generator for 'How many {attribute}s in the components of (least/most) frequent object in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{frequency_type}-frequent tasks") as pbar:
            while len(tasks) < num_tasks:
                try:
                    attribute = self.rng.choice(ATTRIBUTES)
                    
                    # Generate layout with object counts
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)
                    
                    target_name, target_count = self._get_target_object_by_frequency(
                        object_name_to_count, frequency_type)
                    
                    # Find target object
                    target_obj = None
                    for obj in self.metadata.objects:
                        if obj["object_name"] == target_name:
                            target_obj = obj
                            break
                    
                    if not target_obj or not self.metadata.has_components_with_attribute(target_obj, attribute):
                        continue
                    
                    # Ensure different counts exist
                    unique_counts = set(object_name_to_count.values())
                    if len(unique_counts) < 2:
                        continue
                    
                    # Check uniqueness
                    combo_key = (frequency_type, attribute, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)
                    
                    # Create scene using layout system
                    point_cloud = self._create_point_cloud_from_layout(layout, object_mapping)
                    
                    # Count unique attribute values
                    components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                    attribute_values = set()
                    for component in components_with_attr:
                        attribute_values.add(component[attribute])
                    
                    if not attribute_values:
                        continue
                    
                    question = f"How many {attribute}s are in the components of the {frequency_type} frequent object in the scene?"
                    correct_count = len(attribute_values)
                    correct_answer = str(correct_count)
                    
                    candidates = set()
                    
                    # Get unique objects from object_mapping
                    unique_objects_in_scene = list({obj["object_name"]: obj for obj in object_mapping.values()}.values())
                    for obj in unique_objects_in_scene:
                        if obj["object_name"] != target_name:
                            obj_components = self.metadata.get_object_components_with_attribute(obj, attribute)
                            obj_values = set()
                            for comp in obj_components:
                                obj_values.add(comp[attribute])
                            
                            obj_count = len(obj_values)
                            if obj_count != correct_count:
                                candidates.add(str(obj_count))
                    
                    # Generate nearby numbers if needed
                    needed = task_plan.num_options - 1
                    if len(candidates) < needed:
                        used_numbers = {correct_count} | {int(c) for c in candidates}
                        
                        offset = 1
                        while len(candidates) < needed:
                            if correct_count + offset not in used_numbers:
                                candidates.add(str(correct_count + offset))
                                used_numbers.add(correct_count + offset)
                            
                            if len(candidates) >= needed:
                                break
                            
                            if correct_count - offset >= 0 and correct_count - offset not in used_numbers:
                                candidates.add(str(correct_count - offset))
                                used_numbers.add(correct_count - offset)
                            
                            if len(candidates) >= needed:
                                break
                            
                            offset += 1
                    
                    candidates = list(candidates)
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
                    
                except (ValueError, IndexError):
                    continue

        return tasks 