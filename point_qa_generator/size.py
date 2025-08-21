import numpy as np
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import GRID_POSITIONS, get_relative_position, ATTRIBUTES


class SizeGenerator(BasePointQAGenerator):
    """Base class for size-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate size generator configuration."""
        if 'size_type' not in config:
            raise ValueError("size_type must be specified in generator_config")
        if config['size_type'] not in ['largest', 'smallest']:
            raise ValueError("size_type must be 'largest' or 'smallest'")

    def _get_size_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['size_type']

    def _generate_scale_factors(self, all_objects: List[Dict], size_type: str) -> List[float]:
        """Generate size-differentiated scale factors for objects."""
        base_scale = 0.5
        scale_variation = 0.3
        
        scale_factors = []
        for i, obj in enumerate(all_objects):
            if i == 0:  # target object
                if size_type == "largest":
                    scale_factor = base_scale + scale_variation
                else:  # smallest
                    scale_factor = base_scale - scale_variation
            else:
                if size_type == "largest":
                    scale_factor = base_scale + self.rng.uniform(-scale_variation, scale_variation * 0.5)
                else:  # smallest
                    scale_factor = base_scale + self.rng.uniform(scale_variation * 0.5, scale_variation)
            
            scale_factor = max(0.1, min(1.0, scale_factor))
            scale_factors.append(scale_factor)
        
        return scale_factors

    def _generate_size_differentiated_scene(self, task_plan: TaskPlan, target_obj: Dict, 
                                          size_type: str) -> Tuple[List[Dict], List[int], List[float], List[float]]:
        """Generate scene with size-differentiated objects."""
        used_objects = {target_obj["object_id"]}
        available_objects = [obj for obj in self.metadata.objects
                           if obj["object_id"] not in used_objects]
        
        num_distractors = min(task_plan.num_scene_distractors, len(available_objects))
        distractor_objects = self.rng.choice(available_objects, size=num_distractors,
                                           replace=False).tolist() if num_distractors > 0 else []
        
        all_objects = [target_obj] + distractor_objects
        
        available_grids = list(range(9))
        selected_grids = self.rng.choice(available_grids, size=len(all_objects), replace=False).tolist()
        
        angles = [self.metadata.sample_angle() for _ in all_objects]
        scale_factors = self._generate_scale_factors(all_objects, size_type)
        
        return all_objects, selected_grids, angles, scale_factors

    def _generate_scene_distractors(self, task_plan: TaskPlan, target_obj: Dict, 
                                   ref_obj: Dict = None) -> List[Dict]:
        """Generate scene distractor objects for size questions."""
        used_objects = {target_obj["object_id"]}
        if ref_obj:
            used_objects.add(ref_obj["object_id"])
            
        available_objects = [obj for obj in self.metadata.objects
                           if obj["object_id"] not in used_objects]
        
        num_distractors = min(task_plan.num_scene_distractors, len(available_objects))
        return self.rng.choice(available_objects, size=num_distractors,
                               replace=False).tolist() if num_distractors > 0 else []


class WhatSizeGenerator(SizeGenerator):
    """Generator for 'What is the (largest/smallest) object in the scene?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        max_distractors = min(task_plan.num_scene_distractors, len(self.metadata.objects) - 1)
        return len(self.metadata.objects) * (max_distractors + 1) * 10

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate what-size tasks."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating what-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                target_obj = self.rng.choice(self.metadata.objects)
                
                all_objects, all_grids, angles, scale_factors = self._generate_size_differentiated_scene(
                    task_plan, target_obj, size_type)
                
                combo_key = (target_obj["object_id"], size_type, tuple(obj["object_id"] for obj in all_objects[1:]))
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles, scale_factors)
                
                question = f"What is the {size_type} object in the scene?"
                correct_answer = target_obj["object_name"]
                
                scene_object_names_wo_answer = [obj["object_name"] for obj in all_objects if obj != target_obj]
                
                num_distractors = task_plan.num_options - 1
                if len(scene_object_names_wo_answer) >= num_distractors:
                    candidates = scene_object_names_wo_answer
                else:
                    num_needed_from_global = num_distractors - len(scene_object_names_wo_answer)
                    available_global_objects = [obj for obj in self.metadata.objects 
                                              if obj not in all_objects]
                    
                    if len(available_global_objects) >= num_needed_from_global:
                        random_object_names = [obj["object_name"] for obj in self.rng.choice(
                            available_global_objects, size=num_needed_from_global, replace=False)]
                    else:
                        random_object_names = [obj["object_name"] for obj in available_global_objects]
                    
                    candidates = list(set(scene_object_names_wo_answer + random_object_names))
                
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


class ListAttributeSizeGenerator(SizeGenerator):
    """Generator for 'List all {attribute}s in the components of the (largest/smallest) object in the scene?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        
        valid_targets = 0
        for obj in self.metadata.objects:
            for attribute in ATTRIBUTES:
                if self.metadata.has_components_with_attribute(obj, attribute):
                    valid_targets += 1
                    break
        
        max_distractors = min(task_plan.num_scene_distractors, len(self.metadata.objects) - 1)
        return valid_targets * 4 * (max_distractors + 1) * 5

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate list-attribute-size tasks."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                attribute = self.rng.choice(ATTRIBUTES)
                
                valid_targets = [obj for obj in self.metadata.objects
                               if self.metadata.has_components_with_attribute(obj, attribute)]
                if not valid_targets:
                    continue
                
                target_obj = self.rng.choice(valid_targets)
                
                all_objects, all_grids, angles, scale_factors = self._generate_size_differentiated_scene(
                    task_plan, target_obj, size_type)
                
                combo_key = (target_obj["object_id"], size_type, attribute, 
                           tuple(obj["object_id"] for obj in all_objects[1:]))
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles, scale_factors)
                
                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])
                
                if not attribute_values:
                    continue
                
                question = f"List all {attribute}s in the components of the {size_type} object in the scene?"
                correct_answer = ", ".join(sorted(attribute_values))
                
                candidates = set()
                
                for obj in all_objects[1:]:
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


class CountAttributeSizeGenerator(SizeGenerator):
    """Generator for 'How many {attribute}s in the components of the (largest/smallest) object in the scene?' questions."""

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        
        valid_targets = 0
        for obj in self.metadata.objects:
            for attribute in ATTRIBUTES:
                if self.metadata.has_components_with_attribute(obj, attribute):
                    valid_targets += 1
                    break
        
        max_distractors = min(task_plan.num_scene_distractors, len(self.metadata.objects) - 1)
        return valid_targets * 4 * (max_distractors + 1) * 5

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate count-attribute-size tasks."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{size_type} tasks") as pbar:
            while len(tasks) < num_tasks:
                attribute = self.rng.choice(ATTRIBUTES)
                
                valid_targets = [obj for obj in self.metadata.objects
                               if self.metadata.has_components_with_attribute(obj, attribute)]
                if not valid_targets:
                    continue
                
                target_obj = self.rng.choice(valid_targets)
                
                all_objects, all_grids, angles, scale_factors = self._generate_size_differentiated_scene(
                    task_plan, target_obj, size_type)
                
                combo_key = (target_obj["object_id"], size_type, attribute,
                           tuple(obj["object_id"] for obj in all_objects[1:]))
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles, scale_factors)
                
                components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                attribute_values = set()
                for component in components_with_attr:
                    attribute_values.add(component[attribute])
                
                if not attribute_values:
                    continue
                
                question = f"How many {attribute}s are in the components of the {size_type} object in the scene?"
                correct_count = len(attribute_values)
                correct_answer = str(correct_count)
                
                candidates = set()
                
                for obj in all_objects[1:]:
                    obj_components = self.metadata.get_object_components_with_attribute(obj, attribute)
                    obj_values = set()
                    for comp in obj_components:
                        obj_values.add(comp[attribute])
                    
                    obj_count = len(obj_values)
                    if obj_count != correct_count:
                        candidates.add(str(obj_count))
                
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

    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count possible task combinations."""
        self.validate_generator_config(task_plan.generator_config)
        reference_mode = self._get_reference_mode(task_plan)
        
        if reference_mode == 'no_reference':
            max_distractors = min(task_plan.num_scene_distractors, len(self.metadata.objects) - 1)
            return len(self.metadata.objects) * 9 * (max_distractors + 1) * 5
        else:
            return len(self.metadata.objects) * (len(self.metadata.objects) - 1) * 9 * 9 * 3

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate where-size tasks."""
        self.validate_generator_config(task_plan.generator_config)
        size_type = self._get_size_type(task_plan)
        reference_mode = self._get_reference_mode(task_plan)
        
        possible_tasks = self.count_possible_tasks(task_plan)
        if num_tasks > possible_tasks:
            raise ValueError(f"Requested {num_tasks} tasks but only {possible_tasks} possible")

        tasks = []
        seen_combinations = set()

        desc = f"Generating where-{size_type}-{reference_mode} tasks"
        with tqdm(total=num_tasks, desc=desc) as pbar:
            while len(tasks) < num_tasks:
                if reference_mode == 'no_reference':
                    target_obj = self.rng.choice(self.metadata.objects)
                    ref_obj = None
                    
                    all_objects, all_grids, angles, scale_factors = self._generate_size_differentiated_scene(
                        task_plan, target_obj, size_type)
                    
                    combo_key = (target_obj["object_id"], size_type, 'no_reference',
                               tuple(obj["object_id"] for obj in all_objects[1:]), all_grids[0])
                    
                else:
                    target_obj, ref_obj = self.rng.choice(self.metadata.objects, size=2, replace=False)
                    
                    distractor_objects = self._generate_scene_distractors(task_plan, target_obj, ref_obj)
                    
                    all_objects = [target_obj, ref_obj] + distractor_objects
                    
                    available_grids = list(range(9))
                    all_grids = self.rng.choice(available_grids, size=len(all_objects), replace=False).tolist()
                    
                    angles = [self.metadata.sample_angle() for _ in all_objects]
                    scale_factors = self._generate_scale_factors(all_objects, size_type)
                    
                    combo_key = (target_obj["object_id"], ref_obj["object_id"], size_type, reference_mode,
                               tuple(obj["object_id"] for obj in all_objects[2:]), all_grids[0], all_grids[1])

                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)
                
                point_cloud = self._create_point_cloud_scene(all_objects, all_grids, angles, scale_factors)
                
                if reference_mode == 'no_reference':
                    question = f"Where is the {size_type} object in the scene?"
                    correct_answer = GRID_POSITIONS[all_grids[0]]
                    candidates = [GRID_POSITIONS[g] for g in range(9) if GRID_POSITIONS[g] != correct_answer]
                    
                elif reference_mode == 'with_reference':
                    question = f"Where is the {size_type} object in the scene with respect to the {ref_obj['object_name']}?"
                    correct_answer = get_relative_position(all_grids[0], all_grids[1])
                    candidates = []
                    for g in range(9):
                        if g != all_grids[0]:
                            rel_pos = get_relative_position(g, all_grids[1])
                            if rel_pos != correct_answer:
                                candidates.append(rel_pos)
                    candidates = list(set(candidates))
                    
                else:  # reference_to_target
                    question = f"Where is the {ref_obj['object_name']} with respect to the {size_type} object in the scene?"
                    correct_answer = get_relative_position(all_grids[1], all_grids[0])
                    candidates = []
                    for g in range(9):
                        if g != all_grids[1]:
                            rel_pos = get_relative_position(g, all_grids[0])
                            if rel_pos != correct_answer:
                                candidates.append(rel_pos)
                    candidates = list(set(candidates))
                
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