import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from .base import BasePointQAGenerator, TaskPlan, Task
from .utils import NUMBER_GENERATOR_CONFIGS, ATTRIBUTES
from .templates import sample_template


class NumberGenerator(BasePointQAGenerator):
    """Base class for number-related generators."""

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        if 'frequency_type' not in config:
            raise ValueError("frequency_type must be specified in generator_config")
        if config['frequency_type'] not in ['least', 'most']:
            raise ValueError("frequency_type must be 'least' or 'most'")

    def _get_frequency_type(self, task_plan: TaskPlan) -> str:
        return task_plan.generator_config['frequency_type']

    def _generate_layout_with_object_counts(self, task_plan: TaskPlan) -> Tuple[Dict, Dict[str, Dict], Dict[str, int]]:
        """Generate layout-driven object counts with many-to-one mapping."""
        usable_layouts = [l for l in self.layouts if 3 <= len(l["objects"]) <= 9]

        if not usable_layouts:
            raise ValueError("No usable layouts for Number generators (need 3-9 objects)")

        layout = self.rng.choice(usable_layouts)
        total_objects = len(layout["objects"])

        config_options = NUMBER_GENERATOR_CONFIGS[total_objects]
        chosen_idx = self.rng.randint(len(config_options))
        num_types, object_counts = config_options[chosen_idx]
        object_counts = object_counts.copy()

        all_scene_objects = self.rng.choice(self.metadata.objects, size=num_types, replace=False).tolist()
        self.rng.shuffle(object_counts)

        object_mapping = {}
        placeholder_idx = 0
        for obj, count in zip(all_scene_objects, object_counts):
            for _ in range(count):
                placeholder_name = layout["objects"][placeholder_idx]["name"]
                object_mapping[placeholder_name] = obj
                placeholder_idx += 1

        object_name_to_count = {}
        for obj, count in zip(all_scene_objects, object_counts):
            object_name_to_count[obj["object_name"]] = count

        return layout, object_mapping, object_name_to_count

    def _get_target_object_by_frequency(self, object_name_to_count: Dict[str, int],
                                       frequency_type: str) -> Tuple[str, int]:
        if frequency_type == "most":
            target_name = max(object_name_to_count.keys(), key=lambda k: object_name_to_count[k])
        else:
            target_name = min(object_name_to_count.keys(), key=lambda k: object_name_to_count[k])
        return target_name, object_name_to_count[target_name]


class CountObjectGenerator(NumberGenerator):
    """Generator for 'How many {object} in the scene?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating count-object tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("CountObjectGenerator", num_tasks, len(tasks), attempts)
                attempts += 1
                try:
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)

                    target_obj_name = self.rng.choice(list(object_name_to_count.keys()))
                    target_count = object_name_to_count[target_obj_name]

                    combo_key = (target_obj_name, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)

                    question = sample_template(self.rng, "count_object").format(
                        obj=target_obj_name
                    )
                    correct_answer = str(target_count)

                    scene_counts = [c for c in object_name_to_count.values()]
                    candidates = self._compose_count_distractors(
                        target_count, scene_counts, task_plan.num_options - 1
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

                except (ValueError, IndexError):
                    continue

        return tasks


class FrequentObjectGenerator(NumberGenerator):
    """Generator for 'What is the most/least frequent object?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating {frequency_type}-frequent-object tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("FrequentObjectGenerator", num_tasks, len(tasks), attempts)
                attempts += 1
                try:
                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)

                    unique_counts = set(object_name_to_count.values())
                    if len(unique_counts) < 2:
                        continue

                    target_name, target_count = self._get_target_object_by_frequency(
                        object_name_to_count, frequency_type)

                    combo_key = (frequency_type, tuple(sorted(object_name_to_count.items())))
                    if combo_key in seen_combinations:
                        continue
                    seen_combinations.add(combo_key)

                    question = sample_template(
                        self.rng, "frequent_object", frequency_type=frequency_type
                    )
                    correct_answer = target_name

                    # Distractors: scene objects first, supplement globally if needed
                    scene_names = [n for n in object_name_to_count.keys() if n != target_name]
                    num_distractors = task_plan.num_options - 1

                    if len(scene_names) >= num_distractors:
                        candidates = scene_names
                    else:
                        used = set(object_name_to_count.keys())
                        global_names = [obj["object_name"] for obj in self.metadata.objects
                                       if obj["object_name"] not in used]
                        extra = self.rng.choice(
                            global_names,
                            size=min(num_distractors - len(scene_names), len(global_names)),
                            replace=False
                        ).tolist() if global_names else []
                        candidates = scene_names + extra

                    if len(candidates) < num_distractors:
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

                except (ValueError, IndexError):
                    continue

        return tasks


class ListAttributeFrequentGenerator(NumberGenerator):
    """Generator for 'List all {attr}s of the most/least frequent object.' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating list-attribute-{frequency_type}-frequent tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("ListAttributeFrequentGenerator", num_tasks, len(tasks), attempts)
                attempts += 1
                try:
                    attribute = self.rng.choice(ATTRIBUTES)

                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)

                    target_name, _ = self._get_target_object_by_frequency(
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

                    components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                    attribute_values = set(comp[attribute] for comp in components_with_attr)

                    if not attribute_values:
                        continue

                    question = sample_template(
                        self.rng, "list_attribute_frequent", frequency_type=frequency_type
                    ).format(attr=attribute)
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

                except (ValueError, IndexError):
                    continue

        return tasks


class CountAttributeFrequentGenerator(NumberGenerator):
    """Generator for 'How many {attr}s of the most/least frequent object?' questions."""

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        self.validate_generator_config(task_plan.generator_config)
        frequency_type = self._get_frequency_type(task_plan)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc=f"Generating count-attribute-{frequency_type}-frequent tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("CountAttributeFrequentGenerator", num_tasks, len(tasks), attempts)
                attempts += 1
                try:
                    attribute = self.rng.choice(ATTRIBUTES)

                    layout, object_mapping, object_name_to_count = self._generate_layout_with_object_counts(task_plan)

                    target_name, _ = self._get_target_object_by_frequency(
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

                    components_with_attr = self.metadata.get_object_components_with_attribute(target_obj, attribute)
                    attribute_values = set(comp[attribute] for comp in components_with_attr)

                    if not attribute_values:
                        continue

                    question = sample_template(
                        self.rng, "count_attribute_frequent", frequency_type=frequency_type
                    ).format(attr=attribute)
                    correct_count = len(attribute_values)
                    correct_answer = str(correct_count)

                    unique_objs = list({obj["object_name"]: obj for obj in object_mapping.values()}.values())
                    scene_counts = []
                    for obj in unique_objs:
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

                except (ValueError, IndexError):
                    continue

        return tasks
