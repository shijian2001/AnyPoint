"""Relation-based QA generators using layout relation fields."""

from collections import Counter
from typing import Dict, List, Any, Tuple

import numpy as np
from tqdm import tqdm

from .base import BasePointQAGenerator, TaskPlan, Task
from .templates import sample_template


class WhatRelationGenerator(BasePointQAGenerator):
    """Generator for 'What is the object that is [relation] the {reference}?' questions.

    Uses the layout's ``relations`` field directly as ground truth,
    selecting only relations whose (relation, reference) pair is unique
    to guarantee a single correct answer.
    """

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        pass

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        min_objs = task_plan.num_options + 1

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating what-relation tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("WhatRelationGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=min_objs)

                relations = layout.get("relations", [])
                if not relations:
                    continue

                # Only use relations with unique (relation, reference) pairs
                pair_counts = Counter(
                    (r["relation"], r["reference"]) for r in relations
                )
                unique_relations = [
                    r for r in relations
                    if pair_counts[(r["relation"], r["reference"])] == 1
                ]
                if not unique_relations:
                    continue

                rel = unique_relations[self.rng.randint(len(unique_relations))]
                subject_placeholder = rel["subject"]
                reference_placeholder = rel["reference"]
                relation_type = rel["relation"]

                subject_obj = object_mapping[subject_placeholder]
                reference_obj = object_mapping[reference_placeholder]

                combo_key = (subject_obj["object_id"], reference_obj["object_id"], relation_type)
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(self.rng, "what_relation").format(
                    rel=relation_type, ref=reference_obj['object_name']
                )
                correct_answer = subject_obj["object_name"]

                # Distractors from scene objects (exclude subject and reference)
                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] != subject_placeholder
                       and obj["name"] != reference_placeholder
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


class MultiHopRelationGenerator(BasePointQAGenerator):
    """Generator for 2-hop relation reasoning questions.

    Pattern: 'What is the object [rel2] the object [rel1] the {anchor}?'
    Requires at least 4 objects (anchor + middle + answer + 1 distractor).
    """

    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        pass

    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        min_objs = max(task_plan.num_options + 1, 4)

        tasks = []
        seen_combinations = set()
        attempts = 0
        max_attempts = self._max_generation_attempts(num_tasks)

        with tqdm(total=num_tasks, desc="Generating multi-hop-relation tasks") as pbar:
            while len(tasks) < num_tasks:
                if attempts >= max_attempts:
                    self._raise_generation_failure("MultiHopRelationGenerator", num_tasks, len(tasks), attempts)
                attempts += 1

                layout, object_mapping = self._sample_layout_and_map_objects(min_objects=min_objs)

                relations = layout.get("relations", [])
                if len(relations) < 2:
                    continue

                chains = self._find_two_hop_chains(relations)
                if not chains:
                    continue

                chain = chains[self.rng.randint(len(chains))]
                anchor_placeholder = chain["anchor"]
                answer_placeholder = chain["answer"]
                middle_placeholder = chain["middle"]

                anchor_obj = object_mapping[anchor_placeholder]
                answer_obj = object_mapping[answer_placeholder]

                combo_key = (
                    answer_obj["object_id"],
                    anchor_obj["object_id"],
                    chain["rel1"],
                    chain["rel2"],
                )
                if combo_key in seen_combinations:
                    continue
                seen_combinations.add(combo_key)

                question = sample_template(self.rng, "multi_hop_relation").format(
                    rel1=chain["rel1"],
                    rel2=chain["rel2"],
                    anchor=anchor_obj["object_name"],
                )
                correct_answer = answer_obj["object_name"]

                # Distractors from scene (exclude answer, anchor, middle)
                exclude = {answer_placeholder, anchor_placeholder, middle_placeholder}
                scene_candidates = [
                    object_mapping[obj["name"]]["object_name"]
                    for obj in layout["objects"]
                    if obj["name"] not in exclude
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

    def _find_two_hop_chains(self, relations: List[Dict]) -> List[Dict]:
        """Find all valid 2-hop chains from the relation graph.

        A chain is: anchor <--(rel1)-- middle <--(rel2)-- answer
        i.e., r1.subject == middle, r1.reference == anchor
              r2.subject == answer, r2.reference == middle
        """
        chains = []
        for r1 in relations:
            for r2 in relations:
                if r2["reference"] == r1["subject"] and r2["subject"] != r1["reference"]:
                    chains.append({
                        "anchor": r1["reference"],
                        "rel1": r1["relation"],
                        "middle": r1["subject"],
                        "rel2": r2["relation"],
                        "answer": r2["subject"],
                    })

        return self._filter_unique_answer_chains(chains)

    @staticmethod
    def _filter_unique_answer_chains(chains: List[Dict]) -> List[Dict]:
        """Keep only chains where (anchor, rel1, rel2) yields a unique answer."""
        key_counts = Counter(
            (c["anchor"], c["rel1"], c["rel2"]) for c in chains
        )
        return [
            c for c in chains
            if key_counts[(c["anchor"], c["rel1"], c["rel2"])] == 1
        ]
