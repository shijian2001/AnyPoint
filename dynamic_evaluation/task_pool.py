"""Task candidate pool generator."""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import numpy as np

from point_qa_generator.base import Task, TaskPlan
from point_qa_generator.generator import PointQAGenerator


@dataclass
class PoolItem:
    """Single pre-generated candidate task."""

    item_id: int
    task: Task
    point_cloud: Optional[np.ndarray] = None


class TaskPool:
    """
    Pre-generate a fixed candidate pool, then sample from the remaining items.

    ``pool_size`` now means the total number of candidate tasks generated once
    at startup, rather than the number regenerated in each iteration.
    """

    def __init__(self, qa_generator: PointQAGenerator, seed: int = 42, pool_size: int = 1000):
        self.gen = qa_generator
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.pool_size = pool_size
        self._remaining: List[PoolItem] = []

    def ensure_ready(self, cache_dir: str) -> None:
        """Load a cached task pool or build one if the inputs changed."""
        os.makedirs(cache_dir, exist_ok=True)
        manifest_path = os.path.join(cache_dir, "task_pool_manifest.json")

        expected_signature = self._cache_signature()
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if manifest.get("signature") == expected_signature:
                    self._remaining = self._load_cached_pool(manifest)
                    print(f"[INFO] Loaded cached task pool: {len(self._remaining)} items")
                    return
                print("[INFO] Task synthesis inputs changed, regenerating cached task pool")
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                print(f"[INFO] Failed to load cached task pool ({exc}), regenerating")

        self._remaining = self._build_pool(self.pool_size)
        self._save_cached_pool(cache_dir, expected_signature)

    def _build_pool(self, size: int) -> List[PoolItem]:
        if size < 1:
            raise ValueError("pool_size must be positive")

        task_plans = self._build_task_plans()
        per_plan = max(1, int(np.ceil(size / len(task_plans))))

        generated_items: List[PoolItem] = []
        next_item_id = 0

        for plan in task_plans:
            generator = self.gen.generators[plan.generator_type]
            try:
                batch = generator.generate_tasks(plan, per_plan)
            except (ValueError, IndexError) as exc:
                print(f"⚠️  {plan.generator_type}: {exc}")
                continue

            for task, point_cloud in batch:
                generated_items.append(
                    PoolItem(
                        item_id=next_item_id,
                        task=task,
                        point_cloud=point_cloud,
                    )
                )
                next_item_id += 1

        self.rng.shuffle(generated_items)
        pool = generated_items[:size]
        print(f"[INFO] Pre-generated fixed task pool: {len(pool)} items")
        return pool

    def _cache_signature(self) -> Dict[str, Any]:
        task_plans = [
            {
                "generator_type": plan.generator_type,
                "num_options": plan.num_options,
                "seed": plan.seed,
                "generator_config": plan.generator_config,
            }
            for plan in self._build_task_plans()
        ]
        return {
            "pool_size": self.pool_size,
            "seed": self.seed,
            "task_plans": task_plans,
            "source_signature": self.gen.get_source_signature(),
            "cache_format_version": 2,
        }

    def _save_cached_pool(self, cache_dir: str, signature: Dict[str, Any]) -> None:
        manifest_path = os.path.join(cache_dir, "task_pool_manifest.json")

        manifest = {
            "signature": signature,
            "items": [
                {
                    "item_id": item.item_id,
                    "task": {
                        "point": item.task.point,
                        "question": item.task.question,
                        "options": item.task.options,
                        "answer": item.task.answer,
                        "metadata": item.task.metadata,
                    },
                }
                for item in self._remaining
            ],
        }

        print(f"[INFO] Writing task pool manifest: {manifest_path}")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved synthesized task pool cache: {len(self._remaining)} items")

    def _load_cached_pool(self, manifest: Dict[str, Any]) -> List[PoolItem]:
        pool: List[PoolItem] = []
        for item_data in manifest["items"]:
            task_data = item_data["task"]
            item_id = item_data["item_id"]

            pool.append(
                PoolItem(
                    item_id=item_id,
                    task=Task(
                        point=task_data["point"],
                        question=task_data["question"],
                        options=task_data["options"],
                        answer=task_data["answer"],
                        metadata=task_data.get("metadata"),
                    ),
                    point_cloud=None,
                )
            )
        return pool

    def _build_task_plans(self) -> List[TaskPlan]:
        plan_specs: List[tuple[str, Dict[str, object]]] = [
            ("what_distance", {"distance_type": "closest"}),
            ("what_distance", {"distance_type": "farthest"}),
            ("where_distance", {"distance_type": "closest"}),
            ("where_distance", {"distance_type": "farthest"}),
            ("list_attribute_distance", {"distance_type": "closest"}),
            ("list_attribute_distance", {"distance_type": "farthest"}),
            ("count_attribute_distance", {"distance_type": "closest"}),
            ("count_attribute_distance", {"distance_type": "farthest"}),
            ("what_attribute", {}),
            ("list_attribute", {}),
            ("count_attribute", {}),
            ("count_object", {}),
            ("frequent_object", {"frequency_type": "most"}),
            ("frequent_object", {"frequency_type": "least"}),
            ("list_attribute_frequent", {"frequency_type": "most"}),
            ("list_attribute_frequent", {"frequency_type": "least"}),
            ("count_attribute_frequent", {"frequency_type": "most"}),
            ("count_attribute_frequent", {"frequency_type": "least"}),
            ("what_size", {"size_type": "largest"}),
            ("what_size", {"size_type": "smallest"}),
            ("list_attribute_size", {"size_type": "largest"}),
            ("list_attribute_size", {"size_type": "smallest"}),
            ("count_attribute_size", {"size_type": "largest"}),
            ("count_attribute_size", {"size_type": "smallest"}),
            ("where_size", {"size_type": "largest", "reference_mode": "with_reference"}),
            ("where_size", {"size_type": "smallest", "reference_mode": "with_reference"}),
        ]

        task_plans: List[TaskPlan] = []
        for idx, (generator_type, generator_config) in enumerate(plan_specs):
            task_plans.append(
                TaskPlan(
                    generator_type=generator_type,
                    num_options=4,
                    seed=self.seed + idx,
                    generator_config=generator_config,
                )
            )
        return task_plans

    def remaining(self) -> List[PoolItem]:
        """Return all unevaluated candidates."""
        return list(self._remaining)

    def remaining_count(self) -> int:
        return len(self._remaining)

    def pop_random(self, size: int) -> List[PoolItem]:
        """Randomly select and remove candidates from the remaining pool."""
        if size < 1:
            return []
        size = min(size, len(self._remaining))
        if size == 0:
            return []

        indices = self.rng.choice(len(self._remaining), size=size, replace=False)
        return self.pop_indices(indices.tolist())

    def pop_indices(self, indices: List[int]) -> List[PoolItem]:
        """Select and remove candidates by index within the remaining pool."""
        ordered_unique = list(dict.fromkeys(indices))
        removed_by_index = {}
        for idx in sorted(ordered_unique, reverse=True):
            removed_by_index[idx] = self._remaining.pop(idx)
        return [removed_by_index[idx] for idx in ordered_unique]
