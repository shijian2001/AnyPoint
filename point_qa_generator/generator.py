import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from .base import TaskPlan, Task
from .metadata import PointCloudMetadata
from .distance import (WhatDistanceGenerator, WhereDistanceGenerator,
                       ListAttributeDistanceGenerator, CountAttributeDistanceGenerator)
from .attribute import (WhatAttributeGenerator, ListAttributeGenerator,
                        CountAttributeGenerator)
from .number import (CountObjectGenerator, FrequentObjectGenerator,
                     ListAttributeFrequentGenerator, CountAttributeFrequentGenerator)
from .size import (WhatSizeGenerator, ListAttributeSizeGenerator,
                   CountAttributeSizeGenerator, WhereSizeGenerator)
from .relation import WhatRelationGenerator, MultiHopRelationGenerator


class PointQAGenerator:
    """Main interface for Point QA generation."""

    def __init__(
        self,
        metadata_file: str,
        pcd_dir: str,
        layouts_file: str,
        seed: int = 42,
        background_dir: str = None,
        cache_mb: int = 0,
    ):
        self.metadata = PointCloudMetadata(metadata_file, pcd_dir, seed,
                                           cache_mb=cache_mb)
        self.metadata_file = metadata_file
        self.pcd_dir = pcd_dir
        self.layouts_file = layouts_file
        self.background_dir = background_dir or os.path.join(
            os.path.dirname(os.path.abspath(metadata_file)), "background"
        )
        self.layouts = self._load_layouts(layouts_file)
        self.layouts_by_id = {
            layout.get("id"): layout
            for layout in self.layouts
            if layout.get("id") is not None
        }
        self.objects_by_id = {
            obj["object_id"]: obj
            for obj in self.metadata.objects
        }
        self.rng = np.random.RandomState(seed)

        gen_kwargs = dict(background_dir=self.background_dir)
        self.generators = {
            "what_distance": WhatDistanceGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "where_distance": WhereDistanceGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "list_attribute_distance": ListAttributeDistanceGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "count_attribute_distance": CountAttributeDistanceGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "what_attribute": WhatAttributeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "list_attribute": ListAttributeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "count_attribute": CountAttributeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "count_object": CountObjectGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "frequent_object": FrequentObjectGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "list_attribute_frequent": ListAttributeFrequentGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "count_attribute_frequent": CountAttributeFrequentGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "what_size": WhatSizeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "list_attribute_size": ListAttributeSizeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "count_attribute_size": CountAttributeSizeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "where_size": WhereSizeGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "what_relation": WhatRelationGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
            "multi_hop_relation": MultiHopRelationGenerator(self.metadata, seed, self.layouts, **gen_kwargs),
        }

    GENERATOR_VARIANTS: Dict[str, List[Dict[str, Any]]] = {
        "what_distance": [
            {"distance_type": "closest"},
            {"distance_type": "farthest"},
        ],
        "where_distance": [
            {"distance_type": "closest"},
            {"distance_type": "farthest"},
        ],
        "list_attribute_distance": [
            {"distance_type": "closest"},
            {"distance_type": "farthest"},
        ],
        "count_attribute_distance": [
            {"distance_type": "closest"},
            {"distance_type": "farthest"},
        ],
        "what_attribute": [{}],
        "list_attribute": [{}],
        "count_attribute": [{}],
        "count_object": [{}],
        "frequent_object": [
            {"frequency_type": "most"},
            {"frequency_type": "least"},
        ],
        "list_attribute_frequent": [
            {"frequency_type": "most"},
            {"frequency_type": "least"},
        ],
        "count_attribute_frequent": [
            {"frequency_type": "most"},
            {"frequency_type": "least"},
        ],
        "what_size": [
            {"size_type": "largest"},
            {"size_type": "smallest"},
        ],
        "list_attribute_size": [
            {"size_type": "largest"},
            {"size_type": "smallest"},
        ],
        "count_attribute_size": [
            {"size_type": "largest"},
            {"size_type": "smallest"},
        ],
        "where_size": [
            {"size_type": "largest", "reference_mode": "with_reference"},
            {"size_type": "smallest", "reference_mode": "with_reference"},
            {"size_type": "largest", "reference_mode": "reference_to_target"},
            {"size_type": "smallest", "reference_mode": "reference_to_target"},
        ],
        "what_relation": [{}],
        "multi_hop_relation": [{}],
    }


    @staticmethod
    def _allocate_evenly(
        total: int, types: List[str], rng: np.random.RandomState
    ) -> Dict[str, int]:
        base = total // len(types)
        remainder = total % len(types)
        counts = {t: base for t in types}
        bonus_types = rng.choice(types, size=remainder, replace=False)
        for t in bonus_types:
            counts[t] += 1
        return counts

    @staticmethod
    def _allocate_by_weights(
        total: int, types: List[str], weights: Dict[str, float]
    ) -> Dict[str, int]:
        raw = {t: weights[t] * total for t in types}
        counts = {t: int(v) for t, v in raw.items()}
        remainder = total - sum(counts.values())
        fractional = sorted(types, key=lambda t: raw[t] - counts[t], reverse=True)
        for t in fractional[:remainder]:
            counts[t] += 1
        return counts

    def get_source_signature(self) -> Dict[str, Any]:
        return {
            "metadata_file": self._file_signature(self.metadata_file),
            "layouts_file": self._file_signature(self.layouts_file),
            "pcd_dir": self._directory_signature(self.pcd_dir),
            "background_dir": self._directory_signature(self.background_dir)
            if self.background_dir and os.path.isdir(self.background_dir) else None,
        }

    def get_source_signature_hash(self) -> str:
        signature_json = json.dumps(self.get_source_signature(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(signature_json.encode("utf-8")).hexdigest()

    @staticmethod
    def _file_signature(path: str) -> Dict[str, Any]:
        stat = os.stat(path)
        return {
            "path": os.path.abspath(path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    @staticmethod
    def _directory_signature(path: str) -> Dict[str, Any]:
        entries = []
        for root, _, files in os.walk(path):
            for name in sorted(files):
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, path)
                stat = os.stat(full_path)
                entries.append({
                    "path": rel_path,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                })
        return {
            "path": os.path.abspath(path),
            "files": entries,
        }

    def _load_layouts(self, layouts_file: str) -> List[Dict]:
        """Load layouts from JSON or JSONL file."""
        with open(layouts_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                return json.load(f)
            else:
                layouts = []
                for line in f:
                    line = line.strip()
                    if line:
                        layouts.append(json.loads(line))
                return layouts

    def materialize_point_cloud(self, task: Task) -> np.ndarray:
        """Rebuild a synthesized scene point cloud from task metadata."""
        if not task.metadata:
            raise ValueError("Task metadata is required to rebuild point cloud")

        layout_id = task.metadata.get("layout_id")
        layout = self.layouts_by_id.get(layout_id)
        if layout is None:
            raise ValueError(f"Layout not found for task metadata: {layout_id}")

        object_mapping: Dict[str, Dict[str, Any]] = {}
        for obj_info in task.metadata.get("objects", []):
            placeholder = obj_info.get("placeholder")
            object_id = obj_info.get("object_id")
            if placeholder is None or object_id is None:
                raise ValueError("Task metadata objects must contain placeholder and object_id")

            actual_obj = self.objects_by_id.get(object_id)
            if actual_obj is None:
                raise ValueError(f"Object metadata not found for object_id={object_id}")
            object_mapping[placeholder] = actual_obj

        # Use the recorded background so the scene is rebuilt bit-for-bit.
        # Older metadata without background_id falls back to a random pick.
        background_id = task.metadata.get("background_id", "__random__")
        gen_type = task.metadata["generator_type"]
        if isinstance(gen_type, (list, tuple)):  # tolerate list-typed metadata
            gen_type = gen_type[0]
        return self.generators[gen_type]._create_point_cloud_from_layout(
            layout, object_mapping, background_id=background_id
        )

    def generate(self, task_plan: TaskPlan, num_tasks: int, output_dir: str,
                 num_points: Optional[int] = None) -> Dict[str, Any]:
        """Generate QA tasks and save to output directory.

        Unified interface: TaskPlan.generator_type can be a single str,
        a list of str (equal weight), or a dict of str->float (weighted).
        generator_config can be omitted for random variant selection.
        num_options can be int (fixed) or (min, max) tuple (random per batch).

        num_points: if set, each scene point cloud is randomly downsampled to
        this many points before saving (scenes with fewer points are kept as-is).
        Default None = save full-resolution point cloud.
        """
        rng = np.random.RandomState(task_plan.seed)
        # default_rng's choice(replace=False) is ~100x faster than legacy
        # RandomState for large N (49ms -> 0.5ms per downsample). Only affects
        # WHICH points are sampled (random anyway), not QA content or geometry.
        sampler_rng = np.random.default_rng(task_plan.seed ^ 0x5DEECE66)

        gen_types = task_plan.resolve_generator_types()
        for gt in gen_types:
            if gt not in self.generators:
                raise ValueError(f"Unknown generator type: {gt}")

        # Re-seed each generator's own rng from this plan's seed. Without this,
        # every generate() call (e.g. each parallel shard) reuses the rng state
        # seeded at construction time with the shared base_seed, so all shards
        # sample the *same* objects/templates and produce duplicate questions.
        # Mixing the generator-type name keeps distinct types from colliding.
        for gt in self.generators:
            type_seed = (task_plan.seed
                         + (int(hashlib.sha256(gt.encode()).hexdigest(), 16) & 0x7FFFFFFF)) % (2 ** 31)
            self.generators[gt].rng = np.random.RandomState(type_seed)

        weights = task_plan.resolve_weights()
        if weights:
            counts = self._allocate_by_weights(num_tasks, gen_types, weights)
        else:
            counts = self._allocate_evenly(num_tasks, gen_types, rng)

        # Build resolved plans for each generator type
        plans_and_counts: List[Tuple[str, TaskPlan, int]] = []
        for gen_type in gen_types:
            n = counts[gen_type]
            if n == 0:
                continue

            if task_plan.generator_config:
                config = task_plan.generator_config
            else:
                variants = self.GENERATOR_VARIANTS[gen_type]
                config = variants[rng.randint(len(variants))]

            num_options = task_plan.resolve_num_options(rng)
            resolved_plan = TaskPlan(
                generator_type=gen_type,
                num_options=num_options,
                seed=task_plan.seed,
                generator_config=config,
            )
            plans_and_counts.append((gen_type, resolved_plan, n))

        # Generate in parallel across generator types.
        # Dedup key is (question, answer, sorted options) rather than the
        # question text alone: "same question text + different scene/answer" is a
        # VALID distinct training sample (the model must read the point cloud to
        # answer). Generators with a small question-text space (what_size,
        # frequent_object) would otherwise be over-pruned, sending the fill loop
        # below into a near-infinite retry that burns CPU producing nothing.
        all_results: List[Tuple[Task, np.ndarray, str]] = []
        seen_keys: set = set()

        def _key(task):
            return (task.question, task.answer, tuple(sorted(task.options)))

        def _run_generator(gen_type, plan, count):
            generator = self.generators[gen_type]
            return gen_type, plan, generator.generate_tasks(plan, count)

        with ThreadPoolExecutor(max_workers=min(len(plans_and_counts), 32)) as pool:
            futures = {
                pool.submit(_run_generator, gt, plan, n): gt
                for gt, plan, n in plans_and_counts
            }
            for future in as_completed(futures):
                gen_type, plan, results = future.result()
                category = self._build_category(plan)
                for task, pc in results:
                    k = _key(task)
                    if k not in seen_keys:
                        seen_keys.add(k)
                        all_results.append((task, pc, category))

        # If global dedup reduced count below target, fill from random generators.
        # Bail out after a bounded number of non-productive rounds so a generator
        # whose unique-question space is genuinely exhausted cannot spin forever.
        max_empty_rounds = 20
        empty_rounds = 0
        while len(all_results) < num_tasks and empty_rounds < max_empty_rounds:
            before = len(all_results)
            fill_type = gen_types[rng.randint(len(gen_types))]
            variants = self.GENERATOR_VARIANTS[fill_type]
            config = variants[rng.randint(len(variants))]
            num_options = task_plan.resolve_num_options(rng)
            fill_plan = TaskPlan(
                generator_type=fill_type,
                num_options=num_options,
                seed=task_plan.seed + 100003 * (empty_rounds + 1) + len(all_results),
                generator_config=config,
            )
            generator = self.generators[fill_type]
            results = generator.generate_tasks(fill_plan, num_tasks - len(all_results))
            category = self._build_category(fill_plan)
            for task, pc in results:
                k = _key(task)
                if k not in seen_keys:
                    seen_keys.add(k)
                    all_results.append((task, pc, category))
                    if len(all_results) >= num_tasks:
                        break
            empty_rounds = 0 if len(all_results) > before else empty_rounds + 1

        if len(all_results) < num_tasks:
            print(f"  NOTE: produced {len(all_results)}/{num_tasks} unique tasks "
                  f"(unique-question space exhausted for this seed/config)")

        all_results = all_results[:num_tasks]

        # Shuffle order
        rng.shuffle(all_results)

        # Balance answer positions (grouped by num_options)
        self._balance_positions_inplace(all_results, rng)

        # Save
        os.makedirs(output_dir, exist_ok=True)
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        task_records = []
        all_scene_metadata = []

        for i, (task, point_cloud, category) in enumerate(all_results):
            task.point = f"{i:08d}.npy"
            if num_points is not None and len(point_cloud) > num_points:
                idx = sampler_rng.choice(len(point_cloud), size=num_points, replace=False)
                point_cloud = point_cloud[idx]
            np.save(os.path.join(pcd_dir, task.point), point_cloud)

            if task.metadata:
                layout_template = task.metadata.get("layout_description", "")
                layout_description = layout_template
                for obj_info in task.metadata.get("objects", []):
                    placeholder = "[" + obj_info["placeholder"] + "]"
                    layout_description = layout_description.replace(
                        placeholder, obj_info["name"]
                    )

                # Persist EVERY field needed to deterministically rebuild the
                # scene point cloud via materialize_point_cloud():
                # generator_type, generator_config, layout_id, background_id,
                # and the placeholder->object_id mapping.
                all_scene_metadata.append({
                    "scene_id": i,
                    "point_cloud": task.point,
                    "generator_type": task.metadata.get("generator_type"),
                    "generator_config": task.metadata.get("generator_config"),
                    "layout_id": task.metadata.get("layout_id"),
                    "background_id": task.metadata.get("background_id"),
                    "num_points_saved": len(point_cloud),
                    "layout_template": layout_template,
                    "layout_description": layout_description,
                    "objects": {
                        "count": len(task.metadata.get("objects", [])),
                        "details": task.metadata.get("objects", [])
                    }
                })

            task_records.append({
                "question_id": i,
                "scene_id": i,
                "point": task.point,
                "category": category,
                "question": task.question,
                "options": task.options,
                "answer": task.answer,
            })

        if all_scene_metadata:
            with open(os.path.join(output_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
                for rec in all_scene_metadata:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        tasks_file = os.path.join(output_dir, "tasks.jsonl")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for record in task_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        from collections import Counter
        category_counts = dict(Counter(r["category"] for r in task_records))

        task_info = {
            "task_plan": {
                "generator_type": task_plan.generator_type,
                "num_options": task_plan.num_options,
                "seed": task_plan.seed,
                "generator_config": task_plan.generator_config,
            },
            "generation_stats": {
                "num_tasks_requested": num_tasks,
                "num_tasks_generated": len(task_records),
                "output_directory": output_dir,
                "category_distribution": category_counts,
            }
        }

        with open(os.path.join(output_dir, "tasks_info.json"), 'w', encoding='utf-8') as f:
            json.dump(task_info, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(task_records)} tasks:")
        print(f"  Tasks file: {tasks_file}")
        print(f"  Point clouds: {pcd_dir}")
        if len(category_counts) > 1:
            print(f"  Categories: {len(category_counts)}")

        return task_info

    def _balance_positions_inplace(
        self,
        task_results: List[Tuple[Task, np.ndarray, str]],
        rng: np.random.RandomState,
    ) -> None:
        """Balance answer positions grouped by num_options."""
        from collections import defaultdict

        groups: Dict[int, List[int]] = defaultdict(list)
        for i, (task, _, _) in enumerate(task_results):
            groups[len(task.options)].append(i)

        for num_options, indices in groups.items():
            n = len(indices)
            positions = []
            for pos in range(num_options):
                count = n // num_options + (1 if pos < n % num_options else 0)
                positions.extend([pos] * count)
            rng.shuffle(positions)

            for idx, pos in zip(indices, positions):
                task = task_results[idx][0]
                task.options = self._place_answer_at_position(
                    rng, task.answer, task.options, pos
                )

    @staticmethod
    def _build_category(task_plan: TaskPlan) -> str:
        """Build a human-readable category string from the task plan."""
        parts = [task_plan.generator_type]
        for key in ("distance_type", "frequency_type", "size_type", "reference_mode"):
            val = task_plan.generator_config.get(key)
            if val:
                parts.append(val)
        return "_".join(parts)

    def _balance_answer_positions(
        self,
        task_results: List[Tuple[Task, np.ndarray]],
        task_plan: TaskPlan,
    ) -> List[Tuple[Task, np.ndarray]]:
        """Ensure correct answer appears at each position equally often."""
        num_tasks = len(task_results)
        num_options = task_plan.num_options

        # Build a perfectly balanced position assignment
        positions = []
        for pos in range(num_options):
            count = num_tasks // num_options + (1 if pos < num_tasks % num_options else 0)
            positions.extend([pos] * count)
        self.rng.shuffle(positions)

        result = []
        for (task, pc), pos in zip(task_results, positions):
            task.options = self._place_answer_at_position(
                self.rng, task.answer, task.options, pos
            )
            result.append((task, pc))
        return result

    @staticmethod
    def _place_answer_at_position(
        rng: np.random.RandomState,
        correct_answer: str,
        options: List[str],
        target_pos: int,
    ) -> List[str]:
        """Rearrange options so correct_answer sits at target_pos."""
        distractors = [o for o in options if o != correct_answer]
        rng.shuffle(distractors)
        return distractors[:target_pos] + [correct_answer] + distractors[target_pos:]
