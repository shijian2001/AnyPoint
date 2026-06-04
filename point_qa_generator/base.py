from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import os
import numpy as np
from .metadata import PointCloudMetadata
from .scene_builder import (
    fit_background_to_layout,
    get_support_height,
    transform_object_point_cloud,
)


@dataclass
class TaskPlan:
    """Task plan configuration.

    Supports single or multiple generator types with optional weights,
    and fixed or range-based num_options.

    Examples:
        # Single type, fixed options
        TaskPlan(generator_type="what_distance", num_options=4,
                 generator_config={"distance_type": "closest"})

        # Multiple types equally weighted, random 4-6 options
        TaskPlan(generator_type=["what_distance", "what_size", "what_relation"],
                 num_options=(4, 6))

        # Multiple types with weights
        TaskPlan(generator_type={"what_distance": 0.5, "what_relation": 0.3, "multi_hop_relation": 0.2},
                 num_options=(4, 6))
    """
    generator_type: Any  # str, List[str], or Dict[str, float]
    num_options: Any = 4  # int or Tuple[int, int] for range (inclusive)
    seed: int = 42
    generator_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.num_options, int):
            if not 2 <= self.num_options <= 6:
                raise ValueError("num_options must be between 2 and 6")
        elif isinstance(self.num_options, (tuple, list)):
            lo, hi = self.num_options
            if not (2 <= lo <= hi <= 6):
                raise ValueError("num_options range must be within [2, 6]")
        else:
            raise ValueError("num_options must be int or (min, max) tuple")

    @property
    def is_mixed(self) -> bool:
        return isinstance(self.generator_type, (list, dict))

    def resolve_num_options(self, rng) -> int:
        if isinstance(self.num_options, int):
            return self.num_options
        lo, hi = self.num_options
        return int(rng.randint(lo, hi + 1))

    def resolve_generator_types(self) -> List[str]:
        if isinstance(self.generator_type, str):
            return [self.generator_type]
        elif isinstance(self.generator_type, list):
            return self.generator_type
        elif isinstance(self.generator_type, dict):
            return list(self.generator_type.keys())
        return [self.generator_type]

    def resolve_weights(self) -> Optional[Dict[str, float]]:
        if isinstance(self.generator_type, dict):
            total = sum(self.generator_type.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"weights must sum to 1.0, got {total}")
            return self.generator_type
        return None


@dataclass
class Task:
    """Generated task data structure."""
    point: str  # Point cloud filename
    question: str
    options: List[str]  # ["option1", "option2", ...] (shuffled)
    answer: str  # The correct answer content
    metadata: Optional[Dict[str, Any]] = None  # Scene metadata (layout, objects)


class BasePointQAGenerator(ABC):
    """Base class for point cloud QA generators."""

    def __init__(
        self,
        metadata: PointCloudMetadata,
        seed: int = 42,
        layouts=None,
        background_dir: Optional[str] = None,
    ):
        self.metadata = metadata
        self.rng = np.random.RandomState(seed)
        self.background_dir = background_dir
        self.background_files = self._scan_backgrounds(background_dir)
        # Records the background filename (or None) used by the most recent
        # _create_point_cloud_from_layout call, so generators can persist it
        # into task metadata for deterministic reproduction.
        self._last_background_id = None
        self.layouts = layouts or []
        self._layouts_by_min_count = self._build_layout_index(self.layouts)

        if not self.metadata.objects:
            raise ValueError("Metadata contains no objects")

    @staticmethod
    def _build_layout_index(layouts: List[Dict]) -> Dict[int, List[Dict]]:
        """Pre-group layouts by minimum object count for O(1) lookup."""
        index: Dict[int, List[Dict]] = {}
        for n in range(1, 10):
            index[n] = [l for l in layouts if len(l["objects"]) >= n]
        return index

    @staticmethod
    def _scan_backgrounds(background_dir: Optional[str]) -> List[str]:
        if not background_dir or not os.path.isdir(background_dir):
            return []

        background_files = []
        for name in sorted(os.listdir(background_dir)):
            if name.endswith(".npy"):
                background_files.append(os.path.join(background_dir, name))
        return background_files

    @abstractmethod
    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate generator-specific configuration."""
        pass

    @abstractmethod
    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate specified number of tasks."""
        pass

    def _max_generation_attempts(self, num_tasks: int) -> int:
        return max(1000, num_tasks * 200)

    def _raise_generation_failure(self, generator_name: str, num_tasks: int, generated: int, attempts: int) -> None:
        raise RuntimeError(
            f"{generator_name}: generated {generated}/{num_tasks} after {attempts} attempts. "
            f"Check that metadata and layouts are sufficient."
        )
    
    def _sample_layout_and_map_objects(self, min_objects: int = 3) -> Tuple[Dict, Dict[str, Dict]]:
        """Sample a layout and map placeholders to actual objects."""
        if not self.layouts:
            raise ValueError("No layouts available. Initialize generator with layouts.")

        valid_layouts = self._layouts_by_min_count.get(min_objects, [])
        if not valid_layouts:
            raise ValueError(f"No layouts with at least {min_objects} objects found.")

        layout = self.rng.choice(valid_layouts)

        num_objects = len(layout["objects"])
        sampled_objects = self.rng.choice(self.metadata.objects, size=num_objects, replace=False)

        object_mapping = {
            layout["objects"][i]["name"]: sampled_objects[i]
            for i in range(num_objects)
        }

        return layout, object_mapping

    def _create_point_cloud_from_layout(
        self,
        layout: Dict,
        object_mapping: Dict[str, Dict],
        background_id: Optional[str] = "__random__",
    ) -> np.ndarray:
        """Create point cloud scene from layout and object mapping.
        
        Transforms each object according to layout specification:
        1. Load normalized point cloud ([-0.5, 0.5]^3)
        2. Scale by size * 2 (size is half-extents)
        3. Rotate around Y-axis
        4. Translate to position
        
        Args:
            layout: Layout dict with 'objects' list containing position/rotation/size
            object_mapping: Maps placeholder names (obj_0) to actual objects
            
        Returns:
            Combined scene point cloud (N, 3+)
        """
        point_clouds = []
        background = self._load_background(layout, object_mapping, background_id=background_id)
        support_y = get_support_height(background)

        for obj_spec in layout["objects"]:
            obj_name = obj_spec["name"]
            actual_obj = object_mapping[obj_name]

            pcd = self.metadata.load_point_cloud(actual_obj["object_id"])
            point_clouds.append(transform_object_point_cloud(pcd, obj_spec, support_y=support_y))

        if background is not None:
            point_clouds.insert(0, background)

        return np.vstack(point_clouds)

    def _load_background(
        self,
        layout: Dict,
        object_mapping: Dict[str, Dict],
        background_id: Optional[str] = "__random__",
    ) -> Optional[np.ndarray]:
        # background_id controls which background is used so scenes can be
        # rebuilt deterministically from metadata:
        #   "__random__" -> pick randomly (generation time); records the choice
        #                   in self._last_background_id for the caller to persist
        #   None         -> no background
        #   "<filename>" -> use that specific background file (reproduction)
        if background_id == "__random__":
            if not self.background_files:
                self._last_background_id = None
                return None
            # Randomly pick from backgrounds + "no background" as an equal option
            choice = self.rng.randint(len(self.background_files) + 1)
            if choice == len(self.background_files):
                self._last_background_id = None
                return None
            path = self.background_files[choice]
            self._last_background_id = os.path.basename(path)
            background = np.load(path).astype(np.float32)
            return fit_background_to_layout(background, layout)

        # Deterministic path: explicit id (or None)
        self._last_background_id = background_id
        if background_id is None:
            return None
        path = next((p for p in self.background_files
                     if os.path.basename(p) == background_id), None)
        if path is None:
            raise FileNotFoundError(f"Background not found for reproduction: {background_id}")
        background = np.load(path).astype(np.float32)
        return fit_background_to_layout(background, layout)

    def _compose_options(
        self,
        correct_answer: str,
        candidates: List[str],
        num_options: int
    ) -> List[str]:
        """Compose shuffled multiple choice options.

        Filters out any candidate equal to correct_answer before sampling.
        """
        filtered = [c for c in candidates if c != correct_answer]
        num_distractors = num_options - 1
        if len(filtered) < num_distractors:
            raise ValueError(
                f"Not enough distractors: need {num_distractors}, got {len(filtered)}"
            )
        distractors = self.rng.choice(filtered, size=num_distractors, replace=False).tolist()

        all_options = [correct_answer] + distractors
        self.rng.shuffle(all_options)
        return all_options

    def _compose_list_distractors(
        self,
        correct_values: Set[str],
        all_values: List[str],
        num_distractors: int,
    ) -> List[str]:
        """Generate list-type distractors that differ from correct by 1-2 elements."""
        candidates: set = set()
        correct_sorted = ", ".join(sorted(correct_values))
        other_values = [v for v in all_values if v not in correct_values]

        if not other_values:
            return []

        # Strategy 1: Replace one element
        for val in sorted(correct_values):
            replacements = self.rng.choice(
                other_values, size=min(3, len(other_values)), replace=False
            )
            for rep in replacements:
                new_set = (correct_values - {val}) | {rep}
                candidates.add(", ".join(sorted(new_set)))
                if len(candidates) >= num_distractors * 2:
                    break
            if len(candidates) >= num_distractors * 2:
                break

        # Strategy 2: Add one element
        if len(candidates) < num_distractors:
            extras = self.rng.choice(
                other_values, size=min(3, len(other_values)), replace=False
            )
            for extra in extras:
                candidates.add(", ".join(sorted(correct_values | {extra})))

        # Strategy 3: Remove one element (if correct has >1 values)
        if len(candidates) < num_distractors and len(correct_values) > 1:
            for val in sorted(correct_values):
                subset = correct_values - {val}
                if subset:
                    candidates.add(", ".join(sorted(subset)))

        candidates.discard(correct_sorted)
        result = list(candidates)
        if len(result) > num_distractors:
            result = self.rng.choice(result, size=num_distractors, replace=False).tolist()
        return result

    def _compose_count_distractors(
        self,
        correct_count: int,
        scene_counts: List[int],
        num_distractors: int,
    ) -> List[str]:
        """Generate count distractors: prefer real scene counts, fallback +-1/2.

        Never produces 0 as a distractor since all count generators
        require at least 1 matching attribute value to produce a task.
        """
        candidates: set = set()

        for c in scene_counts:
            if c != correct_count and c >= 1:
                candidates.add(str(c))

        for offset in [1, -1, 2, -2]:
            if len(candidates) >= num_distractors:
                break
            val = correct_count + offset
            if val >= 1 and str(val) not in candidates:
                candidates.add(str(val))

        return list(candidates)[:num_distractors]
