from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from .metadata import PointCloudMetadata
from .utils import GRID_COORDINATES, rotate_x_axis, center_and_scale_point_cloud, translate_point_cloud


@dataclass
class TaskPlan:
    """Task plan configuration."""
    generator_type: str
    ## Note: options != correct answer + num_scene_distractors
    num_options: int = 4  # Number of multiple choice options (2-6)
    num_scene_distractors: int = 2  # Number of additional objects in scene (0-7)
    seed: int = 42
    generator_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task plan parameters."""
        if not 2 <= self.num_options <= 6:
            raise ValueError("num_options must be between 2 and 6")
        if not 0 <= self.num_scene_distractors <= 7:
            raise ValueError("num_scene_distractors must be between 0 and 7 (grid system limitation)")


@dataclass
class Task:
    """Generated task data structure."""
    point: str  # Point cloud filename
    question: str
    options: List[str]  # ["A. option1", "B. option2", ...]
    answer: str  # The correct answer content
    answer_id: str  # The correct answer ID (A/B/C/...)


class BasePointQAGenerator(ABC):
    """Base class for point cloud QA generators."""

    def __init__(self, metadata: PointCloudMetadata, seed: int = 42):
        self.metadata = metadata
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def validate_generator_config(self, config: Dict[str, Any]) -> None:
        """Validate generator-specific configuration."""
        pass

    @abstractmethod
    def count_possible_tasks(self, task_plan: TaskPlan) -> int:
        """Count total number of possible tasks for given task plan."""
        pass

    @abstractmethod
    def generate_tasks(self, task_plan: TaskPlan, num_tasks: int) -> List[Tuple[Task, np.ndarray]]:
        """Generate specified number of tasks."""
        pass

    def _create_point_cloud_scene(self, objects: List[Dict], grids: List[int],
                                  angles: List[float]) -> np.ndarray:
        """Create combined point cloud scene from multiple objects."""
        point_clouds = []

        ## TODO: more complex design
        scale_factor = 0.5

        for obj, grid, angle in zip(objects, grids, angles):
            pcd = self.metadata.load_point_cloud(obj["object_id"])
            pcd = rotate_x_axis(pcd, angle)
            pcd = center_and_scale_point_cloud(pcd, scale_factor)
            x_offset, y_offset = GRID_COORDINATES[grid]
            pcd = translate_point_cloud(pcd, x_offset, y_offset)
            point_clouds.append(pcd)

        return np.vstack(point_clouds)

    def _compose_options(self, correct_answer: str, candidates: List[str],
                         num_options: int) -> Tuple[List[str], str]:
        """Compose multiple choice options and determine answer ID."""
        num_distractors = num_options - 1

        unique_candidates = list(set(candidates))

        if len(unique_candidates) < num_distractors:
            raise ValueError(f"Not enough unique candidates: need {num_distractors}, got {len(unique_candidates)}. "
                             f"This indicates a generator bug or insufficient metadata diversity.")

        distractors = self.rng.choice(unique_candidates, size=num_distractors, replace=False).tolist()

        all_options = [correct_answer] + distractors
        option_labels = ['A', 'B', 'C', 'D', 'E', 'F'][:len(all_options)]

        # Shuffle positions but maintain A,B,C,D labels
        indices = list(range(len(all_options)))
        self.rng.shuffle(indices)
        answer_id = option_labels[indices.index(0)]

        formatted_options = []
        for i, label in enumerate(option_labels):
            option_content = all_options[indices[i]]
            formatted_options.append(f"{label}. {option_content}")

        return formatted_options, answer_id
