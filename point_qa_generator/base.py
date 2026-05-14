from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
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
    """Task plan configuration."""
    generator_type: str
    num_options: int = 4  # Number of multiple choice options (2-6)
    seed: int = 42
    generator_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate task plan parameters."""
        if not 2 <= self.num_options <= 6:
            raise ValueError("num_options must be between 2 and 6")


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
        
        # Support both dict (classified) and list (raw) layouts
        if isinstance(layouts, dict):
            self.layouts = layouts.get('all', [])
            self.layouts_by_type = layouts
        else:
            self.layouts = layouts or []
            self.layouts_by_type = None
        
        # Validate metadata
        if not self.metadata.objects:
            raise ValueError("Metadata contains no objects")

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
    
    def _sample_layout_and_map_objects(self, min_objects: int = 3) -> Tuple[Dict, Dict[str, Dict]]:
        """Sample a layout and map placeholders to actual objects.
        
        Args:
            min_objects: Minimum number of objects required in layout
            
        Returns:
            Tuple of (layout, object_mapping)
        """
        if not self.layouts:
            raise ValueError("No layouts available. Initialize generator with layouts.")
        
        # Filter layouts by minimum object count
        valid_layouts = [l for l in self.layouts if len(l["objects"]) >= min_objects]
        if not valid_layouts:
            raise ValueError(f"No layouts with at least {min_objects} objects found.")
        
        layout = self.rng.choice(valid_layouts)
        
        # Map placeholders to actual objects
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
        object_mapping: Dict[str, Dict]
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
        background = self._load_background(layout, object_mapping)
        support_y = get_support_height(background)
        
        for obj_spec in layout["objects"]:
            obj_name = obj_spec["name"]
            actual_obj = object_mapping[obj_name]

            pcd = self.metadata.load_point_cloud(actual_obj["object_id"])
            point_clouds.append(transform_object_point_cloud(pcd, obj_spec, support_y=support_y))

        if background is not None:
            point_clouds.insert(0, background)

        return np.vstack(point_clouds)

    def _load_background(self, layout: Dict, object_mapping: Dict[str, Dict]) -> Optional[np.ndarray]:
        if not self.background_files:
            return None

        descriptor = {
            "layout_id": layout.get("id"),
            "objects": sorted(
                (placeholder, actual_obj["object_id"])
                for placeholder, actual_obj in object_mapping.items()
            ),
        }
        descriptor_json = json.dumps(descriptor, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
        bg_index = int(digest[:16], 16) % len(self.background_files)
        background = np.load(self.background_files[bg_index]).astype(np.float32)
        return fit_background_to_layout(background, layout)

    def _compose_options(
        self, 
        correct_answer: str, 
        candidates: List[str],
        num_options: int
    ) -> List[str]:
        """Compose shuffled multiple choice options.
        
        Args:
            correct_answer: The correct answer string
            candidates: List of candidate answers (distractors)
            num_options: Total number of options to generate
            
        Returns:
            Shuffled list of options (correct answer at random position)
        """
        num_distractors = num_options - 1
        distractors = self.rng.choice(candidates, size=num_distractors, replace=False).tolist()
        
        # Combine and shuffle all options
        all_options = [correct_answer] + distractors
        self.rng.shuffle(all_options)
        
        return all_options
