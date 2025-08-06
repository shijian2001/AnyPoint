import json
import os
from typing import List, Dict
import numpy as np


class PointCloudMetadata:
    """Handles point cloud metadata from JSON file."""

    def __init__(self, json_file: str, pcd_dir: str, seed: int = 42):
        """
        Initialize metadata.

        Args:
            json_file: Path to JSON metadata file
            pcd_dir: Directory containing point cloud .npy files
            seed: Random seed
        """
        self.json_file = json_file
        self.pcd_dir = pcd_dir
        self.rng = np.random.RandomState(seed)
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load metadata from JSON file."""
        self.objects = []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                object_id = item.get("object_id")
                object_name = item.get("object")
                if object_id and object_name:
                    self.objects.append({
                        "object_id": object_id,
                        "object_name": object_name
                    })

    def sample_objects(self, num_samples: int) -> List[Dict[str, str]]:
        """Sample random objects from metadata."""
        if num_samples > len(self.objects):
            raise ValueError(f"Requested {num_samples} samples but only {len(self.objects)} available")
        return self.rng.choice(self.objects, size=num_samples, replace=False).tolist()

    def load_point_cloud(self, object_id: str) -> np.ndarray:
        """Load point cloud for given object ID."""
        file_path = os.path.join(self.pcd_dir, f"{object_id}_8192.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        return np.load(file_path)

    def sample_angle(self) -> float:
        """Sample random rotation angle."""
        return self.rng.uniform(0, 360)