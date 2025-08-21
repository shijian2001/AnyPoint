import json
import os
from typing import List, Dict, Set, Any
import numpy as np


class PointCloudMetadata:
    """Handles point cloud metadata."""

    def __init__(self, jsonl_file: str, pcd_dir: str, seed: int = 42):
        """
        Initialize metadata.

        Args:
            jsonl_file: Path to JSONL metadata file
            pcd_dir: Directory containing point cloud .npy files
            seed: Random seed
        """
        self.jsonl_file = jsonl_file
        self.pcd_dir = pcd_dir
        self.rng = np.random.RandomState(seed)
        self.attributes_file = "./data/metadata/attributes.json"
        self._load_metadata()
        self._load_or_create_attributes()

    def _load_metadata(self) -> None:
        """Load metadata from JSONL file."""
        self.objects = []
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    for object_id, obj_data in data.items():
                        self.objects.append({
                            "object_id": object_id,
                            "object_name": obj_data.get("object", ""),
                            "components": obj_data.get("components", [])
                        })

    def _load_or_create_attributes(self) -> None:
        """Load or create attributes summary file."""
        if os.path.exists(self.attributes_file):
            with open(self.attributes_file, 'r', encoding='utf-8') as f:
                self.attributes = json.load(f)
        else:
            self._create_attributes_summary()

    def _create_attributes_summary(self) -> None:
        """Create attributes summary from all objects."""
        materials = set()
        colors = set()
        shapes = set()
        textures = set()

        for obj in self.objects:
            for component in obj.get("components", []):
                if component.get("material"):
                    materials.add(component["material"])
                if component.get("color"):
                    colors.add(component["color"])
                if component.get("shape"):
                    shapes.add(component["shape"])
                if component.get("texture"):
                    textures.add(component["texture"])

        self.attributes = {
            "material": sorted(list(materials)),
            "color": sorted(list(colors)),
            "shape": sorted(list(shapes)),
            "texture": sorted(list(textures))
        }

        os.makedirs(os.path.dirname(self.attributes_file), exist_ok=True)
        with open(self.attributes_file, 'w', encoding='utf-8') as f:
            json.dump(self.attributes, f, indent=2, ensure_ascii=False)

    def sample_objects(self, num_samples: int) -> List[Dict[str, Any]]:
        """Sample random objects from metadata."""
        if num_samples > len(self.objects):
            raise ValueError(f"Requested {num_samples} samples but only {len(self.objects)} available")
        return self.rng.choice(self.objects, size=num_samples, replace=False).tolist()

    def load_point_cloud(self, object_id: str) -> np.ndarray:
        """Load point cloud for given object ID."""
        file_path = os.path.join(self.pcd_dir, f"{object_id}.npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        return np.load(file_path)

    def sample_angle(self) -> float:
        """Sample random rotation angle."""
        return self.rng.uniform(0, 360)

    def get_attribute_values(self, attribute: str) -> List[str]:
        """Get all possible values for an attribute."""
        return self.attributes.get(attribute, [])

    def get_object_components_with_attribute(self, obj: Dict, attribute: str) -> List[Dict]:
        """Get components that have non-empty value for the attribute."""
        components = []
        for component in obj.get("components", []):
            if component.get(attribute):
                components.append(component)
        return components

    def has_components_with_attribute(self, obj: Dict, attribute: str) -> bool:
        """Check if object has any components with non-empty attribute."""
        return len(self.get_object_components_with_attribute(obj, attribute)) > 0