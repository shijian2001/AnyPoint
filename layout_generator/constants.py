"""Constants for layout generation: size mappings, spatial relations, and constraints."""

from typing import Dict, Tuple, List
import numpy as np

# Size category mappings to AABB half-extents ranges (min, max) for each dimension
# Each size category returns (min_half_extents, max_half_extents) where each is (x, y, z)
# Note: These ranges match the previous sphere radius ranges for consistency
SIZE_RANGES: Dict[str, Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = {
    "largest": ((2.0, 2.0, 2.0), (2.5, 2.5, 2.5)),
    "large": ((1.5, 1.5, 1.5), (2.0, 2.0, 2.0)),
    "medium": ((1.0, 1.0, 1.0), (1.5, 1.5, 1.5)),
    "small": ((0.6, 0.6, 0.6), (1.0, 1.0, 1.0)),
    "smallest": ((0.3, 0.3, 0.3), (0.6, 0.6, 0.6)),
}

# All valid size categories
VALID_SIZES: List[str] = list(SIZE_RANGES.keys())

# Valid spatial relations (organized by type)
VALID_RELATIONS: List[str] = [
    # Horizontal directional (precise direction)
    "in front of", "behind", "to the left of", "to the right of",
    # Horizontal proximity (radial)
    "beside", "next to", "near", "far from",
    # Vertical
    "on", "above", "below", "under",
    # Special
    "surrounding", "at the center of",
]

# Scene configuration
MIN_OBJECTS: int = 2
MAX_OBJECTS: int = 9
SCENE_BOUNDS: float = 8.0  # Scene bounding box half-size (larger → more success)

# Physical constraints
MIN_SEPARATION: float = 0.05  # Minimum gap between objects (5cm, more tolerant)
GROUND_TOLERANCE: float = 0.05  # Ground contact tolerance

# Relation-specific distance (multiplier of sum of radii)
# Key insight: Larger ranges → less collision → higher success rate
RELATION_DISTANCE: Dict[str, Tuple[float, float]] = {
    "in front of": (1.2, 2.0),
    "behind": (1.2, 2.0),
    "to the left of": (1.2, 2.0),
    "to the right of": (1.2, 2.0),
    "beside": (1.2, 2.0),
    "next to": (1.1, 1.8),
    "near": (1.5, 2.5),
    "far from": (3.0, 4.5),
    "on": (1.0, 1.0),  # Contact
    "above": (1.5, 2.5),
    "below": (1.5, 2.5),
    "under": (1.2, 2.0),
    "surrounding": (2.0, 3.0),
    "at the center of": (0.0, 0.0),
}

# Direction vectors for horizontal relations (x, z plane)
RELATION_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "in front of": (0.0, 1.0),   # +z direction
    "behind": (0.0, -1.0),      # -z direction
    "to the left of": (-1.0, 0.0),  # -x direction
    "to the right of": (1.0, 0.0),   # +x direction
}

