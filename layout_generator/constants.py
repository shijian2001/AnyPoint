"""Constants for layout generation: size mappings, spatial relations, and constraints."""

from typing import Dict, Tuple, List

# Size category mappings to scale ranges (min, max) relative to unit sphere
SIZE_RANGES: Dict[str, Tuple[float, float]] = {
    "largest": (2.0, 2.5),
    "large": (1.5, 2.0),
    "medium": (1.0, 1.5),
    "small": (0.6, 1.0),
    "smallest": (0.3, 0.6),
}

# All valid size categories
VALID_SIZES: List[str] = list(SIZE_RANGES.keys())

# Spatial relations with solver support
# Format: relation_name -> (is_horizontal, requires_contact)
SPATIAL_RELATIONS: Dict[str, Tuple[bool, bool]] = {
    # Horizontal planar relations
    "in front of": (True, False),
    "behind": (True, False),
    "to the left of": (True, False),
    "to the right of": (True, False),
    "beside": (True, False),
    "next to": (True, False),
    "near": (True, False),
    "far from": (True, False),
    # Vertical relations
    "on": (False, True),  # Subject on top of reference, requires contact
    "above": (False, False),
    "below": (False, False),
    "under": (False, False),
    # Compound relations
    "surrounding": (True, False),
    "at the center of": (True, False),
}

VALID_RELATIONS: List[str] = list(SPATIAL_RELATIONS.keys())

# Scene constraints
MIN_OBJECTS: int = 2
MAX_OBJECTS: int = 9
SCENE_BOUNDS: float = 5.0  # Scene bounding box half-size

# Solver parameters
MIN_SEPARATION: float = 0.1  # Minimum gap between object bounding spheres
MAX_SOLVER_ATTEMPTS: int = 100  # Max attempts per object placement
PERTURBATION_RANGE: float = 0.3  # Random position perturbation range

# Relation-specific distance parameters (as multiplier of sum of radii)
RELATION_DISTANCE: Dict[str, Tuple[float, float]] = {
    "in front of": (1.1, 1.5),
    "behind": (1.1, 1.5),
    "to the left of": (1.1, 1.5),
    "to the right of": (1.1, 1.5),
    "beside": (1.1, 1.5),
    "next to": (1.05, 1.3),
    "near": (1.2, 2.0),
    "far from": (3.0, 4.5),
    "on": (1.0, 1.0),  # Contact, will use height offset
    "above": (1.2, 2.0),
    "below": (1.2, 2.0),
    "under": (1.0, 1.5),
    "surrounding": (1.5, 2.5),
    "at the center of": (0.0, 0.0),  # Center position
}

# Direction vectors for horizontal relations (x, z plane)
RELATION_DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "in front of": (0.0, 1.0),   # +z direction
    "behind": (0.0, -1.0),      # -z direction
    "to the left of": (-1.0, 0.0),  # -x direction
    "to the right of": (1.0, 0.0),   # +x direction
}

