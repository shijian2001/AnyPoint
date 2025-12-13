"""
Constraint handling with dimensional decomposition.

Core idea: Decompose 3D constraints into independent dimensions (Y, XZ).
Algorithm: Sample from constrained regions using rejection sampling.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .constants import RELATION_DIRECTIONS, RELATION_DISTANCE


@dataclass
class PlacedObject:
    """Placed object with position and AABB size."""
    name: str
    position: np.ndarray  # (x, y, z) - center position
    half_extents: np.ndarray  # (x, y, z) - half size in each dimension
    rotation: float


class ConstraintHandler:
    """
    Constraint handling: Decompose R³ → R(Y) × R²(XZ), sample independently.
    """
    
    VERTICAL_RELATIONS = {"on", "above", "below", "under"}
    HORIZONTAL_RELATIONS = set(RELATION_DIRECTIONS.keys())
    RADIAL_RELATIONS = {"beside", "next to", "near", "far from", "surrounding", "at the center of"}
    
    def __init__(self, rng: np.random.RandomState, jitter_scale: float = 0.15):
        self.rng = rng
        self.jitter_scale = jitter_scale
    
    def decompose_constraints(self, constraints: List) -> Tuple[Optional[object], Optional[object]]:
        """Decompose into vertical (Y) and horizontal (XZ) constraints."""
        v_constraint = None
        h_constraint = None
        
        for c in constraints:
            if not v_constraint and c.relation in self.VERTICAL_RELATIONS:
                v_constraint = c
            if not h_constraint and (c.relation in self.HORIZONTAL_RELATIONS or 
                                     c.relation in self.RADIAL_RELATIONS):
                h_constraint = c
        
        return v_constraint, h_constraint
    
    def sample_position(self, constraints: List, placed_objects: dict, half_extents: np.ndarray) -> Optional[np.ndarray]:
        """Sample position: decompose constraints → sample each dimension."""
        if not constraints:
            return None
        
        v_constraint, h_constraint = self.decompose_constraints(constraints)
        
        # Sample Y (vertical)
        if v_constraint:
            ref = placed_objects.get(v_constraint.reference)
            if not ref:
                return None
            y = self._sample_vertical(ref, half_extents, v_constraint.relation)
        else:
            y = half_extents[1]  # Ground - place at bottom of AABB
        
        # Sample XZ (horizontal)
        if h_constraint:
            ref = placed_objects.get(h_constraint.reference)
            if not ref:
                return None
            if h_constraint.relation in self.HORIZONTAL_RELATIONS:
                x, z = self._sample_horizontal(ref, half_extents, h_constraint.relation)
            else:
                x, z = self._sample_radial(ref, half_extents, h_constraint.relation)
        elif v_constraint:
            ref = placed_objects.get(v_constraint.reference)
            
            # Handle different vertical relations
            if v_constraint.relation == "on":
                # "on": sample within ref's footprint
                ref_footprint = max(ref.half_extents[0], ref.half_extents[2])
                obj_footprint = max(half_extents[0], half_extents[2])
                max_offset = max(0, ref_footprint - obj_footprint) * 0.8
                if max_offset > 0:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    offset = self.rng.uniform(0, max_offset)
                    x = ref.position[0] + np.cos(angle) * offset
                    z = ref.position[2] + np.sin(angle) * offset
                else:
                    jitter = self.rng.uniform(-1, 1, size=2) * obj_footprint * 0.2
                    x, z = ref.position[0] + jitter[0], ref.position[2] + jitter[1]
            elif v_constraint.relation in {"below", "under"}:
                # "below"/"under": sample nearby on ground, not same position!
                combined = np.linalg.norm(half_extents[[0, 2]]) + np.linalg.norm(ref.half_extents[[0, 2]])
                angle = self.rng.uniform(0, 2 * np.pi)
                distance = self.rng.uniform(1.2 * combined, 2.0 * combined)
                x = ref.position[0] + np.cos(angle) * distance
                z = ref.position[2] + np.sin(angle) * distance
            else:
                # "above": use ref position with jitter
                jitter = self.rng.uniform(-1, 1, size=2) * max(ref.half_extents[0], ref.half_extents[2]) * 0.3
                x, z = ref.position[0] + jitter[0], ref.position[2] + jitter[1]
        else:
            # Fallback
            ref = placed_objects.get(constraints[0].reference)
            if not ref:
                return None
            x, z = ref.position[0], ref.position[2]
        
        return np.array([x, y, z])
    
    def _sample_vertical(self, ref: PlacedObject, half_extents: np.ndarray, relation: str) -> float:
        """Sample Y: exact for 'on', range for others."""
        if relation == "on":
            # Exact contact (no jitter in Y, handled by XZ variation)
            return ref.position[1] + ref.half_extents[1] + half_extents[1]
        
        # Distance-based: sample from range
        combined_height = half_extents[1] + ref.half_extents[1]
        dist_min, dist_max = RELATION_DISTANCE.get(relation, (1.2, 2.0))
        distance = self.rng.uniform(dist_min * combined_height, dist_max * combined_height)
        
        if relation == "above":
            return ref.position[1] + distance
        else:  # below, under
            # Ensure object is on ground, not floating
            # "below" means below in space, so just place on ground
            return half_extents[1]
    
    def _sample_horizontal(self, ref: PlacedObject, half_extents: np.ndarray, relation: str) -> Tuple[float, float]:
        """Sample XZ: directional offset."""
        # Use horizontal extent (max of x and z) for distance calculation
        obj_horiz = max(half_extents[0], half_extents[2])
        ref_horiz = max(ref.half_extents[0], ref.half_extents[2])
        combined_extent = obj_horiz + ref_horiz
        dist_min, dist_max = RELATION_DISTANCE.get(relation, (1.2, 1.8))
        distance = self.rng.uniform(dist_min * combined_extent, dist_max * combined_extent)
        
        dx, dz = RELATION_DIRECTIONS[relation]
        return ref.position[0] + dx * distance, ref.position[2] + dz * distance
    
    def _sample_radial(self, ref: PlacedObject, half_extents: np.ndarray, relation: str) -> Tuple[float, float]:
        """Sample XZ: random angle."""
        if relation == "at the center of":
            # True center: use ref position with larger jitter for collision avoidance
            obj_horiz = max(half_extents[0], half_extents[2])
            ref_horiz = max(ref.half_extents[0], ref.half_extents[2])
            jitter = self.rng.uniform(-1, 1, size=2) * (obj_horiz + ref_horiz) * 0.2
            return ref.position[0] + jitter[0], ref.position[2] + jitter[1]
        
        # Radial: uniform angle
        obj_horiz = max(half_extents[0], half_extents[2])
        ref_horiz = max(ref.half_extents[0], ref.half_extents[2])
        combined_extent = obj_horiz + ref_horiz
        dist_min, dist_max = RELATION_DISTANCE.get(relation, (1.2, 1.8))
        distance = self.rng.uniform(dist_min * combined_extent, dist_max * combined_extent)
        angle = self.rng.uniform(0, 2 * np.pi)
        
        return (ref.position[0] + np.cos(angle) * distance,
                ref.position[2] + np.sin(angle) * distance)
