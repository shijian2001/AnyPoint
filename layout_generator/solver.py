"""
Layout solver using constraint-driven rejection sampling.

Core algorithm:
1. Topological sort → determine placement order
2. For each object: sample → validate → accept/reject
3. Repeat until success or max attempts reached
"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque

from .schema import DSL, ObjectSpec, RelationSpec, Layout, LayoutObject
from .constraints import ConstraintHandler, PlacedObject
from .constants import SIZE_RANGES, MIN_SEPARATION, SCENE_BOUNDS, GROUND_TOLERANCE


class SolverError(Exception):
    """Raised when solver fails to find valid layout."""
    pass


class LayoutSolver:
    """
    Layout solver via rejection sampling.
    
    Algorithm:
    1. Topological sort → placement order
    2. For each: sample → validate → accept/reject
    """
    
    def __init__(self, seed: Optional[int] = None, max_attempts: int = 1000):
        """
        Args:
            max_attempts: Maximum sampling attempts per object
        """
        self.rng = np.random.RandomState(seed)
        self.constraint_handler = ConstraintHandler(self.rng)
        self.placed: Dict[str, PlacedObject] = {}
        self.max_attempts = max_attempts
    
    def solve(self, dsl: DSL) -> Layout:
        """Solve DSL via rejection sampling."""
        self.placed = {}
        
        # Assign random sizes and determine placement order
        sizes = self._assign_sizes(dsl.objects)
        order = self._topological_sort(dsl.objects, dsl.relations)
        
        # Place objects one by one
        for name in order:
            obj = next(o for o in dsl.objects if o.name == name)
            half_extents = sizes[name]
            constraints = [r for r in dsl.relations if r.subject == name]
            
            # Rejection sampling: try until valid or max attempts
            position = self._place_object(half_extents, constraints)
            if position is None:
                raise SolverError(f"Failed to place '{name}'")
            
            self.placed[name] = PlacedObject(name, position, half_extents, obj.rotation)
        
        return self._build_layout(dsl)
    
    def _assign_sizes(self, objects: List[ObjectSpec]) -> Dict[str, np.ndarray]:
        """Randomly assign AABB sizes within category bounds."""
        sizes = {}
        for obj in objects:
            min_extents, max_extents = SIZE_RANGES[obj.size]
            # Randomly sample half-extents for each dimension
            half_extents = np.array([
                self.rng.uniform(min_extents[i], max_extents[i]) for i in range(3)
            ])
            sizes[obj.name] = half_extents
        return sizes
    
    def _topological_sort(self, objects: List[ObjectSpec], relations: List[RelationSpec]) -> List[str]:
        """
        Topological sort (Kahn's algorithm) to determine placement order.
        Objects with no dependencies are placed first.
        """
        # Build dependency graph
        in_degree = {o.name: 0 for o in objects}
        adj_list = {o.name: [] for o in objects}
        
        for rel in relations:
            if rel.reference in in_degree and rel.subject in in_degree:
                adj_list[rel.reference].append(rel.subject)
                in_degree[rel.subject] += 1
        
        # Process nodes with no dependencies
        queue = deque([name for name, deg in in_degree.items() if deg == 0])
        order = []
        
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Handle cycles (place remaining objects last)
        remaining = [o.name for o in objects if o.name not in order]
        order.extend(remaining)
        
        return order
    
    def _place_object(self, half_extents: np.ndarray, constraints: List[RelationSpec]) -> Optional[np.ndarray]:
        """Rejection sampling: sample → validate → accept/reject."""
        for _ in range(self.max_attempts):
            # Sample
            if constraints:
                pos = self.constraint_handler.sample_position(constraints, self.placed, half_extents)
                if pos is None:
                    continue
            else:
                # Random ground position - center of AABB at y = half_extents[1]
                max_x = SCENE_BOUNDS - half_extents[0]
                max_z = SCENE_BOUNDS - half_extents[2]
                pos = np.array([
                    self.rng.uniform(-max_x, max_x), 
                    half_extents[1],  # Bottom at ground (y=0)
                    self.rng.uniform(-max_z, max_z)
                ])
            
            # Validate
            if self._is_valid(pos, half_extents):
                return pos
        
        return None
    
    def _is_valid(self, pos: np.ndarray, half_extents: np.ndarray) -> bool:
        """Check if position is valid (within bounds and no collision)."""
        return self._in_bounds(pos, half_extents) and not self._has_collision(pos, half_extents)
    
    def _in_bounds(self, pos: np.ndarray, half_extents: np.ndarray) -> bool:
        """Check if AABB is within scene bounds."""
        # Check if bottom is on or above ground
        if pos[1] - half_extents[1] < -GROUND_TOLERANCE:
            return False
        
        # Check if AABB is within scene bounds in X and Z
        if abs(pos[0]) + half_extents[0] > SCENE_BOUNDS:
            return False
        if abs(pos[2]) + half_extents[2] > SCENE_BOUNDS:
            return False
        
        return True
    
    def _has_collision(self, pos: np.ndarray, half_extents: np.ndarray) -> bool:
        """
        AABB collision check: allow contact for vertical stacking.
        
        Uses separating axis theorem (SAT) for AABB-AABB collision detection.
        Special case: allow exact contact for "on" relations (vertical stacking).
        """
        for obj in self.placed.values():
            # AABB collision check on each axis
            overlap_x = (abs(pos[0] - obj.position[0]) < 
                        half_extents[0] + obj.half_extents[0] + MIN_SEPARATION)
            overlap_y = (abs(pos[1] - obj.position[1]) < 
                        half_extents[1] + obj.half_extents[1] + MIN_SEPARATION)
            overlap_z = (abs(pos[2] - obj.position[2]) < 
                        half_extents[2] + obj.half_extents[2] + MIN_SEPARATION)
            
            # Check if vertically stacked (horizontal overlap + vertical contact)
            horiz_overlap = (abs(pos[0] - obj.position[0]) < 
                           max(half_extents[0], obj.half_extents[0]))
            horiz_overlap_z = (abs(pos[2] - obj.position[2]) < 
                             max(half_extents[2], obj.half_extents[2]))
            vert_gap = abs(pos[1] - obj.position[1]) - (half_extents[1] + obj.half_extents[1])
            
            is_stacking = (horiz_overlap and horiz_overlap_z and 
                          abs(vert_gap) < MIN_SEPARATION * 3)
            
            if is_stacking:
                # Allow contact for stacking - only check horizontal separation
                if overlap_x and overlap_z:
                    # But still require they are actually stacked vertically
                    if abs(vert_gap) > MIN_SEPARATION * 5:
                        return True
                # No collision for valid stacking
                continue
            
            # Normal collision: require separation on all axes
            if overlap_x and overlap_y and overlap_z:
                return True
        
        return False
    
    def _build_layout(self, dsl: DSL) -> Layout:
        """Build final layout from placed objects."""
        objects = [
            LayoutObject(
                name=obj.name,
                position=tuple(obj.position),
                rotation=obj.rotation,
                size=tuple(obj.half_extents)
            )
            for obj in self.placed.values()
        ]
        return Layout(
            objects=objects,
            description=dsl.description,
            relations=dsl.relations,
            id=dsl.id
        )


def solve_dsl(dsl: DSL, seed: Optional[int] = None) -> Optional[Layout]:
    """Solve DSL and return layout (or None if failed)."""
    try:
        return LayoutSolver(seed).solve(dsl)
    except SolverError:
        return None
