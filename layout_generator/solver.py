"""Constraint solver for layout generation: maps DSL to geometric coordinates."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .schema import DSL, ObjectSpec, RelationSpec, Layout, LayoutObject
from .constants import (
    SIZE_RANGES, SPATIAL_RELATIONS, RELATION_DISTANCE, RELATION_DIRECTIONS,
    MIN_SEPARATION, MAX_SOLVER_ATTEMPTS, PERTURBATION_RANGE, SCENE_BOUNDS
)


@dataclass
class PlacedObject:
    """Internal representation of a placed object during solving."""
    name: str
    position: np.ndarray  # (x, y, z)
    radius: float  # Bounding sphere radius (size / 2)
    rotation: float


class SolverError(Exception):
    """Raised when solver cannot find valid layout."""
    pass


class LayoutSolver:
    """Constraint solver for mapping DSL to geometric layout."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize solver with optional random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self._placed_objects: Dict[str, PlacedObject] = {}

    def reset(self, seed: Optional[int] = None):
        """Reset solver state with optional new seed."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self._placed_objects = {}

    def solve(self, dsl: DSL) -> Layout:
        """
        Solve DSL constraints to produce geometric layout.

        Args:
            dsl: Validated DSL object

        Returns:
            Layout with resolved positions

        Raises:
            SolverError: If no valid layout can be found
        """
        self._placed_objects = {}

        # Resolve sizes to actual scale values
        size_map = self._resolve_sizes(dsl.objects)

        # Build dependency graph from relations
        placement_order = self._determine_placement_order(dsl.objects, dsl.relations)

        # Place objects in dependency order
        for obj_name in placement_order:
            obj_spec = next(o for o in dsl.objects if o.name == obj_name)
            size = size_map[obj_name]

            # Find relations where this object is the subject
            constraints = [r for r in dsl.relations if r.subject == obj_name]

            position = self._place_object(obj_spec, size, constraints)
            if position is None:
                raise SolverError(f"Failed to place object '{obj_name}' after {MAX_SOLVER_ATTEMPTS} attempts")

            self._placed_objects[obj_name] = PlacedObject(
                name=obj_name,
                position=position,
                radius=size / 2,
                rotation=obj_spec.rotation
            )

        # Build layout from placed objects
        layout_objects = [
            LayoutObject(
                name=obj.name,
                position=tuple(obj.position),
                rotation=obj.rotation,
                size=obj.radius * 2
            )
            for obj in self._placed_objects.values()
        ]

        return Layout(
            objects=layout_objects,
            description=dsl.description,
            id=dsl.id
        )

    def _resolve_sizes(self, objects: List[ObjectSpec]) -> Dict[str, float]:
        """Resolve size categories to actual scale values with random variation."""
        size_map = {}
        for obj in objects:
            min_size, max_size = SIZE_RANGES[obj.size]
            size_map[obj.name] = self.rng.uniform(min_size, max_size)
        return size_map

    def _determine_placement_order(
        self,
        objects: List[ObjectSpec],
        relations: List[RelationSpec]
    ) -> List[str]:
        """
        Determine object placement order based on relation dependencies.
        Reference objects must be placed before subjects.
        """
        # Build adjacency list: reference -> subjects
        dependencies: Dict[str, List[str]] = {obj.name: [] for obj in objects}
        for rel in relations:
            if rel.reference not in dependencies:
                dependencies[rel.reference] = []
            dependencies[rel.reference].append(rel.subject)

        # Topological sort with largest objects first for stability
        object_names = {obj.name for obj in objects}
        placed = set()
        order = []

        # Start with objects that have no dependencies (are only references)
        subjects = {rel.subject for rel in relations}
        references = {rel.reference for rel in relations}
        roots = object_names - subjects  # Objects never positioned relative to others

        # If no clear roots, start with "largest" objects
        if not roots:
            size_priority = {"largest": 0, "large": 1, "medium": 2, "small": 3, "smallest": 4}
            sorted_objects = sorted(objects, key=lambda o: size_priority.get(o.size, 5))
            roots = {sorted_objects[0].name}

        # BFS to determine order
        queue = list(roots)
        while queue:
            current = queue.pop(0)
            if current in placed:
                continue
            # Check if all dependencies are satisfied
            deps_satisfied = all(
                rel.reference in placed
                for rel in relations if rel.subject == current
            )
            if not deps_satisfied and current not in roots:
                queue.append(current)
                continue

            placed.add(current)
            order.append(current)

            # Add dependents to queue
            for dependent in dependencies.get(current, []):
                if dependent not in placed:
                    queue.append(dependent)

        # Add any remaining objects (no relations)
        for obj in objects:
            if obj.name not in placed:
                order.append(obj.name)

        return order

    def _place_object(
        self,
        obj: ObjectSpec,
        size: float,
        constraints: List[RelationSpec]
    ) -> Optional[np.ndarray]:
        """
        Place a single object satisfying constraints.

        Returns:
            Position array or None if placement failed
        """
        radius = size / 2

        for attempt in range(MAX_SOLVER_ATTEMPTS):
            if not constraints:
                # No constraints: place randomly with collision avoidance
                position = self._place_random(radius)
            else:
                # Compute position from constraints
                position = self._compute_constrained_position(obj.name, radius, constraints)
                if position is None:
                    continue

            # Verify no collisions
            if self._check_collision(position, radius):
                continue

            # Verify within scene bounds
            if not self._check_bounds(position, radius):
                continue

            return position

        return None

    def _place_random(self, radius: float) -> np.ndarray:
        """Place object at random position within scene bounds."""
        max_coord = SCENE_BOUNDS - radius
        return np.array([
            self.rng.uniform(-max_coord, max_coord),
            radius,  # y = height, rest on ground
            self.rng.uniform(-max_coord, max_coord)
        ])

    def _compute_constrained_position(
        self,
        obj_name: str,
        radius: float,
        constraints: List[RelationSpec]
    ) -> Optional[np.ndarray]:
        """Compute position satisfying relation constraints."""
        if not constraints:
            return self._place_random(radius)

        # Use first constraint as primary (others provide additional checks)
        primary = constraints[0]
        ref_obj = self._placed_objects.get(primary.reference)
        if ref_obj is None:
            return None

        relation = primary.relation
        is_horizontal, requires_contact = SPATIAL_RELATIONS.get(relation, (True, False))
        dist_range = RELATION_DISTANCE.get(relation, (1.2, 1.5))

        # Compute distance from reference
        combined_radius = radius + ref_obj.radius
        min_dist = dist_range[0] * combined_radius
        max_dist = dist_range[1] * combined_radius
        distance = self.rng.uniform(min_dist, max_dist)

        # Compute direction based on relation
        position = self._compute_position_from_relation(
            ref_obj.position, ref_obj.radius, radius, relation, distance, requires_contact
        )

        # Add random perturbation for diversity
        position += self._perturbation(is_horizontal)

        # Verify additional constraints
        for constraint in constraints[1:]:
            if not self._verify_constraint(position, radius, constraint):
                return None

        return position

    def _compute_position_from_relation(
        self,
        ref_pos: np.ndarray,
        ref_radius: float,
        obj_radius: float,
        relation: str,
        distance: float,
        requires_contact: bool
    ) -> np.ndarray:
        """Compute position based on specific relation type."""
        position = ref_pos.copy()

        if relation in RELATION_DIRECTIONS:
            # Directional horizontal relation
            dx, dz = RELATION_DIRECTIONS[relation]
            direction = np.array([dx, 0.0, dz])
            position += direction * distance
            position[1] = obj_radius  # Ground level

        elif relation == "on":
            # Object on top of reference
            position[1] = ref_pos[1] + ref_radius + obj_radius
            # Small random horizontal offset for natural look
            position[0] += self.rng.uniform(-0.1, 0.1) * ref_radius
            position[2] += self.rng.uniform(-0.1, 0.1) * ref_radius

        elif relation == "above":
            # Above but not necessarily touching
            position[1] = ref_pos[1] + distance

        elif relation in ("below", "under"):
            # Below reference
            position[1] = max(obj_radius, ref_pos[1] - distance)

        elif relation in ("beside", "next to"):
            # Random horizontal direction
            angle = self.rng.uniform(0, 2 * np.pi)
            position[0] += np.cos(angle) * distance
            position[2] += np.sin(angle) * distance
            position[1] = obj_radius

        elif relation == "near":
            # Nearby in random direction
            angle = self.rng.uniform(0, 2 * np.pi)
            position[0] += np.cos(angle) * distance
            position[2] += np.sin(angle) * distance
            position[1] = obj_radius

        elif relation == "far from":
            # Far away in random direction
            angle = self.rng.uniform(0, 2 * np.pi)
            position[0] += np.cos(angle) * distance
            position[2] += np.sin(angle) * distance
            position[1] = obj_radius

        elif relation == "surrounding":
            # Position around reference
            angle = self.rng.uniform(0, 2 * np.pi)
            position[0] += np.cos(angle) * distance
            position[2] += np.sin(angle) * distance
            position[1] = obj_radius

        elif relation == "at the center of":
            # Same position as reference (or center of multiple references)
            position[1] = obj_radius

        return position

    def _perturbation(self, is_horizontal: bool) -> np.ndarray:
        """Generate random perturbation vector."""
        pert = self.rng.uniform(-PERTURBATION_RANGE, PERTURBATION_RANGE, 3)
        if not is_horizontal:
            pert[0] *= 0.5
            pert[2] *= 0.5
        pert[1] = 0  # No y perturbation to keep ground contact
        return pert

    def _verify_constraint(
        self,
        position: np.ndarray,
        radius: float,
        constraint: RelationSpec
    ) -> bool:
        """Verify if position satisfies a constraint."""
        ref_obj = self._placed_objects.get(constraint.reference)
        if ref_obj is None:
            return True  # Can't verify, assume ok

        relation = constraint.relation
        dist_range = RELATION_DISTANCE.get(relation, (1.2, 1.5))
        combined_radius = radius + ref_obj.radius

        # Compute actual distance
        diff = position - ref_obj.position
        actual_dist = np.linalg.norm(diff)

        # Check if distance is within acceptable range (with tolerance)
        min_dist = dist_range[0] * combined_radius * 0.8
        max_dist = dist_range[1] * combined_radius * 1.5

        return min_dist <= actual_dist <= max_dist

    def _check_collision(self, position: np.ndarray, radius: float) -> bool:
        """Check if position collides with any placed object."""
        for obj in self._placed_objects.values():
            diff = position - obj.position
            dist = np.linalg.norm(diff)
            min_dist = radius + obj.radius + MIN_SEPARATION
            if dist < min_dist:
                return True
        return False

    def _check_bounds(self, position: np.ndarray, radius: float) -> bool:
        """Check if object is within scene bounds."""
        for i, coord in enumerate(position):
            if i == 1:  # y coordinate (height)
                if coord - radius < -0.1:  # Allow small tolerance
                    return False
            else:  # x, z coordinates
                if abs(coord) + radius > SCENE_BOUNDS:
                    return False
        return True


def solve_dsl(dsl: DSL, seed: Optional[int] = None) -> Optional[Layout]:
    """
    Convenience function to solve a DSL.

    Args:
        dsl: Validated DSL object
        seed: Random seed for reproducibility

    Returns:
        Layout or None if solving failed
    """
    solver = LayoutSolver(seed)
    try:
        return solver.solve(dsl)
    except SolverError:
        return None

