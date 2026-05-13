"""
Layout solver: decomposed constraint optimization.

The 3D layout problem decomposes by axis independence:
  - Y (vertical): support DAG → closed-form.
  - XZ (horizontal): constrained optimization with analytical gradient.

The XZ solver uses SLSQP with:
  - Objective: relation energy (directional + radial + containment)
  - Hard constraints: collision avoidance, scene boundaries
  - Analytical Jacobian for O(1) gradient cost per iteration.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .schema import DSL, ObjectSpec, RelationSpec, Layout, LayoutObject
from .constants import SIZE_RANGES, SCENE_BOUNDS, RELATION_DISTANCE, RELATION_DIRECTIONS


class SolverError(Exception):
    pass


class LayoutSolver:
    """
    Two-phase constraint solver for 3D scene layouts.

    Phase 1: Analytical vertical placement from support relations.
    Phase 2: SLSQP optimization for horizontal placement with
             collision and boundary constraints.
    """

    def __init__(self, seed: Optional[int] = None, max_restarts: int = 10):
        self.rng = np.random.RandomState(seed)
        self.max_restarts = max_restarts

    def solve(self, dsl: DSL) -> Layout:
        sizes = self._sample_sizes(dsl.objects)
        names = [o.name for o in dsl.objects]
        y_coords = self._solve_vertical(names, sizes, dsl.relations)

        for attempt in range(self.max_restarts):
            xz = self._solve_horizontal(names, sizes, y_coords, dsl.relations, attempt)
            if xz is not None:
                return self._build_layout(dsl, names, sizes, y_coords, xz)

        raise SolverError(f"Could not find valid layout after {self.max_restarts} restarts")

    # ─── Phase 1: Vertical (closed-form) ─────────────────────────────────────

    def _solve_vertical(self, names: List[str], sizes: Dict[str, np.ndarray],
                        relations: List[RelationSpec]) -> Dict[str, float]:
        parent_of = {}
        for rel in relations:
            if rel.relation in ("on", "above"):
                parent_of[rel.subject] = (rel.reference, rel.relation)
            elif rel.relation in ("below", "under"):
                parent_of[rel.reference] = (rel.subject, "on")

        y = {}
        resolving = set()

        def resolve(name: str) -> float:
            if name in y:
                return y[name]
            if name in resolving:
                # Cycle detected — break it by placing on ground
                y[name] = sizes[name][1]
                return y[name]
            resolving.add(name)
            if name not in parent_of:
                y[name] = sizes[name][1]
            else:
                parent, rel_type = parent_of[name]
                py = resolve(parent)
                if rel_type == "on":
                    y[name] = py + sizes[parent][1] + sizes[name][1]
                else:
                    gap = self.rng.uniform(0.3, 0.8) * (sizes[name][1] + sizes[parent][1])
                    y[name] = py + sizes[parent][1] + sizes[name][1] + gap
            resolving.discard(name)
            return y[name]

        for name in names:
            resolve(name)
        return y

    # ─── Phase 2: Horizontal (SLSQP with analytical gradient) ─────────────────

    def _solve_horizontal(self, names: List[str], sizes: Dict[str, np.ndarray],
                          y_coords: Dict[str, float], relations: List[RelationSpec],
                          attempt: int) -> Optional[np.ndarray]:
        n = len(names)
        idx = {name: i for i, name in enumerate(names)}
        ctx = self._build_context(names, sizes, y_coords, relations, idx)

        x0 = self._initialize(names, sizes, relations, idx, attempt)

        bounds = [(-SCENE_BOUNDS + sizes[names[i // 2]][0 if i % 2 == 0 else 2],
                    SCENE_BOUNDS - sizes[names[i // 2]][0 if i % 2 == 0 else 2])
                  for i in range(n * 2)]

        constraints = self._build_slsqp_constraints(n, ctx)

        result = minimize(
            fun=self._objective,
            x0=x0,
            args=(ctx,),
            method='SLSQP',
            jac=self._gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-9}
        )

        # Validate directional constraints
        if self._directions_satisfied(result.x, ctx):
            return result.x
        return None

    def _build_context(self, names, sizes, y_coords, relations, idx) -> dict:
        """Compile all constraint data into a context dict."""
        VERTICAL = {"on", "above", "below", "under"}
        n = len(names)

        stacking = set()
        on_parent = {}
        for rel in relations:
            if rel.relation in VERTICAL:
                i, j = idx[rel.subject], idx[rel.reference]
                stacking.add((min(i, j), max(i, j)))
            if rel.relation == "on":
                on_parent[idx[rel.subject]] = idx[rel.reference]

        # Directional: (si, ri, dx_dir, dz_dir, target)
        directional = []
        # Radial: (si, ri, target)
        radial = []
        # Containment: (si, ri)
        containment = []

        for rel in relations:
            si, ri = idx[rel.subject], idx[rel.reference]
            s_ext, r_ext = sizes[rel.subject], sizes[rel.reference]
            combined = max(s_ext[0], s_ext[2]) + max(r_ext[0], r_ext[2])

            if rel.relation in RELATION_DIRECTIONS:
                dx_d, dz_d = RELATION_DIRECTIONS[rel.relation]
                d_min, d_max = RELATION_DISTANCE[rel.relation]
                directional.append((si, ri, dx_d, dz_d, (d_min + d_max) / 2 * combined))
            elif rel.relation in ("beside", "next to", "near", "far from", "surrounding"):
                d_min, d_max = RELATION_DISTANCE[rel.relation]
                radial.append((si, ri, (d_min + d_max) / 2 * combined))
            elif rel.relation == "at the center of":
                radial.append((si, ri, 0.0))
            elif rel.relation in ("on", "above", "below", "under"):
                containment.append((si, ri))

        # Collision pairs
        collisions = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in stacking:
                    continue
                vert_sep = abs(y_coords[names[i]] - y_coords[names[j]]) \
                           - (sizes[names[i]][1] + sizes[names[j]][1])
                if vert_sep > 0.05:
                    continue
                share_surface = (on_parent.get(i) == on_parent.get(j) and i in on_parent)
                gap = 0.0 if share_surface else 0.05
                sep_x = sizes[names[i]][0] + sizes[names[j]][0] + gap
                sep_z = sizes[names[i]][2] + sizes[names[j]][2] + gap
                collisions.append((i, j, sep_x, sep_z))

        # Surface spreading: pairs sharing the same parent should spread out
        spreading = []
        children_of = {}
        for child, parent in on_parent.items():
            children_of.setdefault(parent, []).append(child)
        for parent, children in children_of.items():
            if len(children) < 2:
                continue
            # Target separation: spread evenly across parent footprint
            parent_footprint = max(sizes[names[parent]][0], sizes[names[parent]][2])
            target_sep = parent_footprint * 0.8 / max(len(children) - 1, 1)
            for ci in range(len(children)):
                for cj in range(ci + 1, len(children)):
                    spreading.append((children[ci], children[cj], target_sep))

        return {
            'n': n, 'directional': directional, 'radial': radial,
            'containment': containment, 'collisions': collisions,
            'spreading': spreading,
        }

    def _objective(self, xz: np.ndarray, ctx: dict) -> float:
        """Relation energy (smooth, differentiable)."""
        E = 0.0

        for si, ri, dx_d, dz_d, target in ctx['directional']:
            disp_x, disp_z = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            proj = disp_x * dx_d + disp_z * dz_d
            E += (proj - target) ** 2
            lateral = disp_x * (-dz_d) + disp_z * dx_d
            E += 0.1 * lateral * lateral

        for si, ri, target in ctx['radial']:
            dx, dz = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            dist = np.sqrt(dx*dx + dz*dz + 1e-6)
            E += (dist - target) ** 2

        for si, ri in ctx['containment']:
            dx, dz = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            E += 0.5 * (dx*dx + dz*dz)

        # Spreading: objects on same surface should maintain minimum separation
        for ci, cj, target_sep in ctx['spreading']:
            dx, dz = xz[ci*2] - xz[cj*2], xz[ci*2+1] - xz[cj*2+1]
            dist = np.sqrt(dx*dx + dz*dz + 1e-6)
            if dist < target_sep:
                E += 2.0 * (target_sep - dist) ** 2

        return E

    def _gradient(self, xz: np.ndarray, ctx: dict) -> np.ndarray:
        """Analytical gradient of relation energy."""
        n = ctx['n']
        grad = np.zeros(n * 2)

        for si, ri, dx_d, dz_d, target in ctx['directional']:
            disp_x, disp_z = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            proj = disp_x * dx_d + disp_z * dz_d
            lateral = disp_x * (-dz_d) + disp_z * dx_d

            # d/d(si_x): 2*(proj-target)*dx_d + 2*0.1*lateral*(-dz_d)
            dEdsx = 2 * (proj - target) * dx_d + 0.2 * lateral * (-dz_d)
            dEdsz = 2 * (proj - target) * dz_d + 0.2 * lateral * dx_d

            grad[si*2] += dEdsx
            grad[si*2+1] += dEdsz
            grad[ri*2] -= dEdsx
            grad[ri*2+1] -= dEdsz

        for si, ri, target in ctx['radial']:
            dx, dz = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            dist = np.sqrt(dx*dx + dz*dz + 1e-6)
            coeff = 2 * (dist - target) / dist

            grad[si*2] += coeff * dx
            grad[si*2+1] += coeff * dz
            grad[ri*2] -= coeff * dx
            grad[ri*2+1] -= coeff * dz

        for si, ri in ctx['containment']:
            dx, dz = xz[si*2] - xz[ri*2], xz[si*2+1] - xz[ri*2+1]
            grad[si*2] += dx
            grad[si*2+1] += dz
            grad[ri*2] -= dx
            grad[ri*2+1] -= dz

        # Spreading gradient
        for ci, cj, target_sep in ctx['spreading']:
            dx, dz = xz[ci*2] - xz[cj*2], xz[ci*2+1] - xz[cj*2+1]
            dist = np.sqrt(dx*dx + dz*dz + 1e-6)
            if dist < target_sep:
                coeff = -4.0 * (target_sep - dist) / dist
                grad[ci*2] += coeff * dx
                grad[ci*2+1] += coeff * dz
                grad[cj*2] -= coeff * dx
                grad[cj*2+1] -= coeff * dz

        return grad

    def _build_slsqp_constraints(self, n: int, ctx: dict) -> List[dict]:
        """Build inequality constraints for SLSQP (each must be >= 0)."""
        constraints = []

        # Collision: for each pair, at least one axis must be separated
        # |xi - xj| >= sep_x  OR  |zi - zj| >= sep_z
        # Reformulation: max(|dx|-sep_x, |dz|-sep_z) >= 0
        for i, j, sep_x, sep_z in ctx['collisions']:
            def make_collision_constraint(i=i, j=j, sx=sep_x, sz=sep_z):
                def constraint(xz):
                    dx = abs(xz[i*2] - xz[j*2])
                    dz = abs(xz[i*2+1] - xz[j*2+1])
                    return max(dx - sx, dz - sz)
                return constraint
            constraints.append({'type': 'ineq', 'fun': make_collision_constraint()})

        return constraints

    def _directions_satisfied(self, xz: np.ndarray, ctx: dict) -> bool:
        """Verify all directional constraints have correct sign."""
        for si, ri, dx_d, dz_d, _ in ctx['directional']:
            proj = (xz[si*2] - xz[ri*2]) * dx_d + (xz[si*2+1] - xz[ri*2+1]) * dz_d
            if proj <= 0:
                return False
        return True

    # ─── Initialization ───────────────────────────────────────────────────────

    def _initialize(self, names: List[str], sizes: Dict[str, np.ndarray],
                    relations: List[RelationSpec], idx: Dict[str, int],
                    attempt: int) -> np.ndarray:
        n = len(names)
        x0 = np.zeros(n * 2)

        ref_count = defaultdict(int)
        for rel in relations:
            ref_count[rel.reference] += 1
        root = max(ref_count, key=ref_count.get) if ref_count else names[0]

        root_i = idx[root]
        jitter = self.rng.uniform(-0.3, 0.3, size=2) * (attempt + 1)
        x0[root_i*2], x0[root_i*2+1] = jitter[0], jitter[1]

        placed = {root}
        queue = [root]

        while queue:
            cur = queue.pop(0)
            ci = idx[cur]
            cx, cz = x0[ci*2], x0[ci*2+1]

            for rel in relations:
                if rel.reference == cur and rel.subject not in placed:
                    tgt, sign = rel.subject, 1.0
                elif rel.subject == cur and rel.reference not in placed:
                    tgt, sign = rel.reference, -1.0
                else:
                    continue

                ti = idx[tgt]
                combined = max(sizes[tgt][0], sizes[tgt][2]) + max(sizes[cur][0], sizes[cur][2])

                if rel.relation in RELATION_DIRECTIONS:
                    dx, dz = RELATION_DIRECTIONS[rel.relation]
                    d_min, d_max = RELATION_DISTANCE[rel.relation]
                    dist = (d_min + d_max) / 2 * combined
                    x0[ti*2] = cx + dx * dist * sign
                    x0[ti*2+1] = cz + dz * dist * sign
                elif rel.relation in ("on", "at the center of"):
                    x0[ti*2] = cx + self.rng.uniform(-0.3, 0.3)
                    x0[ti*2+1] = cz + self.rng.uniform(-0.3, 0.3)
                else:
                    angle = self.rng.uniform(0, 2 * np.pi)
                    d_min, d_max = RELATION_DISTANCE.get(rel.relation, (1.2, 2.0))
                    dist = (d_min + d_max) / 2 * combined
                    x0[ti*2] = cx + np.cos(angle) * dist * sign
                    x0[ti*2+1] = cz + np.sin(angle) * dist * sign

                placed.add(tgt)
                queue.append(tgt)

        for name in names:
            if name not in placed:
                i = idx[name]
                angle = self.rng.uniform(0, 2 * np.pi)
                r = self.rng.uniform(1.0, 3.0)
                x0[i*2], x0[i*2+1] = np.cos(angle) * r, np.sin(angle) * r

        return x0

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _sample_sizes(self, objects: List[ObjectSpec]) -> Dict[str, np.ndarray]:
        sizes = {}
        for obj in objects:
            lo, hi = SIZE_RANGES[obj.size]
            sizes[obj.name] = np.array([self.rng.uniform(lo[i], hi[i]) for i in range(3)])
        return sizes

    def _build_layout(self, dsl: DSL, names: List[str], sizes: Dict[str, np.ndarray],
                      y_coords: Dict[str, float], xz: np.ndarray) -> Layout:
        objects = []
        for obj in dsl.objects:
            i = names.index(obj.name)
            objects.append(LayoutObject(
                name=obj.name,
                position=(float(xz[i*2]), float(y_coords[obj.name]), float(xz[i*2+1])),
                rotation=obj.rotation,
                size=tuple(sizes[obj.name])
            ))
        return Layout(objects=objects, description=dsl.description, relations=dsl.relations, id=dsl.id)


def solve_dsl(dsl: DSL, seed: Optional[int] = None) -> Optional[Layout]:
    """Convenience function: solve DSL, return Layout or None on failure."""
    try:
        return LayoutSolver(seed).solve(dsl)
    except SolverError:
        return None
