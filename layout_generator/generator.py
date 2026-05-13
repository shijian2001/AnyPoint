"""Layout generator: LLM-based DSL generation with constraint solving."""

import asyncio
import re
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from .schema import DSL, ObjectSpec, RelationSpec, Layout, Template
from .validator import parse_dsl, DSLValidationError
from .api.json_parser import JSONParser
from .solver import LayoutSolver, SolverError
from .constants import (
    VALID_SIZES, VALID_RELATIONS, MIN_OBJECTS, MAX_OBJECTS, RELATION_DIRECTIONS
)
from .api import StreamGenerator

logger = logging.getLogger(__name__)

# ─── System Prompt ────────────────────────────────────────────────────────────

DSL_SYSTEM_PROMPT = """You are an expert at generating spatial scene layouts for 3D object placement.

Given a list of objects, generate a JSON DSL describing their spatial arrangement. Consider each object's real-world semantics (what it is, how it's typically used, what scale it would be) to determine plausible positions and size assignments. The layout should have rich spatial structure — objects at varying distances, distributed across multiple directions, with a mix of vertical stacking and horizontal spread.

## CRITICAL REQUIREMENTS:
1. ⚠️ MUST include BOTH 'largest' and 'smallest' size categories (one object each minimum)
2. ⚠️ All relation references MUST be actual object names from the objects list (NO 'scene', 'ground', 'floor', etc.)
3. ⚠️ Description MUST mention every object by its EXACT name (not aliases)
4. ⚠️ Description should ONLY state spatial relationships - NO adjectives or attributes (e.g., "large table" → "table", "wooden chair" → "chair")
5. ⚠️ Use diverse spatial relations — combine vertical (on/above), directional (in front of/behind/left/right), and proximity (near/far/beside) relations rather than relying on a single relation type

## JSON FORMAT:
```json
{
  "description": "Pure spatial layout description mentioning ALL objects by name without adjectives",
  "objects": [
    {"name": "exact_object_name", "size": "size_category", "rotation": degrees}
  ],
  "relations": [
    {"subject": "object_name", "relation": "spatial_relation", "reference": "another_object_name"}
  ]
}
```

## VALID SIZE CATEGORIES:
- largest (REQUIRED - assign to one object)
- large
- medium
- small
- smallest (REQUIRED - assign to one object)

## VALID SPATIAL RELATIONS:
- Horizontal: "in front of", "behind", "to the left of", "to the right of", "beside", "next to", "near", "far from"
- Vertical: "on", "above", "below", "under"
- Other: "surrounding", "at the center of"

## LAYOUT GUIDELINES:
- **Semantic awareness**: Consider what each object is and assign sizes/positions that reflect real-world proportions (e.g., a building is larger than a cup; a coin sits on a table, not vice versa)
- **Physical plausibility**: Vertical stacking must respect gravity — heavier/larger objects support lighter/smaller ones
- **Spatial richness**: Distribute objects across the scene — some near, some far; some stacked vertically, some spread horizontally; some in front, some behind
- **Relation diversity**: Each layout should use multiple relation types (not just "on" for everything, or just "beside" for everything)

## EXAMPLES (showing diverse layout patterns):

### Example 1 (surface stacking): ["table", "lamp", "book"]
```json
{
  "description": "The table is positioned at the center. The lamp is on the table. The book is on the table beside the lamp.",
  "objects": [
    {"name": "table", "size": "largest", "rotation": 0},
    {"name": "lamp", "size": "medium", "rotation": 0},
    {"name": "book", "size": "smallest", "rotation": 45}
  ],
  "relations": [
    {"subject": "lamp", "relation": "on", "reference": "table"},
    {"subject": "book", "relation": "on", "reference": "table"},
    {"subject": "book", "relation": "beside", "reference": "lamp"}
  ]
}
```

### Example 2 (linear chain): ["bookshelf", "desk", "chair", "lamp"]
```json
{
  "description": "The bookshelf is behind the desk. The desk is behind the chair. The lamp is on the desk.",
  "objects": [
    {"name": "bookshelf", "size": "largest", "rotation": 0},
    {"name": "desk", "size": "large", "rotation": 0},
    {"name": "chair", "size": "medium", "rotation": 0},
    {"name": "lamp", "size": "smallest", "rotation": 0}
  ],
  "relations": [
    {"subject": "desk", "relation": "in front of", "reference": "bookshelf"},
    {"subject": "chair", "relation": "in front of", "reference": "desk"},
    {"subject": "lamp", "relation": "on", "reference": "desk"}
  ]
}
```

### Example 3 (distributed cluster): ["tree", "bench", "fountain", "statue", "bird"]
```json
{
  "description": "The fountain is at the center of the tree. The bench is to the left of the fountain. The statue is to the right of the fountain. The bird is on the statue.",
  "objects": [
    {"name": "tree", "size": "largest", "rotation": 0},
    {"name": "fountain", "size": "large", "rotation": 0},
    {"name": "bench", "size": "medium", "rotation": 0},
    {"name": "statue", "size": "small", "rotation": 0},
    {"name": "bird", "size": "smallest", "rotation": 15}
  ],
  "relations": [
    {"subject": "fountain", "relation": "at the center of", "reference": "tree"},
    {"subject": "bench", "relation": "to the left of", "reference": "fountain"},
    {"subject": "statue", "relation": "to the right of", "reference": "fountain"},
    {"subject": "bird", "relation": "on", "reference": "statue"}
  ]
}
```

### Example 4 (multi-level): ["floor", "cabinet", "tv", "speaker", "remote", "plant"]
```json
{
  "description": "The cabinet is on the floor. The tv is on the cabinet. The speaker is beside the cabinet. The remote is in front of the cabinet. The plant is far from the cabinet.",
  "objects": [
    {"name": "floor", "size": "largest", "rotation": 0},
    {"name": "cabinet", "size": "large", "rotation": 0},
    {"name": "tv", "size": "medium", "rotation": 0},
    {"name": "speaker", "size": "small", "rotation": 0},
    {"name": "remote", "size": "smallest", "rotation": 0},
    {"name": "plant", "size": "medium", "rotation": 0}
  ],
  "relations": [
    {"subject": "cabinet", "relation": "on", "reference": "floor"},
    {"subject": "tv", "relation": "on", "reference": "cabinet"},
    {"subject": "speaker", "relation": "beside", "reference": "cabinet"},
    {"subject": "remote", "relation": "in front of", "reference": "cabinet"},
    {"subject": "plant", "relation": "far from", "reference": "cabinet"}
  ]
}
```

Generate a spatial arrangement that fits the given objects naturally. Use diverse layout patterns — not always centered placement. Consider the objects' real-world semantics to choose appropriate spatial structures."""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _create_user_prompt(object_names: List[str]) -> str:
    names_str = ", ".join(f'"{name}"' for name in object_names)
    return f"Generate a spatial scene DSL for these objects: [{names_str}]"


def _abstract_to_template(dsl: DSL, template_id: int) -> Template:
    """Abstract DSL to reusable template by replacing object names with placeholders."""
    name_mapping = {obj.name: f"obj_{i}" for i, obj in enumerate(dsl.objects)}

    description = dsl.description
    # Replace longest names first to avoid partial matches (e.g. "table lamp" before "table")
    for original in sorted(name_mapping.keys(), key=len, reverse=True):
        placeholder = name_mapping[original]
        description = description.replace(original, f"[{placeholder}]")
        description = description.replace(original.capitalize(), f"[{placeholder}]")
        description = description.replace(original.title(), f"[{placeholder}]")

    template_objects = [
        ObjectSpec(name=name_mapping[obj.name], size=obj.size, rotation=obj.rotation)
        for obj in dsl.objects
    ]
    template_relations = [
        RelationSpec(
            subject=name_mapping[rel.subject],
            relation=rel.relation,
            reference=name_mapping[rel.reference]
        )
        for rel in dsl.relations
    ]

    return Template(
        id=template_id, count=len(dsl.objects),
        description=description, objects=template_objects, relations=template_relations
    )


def _template_to_dsl(template: Template) -> DSL:
    return DSL(
        description=template.description,
        objects=template.objects.copy(),
        relations=template.relations.copy(),
        id=template.id,
        count=template.count
    )


def _structure_key(template: Template) -> tuple:
    """Compute a hashable key representing the full structure (topology + sizes)."""
    relations_key = tuple(
        (r.subject, r.relation, r.reference)
        for r in sorted(template.relations, key=lambda r: (r.subject, r.relation, r.reference))
    )
    sizes_key = tuple((o.name, o.size) for o in template.objects)
    return (relations_key, sizes_key)


def _verify_directional_relations(layout: Layout) -> bool:
    """Verify all directional relations hold in solved coordinates."""
    idx = {o.name: o for o in layout.objects}
    for rel in layout.relations:
        if rel.relation in RELATION_DIRECTIONS:
            subj, ref = idx[rel.subject], idx[rel.reference]
            dx_d, dz_d = RELATION_DIRECTIONS[rel.relation]
            proj = (subj.position[0] - ref.position[0]) * dx_d + \
                   (subj.position[2] - ref.position[2]) * dz_d
            if proj <= 0:
                return False
    return True


# ─── Main Generator ──────────────────────────────────────────────────────────

class LayoutGenerator:
    """
    Layout generator with built-in quality assurance.

    Quality pipeline per layout:
      1. LLM generates DSL
      2. Validator checks structure, density, and relation diversity
      3. Solver produces coordinates
      4. Post-verification: directional relations checked against coordinates
      5. Deduplication: no repeated object sets or relation topologies

    Usage:
        generator = LayoutGenerator(model_name="...", api_keys=[...])
        templates, layouts = await generator.generate_batch(object_lists)
    """

    def __init__(
        self,
        model_name: str,
        api_keys: List[str],
        max_concurrent_per_key: int = 100,
        max_retries: int = 5,
        solver_threads: int = 4,
        seed: Optional[int] = None
    ):
        self.stream_generator = StreamGenerator(
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key,
            max_retries=max_retries,
            with_unique_id=True
        )
        self.solver_pool = ThreadPoolExecutor(max_workers=solver_threads)
        self.seed = seed
        self._template_counter = 0
        self._lock = asyncio.Lock()
        self._seen_structures: Set[tuple] = set()

    async def generate_batch(
        self,
        object_lists: List[List[str]],
        layouts_per_template: int = 1
    ) -> Tuple[List[Template], List[Layout]]:
        """Generate layouts with quality assurance and deduplication."""
        prompts = [
            {"id": str(i), "prompt": _create_user_prompt(obj_list)}
            for i, obj_list in enumerate(object_lists)
        ]

        templates: List[Template] = []
        processed_ids: Set[str] = set()

        async for result in self.stream_generator.generate_stream(
            prompts=prompts,
            system_prompt=DSL_SYSTEM_PROMPT,
            validate_func=self._validate_dsl_response
        ):
            if result is None or "id" not in result or "result" not in result:
                continue
            if result["id"] in processed_ids:
                continue
            processed_ids.add(result["id"])

            try:
                dsl = self._parse_response(result["result"])
                if dsl is None:
                    continue

                async with self._lock:
                    template = _abstract_to_template(dsl, self._template_counter)

                    # Dedup structure
                    sk = _structure_key(template)
                    if sk in self._seen_structures:
                        logger.debug(f"Duplicate structure for prompt {result['id']}, skipping")
                        continue
                    self._seen_structures.add(sk)
                    self._template_counter += 1

                templates.append(template)

            except Exception as e:
                logger.error(f"Error processing response {result['id']}: {e}")

        # Solve + verify
        layouts = await self._solve_and_verify(templates, layouts_per_template)
        return templates, layouts

    async def generate_single(
        self,
        object_names: List[str],
        num_layouts: int = 1
    ) -> Tuple[Optional[Template], List[Layout]]:
        templates, layouts = await self.generate_batch([object_names], num_layouts)
        template = templates[0] if templates else None
        return template, layouts

    def _validate_dsl_response(self, response: str) -> Optional[str]:
        data = JSONParser.parse(response)
        if data is None:
            return None
        try:
            parse_dsl(data)
            return response
        except DSLValidationError:
            return None

    def _parse_response(self, response: str) -> Optional[DSL]:
        data = JSONParser.parse(response)
        if data is None:
            return None
        try:
            return parse_dsl(data)
        except DSLValidationError as e:
            logger.debug(f"DSL validation failed: {e}")
            return None

    async def _solve_and_verify(
        self, templates: List[Template], layouts_per_template: int
    ) -> List[Layout]:
        """Solve templates and verify directional relations."""
        loop = asyncio.get_event_loop()
        tasks = []

        for template in templates:
            for i in range(layouts_per_template):
                seed = self.seed + template.id * 1000 + i if self.seed else None
                tasks.append(
                    loop.run_in_executor(
                        self.solver_pool, self._solve_and_verify_single, template, seed
                    )
                )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        layouts = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Solver error: {result}")
            elif result is not None:
                layouts.append(result)

        return layouts

    def _solve_and_verify_single(self, template: Template, seed: Optional[int]) -> Optional[Layout]:
        """Solve and post-verify a single template."""
        solver = LayoutSolver(seed)
        dsl = _template_to_dsl(template)

        try:
            layout = solver.solve(dsl)
        except SolverError as e:
            logger.warning(f"Failed to solve template {template.id}: {e}")
            return None

        if not _verify_directional_relations(layout):
            logger.warning(f"Template {template.id}: directional verification failed")
            return None

        return layout


# ─── Utility ──────────────────────────────────────────────────────────────────

def sample_object_names(
    available_objects: List[str],
    count: Optional[int] = None,
    seed: Optional[int] = None
) -> List[str]:
    rng = np.random.RandomState(seed)
    if count is None:
        count = rng.randint(MIN_OBJECTS, MAX_OBJECTS + 1)
    count = max(MIN_OBJECTS, min(MAX_OBJECTS, count, len(available_objects)))
    return list(rng.choice(available_objects, size=count, replace=False))
