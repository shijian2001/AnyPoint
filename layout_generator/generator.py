"""Layout generator: LLM-based DSL generation with solver-based layout computation."""

import asyncio
import json
import re
import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from .schema import DSL, ObjectSpec, RelationSpec, Layout, Template
from .validator import parse_dsl, DSLValidationError
from .api.json_parser import JSONParser
from .solver import LayoutSolver, SolverError
from .constants import VALID_SIZES, VALID_RELATIONS, MIN_OBJECTS, MAX_OBJECTS
from .api import StreamGenerator

logger = logging.getLogger(__name__)

# System prompt for DSL generation
DSL_SYSTEM_PROMPT = """You are an expert at generating realistic spatial scene layouts for 3D object placement.

Given a list of objects, generate a JSON DSL describing their spatial arrangement in a physically plausible, real-world scene.

## CRITICAL REQUIREMENTS:
1. ⚠️ MUST include BOTH 'largest' and 'smallest' size categories (one object each minimum)
2. ⚠️ All relation references MUST be actual object names from the objects list (NO 'scene', 'ground', 'floor', etc.)
3. ⚠️ Description MUST mention every object by its EXACT name (not aliases)
4. ⚠️ Description should ONLY state spatial relationships - NO adjectives or attributes (e.g., "large table" → "table", "wooden chair" → "chair")
5. ⚠️ Layout must be realistic and follow real-world scene logic

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

## REALISTIC SCENE CONSTRAINTS:
- **Physical stability**: Objects must have stable support (e.g., items "on" tables, not floating)
- **Size logic**: Smaller objects typically "on" or "near" larger ones; larger objects rarely "on" smaller ones
- **Functional relationships**: Objects should be arranged as they would in real scenes (e.g., lamp on desk, cup on table, books on shelf)
- **Spatial coherence**: Related objects should be grouped (e.g., dining items together, work items together)
- **Gravity compliance**: Vertical relations must respect gravity (heavy items below, light items above)
- **Accessibility**: Objects should be reachable and usable in the arrangement
- **Rotation realism**: Rotation should match typical object orientations (0° for most furniture)

## EXAMPLES:

### Example 1: ["table", "lamp", "book"]
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

### Example 2: ["sofa", "coffee_table", "vase", "remote"]
```json
{
  "description": "The sofa is positioned at the center. The coffee_table is in front of the sofa. The vase is on the coffee_table. The remote is on the coffee_table beside the vase.",
  "objects": [
    {"name": "sofa", "size": "largest", "rotation": 0},
    {"name": "coffee_table", "size": "large", "rotation": 0},
    {"name": "vase", "size": "small", "rotation": 0},
    {"name": "remote", "size": "smallest", "rotation": 30}
  ],
  "relations": [
    {"subject": "coffee_table", "relation": "in front of", "reference": "sofa"},
    {"subject": "vase", "relation": "on", "reference": "coffee_table"},
    {"subject": "remote", "relation": "on", "reference": "coffee_table"},
    {"subject": "remote", "relation": "beside", "reference": "vase"}
  ]
}
```

### Example 3: ["desk", "monitor", "keyboard", "mouse", "pen"]
```json
{
  "description": "The desk is at the center. The monitor is on the desk. The keyboard is on the desk in front of the monitor. The mouse is on the desk to the right of the keyboard. The pen is on the desk beside the keyboard.",
  "objects": [
    {"name": "desk", "size": "largest", "rotation": 0},
    {"name": "monitor", "size": "large", "rotation": 0},
    {"name": "keyboard", "size": "medium", "rotation": 0},
    {"name": "mouse", "size": "small", "rotation": 0},
    {"name": "pen", "size": "smallest", "rotation": 45}
  ],
  "relations": [
    {"subject": "monitor", "relation": "on", "reference": "desk"},
    {"subject": "keyboard", "relation": "on", "reference": "desk"},
    {"subject": "keyboard", "relation": "in front of", "reference": "monitor"},
    {"subject": "mouse", "relation": "on", "reference": "desk"},
    {"subject": "mouse", "relation": "to the right of", "reference": "keyboard"},
    {"subject": "pen", "relation": "on", "reference": "desk"},
    {"subject": "pen", "relation": "beside", "reference": "keyboard"}
  ]
}
```

Now generate a realistic, physically plausible spatial arrangement that follows real-world scene logic. Focus purely on spatial relationships without any descriptive adjectives."""


def _create_user_prompt(object_names: List[str]) -> str:
    """Create user prompt for DSL generation."""
    names_str = ", ".join(f'"{name}"' for name in object_names)
    return f"Generate a spatial scene DSL for these objects: [{names_str}]"


def _abstract_to_template(dsl: DSL, template_id: int) -> Template:
    """
    Abstract DSL to reusable template by replacing object names with placeholders.

    Args:
        dsl: Validated DSL object
        template_id: Unique template identifier

    Returns:
        Template with placeholder names
    """
    # Create name mapping: original -> placeholder
    name_mapping = {obj.name: f"obj_{i}" for i, obj in enumerate(dsl.objects)}

    # Transform description: replace names with placeholders in brackets
    description = dsl.description
    for original, placeholder in name_mapping.items():
        # Replace exact matches, handling word boundaries
        pattern = r'\b' + re.escape(original) + r'\b'
        description = re.sub(pattern, f"[{placeholder}]", description, flags=re.IGNORECASE)

    # Transform objects
    template_objects = [
        ObjectSpec(
            name=name_mapping[obj.name],
            size=obj.size,
            rotation=obj.rotation
        )
        for obj in dsl.objects
    ]

    # Transform relations
    template_relations = [
        RelationSpec(
            subject=name_mapping[rel.subject],
            relation=rel.relation,
            reference=name_mapping[rel.reference]
        )
        for rel in dsl.relations
    ]

    return Template(
        id=template_id,
        count=len(dsl.objects),
        description=description,
        objects=template_objects,
        relations=template_relations
    )


def _template_to_dsl(template: Template) -> DSL:
    """Convert template back to DSL for solving."""
    return DSL(
        description=template.description,
        objects=template.objects.copy(),
        relations=template.relations.copy(),
        id=template.id,
        count=template.count
    )


class LayoutGenerator:
    """
    Main layout generator combining LLM DSL generation with constraint solving.

    Usage:
        generator = LayoutGenerator(model_name="...", api_keys=[...])
        layouts = await generator.generate_batch(object_lists, num_layouts=100)
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
        """
        Initialize layout generator.

        Args:
            model_name: LLM model name for DSL generation
            api_keys: API keys for LLM access
            max_concurrent_per_key: Max concurrent API requests per key
            max_retries: Max retries per request
            solver_threads: Number of threads for parallel solving
            seed: Random seed for reproducibility
        """
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

    async def generate_batch(
        self,
        object_lists: List[List[str]],
        layouts_per_template: int = 1
    ) -> Tuple[List[Template], List[Layout]]:
        """
        Generate layouts for multiple object lists.

        Args:
            object_lists: List of object name lists (2-9 objects each)
            layouts_per_template: Number of layout variations per template

        Returns:
            Tuple of (templates, layouts)
        """
        # Generate DSLs via LLM
        prompts = [
            {"id": str(i), "prompt": _create_user_prompt(obj_list)}
            for i, obj_list in enumerate(object_lists)
        ]

        templates: List[Template] = []
        dsls: List[DSL] = []
        processed_ids: Set[str] = set()

        # Stream DSL generation
        async for result in self.stream_generator.generate_stream(
            prompts=prompts,
            system_prompt=DSL_SYSTEM_PROMPT,
            validate_func=self._validate_dsl_response
        ):
            # Defensive check for valid result
            if result is None or "id" not in result or "result" not in result:
                continue

            # Skip duplicates
            if result["id"] in processed_ids:
                continue
            processed_ids.add(result["id"])

            prompt_id = result["id"]
            response = result["result"]

            try:
                dsl = self._parse_response(response)
                if dsl is None:
                    logger.warning(f"Failed to parse DSL for prompt {prompt_id}")
                    continue

                async with self._lock:
                    template = _abstract_to_template(dsl, self._template_counter)
                    self._template_counter += 1

                templates.append(template)
                dsls.append(dsl)
                
                # Log template details
                relation_summary = [f"{r.subject} {r.relation} {r.reference}" for r in dsl.relations[:3]]
                if len(dsl.relations) > 3:
                    relation_summary.append(f"... +{len(dsl.relations)-3} more")
                logger.info(
                    f"Generated template {template.id}: {template.count} objects, "
                    f"{len(dsl.relations)} relations. Sample: {'; '.join(relation_summary)}"
                )

            except Exception as e:
                logger.error(f"Error processing response {prompt_id}: {e}")

        # Solve templates to layouts in parallel
        layouts = await self._solve_templates_parallel(templates, layouts_per_template)

        return templates, layouts

    async def generate_single(
        self,
        object_names: List[str],
        num_layouts: int = 1
    ) -> Tuple[Optional[Template], List[Layout]]:
        """
        Generate layouts for a single object list.

        Args:
            object_names: List of 2-9 object names
            num_layouts: Number of layout variations

        Returns:
            Tuple of (template, layouts)
        """
        templates, layouts = await self.generate_batch([object_names], num_layouts)
        template = templates[0] if templates else None
        return template, layouts

    def _validate_dsl_response(self, response: str) -> Optional[str]:
        """Validate LLM response contains valid DSL JSON."""
        data = JSONParser.parse(response)
        if data is None:
            return None
        
        try:
            parse_dsl(data)
            return response
        except DSLValidationError:
            return None

    def _parse_response(self, response: str) -> Optional[DSL]:
        """Parse LLM response to DSL object."""
        data = JSONParser.parse(response)
        if data is None:
            logger.warning("Failed to parse JSON from response")
            return None
        try:
            dsl = parse_dsl(data)
            logger.info(f"Successfully parsed DSL: {len(data.get('objects', []))} objects, {len(data.get('relations', []))} relations")
            return dsl
        except DSLValidationError as e:
            logger.warning(f"DSL validation failed: {e}")
            return None

    async def _solve_templates_parallel(
        self,
        templates: List[Template],
        layouts_per_template: int
    ) -> List[Layout]:
        """Solve templates to layouts using thread pool."""
        loop = asyncio.get_event_loop()
        tasks = []

        for template in templates:
            for i in range(layouts_per_template):
                seed = self.seed + template.id * 1000 + i if self.seed else None
                tasks.append(
                    loop.run_in_executor(
                        self.solver_pool,
                        self._solve_single,
                        template,
                        seed
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

    def _solve_single(self, template: Template, seed: Optional[int]) -> Optional[Layout]:
        """Solve single template to layout (runs in thread pool)."""
        solver = LayoutSolver(seed)
        dsl = _template_to_dsl(template)

        try:
            return solver.solve(dsl)
        except SolverError as e:
            logger.warning(f"Failed to solve template {template.id}: {e}")
            return None


def sample_object_names(
    available_objects: List[str],
    count: Optional[int] = None,
    seed: Optional[int] = None
) -> List[str]:
    """
    Sample random object names for DSL generation.

    Args:
        available_objects: List of available object names
        count: Number of objects (random 2-9 if None)
        seed: Random seed

    Returns:
        List of sampled object names
    """
    import numpy as np
    rng = np.random.RandomState(seed)

    if count is None:
        count = rng.randint(MIN_OBJECTS, MAX_OBJECTS + 1)

    count = max(MIN_OBJECTS, min(MAX_OBJECTS, count, len(available_objects)))
    return list(rng.choice(available_objects, size=count, replace=False))
