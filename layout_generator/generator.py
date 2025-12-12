"""Layout generator: LLM-based DSL generation with solver-based layout computation."""

import asyncio
import json
import re
import logging
from typing import List, Optional, Dict, Any, Tuple, Set
from concurrent.futures import ThreadPoolExecutor
from .schema import DSL, ObjectSpec, RelationSpec, Layout, Template
from .validator import parse_dsl, extract_json_from_response, DSLValidationError
from .solver import LayoutSolver, SolverError
from .constants import VALID_SIZES, VALID_RELATIONS, MIN_OBJECTS, MAX_OBJECTS
from .api import StreamGenerator

logger = logging.getLogger(__name__)

# System prompt for DSL generation
DSL_SYSTEM_PROMPT = """You are an expert at generating spatial scene descriptions for 3D object placement.

Given a list of objects, generate a JSON DSL describing their spatial arrangement. The DSL must follow this exact format:

```json
{
  "description": "A detailed description mentioning ALL objects by their exact names",
  "objects": [
    {"name": "object_name", "size": "size_category", "rotation": angle_in_degrees}
  ],
  "relations": [
    {"subject": "object1", "relation": "spatial_relation", "reference": "object2"}
  ]
}
```

RULES:
1. Size categories (must include both 'largest' and 'smallest'):
   - largest, large, medium, small, smallest

2. Valid spatial relations:
   - Horizontal: "in front of", "behind", "to the left of", "to the right of", "beside", "next to", "near", "far from"
   - Vertical: "on", "above", "below", "under"
   - Other: "surrounding", "at the center of"

3. The description MUST mention each object by its EXACT name (not aliases)
4. Rotation: 0-360 degrees, choose reasonable orientations
5. Relations should be semantically meaningful (e.g., small objects "on" large surfaces)
6. Avoid impossible configurations (e.g., large object "on" small object)

Generate a creative but physically plausible arrangement."""


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
                logger.info(f"Generated template {template.id} with {template.count} objects")

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
        json_str = extract_json_from_response(response)
        if json_str is None:
            return None
        try:
            parse_dsl(json_str)
            return response
        except DSLValidationError:
            return None

    def _parse_response(self, response: str) -> Optional[DSL]:
        """Parse LLM response to DSL object."""
        json_str = extract_json_from_response(response)
        if json_str is None:
            return None
        try:
            return parse_dsl(json_str)
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
