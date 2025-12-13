"""
Layout Generator Module

Generates 3D scene layouts from object lists using LLM-based DSL generation
and constraint-based position solving.

Design:
    LLM generates semantic DSL → Abstract to reusable templates → Solver maps to coordinates

Components:
    - schema: Data structures (DSL, Template, Layout)
    - constants: Size mappings, spatial relations, constraints
    - validator: DSL validation and parsing
    - solver: Constraint solver for position generation
    - generator: Main layout generation pipeline
"""

from .schema import DSL, ObjectSpec, RelationSpec, Layout, LayoutObject, Template
from .validator import (
    parse_dsl,
    validate_dsl_dict,
    DSLValidationError
)
from .solver import LayoutSolver, SolverError, solve_dsl
from .generator import LayoutGenerator, sample_object_names
from .constants import VALID_SIZES, VALID_RELATIONS

__all__ = [
    # Schema
    "DSL",
    "ObjectSpec",
    "RelationSpec",
    "Layout",
    "LayoutObject",
    "Template",
    # Validation
    "parse_dsl",
    "validate_dsl_dict",
    "DSLValidationError",
    # Solving
    "LayoutSolver",
    "SolverError",
    "solve_dsl",
    # Generation
    "LayoutGenerator",
    "sample_object_names",
    # Constants
    "VALID_SIZES",
    "VALID_RELATIONS",
]

