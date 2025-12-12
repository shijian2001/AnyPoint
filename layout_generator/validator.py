"""DSL validation for layout generation."""

import json
import re
from typing import Tuple, Optional, List, Set
from .schema import DSL, ObjectSpec, RelationSpec
from .constants import VALID_SIZES, VALID_RELATIONS, MIN_OBJECTS, MAX_OBJECTS


class DSLValidationError(Exception):
    """Raised when DSL validation fails."""
    pass


def validate_json_format(json_str: str) -> Tuple[bool, Optional[dict], str]:
    """
    Validate JSON format and parse content.

    Returns:
        (is_valid, parsed_dict, error_message)
    """
    try:
        data = json.loads(json_str)
        return True, data, ""
    except json.JSONDecodeError as e:
        return False, None, f"Invalid JSON: {str(e)}"


def validate_object_spec(obj_data: dict, index: int) -> Tuple[bool, str]:
    """Validate a single object specification."""
    required_fields = ["name", "size", "rotation"]

    for field in required_fields:
        if field not in obj_data:
            return False, f"Object {index}: missing required field '{field}'"

    if not isinstance(obj_data["name"], str) or not obj_data["name"].strip():
        return False, f"Object {index}: 'name' must be a non-empty string"

    if obj_data["size"] not in VALID_SIZES:
        return False, f"Object {index}: invalid size '{obj_data['size']}'. Must be one of {VALID_SIZES}"

    if not isinstance(obj_data["rotation"], (int, float)):
        return False, f"Object {index}: 'rotation' must be a number"

    return True, ""


def validate_relation_spec(rel_data: dict, index: int, object_names: Set[str]) -> Tuple[bool, str]:
    """Validate a single relation specification."""
    required_fields = ["subject", "relation", "reference"]

    for field in required_fields:
        if field not in rel_data:
            return False, f"Relation {index}: missing required field '{field}'"

    if rel_data["subject"] not in object_names:
        return False, f"Relation {index}: subject '{rel_data['subject']}' not found in objects"

    if rel_data["reference"] not in object_names:
        return False, f"Relation {index}: reference '{rel_data['reference']}' not found in objects"

    if rel_data["subject"] == rel_data["reference"]:
        return False, f"Relation {index}: subject and reference cannot be the same object"

    if rel_data["relation"] not in VALID_RELATIONS:
        return False, f"Relation {index}: invalid relation '{rel_data['relation']}'. Must be one of {VALID_RELATIONS}"

    return True, ""


def validate_size_constraints(objects: List[dict]) -> Tuple[bool, str]:
    """Validate that there is at least one largest and one smallest object."""
    sizes = [obj["size"] for obj in objects]

    if "largest" not in sizes:
        return False, "Missing required 'largest' sized object"
    if "smallest" not in sizes:
        return False, "Missing required 'smallest' sized object"

    return True, ""


def validate_description_contains_names(description: str, object_names: Set[str]) -> Tuple[bool, str]:
    """Validate that description contains all object names (not aliases)."""
    description_lower = description.lower()

    for name in object_names:
        if name.lower() not in description_lower:
            return False, f"Description must contain object name '{name}'"

    return True, ""


def validate_dsl_dict(data: dict) -> Tuple[bool, str]:
    """
    Validate a complete DSL dictionary.

    Returns:
        (is_valid, error_message)
    """
    # Check required top-level fields
    # Note: "descirption" typo in spec is intentional to match the provided DSL format
    if "description" not in data and "descirption" not in data:
        return False, "Missing required field 'description'"

    description = data.get("description", data.get("descirption", ""))

    if "objects" not in data:
        return False, "Missing required field 'objects'"

    if not isinstance(data["objects"], list):
        return False, "'objects' must be a list"

    # Validate object count
    num_objects = len(data["objects"])
    if num_objects < MIN_OBJECTS or num_objects > MAX_OBJECTS:
        return False, f"Object count must be between {MIN_OBJECTS} and {MAX_OBJECTS}, got {num_objects}"

    # Validate each object
    object_names: Set[str] = set()
    for i, obj in enumerate(data["objects"]):
        is_valid, error = validate_object_spec(obj, i)
        if not is_valid:
            return False, error
        object_names.add(obj["name"])

    # Validate size constraints
    is_valid, error = validate_size_constraints(data["objects"])
    if not is_valid:
        return False, error

    # Validate relations
    if "relations" not in data:
        return False, "Missing required field 'relations'"

    if not isinstance(data["relations"], list):
        return False, "'relations' must be a list"

    for i, rel in enumerate(data["relations"]):
        is_valid, error = validate_relation_spec(rel, i, object_names)
        if not is_valid:
            return False, error

    # Validate description contains all object names
    is_valid, error = validate_description_contains_names(description, object_names)
    if not is_valid:
        return False, error

    return True, ""


def parse_dsl(json_str: str) -> DSL:
    """
    Parse and validate a DSL JSON string.

    Args:
        json_str: JSON string containing DSL

    Returns:
        Validated DSL object

    Raises:
        DSLValidationError: If validation fails
    """
    # Parse JSON
    is_valid, data, error = validate_json_format(json_str)
    if not is_valid:
        raise DSLValidationError(error)

    # Validate structure
    is_valid, error = validate_dsl_dict(data)
    if not is_valid:
        raise DSLValidationError(error)

    # Build DSL object
    description = data.get("description", data.get("descirption", ""))

    objects = [
        ObjectSpec(
            name=obj["name"],
            size=obj["size"],
            rotation=float(obj["rotation"])
        )
        for obj in data["objects"]
    ]

    relations = [
        RelationSpec(
            subject=rel["subject"],
            relation=rel["relation"],
            reference=rel["reference"]
        )
        for rel in data["relations"]
    ]

    return DSL(
        description=description,
        objects=objects,
        relations=relations,
        id=data.get("id"),
        count=data.get("count")
    )


def extract_json_from_response(response: str) -> Optional[str]:
    """
    Extract JSON content from LLM response that may contain markdown code blocks.

    Args:
        response: LLM response string

    Returns:
        Extracted JSON string or None if not found
    """
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
    matches = re.findall(code_block_pattern, response)

    for match in matches:
        try:
            json.loads(match.strip())
            return match.strip()
        except json.JSONDecodeError:
            continue

    # Try to find raw JSON object
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, response)

    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue

    return None

