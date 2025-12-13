"""Data structures for layout generation DSL and output formats."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ObjectSpec:
    """Specification for a single object in the DSL."""
    name: str
    size: str  # "largest", "large", "medium", "small", "smallest"
    rotation: float  # Rotation angle in degrees

    def to_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "rotation": self.rotation}


@dataclass
class RelationSpec:
    """Spatial relation between two objects."""
    subject: str  # Object being positioned
    relation: str  # Spatial relation type
    reference: str  # Reference object

    def to_dict(self) -> dict:
        return {"subject": self.subject, "relation": self.relation, "reference": self.reference}


@dataclass
class DSL:
    """Domain-specific language for scene layout description."""
    description: str
    objects: List[ObjectSpec]
    relations: List[RelationSpec]
    id: Optional[int] = None
    count: Optional[int] = None

    def __post_init__(self):
        if self.count is None:
            self.count = len(self.objects)

    def to_dict(self) -> dict:
        result = {
            "description": self.description,
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations]
        }
        if self.id is not None:
            result["id"] = self.id
            result["count"] = self.count
        return result


@dataclass
class LayoutObject:
    """Object with resolved geometric properties."""
    name: str
    position: Tuple[float, float, float]  # (x, y, z) coordinates
    rotation: float  # Rotation angle in degrees
    size: Tuple[float, float, float]  # AABB half-extents (x, y, z)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": list(self.position),
            "rotation": self.rotation,
            "size": list(self.size)
        }


@dataclass
class Layout:
    """Final layout with resolved geometric positions."""
    objects: List[LayoutObject]
    description: str
    id: Optional[int] = None

    def to_dict(self) -> dict:
        result = {
            "description": self.description,
            "objects": [obj.to_dict() for obj in self.objects]
        }
        if self.id is not None:
            result["id"] = self.id
        return result


@dataclass
class Template:
    """Reusable template with placeholder object names."""
    id: int
    count: int
    description: str  # Template description with placeholders
    objects: List[ObjectSpec]
    relations: List[RelationSpec]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "count": self.count,
            "description": self.description,
            "objects": [obj.to_dict() for obj in self.objects],
            "relations": [rel.to_dict() for rel in self.relations]
        }

