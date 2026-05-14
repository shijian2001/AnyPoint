from .scene_builder import (
    build_scene_point_cloud,
    estimate_background_support_y,
    estimate_usable_background_bounds,
    fit_background_to_layout,
    get_support_height,
    transform_object_point_cloud,
)

__version__ = "1.0.0"

__all__ = [
    "PointQAGenerator",
    "TaskPlan",
    "PointCloudMetadata",
    "WhatDistanceGenerator",
    "WhereDistanceGenerator",
    "ListAttributeDistanceGenerator",
    "CountAttributeDistanceGenerator",
    "WhatAttributeGenerator",
    "ListAttributeGenerator",
    "CountAttributeGenerator",
    "CountObjectGenerator",
    "FrequentObjectGenerator",
    "ListAttributeFrequentGenerator",
    "CountAttributeFrequentGenerator",
    "WhatSizeGenerator",
    "ListAttributeSizeGenerator",
    "CountAttributeSizeGenerator",
    "WhereSizeGenerator",
    "build_scene_point_cloud",
    "estimate_background_support_y",
    "estimate_usable_background_bounds",
    "fit_background_to_layout",
    "get_support_height",
    "transform_object_point_cloud",
]


def __getattr__(name: str):
    if name == "PointQAGenerator":
        from .generator import PointQAGenerator

        return PointQAGenerator
    if name == "TaskPlan":
        from .base import TaskPlan

        return TaskPlan
    if name == "PointCloudMetadata":
        from .metadata import PointCloudMetadata

        return PointCloudMetadata
    if name in {
        "WhatDistanceGenerator",
        "WhereDistanceGenerator",
        "ListAttributeDistanceGenerator",
        "CountAttributeDistanceGenerator",
    }:
        from .distance import (
            CountAttributeDistanceGenerator,
            ListAttributeDistanceGenerator,
            WhatDistanceGenerator,
            WhereDistanceGenerator,
        )

        return {
            "WhatDistanceGenerator": WhatDistanceGenerator,
            "WhereDistanceGenerator": WhereDistanceGenerator,
            "ListAttributeDistanceGenerator": ListAttributeDistanceGenerator,
            "CountAttributeDistanceGenerator": CountAttributeDistanceGenerator,
        }[name]
    if name in {
        "WhatAttributeGenerator",
        "ListAttributeGenerator",
        "CountAttributeGenerator",
    }:
        from .attribute import CountAttributeGenerator, ListAttributeGenerator, WhatAttributeGenerator

        return {
            "WhatAttributeGenerator": WhatAttributeGenerator,
            "ListAttributeGenerator": ListAttributeGenerator,
            "CountAttributeGenerator": CountAttributeGenerator,
        }[name]
    if name in {
        "CountObjectGenerator",
        "FrequentObjectGenerator",
        "ListAttributeFrequentGenerator",
        "CountAttributeFrequentGenerator",
    }:
        from .number import (
            CountAttributeFrequentGenerator,
            CountObjectGenerator,
            FrequentObjectGenerator,
            ListAttributeFrequentGenerator,
        )

        return {
            "CountObjectGenerator": CountObjectGenerator,
            "FrequentObjectGenerator": FrequentObjectGenerator,
            "ListAttributeFrequentGenerator": ListAttributeFrequentGenerator,
            "CountAttributeFrequentGenerator": CountAttributeFrequentGenerator,
        }[name]
    if name in {
        "WhatSizeGenerator",
        "ListAttributeSizeGenerator",
        "CountAttributeSizeGenerator",
        "WhereSizeGenerator",
    }:
        from .size import (
            CountAttributeSizeGenerator,
            ListAttributeSizeGenerator,
            WhatSizeGenerator,
            WhereSizeGenerator,
        )

        return {
            "WhatSizeGenerator": WhatSizeGenerator,
            "ListAttributeSizeGenerator": ListAttributeSizeGenerator,
            "CountAttributeSizeGenerator": CountAttributeSizeGenerator,
            "WhereSizeGenerator": WhereSizeGenerator,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
