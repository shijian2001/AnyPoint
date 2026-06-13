"""Task-metadata helpers shared by compare-run baselines."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence


def task_category(task: Any) -> str:
    """Return a stable per-task capability label.

    For AnyPoint's fixed pool each task carries a fine-grained ``category`` (30
    question types, e.g. ``count_attribute_distance_closest``) plus a coarser
    ``generator_type`` (17 families). Capability-region baselines (ACD's region
    scoring, AutoBencher's per-category quotas) operate on the FINE category so
    they match the unit used in our error analysis; ACD additionally uses the
    coarse ``generator_type`` as the sibling *family* (see ``_task_family``).
    Falls back to the old ``generator_type[_k_v...]`` form when no fine category
    is present.
    """
    metadata = getattr(task, "metadata", None) or {}
    fine = metadata.get("category")
    if fine:
        return fine
    generator_type = metadata.get("generator_type", "unknown")
    config = metadata.get("generator_config", {}) or {}
    suffix = "_".join(
        f"{key}_{config[key]}" for key in sorted(config) if config[key] not in (None, "")
    )
    return f"{generator_type}_{suffix}" if suffix else generator_type


def indices_by_category(items: Sequence[Any]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        grouped[task_category(item.task)].append(idx)
    return dict(grouped)
