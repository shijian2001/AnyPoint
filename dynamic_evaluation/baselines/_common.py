"""Task-metadata helpers shared by compare-run baselines."""

from collections import defaultdict
from typing import Any, Dict, List, Sequence


def task_category(task: Any) -> str:
    """Return a stable ``generator_type[_k_v...]`` identifier for a task."""
    metadata = getattr(task, "metadata", None) or {}
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
