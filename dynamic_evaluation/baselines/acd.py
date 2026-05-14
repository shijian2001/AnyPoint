"""Automated Capability Discovery (Lu et al., 2025) adapter.

Cross-domain port: ACD's open-ended capability exploration becomes adaptive
search over AnyPoint's fixed generator capability space. The selector keeps
ACD's core loop:

1. Treat task categories as discovered capability regions.
2. Use model feedback to favor regions with observed failures.
3. Keep exploring under-tested regions through an uncertainty bonus.
4. Propagate failures to sibling categories in the same generator family.

Within a selected capability region, candidates are ordered by a lightweight
task-complexity proxy so the next probe is a stronger capability test rather
than an arbitrary first item from the fixed pool.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

from ._common import indices_by_category, task_category


def select_acd_style_indices(
    remaining_items: Sequence[Any],
    correct_tasks: Sequence[Any],
    error_tasks: Sequence[Any],
    k: int,
    alpha: float = 1.0,
) -> List[int]:
    """Pick ``k`` indices by adaptive capability discovery over the fixed pool."""
    if k <= 0:
        return []

    category_stats = _category_stats(correct_tasks, error_tasks)
    family_stats = _family_stats(correct_tasks, error_tasks)
    total_tested = max(1, len(correct_tasks) + len(error_tasks))
    category_scores = {
        category: _acd_category_score(
            category_stats.get(category, {"tested": 0, "errors": 0}),
            family_stats.get(
                _category_family(remaining_items, category),
                {"tested": 0, "errors": 0},
            ),
            total_tested,
            alpha,
        )
        for category in {task_category(item.task) for item in remaining_items}
    }
    grouped = indices_by_category(remaining_items)
    for indices in grouped.values():
        indices.sort(
            key=lambda idx: (
                -_task_complexity(remaining_items[idx].task),
                remaining_items[idx].item_id,
            )
        )
    ranked = sorted(category_scores, key=lambda c: (-category_scores[c], c))

    selected: List[int] = []
    while len(selected) < min(k, len(remaining_items)) and ranked:
        progressed = False
        for category in ranked:
            if len(selected) >= k:
                break
            bucket = grouped.get(category, [])
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return selected


def _category_stats(
    correct_tasks: Sequence[Any], error_tasks: Sequence[Any]
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tested": 0, "errors": 0})
    for task in correct_tasks:
        stats[task_category(task)]["tested"] += 1
    for task in error_tasks:
        category = task_category(task)
        stats[category]["tested"] += 1
        stats[category]["errors"] += 1
    return dict(stats)


def _family_stats(
    correct_tasks: Sequence[Any], error_tasks: Sequence[Any]
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tested": 0, "errors": 0})
    for task in correct_tasks:
        stats[_task_family(task)]["tested"] += 1
    for task in error_tasks:
        family = _task_family(task)
        stats[family]["tested"] += 1
        stats[family]["errors"] += 1
    return dict(stats)


def _acd_category_score(
    stats: Dict[str, int],
    family_stats: Dict[str, int],
    total_tested: int,
    alpha: float,
) -> float:
    tested = stats["tested"]
    family_error_rate = _error_rate(family_stats)
    family_uncertainty = _uncertainty_bonus(family_stats["tested"], total_tested)
    if tested == 0:
        return 1.0 + 0.35 * family_error_rate + 0.15 * family_uncertainty
    error_rate = stats["errors"] / tested
    category_uncertainty = _uncertainty_bonus(tested, total_tested)
    return (
        error_rate
        + alpha * category_uncertainty
        + 0.35 * family_error_rate
        + 0.15 * family_uncertainty
    )


def _error_rate(stats: Dict[str, int]) -> float:
    tested = stats["tested"]
    if tested == 0:
        return 0.0
    return stats["errors"] / tested


def _uncertainty_bonus(tested: int, total_tested: int) -> float:
    if tested == 0:
        return 1.0
    return math.sqrt(math.log(total_tested + 1) / tested)


def _category_family(remaining_items: Sequence[Any], category: str) -> str:
    for item in remaining_items:
        if task_category(item.task) == category:
            return _task_family(item.task)
    return "unknown"


def _task_family(task: Any) -> str:
    metadata = getattr(task, "metadata", None) or {}
    return metadata.get("generator_type", "unknown")


def _task_complexity(task: Any) -> float:
    metadata = getattr(task, "metadata", None) or {}
    objects = metadata.get("objects", []) or []
    layout_description = metadata.get("layout_description", "") or ""
    options = getattr(task, "options", None) or []
    category = task_category(task)

    relation_count = layout_description.count(".") + layout_description.count(",")
    capability_bonus = 1.0 if any(token in category for token in ("where", "distance", "size", "list")) else 0.0
    return len(objects) + relation_count + 0.1 * len(options) + capability_bonus
