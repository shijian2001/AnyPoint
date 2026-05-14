"""AutoBencher (Li et al., 2024) adapter.

Cross-domain port: AutoBencher's benchmark-construction idea becomes a static
one-shot builder over AnyPoint's fixed candidate pool. It does not consume
model feedback. Instead, it allocates a balanced quota across capability
categories and greedily chooses difficult but non-duplicate tasks inside each
category, mirroring AutoBencher's goal of producing a salient, diverse, and
challenging benchmark for the target domain.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Set

from ._common import indices_by_category, task_category

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "s",
    "the",
    "to",
    "what",
    "which",
    "where",
}


def select_autobencher_style_indices(
    remaining_items: Sequence[Any], budget: int
) -> List[int]:
    """Select a balanced, difficult, and diverse static benchmark."""
    if budget <= 0:
        return []

    grouped = indices_by_category(remaining_items)
    quotas = _category_quotas(grouped, remaining_items, budget)
    selected: List[int] = []
    for category in sorted(quotas):
        selected.extend(
            _select_diverse_hard_items(grouped[category], remaining_items, quotas[category])
        )
    return selected


def _category_quotas(
    grouped: Dict[str, List[int]],
    remaining_items: Sequence[Any],
    budget: int,
) -> Dict[str, int]:
    categories = sorted(grouped)
    target = min(budget, sum(len(indices) for indices in grouped.values()))
    if not categories or target <= 0:
        return {}

    base = target // len(categories)
    quotas = {category: min(base, len(grouped[category])) for category in categories}
    allocated = sum(quotas.values())
    category_priority = sorted(
        categories,
        key=lambda category: (
            -_average_difficulty(grouped[category], remaining_items),
            -len(grouped[category]),
            category,
        ),
    )

    while allocated < target:
        progressed = False
        for category in category_priority:
            if allocated >= target:
                break
            if quotas[category] >= len(grouped[category]):
                continue
            quotas[category] += 1
            allocated += 1
            progressed = True
        if not progressed:
            break
    return {category: quota for category, quota in quotas.items() if quota > 0}


def _select_diverse_hard_items(
    indices: List[int],
    remaining_items: Sequence[Any],
    quota: int,
) -> List[int]:
    selected: List[int] = []
    selected_tokens: List[Set[str]] = []
    candidates = list(indices)

    while candidates and len(selected) < quota:
        best_idx = max(
            candidates,
            key=lambda idx: (
                _mmr_score(remaining_items[idx].task, selected_tokens),
                -remaining_items[idx].item_id,
            ),
        )
        selected.append(best_idx)
        selected_tokens.append(_task_tokens(remaining_items[best_idx].task))
        candidates.remove(best_idx)
    return selected


def _average_difficulty(indices: List[int], remaining_items: Sequence[Any]) -> float:
    if not indices:
        return 0.0
    return sum(_task_difficulty(remaining_items[idx].task) for idx in indices) / len(indices)


def _mmr_score(task: Any, selected_tokens: Sequence[Set[str]]) -> float:
    if not selected_tokens:
        return _task_difficulty(task)
    return _task_difficulty(task) - 5.0 * max(_jaccard(_task_tokens(task), tokens) for tokens in selected_tokens)


def _task_difficulty(task: Any) -> float:
    metadata = getattr(task, "metadata", None) or {}
    objects = metadata.get("objects", []) or []
    layout_description = metadata.get("layout_description", "") or ""
    options = getattr(task, "options", None) or []
    category = task_category(task)

    relation_count = len(
        re.findall(
            r"\b(on|under|near|beside|left|right|front|behind|closest|farthest)\b",
            layout_description.lower(),
        )
    )
    option_ambiguity = _option_ambiguity(options)
    category_bonus = 2.0 if any(token in category for token in ("list", "where", "distance", "size")) else 0.0
    return (
        len(objects)
        + layout_description.count(".")
        + 0.5 * relation_count
        + len(options) * 0.1
        + option_ambiguity
        + category_bonus
    )


def _option_ambiguity(options: Sequence[Any]) -> float:
    option_tokens = [_tokenize(str(option)) for option in options]
    if len(option_tokens) < 2:
        return 0.0
    overlaps: List[float] = []
    for idx, left in enumerate(option_tokens):
        for right in option_tokens[idx + 1:]:
            overlaps.append(_jaccard(left, right))
    return sum(overlaps) / len(overlaps)


def _task_tokens(task: Any) -> Set[str]:
    metadata = getattr(task, "metadata", None) or {}
    text_parts = [
        getattr(task, "question", ""),
        getattr(task, "answer", ""),
        metadata.get("layout_description", ""),
        " ".join(_meaningful_option_tokens(getattr(task, "options", None) or [])),
    ]
    return _tokenize(" ".join(text_parts))


def _tokenize(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
        and token != "obj"
        and not token.isdigit()
        and len(token) > 1
    }


def _meaningful_option_tokens(options: Sequence[Any]) -> List[str]:
    tokens: List[str] = []
    for option in options:
        text = str(option).strip().lower()
        if text in {"a", "b", "c", "d", "e", "f"}:
            continue
        tokens.extend(re.findall(r"[a-z0-9]+", text))
    return tokens


def _jaccard(left: Set[str], right: Set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
