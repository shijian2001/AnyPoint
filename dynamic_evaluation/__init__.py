"""Dynamic evaluation framework."""

from importlib import import_module
from typing import Any

from .config import EvalConfig, TaskResult

__all__ = [
    "EvalConfig",
    "TaskResult",
    "TaskEmbedder",
    "UtilityCalculator",
    "TaskPool",
    "DynamicEvaluator",
    "SEAState",
    "select_acd_style_indices",
    "select_autobencher_style_indices",
]

_LAZY_ATTRS = {
    "TaskEmbedder": (".embedder", "TaskEmbedder"),
    "UtilityCalculator": (".utility", "UtilityCalculator"),
    "TaskPool": (".task_pool", "TaskPool"),
    "DynamicEvaluator": (".evaluator", "DynamicEvaluator"),
    "SEAState": (".baselines", "SEAState"),
    "select_acd_style_indices": (".baselines", "select_acd_style_indices"),
    "select_autobencher_style_indices": (".baselines", "select_autobencher_style_indices"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
