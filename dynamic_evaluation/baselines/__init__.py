"""Baseline selectors for compare-run experiments.

Each module hosts one baseline and is named after the method it adapts:

- :mod:`acd` — Automated Capability Discovery (Lu et al., 2025).
- :mod:`autobencher` — AutoBencher (Li et al., 2024).
- :mod:`sea` — Stochastic Error Ascent (Song et al., 2025).

Baselines share a small set of helpers from :mod:`_common`.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "select_acd_style_indices",
    "select_autobencher_style_indices",
    "SEAState",
]

_LAZY_ATTRS = {
    "select_acd_style_indices": (".acd", "select_acd_style_indices"),
    "select_autobencher_style_indices": (".autobencher", "select_autobencher_style_indices"),
    "SEAState": (".sea", "SEAState"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
