import os


POINTALIGN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WEIGHTS_ROOT = os.path.join(POINTALIGN_ROOT, "params_weight")


def get_weights_root() -> str:
    return os.environ.get("POINTALIGN_WEIGHTS_ROOT", DEFAULT_WEIGHTS_ROOT)


def resolve_weight_path(*parts: str) -> str:
    return os.path.join(get_weights_root(), *parts)
