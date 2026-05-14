from typing import List, Optional, Sequence, TypeVar


T = TypeVar("T")


def resolve_devices(devices: str, cuda_visible_devices: Optional[str] = None) -> List[str]:
    """Resolve CLI device arguments into an ordered, non-empty device list."""
    resolved = [item.strip() for item in devices.split(",") if item.strip()]
    if not resolved:
        raise ValueError("至少需要提供一个 device")
    _validate_cuda_ordinals(resolved, cuda_visible_devices)
    return resolved


def _validate_cuda_ordinals(devices: List[str], cuda_visible_devices: Optional[str]) -> None:
    if not cuda_visible_devices:
        return

    visible = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
    if not visible:
        return

    visible_count = len(visible)
    for device in devices:
        if not device.startswith("cuda:"):
            continue
        try:
            ordinal = int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"无法解析 CUDA device: {device}") from exc

        if ordinal >= visible_count:
            raise ValueError(
                f"{device} is unavailable because CUDA_VISIBLE_DEVICES={cuda_visible_devices!r} exposes "
                f"only {visible_count} logical CUDA device(s). Use logical ids cuda:0..cuda:{visible_count - 1}, "
                "or expose more physical GPUs."
            )


def split_evenly(items: Sequence[T], num_chunks: int) -> List[List[T]]:
    """Split items across workers while preserving each worker's local order."""
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    chunks: List[List[T]] = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append(item)
    return [chunk for chunk in chunks if chunk]
