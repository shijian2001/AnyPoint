from __future__ import annotations

from pathlib import Path

import numpy as np


def build_scene_point_cloud(
    layout: dict,
    object_specs: list[dict],
    pcd_dir: str | Path,
    background_npy: str | Path | None = None,
    room: dict | None = None,
) -> np.ndarray:
    point_clouds: list[np.ndarray] = []
    background, usable_bounds = _load_background(background_npy, layout, room=room)
    support_y = get_support_height(background)

    for object_spec in object_specs:
        point_cloud = np.load(_resolve_object_point_cloud_path(pcd_dir, object_spec["object_id"])).astype(np.float32)
        transformed = transform_object_point_cloud(point_cloud, object_spec, support_y=support_y)
        if usable_bounds is not None:
            _validate_object_within_usable_bounds(transformed[:, :3], usable_bounds, object_spec.get("name", "object"))
        point_clouds.append(transformed)

    if background is not None:
        point_clouds.insert(0, background)

    if not point_clouds:
        raise ValueError("Scene contains no point clouds")

    return np.vstack(point_clouds)


def transform_object_point_cloud(point_cloud: np.ndarray, object_spec: dict, support_y: float = 0.0) -> np.ndarray:
    coords = point_cloud[:, :3]
    colors = point_cloud[:, 3:] if point_cloud.shape[1] > 3 else np.zeros((len(point_cloud), 3), dtype=np.float32)

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    extent = np.where((max_coords - min_coords) > 1e-6, max_coords - min_coords, 1.0)
    normalized = (coords - min_coords) / extent - 0.5

    position = np.asarray(object_spec["position"], dtype=np.float32).copy()
    rotation = float(object_spec.get("rotation", 0.0))
    size = np.asarray(object_spec["size"], dtype=np.float32)

    transformed = normalized * (size * 2.0)
    angle_rad = np.radians(rotation)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        dtype=np.float32,
    )
    transformed = transformed @ rotation_matrix.T

    position[1] = support_y + position[1]
    transformed = transformed + position
    return np.hstack((transformed.astype(np.float32), colors.astype(np.float32)))


def fit_background_to_layout(background: np.ndarray, layout: dict, room: dict | None = None) -> np.ndarray:
    if background is None or len(background) == 0:
        return background

    transformed, _ = _fit_background_to_layout_with_bounds(background, layout, room=room)
    return transformed


def estimate_usable_background_bounds(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if coords is None or len(coords) == 0:
        raise ValueError("Background coordinates are empty")

    usable_min = np.zeros(3, dtype=np.float32)
    usable_max = np.zeros(3, dtype=np.float32)
    for axis in range(3):
        axis_values = coords[:, axis]
        axis_min = float(np.min(axis_values))
        axis_max = float(np.max(axis_values))
        axis_extent = max(axis_max - axis_min, 1e-6)
        low_candidate = float(np.quantile(axis_values, 0.10))
        high_candidate = float(np.quantile(axis_values, 0.90 if axis != 1 else 0.95))
        if abs(low_candidate - axis_min) <= axis_extent * 0.02:
            low_candidate = float(np.quantile(axis_values, 0.20))
        if abs(high_candidate - axis_max) <= axis_extent * 0.02:
            high_candidate = float(np.quantile(axis_values, 0.80))
        usable_min[axis] = low_candidate
        usable_max[axis] = high_candidate
    support_height = get_support_height(np.hstack((coords.astype(np.float32), np.zeros((len(coords), 3), dtype=np.float32))))
    usable_min[1] = support_height
    return usable_min, usable_max


def get_support_height(background: np.ndarray | None) -> float:
    if background is None or len(background) == 0:
        return 0.0

    y_coords = background[:, 1]
    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    y_extent = y_max - y_min
    x_extent = float(np.max(background[:, 0]) - np.min(background[:, 0]))
    z_extent = float(np.max(background[:, 2]) - np.min(background[:, 2]))
    horizontal_extent = max(x_extent, z_extent, 1e-6)

    if y_extent <= 0.2 * horizontal_extent:
        return y_max
    return float(np.quantile(y_coords, 0.02))


def estimate_background_support_y(coords: np.ndarray) -> float:
    if coords.size == 0:
        raise ValueError("Background coordinates are empty")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("Background coordinates must have shape (N, 3)")

    y_coords = coords[:, 1].astype(np.float32)
    y_min = float(np.min(y_coords))
    y_max = float(np.max(y_coords))
    y_extent = y_max - y_min
    if y_extent <= 1e-6:
        return y_max

    candidate_limit = y_min + max(y_extent * 0.35, 0.05)
    candidate_mask = y_coords <= candidate_limit
    candidate_coords = coords[candidate_mask]
    if len(candidate_coords) == 0:
        return y_min

    bucket_size = max(y_extent * 0.01, 1e-3)
    bucket_ids = np.round((candidate_coords[:, 1] - y_min) / bucket_size).astype(np.int32)

    best_area = -1.0
    best_height = y_min
    for bucket_id in np.unique(bucket_ids):
        bucket_coords = candidate_coords[bucket_ids == bucket_id]
        if len(bucket_coords) == 0:
            continue

        x_extent = float(np.max(bucket_coords[:, 0]) - np.min(bucket_coords[:, 0]))
        z_extent = float(np.max(bucket_coords[:, 2]) - np.min(bucket_coords[:, 2]))
        footprint_area = x_extent * z_extent
        bucket_height = float(np.median(bucket_coords[:, 1]))

        if footprint_area > best_area + 1e-6:
            best_area = footprint_area
            best_height = bucket_height
            continue
        if abs(footprint_area - best_area) <= 1e-6 and bucket_height > best_height:
            best_height = bucket_height

    return best_height


def _fit_background_to_layout_with_bounds(
    background: np.ndarray,
    layout: dict,
    room: dict | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    bg = background.copy()
    coords = bg[:, :3]
    colors = bg[:, 3:]

    scene_min_x = min(float(obj["position"][0] - obj["size"][0]) for obj in layout["objects"])
    scene_max_x = max(float(obj["position"][0] + obj["size"][0]) for obj in layout["objects"])
    scene_min_z = min(float(obj["position"][2] - obj["size"][2]) for obj in layout["objects"])
    scene_max_z = max(float(obj["position"][2] + obj["size"][2]) for obj in layout["objects"])
    scene_max_y = max(float(obj["position"][1] + obj["size"][1]) for obj in layout["objects"])
    scene_center_x = 0.5 * (scene_min_x + scene_max_x)
    scene_center_z = 0.5 * (scene_min_z + scene_max_z)
    scene_width_x = max(scene_max_x - scene_min_x, 1e-6)
    scene_width_z = max(scene_max_z - scene_min_z, 1e-6)

    bg_min = coords.min(axis=0)
    bg_max = coords.max(axis=0)
    bg_raw_extent = bg_max - bg_min
    bg_extent = np.maximum(bg_raw_extent, 1e-6)
    usable_min, usable_max = estimate_usable_background_bounds(coords)
    usable_center = 0.5 * (usable_min + usable_max)
    usable_extent = np.maximum(usable_max - usable_min, 1e-6)

    margin = float(room.get("margin", 0.0)) if room else 0.0
    target_width_x = scene_width_x + margin * 2.0 if margin > 0.0 else scene_width_x * 1.2
    target_width_z = scene_width_z + margin * 2.0 if margin > 0.0 else scene_width_z * 1.2

    y_extent = float(bg_raw_extent[1])
    horizontal_extent = float(max(bg_raw_extent[0], bg_raw_extent[2], 1e-6))
    is_plane_like = y_extent <= 0.2 * horizontal_extent

    centered = coords - usable_center
    if is_plane_like:
        scale = np.array(
            [
                target_width_x / bg_extent[0],
                1.0,
                target_width_z / bg_extent[2],
            ],
            dtype=np.float32,
        )
    else:
        target_height = max(float(room.get("wall_height", 0.0)) if room else 0.0, scene_max_y + margin)
        usable_height = max(float(usable_max[1] - usable_min[1]), 1e-6)
        uniform_scale = max(
            target_width_x / usable_extent[0],
            target_width_z / usable_extent[2],
            target_height / usable_height,
        )
        scale = np.array([uniform_scale, uniform_scale, uniform_scale], dtype=np.float32)

    transformed = centered * scale
    transformed[:, 0] += scene_center_x
    transformed[:, 2] += scene_center_z

    transformed_usable_min = (usable_min - usable_center) * scale
    transformed_usable_max = (usable_max - usable_center) * scale
    transformed_usable_min[0] += scene_center_x
    transformed_usable_max[0] += scene_center_x
    transformed_usable_min[2] += scene_center_z
    transformed_usable_max[2] += scene_center_z

    if is_plane_like:
        y_shift = float(np.max(transformed[:, 1]))
        transformed[:, 1] -= y_shift
        transformed_usable_min[1] -= y_shift
        transformed_usable_max[1] -= y_shift
    else:
        support_y_raw = float(usable_min[1])
        y_shift = support_y_raw * scale[1]
        transformed[:, 1] -= y_shift
        transformed_usable_min[1] -= y_shift
        transformed_usable_max[1] -= y_shift

    usable_bounds = None
    if not is_plane_like:
        usable_bounds = (
            transformed_usable_min.astype(np.float32),
            transformed_usable_max.astype(np.float32),
        )

    return np.hstack((transformed.astype(np.float32), colors.astype(np.float32))), usable_bounds


def _load_background(
    background_npy: str | Path | None,
    layout: dict,
    room: dict | None = None,
) -> tuple[np.ndarray | None, tuple[np.ndarray, np.ndarray] | None]:
    if background_npy is None:
        return None, None
    background = np.load(background_npy).astype(np.float32)
    return _fit_background_to_layout_with_bounds(background, layout, room=room)


def _resolve_object_point_cloud_path(pcd_dir: str | Path, object_id: str) -> Path:
    path = Path(pcd_dir) / f"{object_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {path}")
    return path


def _validate_object_within_usable_bounds(
    object_xyz: np.ndarray,
    usable_bounds: tuple[np.ndarray, np.ndarray],
    object_name: str,
) -> None:
    usable_min, usable_max = usable_bounds
    obj_min = object_xyz.min(axis=0)
    obj_max = object_xyz.max(axis=0)
    clearance = 0.02

    if np.any(obj_min[[0, 2]] < usable_min[[0, 2]] - clearance) or np.any(obj_max[[0, 2]] > usable_max[[0, 2]] + clearance):
        raise ValueError(f"Object {object_name} exceeds usable background floor area")
    if obj_min[1] < usable_min[1] - clearance or obj_max[1] > usable_max[1] + clearance:
        raise ValueError(f"Object {object_name} exceeds usable background height")
