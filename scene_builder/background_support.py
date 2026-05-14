from __future__ import annotations

import numpy as np


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
