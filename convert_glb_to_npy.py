#!/usr/bin/env python3
"""
Convert GLB meshes to NPY point clouds (vertex-based, with colors).

Features:
- Multiprocessing for parallel conversion
- Resume support (skips existing files)
- Error tolerance (logs failures, continues)
- Progress bar

Usage:
    python convert_glb_to_npy.py \
        --input-dir /path/to/glbs \
        --output-dir /path/to/npy \
        --workers 16
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import trimesh

trimesh.util.log.setLevel("ERROR")


def sample_texture(image, uv: np.ndarray) -> np.ndarray:
    image_rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    height, width = image_rgba.shape[:2]

    uv = np.asarray(uv, dtype=np.float32)
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)

    x = np.clip(np.rint(u * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint((1.0 - v) * (height - 1)).astype(np.int64), 0, height - 1)
    return image_rgba[y, x]


def resolve_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    visual = mesh.visual
    vertex_count = len(mesh.vertices)

    if hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
        colors = np.asarray(visual.vertex_colors)
        if colors.ndim == 2 and len(colors) == vertex_count:
            if colors.shape[1] == 3:
                alpha = np.full((vertex_count, 1), 255, dtype=colors.dtype)
                return np.concatenate([colors, alpha], axis=1)
            if colors.shape[1] >= 4:
                return colors[:, :4]

    material = getattr(visual, "material", None)
    if hasattr(visual, "uv") and visual.uv is not None and material is not None:
        texture = getattr(material, "baseColorTexture", None)
        if texture is not None:
            return sample_texture(texture, visual.uv)

    main_color = getattr(material, "main_color", None)
    if main_color is not None:
        color = np.asarray(main_color, dtype=np.uint8).reshape(-1)
        if color.size == 3:
            color = np.concatenate([color, np.array([255], dtype=np.uint8)])
        if color.size >= 4:
            return np.repeat(color[:4][None, :], vertex_count, axis=0)

    return np.tile(np.array([[255, 255, 255, 255]], dtype=np.uint8), (vertex_count, 1))


def collect_scene_points(scene: trimesh.Scene) -> np.ndarray:
    point_sets = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]

        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.vertices) == 0:
            continue

        vertices = trimesh.transform_points(np.asarray(geometry.vertices), transform)
        colors = resolve_vertex_colors(geometry)[:, :3].astype(np.float32) / 255.0
        point_sets.append(np.concatenate([vertices.astype(np.float32), colors], axis=1))

    if not point_sets:
        raise ValueError("No mesh vertices found in scene")

    return np.concatenate(point_sets, axis=0)


def convert_single(src_path: Path, dst_dir: Path, skip_existing: bool = True) -> str:
    """Convert one GLB file. Returns status string."""
    stem = src_path.stem
    if stem.endswith("_2048"):
        stem = stem[:-5]

    dst_path = dst_dir / f"{stem}.npy"

    if skip_existing and dst_path.exists():
        return f"SKIP {src_path.name}"

    try:
        scene = trimesh.load(src_path, force="scene")
        points = collect_scene_points(scene)
        np.save(dst_path, points.astype(np.float32))
        return f"OK   {src_path.name} -> {dst_path.name} {points.shape}"
    except Exception as e:
        return f"FAIL {src_path.name}: {e}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GLB meshes to NPY point clouds.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=min(16, cpu_count()), help="Number of parallel workers")
    parser.add_argument("--no-skip", action="store_true", help="Re-convert even if output exists")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(input_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found in {input_dir}")

    print(f"Found {len(glb_files)} GLB files")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {not args.no_skip}")
    print()

    worker_fn = partial(convert_single, dst_dir=output_dir, skip_existing=not args.no_skip)

    ok, skip, fail = 0, 0, 0
    failed_files = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_fn, glb_files), 1):
            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
                failed_files.append(result)

            if i % 100 == 0 or i == len(glb_files):
                print(f"[{i}/{len(glb_files)}] ok={ok} skip={skip} fail={fail}")

    print(f"\nDone! ok={ok}, skip={skip}, fail={fail}")

    if failed_files:
        log_path = output_dir / "convert_failures.log"
        with open(log_path, "w") as f:
            f.write("\n".join(failed_files))
        print(f"Failures logged to: {log_path}")


if __name__ == "__main__":
    main()
