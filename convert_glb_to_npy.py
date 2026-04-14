#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def sample_texture(image, uv: np.ndarray) -> np.ndarray:
    image_rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
    height, width = image_rgba.shape[:2]

    uv = np.asarray(uv, dtype=np.float32)
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)

    # glTF UV origin is bottom-left; image arrays are top-left.
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


def convert_file(src_path: Path, dst_dir: Path) -> tuple[Path, tuple[int, int]]:
    scene = trimesh.load(src_path, force="scene")
    points = collect_scene_points(scene)

    stem = src_path.stem
    if stem.endswith("_2048"):
        stem = stem[:-5]

    dst_path = dst_dir / f"{stem}.npy"
    np.save(dst_path, points.astype(np.float32))
    return dst_path, points.shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GLB meshes to NPY point clouds without vertex sampling.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(input_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found in {input_dir}")

    for src_path in glb_files:
        dst_path, shape = convert_file(src_path, output_dir)
        print(f"{src_path.name} -> {dst_path.name} {shape}")


if __name__ == "__main__":
    main()
