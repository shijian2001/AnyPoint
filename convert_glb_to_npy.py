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


def sample_mesh_surface(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    if len(mesh.faces) == 0:
        vertex_colors = resolve_vertex_colors(mesh)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        colors = vertex_colors[:, :3].astype(np.float32) / 255.0
        if len(vertices) == 0:
            return np.empty((0, 6), dtype=np.float32)
        if len(vertices) >= n_points:
            indices = np.random.choice(len(vertices), n_points, replace=False)
        else:
            indices = np.random.choice(len(vertices), n_points, replace=True)
        return np.concatenate([vertices[indices], colors[indices]], axis=1)

    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    face_indices = np.asarray(face_indices, dtype=np.int64)
    face_vertex_indices = mesh.faces[face_indices]
    triangles = np.asarray(mesh.vertices, dtype=np.float32)[face_vertex_indices]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, points).astype(np.float32)
    sampled_colors = resolve_surface_colors(mesh, face_vertex_indices, barycentric)
    return np.concatenate([points.astype(np.float32), sampled_colors], axis=1)


def resolve_surface_colors(
    mesh: trimesh.Trimesh,
    face_vertex_indices: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    visual = mesh.visual
    material = getattr(visual, "material", None)
    if hasattr(visual, "uv") and visual.uv is not None and material is not None:
        texture = getattr(material, "baseColorTexture", None)
        if texture is not None:
            face_uv = np.asarray(visual.uv, dtype=np.float32)[face_vertex_indices]
            sampled_uv = np.sum(face_uv * barycentric[:, :, None], axis=1)
            return sample_texture(texture, sampled_uv)[:, :3].astype(np.float32) / 255.0

    vertex_colors = resolve_vertex_colors(mesh)[:, :3].astype(np.float32) / 255.0
    face_vertex_colors = vertex_colors[face_vertex_indices]
    return np.sum(face_vertex_colors * barycentric[:, :, None], axis=1)


def collect_scene_surface_samples(scene: trimesh.Scene, sample_points: int) -> np.ndarray:
    mesh_entries = []
    total_area = 0.0
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]

        if not isinstance(geometry, trimesh.Trimesh) or len(geometry.vertices) == 0:
            continue

        world_mesh = geometry.copy()
        world_mesh.apply_transform(transform)
        area = float(max(world_mesh.area, 0.0))
        mesh_entries.append((world_mesh, area))
        total_area += area

    if not mesh_entries:
        raise ValueError("No mesh geometry found in scene")

    if total_area <= 0:
        total_vertices = sum(len(mesh.vertices) for mesh, _ in mesh_entries)
        if total_vertices == 0:
            raise ValueError("No mesh vertices found in scene")
        point_sets = []
        remaining = sample_points
        for idx, (mesh, _) in enumerate(mesh_entries):
            if idx == len(mesh_entries) - 1:
                n_points = remaining
            else:
                n_points = max(1, int(round(sample_points * len(mesh.vertices) / total_vertices)))
                remaining -= n_points
            point_sets.append(sample_mesh_surface(mesh, n_points))
        return np.concatenate(point_sets, axis=0)

    raw_counts = np.array([sample_points * area / total_area for _, area in mesh_entries], dtype=np.float64)
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 1)

    while counts.sum() < sample_points:
        counts[np.argmax(raw_counts - counts)] += 1
    while counts.sum() > sample_points:
        reducible = np.where(counts > 1)[0]
        if len(reducible) == 0:
            break
        reducible_scores = raw_counts[reducible] - counts[reducible]
        counts[reducible[np.argmin(reducible_scores)]] -= 1

    point_sets = [
        sample_mesh_surface(mesh, n_points)
        for (mesh, _), n_points in zip(mesh_entries, counts)
    ]
    return np.concatenate(point_sets, axis=0)


def infer_surface_sample_points(scene: trimesh.Scene) -> int:
    sample_points = 0
    for node_name in scene.graph.nodes_geometry:
        _, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry[geometry_name]
        if isinstance(geometry, trimesh.Trimesh):
            sample_points += len(geometry.vertices)

    if sample_points <= 0:
        raise ValueError("No mesh vertices found in scene")

    return sample_points


def convert_file(
    src_path: Path,
    dst_dir: Path,
    sample_points: int | None = None,
) -> tuple[Path, tuple[int, int]]:
    scene = trimesh.load(src_path, force="scene")
    target_points = sample_points if sample_points is not None else infer_surface_sample_points(scene)
    points = collect_scene_surface_samples(scene, target_points)

    stem = src_path.stem
    if stem.endswith("_2048"):
        stem = stem[:-5]

    dst_path = dst_dir / f"{stem}.npy"
    np.save(dst_path, points.astype(np.float32))
    return dst_path, points.shape


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GLB meshes to NPY point clouds.")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-points", type=int, default=None, help="Optional number of surface points to sample per GLB scene.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(input_dir.glob("*.glb"))
    if not glb_files:
        raise FileNotFoundError(f"No .glb files found in {input_dir}")

    for src_path in glb_files:
        dst_path, shape = convert_file(
            src_path,
            output_dir,
            sample_points=args.sample_points,
        )
        print(f"{src_path.name} -> {dst_path.name} {shape}")


if __name__ == "__main__":
    main()
