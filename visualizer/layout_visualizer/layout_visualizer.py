"""Layout visualizer for rendering 3D scene layouts with object point clouds."""

import numpy as np
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import json

from ..point_visualizer import PointCloudVisualizer, VisualizationConfig, ColorScheme


@dataclass
class LayoutVisualizationConfig(VisualizationConfig):
    """Extended configuration for layout visualization.
    
    Attributes:
        show_bounding_boxes: Whether to show object bounding boxes
        show_labels: Whether to show object labels
        show_ground_plane: Whether to show ground plane
        ground_plane_size: Size of the ground plane
        object_colormap: Color scheme for different objects
    """
    show_bounding_boxes: bool = False
    show_labels: bool = True
    show_ground_plane: bool = False
    ground_plane_size: float = 12.0
    object_colors: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.8, 0.3, 0.3),   # Red
        (0.3, 0.8, 0.3),   # Green
        (0.3, 0.3, 0.8),   # Blue
        (0.8, 0.8, 0.3),   # Yellow
        (0.8, 0.3, 0.8),   # Magenta
        (0.3, 0.8, 0.8),   # Cyan
        (0.8, 0.5, 0.3),   # Orange
        (0.5, 0.3, 0.8),   # Purple
        (0.3, 0.6, 0.5),   # Teal
        (0.6, 0.5, 0.3),   # Brown
    ])


class LayoutVisualizer:
    """Visualizer for 3D scene layouts with object point clouds.
    
    This class takes a layout specification and renders it by:
    1. Loading object point clouds from files
    2. Transforming them according to layout positions/rotations/scales
    3. Rendering the complete scene
    
    Example:
        >>> viz = LayoutVisualizer(objects_dir="data/layout/objects")
        >>> viz.load_layout(layout_dict)
        >>> viz.visualize()
    """
    
    def __init__(
        self,
        objects_dir: Union[str, Path],
        config: Optional[LayoutVisualizationConfig] = None
    ):
        """Initialize layout visualizer.
        
        Args:
            objects_dir: Directory containing object point cloud files (.npy or .ply)
            config: Visualization configuration
        """
        self.objects_dir = Path(objects_dir)
        self.config = config or LayoutVisualizationConfig()
        
        # Cache for loaded object templates
        self._object_cache: Dict[str, np.ndarray] = {}
        
        # Current layout data
        self._layout: Optional[dict] = None
        self._transformed_objects: List[o3d.geometry.PointCloud] = []
        self._object_labels: List[str] = []
        
        # Load available objects
        self._available_objects = self._scan_objects()
    
    def _scan_objects(self) -> Dict[str, Path]:
        """Scan objects directory for available point cloud files."""
        objects = {}
        if self.objects_dir.exists():
            for ext in ['*.npy', '*.ply', '*.pcd']:
                for f in self.objects_dir.glob(ext):
                    objects[f.stem] = f
        return objects
    
    def get_available_objects(self) -> List[str]:
        """Get list of available object names."""
        return list(self._available_objects.keys())
    
    def load_object_template(self, name: str) -> np.ndarray:
        """Load an object template point cloud.
        
        Args:
            name: Object name (without extension)
            
        Returns:
            Point cloud as numpy array (N, 3) or (N, 6) with colors
        """
        if name in self._object_cache:
            return self._object_cache[name]
        
        if name not in self._available_objects:
            raise ValueError(f"Object '{name}' not found. Available: {list(self._available_objects.keys())}")
        
        path = self._available_objects[name]
        
        if path.suffix == '.npy':
            points = np.load(path)
        else:
            pcd = o3d.io.read_point_cloud(str(path))
            points = np.asarray(pcd.points)
            # Preserve colors if available
            if pcd.has_colors():
                colors = np.asarray(pcd.colors)
                points = np.hstack([points, colors])
        
        # Normalize to unit sphere centered at origin
        points = self._normalize_point_cloud(points)
        
        self._object_cache[name] = points
        return points
    
    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit AABB centered at origin.
        
        The normalized AABB has half-extents of (0.5, 0.5, 0.5),
        meaning it ranges from -0.5 to 0.5 in each dimension.
        
        Important: The min/max bounds of the point cloud will be EXACTLY at
        [-0.5, 0.5] in each dimension, ensuring proper contact for "on" relations.
        
        Preserves color information if present (columns 3:6).
        """
        # Separate xyz and optional colors
        xyz = points[:, :3]
        colors = points[:, 3:] if points.shape[1] > 3 else None
        
        # Get actual AABB bounds
        min_coords = xyz.min(axis=0)
        max_coords = xyz.max(axis=0)
        ranges = max_coords - min_coords
        
        # Compute center of AABB (not mean of points!)
        aabb_center = (max_coords + min_coords) / 2
        
        # Center at origin
        xyz = xyz - aabb_center
        
        # Scale each dimension independently so that
        # min -> -0.5 and max -> 0.5 (EXACTLY)
        if np.any(ranges > 0):
            xyz = xyz / ranges  # Now points are in [-0.5, 0.5] exactly
        
        # Recombine with colors if present
        return np.hstack([xyz, colors]) if colors is not None else xyz
    
    def load_layout(self, layout: dict, object_mapping: Optional[Dict[str, str]] = None) -> 'LayoutVisualizer':
        """Load a layout specification.
        
        Args:
            layout: Layout dict with format:
                {
                    "description": "...",
                    "objects": [
                        {"name": "obj_0", "position": [x, y, z], "rotation": deg, "size": scale},
                        ...
                    ]
                }
            object_mapping: Optional dict mapping obj_X -> actual object name
                e.g., {"obj_0": "table", "obj_1": "chair"}
                If provided, will try to load and sample point clouds.
                If None, uses default sphere.
                
        Returns:
            Self for method chaining
        """
        self._layout = layout
        self._transformed_objects = []
        self._object_labels = []
        
        for i, obj_spec in enumerate(layout.get("objects", [])):
            obj_name = obj_spec["name"]
            position = np.array(obj_spec["position"])
            rotation = obj_spec.get("rotation", 0)
            size = obj_spec.get("size", [1.0, 1.0, 1.0])
            
            # Convert size to numpy array (handle both old scalar and new tuple format)
            if isinstance(size, (int, float)):
                # Legacy support: scalar size means uniform scaling
                size = np.array([size, size, size])
            else:
                size = np.array(size)
            
            # Load template: use object mapping if provided, else use cube
            if object_mapping and obj_name in object_mapping:
                actual_name = object_mapping[obj_name]
                try:
                    template = self._load_and_sample_object(actual_name, n_samples=8192)
                except (ValueError, FileNotFoundError):
                    # Fallback to cube if object file not found
                    template = self._create_placeholder_cube()
            else:
                # No mapping: use default cube
                template = self._create_placeholder_cube()
            
            # Separate geometry and colors (if present)
            has_colors = template.shape[1] == 6
            points_xyz = template[:, :3]
            colors_rgb = template[:, 3:6] if has_colors else None
            
            # Transform geometry
            transformed = self._transform_object(points_xyz, position, rotation, size)
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed)
            
            # Use original colors if available, otherwise assign predefined color
            if has_colors:
                # Normalize colors to [0, 1] if needed
                if colors_rgb.max() > 1.0:
                    colors_rgb = colors_rgb / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            else:
                color = self.config.object_colors[i % len(self.config.object_colors)]
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(color, (len(transformed), 1))
                )
            
            self._transformed_objects.append(pcd)
            self._object_labels.append(obj_name)
        
        return self
    
    def _create_placeholder_cube(self, n_points: int = 8192) -> np.ndarray:
        """Create a placeholder cube point cloud with uniform sampling.
        
        Returns points uniformly distributed in a unit cube [-0.5, 0.5]^3.
        """
        # Generate random points uniformly in unit cube
        points = np.random.uniform(-0.5, 0.5, (n_points, 3))
        return points
    
    def _load_and_sample_object(self, name: str, n_samples: int = 8192) -> np.ndarray:
        """Load an object point cloud and randomly sample to n_samples points.
        
        Args:
            name: Object name (e.g., "table", "chair")
            n_samples: Target number of points (default: 8192)
            
        Returns:
            Sampled and normalized point cloud (n_samples, 3) or (n_samples, 6) with colors
        """
        # Load raw point cloud (preserves colors if available)
        raw_points = self.load_object_template(name)
        n_raw = len(raw_points)
        
        # Sample indices
        if n_raw >= n_samples:
            indices = np.random.choice(n_raw, size=n_samples, replace=False)
        else:
            indices = np.random.choice(n_raw, size=n_samples, replace=True)
        
        return raw_points[indices]
    
    def _transform_object(
        self,
        points: np.ndarray,
        position: np.ndarray,
        rotation: float,
        size: np.ndarray
    ) -> np.ndarray:
        """Transform object points according to layout specification.
        
        Args:
            points: Original point cloud (N, 3) - normalized to unit AABB
            position: Target position (x, y, z) - center of AABB
            rotation: Rotation angle in degrees (around Y axis)
            size: AABB half-extents (x, y, z)
            
        Returns:
            Transformed point cloud
        """
        # Scale - apply different scale to each dimension
        # Points are in [-0.5, 0.5]^3, so multiply by 2*half_extents to get full AABB
        transformed = points * (size * 2)
        
        # Rotate around Y axis
        angle_rad = np.radians(rotation)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        transformed = transformed @ rotation_matrix.T
        
        # Translate
        transformed = transformed + position
        
        return transformed
    
    def visualize(
        self,
        save_screenshot: Optional[str] = None,
        show_statistics: bool = False
    ) -> None:
        """Visualize the loaded layout.
        
        Args:
            save_screenshot: Optional path to save screenshot
            show_statistics: Whether to print statistics
        """
        if not self._transformed_objects:
            raise ValueError("No layout loaded. Call load_layout() first.")
        
        geometries = list(self._transformed_objects)
        
        # Add ground plane
        if self.config.show_ground_plane:
            ground = self._create_ground_plane()
            geometries.append(ground)
        
        # Add bounding boxes
        if self.config.show_bounding_boxes:
            for pcd, label in zip(self._transformed_objects, self._object_labels):
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox.color = (0.5, 0.5, 0.5)
                geometries.append(bbox)
        
        # Print statistics
        if show_statistics:
            self._print_layout_statistics()
        
        # Render
        title = self._layout.get("description", "Layout Visualization")[:50]
        self._render(geometries, title, save_screenshot)
    
    def _create_ground_plane(self) -> o3d.geometry.PointCloud:
        """Create a ground plane point cloud."""
        size = self.config.ground_plane_size
        density = 0.3  # Points per unit
        
        n_points = int((size * 2) ** 2 * density)
        x = np.random.uniform(-size, size, n_points)
        z = np.random.uniform(-size, size, n_points)
        y = np.zeros(n_points)
        
        points = np.column_stack([x, y, z])
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.9, 0.9, 0.9], (n_points, 1))
        )
        
        return pcd
    
    def _render(
        self,
        geometries: List,
        title: str,
        save_screenshot: Optional[str]
    ) -> None:
        """Render geometries using Open3D."""
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=title,
            width=self.config.width,
            height=self.config.height
        )
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        if self.config.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coord_frame)
        
        # Configure render options
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray(self.config.background_color)
        render_opt.point_size = self.config.point_size
        render_opt.light_on = self.config.light_on
        
        if save_screenshot:
            vis.run()
            vis.capture_screen_image(save_screenshot)
            print(f"Screenshot saved to: {save_screenshot}")
        else:
            vis.run()
        
        vis.destroy_window()
    
    def _print_layout_statistics(self) -> None:
        """Print layout statistics."""
        print("\n" + "=" * 60)
        print("LAYOUT STATISTICS")
        print("=" * 60)
        
        if self._layout:
            print(f"Description: {self._layout.get('description', 'N/A')[:80]}")
            print(f"Objects: {len(self._transformed_objects)}")
            
            for i, (pcd, label) in enumerate(zip(self._transformed_objects, self._object_labels)):
                points = np.asarray(pcd.points)
                center = points.mean(axis=0)
                print(f"  [{i}] {label}: center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}), "
                      f"points={len(points)}")
        
        print("=" * 60)
    
    def save_layout_image(
        self,
        output_path: str,
        view_angle: float = 45.0
    ) -> None:
        """Save layout as image with specific view angle.
        
        Args:
            output_path: Output image path
            view_angle: Rotation angle for view
        """
        self.visualize(save_screenshot=output_path)
    
    def clear(self) -> None:
        """Clear current layout."""
        self._layout = None
        self._transformed_objects = []
        self._object_labels = []


def visualize_layout(
    layout: dict,
    objects_dir: Union[str, Path],
    save_path: Optional[str] = None
) -> None:
    """Convenience function to visualize a single layout.
    
    Args:
        layout: Layout dictionary
        objects_dir: Path to objects directory
        save_path: Optional path to save screenshot
    """
    viz = LayoutVisualizer(objects_dir)
    viz.load_layout(layout)
    viz.visualize(save_screenshot=save_path, show_statistics=True)


def visualize_layouts_batch(
    layouts: List[dict],
    objects_dir: Union[str, Path],
    output_dir: str,
    prefix: str = "layout"
) -> None:
    """Visualize multiple layouts and save as images.
    
    Args:
        layouts: List of layout dictionaries
        objects_dir: Path to objects directory
        output_dir: Output directory for images
        prefix: Filename prefix
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    viz = LayoutVisualizer(objects_dir)
    
    for i, layout in enumerate(layouts):
        viz.clear()
        viz.load_layout(layout)
        
        save_path = output_path / f"{prefix}_{i:04d}.png"
        viz.visualize(save_screenshot=str(save_path))
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(layouts)} layouts")
    
    print(f"All {len(layouts)} layouts saved to: {output_dir}")
