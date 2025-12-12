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
    show_ground_plane: bool = True
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
            Point cloud as numpy array (N, 3)
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
        
        # Normalize to unit sphere centered at origin
        points = self._normalize_point_cloud(points)
        
        self._object_cache[name] = points
        return points
    
    def _normalize_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """Normalize point cloud to unit sphere centered at origin."""
        # Center at origin
        center = points.mean(axis=0)
        points = points - center
        
        # Scale to fit in unit sphere
        max_dist = np.linalg.norm(points, axis=1).max()
        if max_dist > 0:
            points = points / max_dist * 0.5  # radius = 0.5
        
        return points
    
    def load_layout(self, layout: dict) -> 'LayoutVisualizer':
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
            size = obj_spec.get("size", 1.0)
            
            # Try to load object template, use placeholder if not found
            try:
                template = self.load_object_template(obj_name)
            except ValueError:
                # Use default sphere as placeholder
                template = self._create_placeholder_sphere()
            
            # Transform object
            transformed = self._transform_object(template, position, rotation, size)
            
            # Create Open3D point cloud with color
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed)
            
            # Assign color based on object index
            color = self.config.object_colors[i % len(self.config.object_colors)]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(color, (len(transformed), 1))
            )
            
            self._transformed_objects.append(pcd)
            self._object_labels.append(obj_name)
        
        return self
    
    def _create_placeholder_sphere(self, n_points: int = 500) -> np.ndarray:
        """Create a placeholder sphere point cloud."""
        # Generate points on unit sphere
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        costheta = np.random.uniform(-1, 1, n_points)
        theta = np.arccos(costheta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.column_stack([x, y, z]) * 0.5
    
    def _transform_object(
        self,
        points: np.ndarray,
        position: np.ndarray,
        rotation: float,
        size: float
    ) -> np.ndarray:
        """Transform object points according to layout specification.
        
        Args:
            points: Original point cloud (N, 3)
            position: Target position (x, y, z)
            rotation: Rotation angle in degrees (around Y axis)
            size: Scale factor
            
        Returns:
            Transformed point cloud
        """
        # Scale
        transformed = points * size
        
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
