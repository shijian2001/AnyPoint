import numpy as np
import open3d as o3d
from typing import Union, List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class ColorScheme(Enum):
    """Color scheme enumeration for point cloud visualization."""
    UNIFORM = "uniform"
    HEIGHT = "height" 
    DEPTH = "depth"
    DENSITY = "density"
    CLUSTER = "cluster"
    RAINBOW = "rainbow"
    NORMAL = "normal"


@dataclass
class VisualizationConfig:
    """Configuration for point cloud visualization.
    
    Attributes:
        point_size: Size of rendered points (1.0-10.0)
        background_color: RGB background color (0.0-1.0 for each channel)
        show_coordinate_frame: Whether to display coordinate axes
        window_name: Title of visualization window
        width: Window width in pixels
        height: Window height in pixels
        light_on: Enable lighting effects
    """
    point_size: float = 2.0
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    show_coordinate_frame: bool = True
    window_name: str = "Point Cloud Visualization" 
    width: int = 1200
    height: int = 800
    light_on: bool = True


class PointCloudVisualizer:
    """Professional point cloud visualizer with automatic single/multi-cloud rendering.
    
    This class automatically detects the number of point clouds and renders them
    appropriately - single cloud uses the full window, multiple clouds are 
    distinguished by colors for comparison.
    
    Example:
        # Single cloud
        >>> viz = PointCloudVisualizer()
        >>> viz.add_point_cloud(points, "Scene")
        >>> viz.visualize(ColorScheme.HEIGHT)
        
        # Multiple clouds  
        >>> viz.add_point_cloud(points2, "Scene 2")
        >>> viz.visualize()  # Automatically renders both with different colors
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        """Initialize the point cloud visualizer.
        
        Args:
            config: Visualization configuration. Uses default if None.
        """
        self.config = config or VisualizationConfig()
        self._point_clouds: List[o3d.geometry.PointCloud] = []
        self._labels: List[str] = []
        self._metadata: List[Dict[str, Any]] = []
        
        # Predefined color palette for multi-cloud visualization
        self._uniform_colors = [
            [0.8, 0.2, 0.2],  # Red
            [0.2, 0.8, 0.2],  # Green  
            [0.2, 0.2, 0.8],  # Blue
            [0.8, 0.8, 0.2],  # Yellow
            [0.8, 0.2, 0.8],  # Magenta
            [0.2, 0.8, 0.8],  # Cyan
            [0.8, 0.5, 0.2],  # Orange
            [0.5, 0.2, 0.8],  # Purple
        ]

    def add_point_cloud(
        self,
        points: Union[np.ndarray, str, o3d.geometry.PointCloud],
        label: str = "Point Cloud", 
        colors: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'PointCloudVisualizer':
        """Add a point cloud to the visualizer.
        
        Args:
            points: Point cloud data. Can be:
                   - numpy array (N, 3) or (N, 6) with RGB
                   - string path to point cloud file (.npy, .ply, .pcd, .xyz)
                   - Open3D PointCloud object
            label: Display label for the point cloud
            colors: Optional RGB colors (N, 3), values 0-1 or 0-255
            metadata: Additional metadata dictionary
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If points data is invalid
            FileNotFoundError: If file path doesn't exist
        """
        # Convert input to Open3D point cloud
        if isinstance(points, str):
            pcd = self._load_from_file(points)
        elif isinstance(points, o3d.geometry.PointCloud):
            pcd = points
        elif isinstance(points, np.ndarray):
            pcd = self._create_from_array(points, colors)
        else:
            raise ValueError(f"Unsupported points type: {type(points)}")
        
        self._point_clouds.append(pcd)
        self._labels.append(label)
        self._metadata.append(metadata or {})
        
        return self

    def visualize(
        self,
        color_scheme: ColorScheme = ColorScheme.UNIFORM,
        save_screenshot: Optional[str] = None,
        show_statistics: bool = False
    ) -> None:
        """Visualize point clouds with automatic single/multi-cloud rendering.
        
        Automatically detects number of point clouds:
        - Single cloud: applies the specified color scheme
        - Multiple clouds: uses uniform colors to distinguish clouds
        
        Args:
            color_scheme: Color scheme to apply (ignored for multi-cloud mode)
            save_screenshot: Path to save screenshot
            show_statistics: Whether to print statistics
        """
        if not self._point_clouds:
            raise ValueError("No point clouds added. Use add_point_cloud() first.")
        
        n_clouds = len(self._point_clouds)
        
        if n_clouds == 1:
            # Single cloud mode - apply specified color scheme
            self._apply_color_scheme(0, color_scheme)
            window_title = f"{self.config.window_name} - {self._labels[0]}"
        else:
            # Multi-cloud mode - use uniform colors for distinction
            self._apply_multi_cloud_colors()
            labels_str = ", ".join(self._labels[:3])
            if n_clouds > 3:
                labels_str += f" (+{n_clouds-3} more)"
            window_title = f"{self.config.window_name} - {labels_str}"
        
        if show_statistics:
            self._print_statistics()
        
        # Render with Open3D
        self._render_open3d(window_title, save_screenshot)

    def create_animation_frames(
        self,
        output_dir: str,
        n_frames: int = 36,
        color_scheme: ColorScheme = ColorScheme.HEIGHT
    ) -> None:
        """Create animation frames by rotating the view.
        
        Args:
            output_dir: Directory to save frames
            n_frames: Number of frames to generate
            color_scheme: Color scheme to use (for single cloud)
        """
        if not self._point_clouds:
            raise ValueError("No point clouds added")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Apply colors based on cloud count
        if len(self._point_clouds) == 1:
            self._apply_color_scheme(0, color_scheme)
        else:
            self._apply_multi_cloud_colors()
        
        # Create hidden visualizer for frame generation
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=self.config.width, height=self.config.height, visible=False)
        
        # Add geometries
        for pcd in self._point_clouds:
            vis.add_geometry(pcd)
        
        if self.config.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coord_frame)
        
        self._configure_render_options(vis)
        
        # Generate frames with rotation
        view_control = vis.get_view_control()
        rotation_step = 360.0 / n_frames
        
        for i in range(n_frames):
            view_control.rotate(rotation_step, 0.0)
            vis.poll_events()
            vis.update_renderer()
            
            frame_path = output_path / f"frame_{i:04d}.png"
            vis.capture_screen_image(str(frame_path))
            
            if i % 6 == 0:  # Print progress every 6 frames
                print(f"Generated frames {i+1}/{n_frames}")
        
        vis.destroy_window()
        print(f"Animation frames saved to: {output_dir}")

    def clear(self) -> None:
        """Clear all point clouds."""
        self._point_clouds.clear()
        self._labels.clear()
        self._metadata.clear()

    def export_point_cloud(self, index: int, file_path: str) -> bool:
        """Export a specific point cloud to file.
        
        Args:
            index: Index of point cloud to export
            file_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        if index < 0 or index >= len(self._point_clouds):
            raise IndexError(f"Point cloud index {index} out of range")
        
        success = o3d.io.write_point_cloud(file_path, self._point_clouds[index])
        
        if success:
            print(f"Point cloud '{self._labels[index]}' exported to: {file_path}")
        else:
            print(f"Failed to export point cloud to: {file_path}")
        
        return success

    def get_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded point clouds.
        
        Returns:
            List of dictionaries containing point cloud information
        """
        info_list = []
        
        for i, (pcd, label, metadata) in enumerate(
            zip(self._point_clouds, self._labels, self._metadata)
        ):
            points = np.asarray(pcd.points)
            
            info = {
                'index': i,
                'label': label,
                'n_points': len(points),
                'has_colors': pcd.has_colors(),
                'has_normals': pcd.has_normals(),
                'bounds': {
                    'min': points.min(axis=0).tolist() if len(points) > 0 else None,
                    'max': points.max(axis=0).tolist() if len(points) > 0 else None,
                },
                'metadata': metadata
            }
            info_list.append(info)
        
        return info_list

    # Internal implementation methods
    def _load_from_file(self, file_path: str) -> o3d.geometry.PointCloud:
        """Load point cloud from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {file_path}")
        
        if path.suffix == '.npy':
            data = np.load(file_path)
            return self._create_from_array(data)
        elif path.suffix in ['.ply', '.pcd', '.xyz']:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                raise ValueError(f"Failed to load point cloud from {file_path}")
            return pcd
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _create_from_array(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None
    ) -> o3d.geometry.PointCloud:
        """Create Open3D point cloud from numpy array."""
        points = np.asarray(points, dtype=np.float64)
        
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("Points must be (N, 3) or (N, 6) array")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # Handle colors
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        elif points.shape[1] >= 6:
            colors = points[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def _apply_multi_cloud_colors(self) -> None:
        """Apply uniform colors to distinguish multiple clouds."""
        for i, pcd in enumerate(self._point_clouds):
            points = np.asarray(pcd.points)
            color = self._uniform_colors[i % len(self._uniform_colors)]
            colors = np.tile(color, (len(points), 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)

    def _apply_color_scheme(self, cloud_index: int, scheme: ColorScheme) -> None:
        """Apply color scheme to a specific point cloud."""
        pcd = self._point_clouds[cloud_index]
        points = np.asarray(pcd.points)
        colors = self._compute_colors(points, scheme, cloud_index)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    def _compute_colors(
        self,
        points: np.ndarray,
        scheme: ColorScheme,
        cloud_index: int
    ) -> np.ndarray:
        """Compute colors based on the specified color scheme."""
        n_points = len(points)
        
        if scheme == ColorScheme.UNIFORM:
            color = self._uniform_colors[cloud_index % len(self._uniform_colors)]
            return np.tile(color, (n_points, 1))
        
        elif scheme == ColorScheme.HEIGHT:
            z_values = points[:, 2]
            normalized = self._normalize_values(z_values)
            return plt.cm.viridis(normalized)[:, :3]
        
        elif scheme == ColorScheme.DEPTH:
            distances = np.linalg.norm(points, axis=1)
            normalized = self._normalize_values(distances)
            return plt.cm.plasma(normalized)[:, :3]
        
        elif scheme == ColorScheme.DENSITY:
            return self._compute_density_colors(points)
        
        elif scheme == ColorScheme.CLUSTER:
            return self._compute_cluster_colors(points)
        
        elif scheme == ColorScheme.RAINBOW:
            indices = np.arange(n_points) / max(n_points - 1, 1)
            return plt.cm.hsv(indices)[:, :3]
        
        elif scheme == ColorScheme.NORMAL:
            return self._compute_normal_colors(points)
        
        else:
            # Default to uniform
            return self._compute_colors(points, ColorScheme.UNIFORM, cloud_index)

    def _normalize_values(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to [0, 1] range."""
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def _compute_density_colors(self, points: np.ndarray) -> np.ndarray:
        """Compute colors based on local point density."""
        if len(points) < 10:
            return np.tile([0.5, 0.5, 0.5], (len(points), 1))
        
        k = min(10, len(points) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(points)
        
        distances, _ = nn.kneighbors(points)
        mean_distances = distances[:, 1:].mean(axis=1)
        density = 1.0 / (mean_distances + 1e-8)
        normalized = self._normalize_values(density)
        
        return plt.cm.hot(normalized)[:, :3]

    def _compute_cluster_colors(self, points: np.ndarray) -> np.ndarray:
        """Compute colors based on K-means clustering."""
        n_clusters = min(8, max(2, len(points) // 200))
        
        if len(points) < n_clusters:
            return np.tile([0.5, 0.5, 0.5], (len(points), 1))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(points)
        colors = plt.cm.Set3(labels / n_clusters)[:, :3]
        
        return colors

    def _compute_normal_colors(self, points: np.ndarray) -> np.ndarray:
        """Compute colors based on surface normals."""
        pcd_temp = o3d.geometry.PointCloud()
        pcd_temp.points = o3d.utility.Vector3dVector(points)
        
        pcd_temp.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        normals = np.asarray(pcd_temp.normals)
        colors = (normals + 1.0) / 2.0  # Map [-1,1] to [0,1]
        
        return colors

    def _render_open3d(self, window_title: str, save_screenshot: Optional[str]) -> None:
        """Render using Open3D visualizer."""
        # Update window title
        config = VisualizationConfig(**vars(self.config))
        config.window_name = window_title
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=config.window_name,
            width=config.width,
            height=config.height
        )
        
        # Add all point clouds
        for pcd in self._point_clouds:
            vis.add_geometry(pcd)
        
        # Add coordinate frame if requested
        if config.show_coordinate_frame:
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            vis.add_geometry(coord_frame)
        
        # Configure render options
        self._configure_render_options(vis)
        
        # Run visualization
        if save_screenshot:
            vis.run()
            vis.capture_screen_image(save_screenshot)
            print(f"Screenshot saved to: {save_screenshot}")
        else:
            vis.run()
        
        vis.destroy_window()

    def _configure_render_options(self, vis: o3d.visualization.Visualizer) -> None:
        """Configure Open3D render options."""
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray(self.config.background_color)
        render_opt.point_size = self.config.point_size
        render_opt.light_on = self.config.light_on
        render_opt.show_coordinate_frame = self.config.show_coordinate_frame

    def _print_statistics(self) -> None:
        """Print detailed statistics for all point clouds."""
        print("\n" + "=" * 60)
        print("POINT CLOUD STATISTICS")
        print("=" * 60)
        
        total_points = 0
        
        for i, (pcd, label) in enumerate(zip(self._point_clouds, self._labels)):
            points = np.asarray(pcd.points)
            n_points = len(points)
            total_points += n_points
            
            print(f"\n[{i+1}] {label}:")
            print(f"    Points: {n_points:,}")
            
            if n_points > 0:
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)
                center = points.mean(axis=0)
                
                print(f"    Bounds: X[{min_bound[0]:.3f}, {max_bound[0]:.3f}] "
                      f"Y[{min_bound[1]:.3f}, {max_bound[1]:.3f}] "
                      f"Z[{min_bound[2]:.3f}, {max_bound[2]:.3f}]")
                print(f"    Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                print(f"    Colors: {'Yes' if pcd.has_colors() else 'No'}")
        
        print(f"\nTotal Points: {total_points:,}")
        print(f"Total Point Clouds: {len(self._point_clouds)}")
        print("=" * 60)