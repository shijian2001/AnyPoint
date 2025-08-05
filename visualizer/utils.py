import numpy as np
from .point_visualizer import ColorScheme, VisualizationConfig, PointCloudVisualizer
from typing import Union, List, Optional

# Convenience functions
def quick_visualize(
    points: Union[np.ndarray, str],
    color_scheme: ColorScheme = ColorScheme.HEIGHT,
    title: str = "Quick Visualization"
) -> None:
    """Quick visualization function for single point cloud.
    
    Args:
        points: Point cloud data or file path
        color_scheme: Color scheme to use
        title: Window title
    """
    config = VisualizationConfig(window_name=title)
    viz = PointCloudVisualizer(config)
    viz.add_point_cloud(points, title)
    viz.visualize(color_scheme, show_statistics=True)


def compare_multiple(
    point_clouds: List[Union[np.ndarray, str]],
    labels: Optional[List[str]] = None,
    save_screenshot: Optional[str] = None
) -> None:
    """Compare multiple point clouds in a single view.
    
    Args:
        point_clouds: List of point cloud data or file paths
        labels: Optional labels for each point cloud
        save_screenshot: Optional path to save screenshot
    """
    viz = PointCloudVisualizer()
    
    for i, pcd in enumerate(point_clouds):
        label = labels[i] if labels and i < len(labels) else f"Cloud {i+1}"
        viz.add_point_cloud(pcd, label)
    
    viz.visualize(save_screenshot=save_screenshot, show_statistics=True)