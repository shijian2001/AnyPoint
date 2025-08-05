import numpy as np
from visualizer import PointCloudVisualizer, ColorScheme

points = np.random.randn(1000, 3)

viz = PointCloudVisualizer()
viz.add_point_cloud(points, "My Cloud")
viz.visualize(ColorScheme.DENSITY)