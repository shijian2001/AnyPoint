import numpy as np
from visualizer import PointCloudVisualizer, ColorScheme

# points = np.random.randn(1000, 3)

viz = PointCloudVisualizer()
viz.add_point_cloud("./data/000.npy", "My Cloud")
viz.visualize(ColorScheme.ORIGINAL)