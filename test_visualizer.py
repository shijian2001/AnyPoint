import numpy as np
from visualizer import PointCloudVisualizer, ColorScheme

# points = np.random.randn(1000, 3)

viz = PointCloudVisualizer()
viz.add_point_cloud("./data/test/test_npy/063386fe274e49848a471c65e2fdbd6f.npy", "My Cloud")
viz.visualize(ColorScheme.ORIGINAL)


# ============ Layout Visualization Example ============
# from visualizer.layout_visualizer import LayoutVisualizer
#
# # Example layout (from layout_gen.py output)
# layout = {
#     "description": "A table with a cup on top and a chair beside",
#     "objects": [
#         {"name": "table", "position": [0.0, 1.1, 0.0], "rotation": 0, "size": 2.2},
#         {"name": "cup", "position": [0.1, 2.4, 0.1], "rotation": 0, "size": 0.4},
#         {"name": "chair", "position": [2.5, 0.7, 0.0], "rotation": 90, "size": 1.4},
#     ]
# }
#
# viz = LayoutVisualizer(objects_dir="data/layout/objects")
# viz.load_layout(layout)
# viz.visualize(show_statistics=True)
# ======================================================