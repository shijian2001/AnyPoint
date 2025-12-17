#!/usr/bin/env python3
"""
Visualize a random layout from the generated layouts.

Usage:
    python vis_layout.py              # Use default spheres
    python vis_layout.py --objects    # Load actual object point clouds
"""

import json
import random
import argparse
from visualizer.layout_visualizer import LayoutVisualizer

# Mapping from common object names to point cloud files
# (Only needed if you want to use actual object models)
DEFAULT_OBJECT_NAMES = [
    "table", "chair", "lamp", "book", "cup", 
    "sphere", "cube", "cylinder", "cone", "pyramid"
]

def create_object_mapping(layout, available_objects):
    """Create a mapping from obj_X to actual object names.
    
    Args:
        layout: Layout dict
        available_objects: List of available object names
        
    Returns:
        Dict mapping obj_X to actual object name
    """
    mapping = {}
    n_objects = len(layout["objects"])
    
    # Simple strategy: randomly assign available objects
    for i, obj_spec in enumerate(layout["objects"]):
        obj_name = obj_spec["name"]
        if i < len(available_objects):
            mapping[obj_name] = available_objects[i]
        else:
            # Cycle through if more objects than available
            mapping[obj_name] = available_objects[i % len(available_objects)]
    
    return mapping

def main():
    parser = argparse.ArgumentParser(description="Visualize random layout")
    parser.add_argument('--objects', action='store_true', 
                       help='Use actual object point clouds (default: use spheres)')
    parser.add_argument('--seed', type=int, help='Random seed for layout selection')
    args = parser.parse_args()
    
    # Load layouts
    with open('data/layout/outputs_gpt_oss/layouts.json', 'r') as f:
        layouts = json.load(f)
    
    print(f'Total layouts: {len(layouts)}')
    
    # Pick a random layout
    if args.seed is not None:
        random.seed(args.seed)
    layout = random.choice(layouts)
    
    print(f'\n{"="*60}')
    print(f'Visualizing Layout (ID: {layout.get("id", "N/A")})')
    print(f'{"="*60}')
    print(f'Description:\n  {layout["description"]}')
    print(f'\nObjects ({len(layout["objects"])}):')
    for obj in layout['objects']:
        pos = obj['position']
        size = obj['size']
        # Handle both tuple (new AABB) and scalar (legacy) formats
        if isinstance(size, (list, tuple)):
            size_str = f'({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})'
        else:
            size_str = f'{size:.2f}'
        print(f'  - {obj["name"]}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), '
              f'size={size_str}, rot={obj["rotation"]:.0f}°')
    
    # Visualize
    viz = LayoutVisualizer(objects_dir='data/layout/objects')
    
    if args.objects:
        # Use actual object point clouds
        print(f'\nMode: Loading actual object point clouds (sampled to 8192 points)')
        object_mapping = create_object_mapping(layout, DEFAULT_OBJECT_NAMES)
        print(f'Object mapping: {object_mapping}')
        viz.load_layout(layout, object_mapping=object_mapping)
    else:
        # Use default cubes
        print(f'\nMode: Using default cubes (8192 points each)')
        viz.load_layout(layout)
    
    print()
    viz.visualize(show_statistics=True)

if __name__ == "__main__":
    main()