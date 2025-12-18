import os
import json
from typing import Dict, List, Tuple, Any
import numpy as np
from .base import TaskPlan
from .metadata import PointCloudMetadata
from .distance import (WhatDistanceGenerator, WhereDistanceGenerator,
                       ListAttributeDistanceGenerator, CountAttributeDistanceGenerator)
from .attribute import (WhatAttributeGenerator, ListAttributeGenerator,
                        CountAttributeGenerator)
from .number import (CountObjectGenerator, FrequentObjectGenerator,
                     ListAttributeFrequentGenerator, CountAttributeFrequentGenerator)
from .size import (WhatSizeGenerator, ListAttributeSizeGenerator,
                   CountAttributeSizeGenerator, WhereSizeGenerator)


class PointQAGenerator:
    """Main interface for Point QA generation."""

    def __init__(self, metadata_file: str, pcd_dir: str, layouts_file: str, seed: int = 42):
        """
        Initialize Point QA Generator with layout system.

        Args:
            metadata_file: Path to object metadata JSONL file
            pcd_dir: Directory containing point cloud .npy files
            layouts_file: Path to layouts JSON file
            seed: Random seed
        """
        self.metadata = PointCloudMetadata(metadata_file, pcd_dir, seed)
        self.layouts = self._load_layouts(layouts_file)
        self.layouts_classified = self._classify_layouts(self.layouts)
        self.rng = np.random.RandomState(seed)
        
        self.generators = {
            # Distance-based generators
            "what_distance": WhatDistanceGenerator(self.metadata, seed, self.layouts_classified),
            "where_distance": WhereDistanceGenerator(self.metadata, seed, self.layouts_classified),
            "list_attribute_distance": ListAttributeDistanceGenerator(self.metadata, seed, self.layouts_classified),
            "count_attribute_distance": CountAttributeDistanceGenerator(self.metadata, seed, self.layouts_classified),

            # Attribute-based generators
            "what_attribute": WhatAttributeGenerator(self.metadata, seed, self.layouts_classified),
            "list_attribute": ListAttributeGenerator(self.metadata, seed, self.layouts_classified),
            "count_attribute": CountAttributeGenerator(self.metadata, seed, self.layouts_classified),

            # Number-based generators
            "count_object": CountObjectGenerator(self.metadata, seed, self.layouts_classified),
            "frequent_object": FrequentObjectGenerator(self.metadata, seed, self.layouts_classified),
            "list_attribute_frequent": ListAttributeFrequentGenerator(self.metadata, seed, self.layouts_classified),
            "count_attribute_frequent": CountAttributeFrequentGenerator(self.metadata, seed, self.layouts_classified),

            # Size-based generators
            "what_size": WhatSizeGenerator(self.metadata, seed, self.layouts_classified),
            "list_attribute_size": ListAttributeSizeGenerator(self.metadata, seed, self.layouts_classified),
            "count_attribute_size": CountAttributeSizeGenerator(self.metadata, seed, self.layouts_classified),
            "where_size": WhereSizeGenerator(self.metadata, seed, self.layouts_classified)
        }
    
    def _load_layouts(self, layouts_file: str) -> List[Dict]:
        """Load layouts from JSON file.
        
        Args:
            layouts_file: Path to layouts JSON file
            
        Returns:
            List of layout dictionaries
        """
        with open(layouts_file, 'r', encoding='utf-8') as f:
            layouts = json.load(f)
        return layouts
    
    def _classify_layouts(self, layouts: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify layouts by generator requirements.
        
        Args:
            layouts: List of all layouts
            
        Returns:
            Dictionary with classified layouts:
                - 'standard': Layouts with 2-9 objects (most generators)
                - 'special': Layouts with 3-9 objects (WhatDistance, Number generators)
                - 'all': All layouts
        """
        classified = {
            'standard': [],
            'special': [],
            'all': layouts
        }
        
        for layout in layouts:
            n_objects = len(layout["objects"])
            
            # Most generators need 2-9 objects
            if 2 <= n_objects <= 9:
                classified['standard'].append(layout)
            
            # Special generators (WhatDistance, Number) need 3-9 objects
            if 3 <= n_objects <= 9:
                classified['special'].append(layout)
        
        # Validate requirements
        if not classified['special']:
            print("Warning: No layouts with 3-9 objects found. Special generators may fail.")
        if not classified['standard']:
            raise ValueError("No layouts with at least 2 objects found.")
        
        return classified
    
    def _sample_layout_with_mapping(
        self, 
        task_plan: TaskPlan,
        max_attempts: int = 100
    ) -> Tuple[Dict, Dict[str, Dict]]:
        """Sample a layout and map placeholders to actual objects.
        
        Randomly samples a layout, then randomly maps each placeholder 
        (obj_0, obj_1, ...) to an actual object from metadata. Checks
        compatibility with the generator type.
        
        Args:
            task_plan: Task configuration containing generator type
            max_attempts: Maximum retry attempts if compatibility fails
            
        Returns:
            Tuple of (layout, object_mapping)
            
        Raises:
            RuntimeError: If no compatible layout found after max_attempts
        """
        for _ in range(max_attempts):
            # Sample a random layout
            layout = self.rng.choice(self.layouts)
            
            # Map placeholders to actual objects
            num_objects = len(layout["objects"])
            sampled_objects = self.rng.choice(
                self.metadata.objects, 
                size=num_objects, 
                replace=False
            )
            
            object_mapping = {
                layout["objects"][i]["name"]: sampled_objects[i]
                for i in range(num_objects)
            }
            
            # Check compatibility
            if self._is_layout_compatible(task_plan.generator_type, layout, object_mapping):
                return layout, object_mapping
        
        raise RuntimeError(
            f"Failed to find compatible layout after {max_attempts} attempts "
            f"for generator type: {task_plan.generator_type}"
        )
    
    def _is_layout_compatible(
        self, 
        generator_type: str,
        layout: Dict,
        object_mapping: Dict[str, Dict]
    ) -> bool:
        """Check if layout and object mapping are compatible with generator.
        
        Compatibility rules:
        - Attribute-based: At least one object must have components with attributes
        - Distance/Size-based: At least 2 objects required (guaranteed by layout)
        - Number-based: Always compatible
        
        Args:
            generator_type: Type of generator (e.g., "what_attribute")
            layout: Layout dictionary
            object_mapping: Mapping from placeholders to actual objects
            
        Returns:
            True if compatible, False otherwise
        """
        # Attribute-based generators need objects with component attributes
        if "attribute" in generator_type:
            for actual_obj in object_mapping.values():
                if self.metadata.has_components_with_attribute(actual_obj, "material"):
                    return True
            return False
        
        # Distance/size-based generators need at least 2 objects
        # (layouts guarantee this, but check for safety)
        if "distance" in generator_type or "size" in generator_type:
            return len(layout["objects"]) >= 2
        
        # Number-based generators are always compatible
        return True

    def generate(self, task_plan: TaskPlan, num_tasks: int, output_dir: str) -> Dict[str, Any]:
        """
        Generate QA tasks and save to output directory.

        Args:
            task_plan: Task configuration
            num_tasks: Number of tasks to generate
            output_dir: Output directory path

        Returns:
            Dictionary containing generation statistics
        """
        if task_plan.generator_type not in self.generators:
            raise ValueError(f"Unknown generator type: {task_plan.generator_type}")

        generator = self.generators[task_plan.generator_type]

        os.makedirs(output_dir, exist_ok=True)
        pcd_dir = os.path.join(output_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)

        task_results = generator.generate_tasks(task_plan, num_tasks)

        task_records = []
        all_scene_metadata = []
        
        for i, (task, point_cloud) in enumerate(task_results):
            pcd_path = os.path.join(pcd_dir, task.point)
            np.save(pcd_path, point_cloud)

            # Collect scene metadata
            if task.metadata:
                # Fill placeholders in layout description
                layout_template = task.metadata.get("layout_description", "")
                layout_description = layout_template
                for obj_info in task.metadata.get("objects", []):
                    placeholder = "[" + obj_info["placeholder"] + "]"
                    obj_name = obj_info["name"]
                    layout_description = layout_description.replace(placeholder, obj_name)
                
                scene_metadata = {
                    "scene_id": i,
                    "point_cloud": task.point,
                    "layout_id": task.metadata.get("layout_id"),
                    "layout_template": layout_template,
                    "layout_description": layout_description,
                    "objects": {
                        "count": len(task.metadata.get("objects", [])),
                        "details": task.metadata.get("objects", [])
                    }
                }
                all_scene_metadata.append(scene_metadata)

            task_record = {
                "question_id": i,
                "point": task.point,
                "category": f"{task_plan.generator_type}_{task_plan.generator_config.get('distance_type', '')}",
                "question": task.question,
                "options": task.options,
                "answer": task.answer
            }
            task_records.append(task_record)

        # Save all scene metadata to single file
        if all_scene_metadata:
            metadata_file = os.path.join(pcd_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_scene_metadata, f, indent=2, ensure_ascii=False)

        tasks_file = os.path.join(output_dir, "tasks.jsonl")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for record in task_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        task_info = {
            "task_plan": {
                "generator_type": task_plan.generator_type,
                "num_options": task_plan.num_options,
                "seed": task_plan.seed,
                "generator_config": task_plan.generator_config
            },
            "generation_stats": {
                "num_tasks_requested": num_tasks,
                "num_tasks_generated": len(task_records),
                "output_directory": output_dir
            }
        }

        task_info_file = os.path.join(output_dir, "tasks_info.json")
        with open(task_info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, indent=2, ensure_ascii=False)

        print(f"Generated {len(task_records)} tasks:")
        print(f"  Tasks file: {tasks_file}")
        print(f"  Point clouds: {pcd_dir}")
        print(f"  Task info: {task_info_file}")

        return task_info