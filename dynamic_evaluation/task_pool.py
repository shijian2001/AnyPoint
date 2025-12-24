"""Task candidate pool generator."""

from typing import List, Tuple
import numpy as np

from point_qa_generator.base import Task, TaskPlan
from point_qa_generator.generator import PointQAGenerator


class TaskPool:
    """
    Generate diverse candidates by random sampling.
    
    Returns (Task, point_cloud) tuples for flexible downstream use.
    """
    
    def __init__(self, qa_generator: PointQAGenerator, seed: int = 42):
        self.gen = qa_generator
        self.rng = np.random.RandomState(seed)
        self.gen_types = list(qa_generator.generators.keys())
    
    def sample(self, size: int) -> List[Tuple[Task, np.ndarray]]:
        """
        Sample diverse task candidates.
        
        Args:
            size: Number of candidates to generate
            
        Returns:
            List of (Task, point_cloud) tuples
            
        Usage:
            candidates = pool.sample(1000)
            tasks = [t for t, _ in candidates]
            point_clouds = [pc for _, pc in candidates]
        """
        results = []
        per_type = max(1, size // len(self.gen_types))
        
        for gen_type in self.gen_types:
            generator = self.gen.generators[gen_type]
            
            # Random seed for layout diversity
            plan = TaskPlan(
                generator_type=gen_type,
                num_options=4,
                seed=self.rng.randint(0, 1000000)
            )
            
            try:
                batch = generator.generate_tasks(plan, per_type)
                results.extend(batch)
            except (ValueError, IndexError) as e:
                print(f"⚠️  {gen_type}: {e}")
        
        # Shuffle and trim
        self.rng.shuffle(results)
        return results[:size]

