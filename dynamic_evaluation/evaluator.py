"""Dynamic evaluator: main evaluation engine."""

import os
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from tqdm import tqdm

from point_qa_generator.base import Task
from point_qa_generator.generator import PointQAGenerator
from models.point_qa_model import PointQAModel

from .config import EvalConfig, TaskResult
from .embedder import TaskEmbedder
from .utility import UtilityCalculator
from .task_pool import TaskPool


class DynamicEvaluator:
    """
    Dynamic evaluation with utility-driven sampling.
    
    Algorithm:
        1. Cold start: Random batch → Initialize H, E
        2. Loop while evaluated < budget:
            a. Generate pool_size candidates
            b. Compute U(t) for each
            c. Select top batch_size by utility
            d. Evaluate selected tasks
            e. Update H and E
    
    Key sets:
        H: History (all tested tasks)
        E: Errors (failed tasks)
    """
    
    def __init__(
        self,
        qa_generator: PointQAGenerator,
        model: PointQAModel,
        config: EvalConfig
    ):
        self.gen = qa_generator
        self.model = model
        self.cfg = config
        
        # Components
        self.embedder = TaskEmbedder()
        self.utility = UtilityCalculator(config.lambda_explore)
        self.pool = TaskPool(qa_generator, config.seed)
        
        # State: H and E
        self.H_tasks: List[Task] = []
        self.H_embs: Optional[np.ndarray] = None
        
        self.E_tasks: List[Task] = []
        self.E_embs: Optional[np.ndarray] = None
        self.E_point_clouds: List[np.ndarray] = []
        self.E_indices: List[int] = []
        
        # Results
        self.results: List[TaskResult] = []
        self.n_eval = 0
    
    def run(self, output_dir: str) -> Dict[str, Any]:
        """Execute evaluation pipeline."""
        os.makedirs(output_dir, exist_ok=True)
        
        self._print_header()
        
        # Phase 1: Cold start
        self._cold_start()
        
        # Phase 2: Iterative
        iteration = 1
        while self.n_eval < self.cfg.budget:
            self._iterate(iteration)
            iteration += 1
        
        # Save
        summary = self._save(output_dir)
        self._print_summary(summary)
        
        return summary
    
    def _cold_start(self):
        """Initialize with random batch."""
        print("🔥 Cold Start\n")
        
        candidates = self.pool.sample(self.cfg.batch_size)
        self._evaluate(candidates, phase="cold_start")
        self._update()
        
        err_rate = len(self.E_tasks) / len(self.H_tasks)
        print(f"✓ Initial: |H|={len(self.H_tasks)}, |E|={len(self.E_tasks)} ({err_rate:.1%})\n")
    
    def _iterate(self, iteration: int):
        """Single evaluation iteration."""
        remaining = self.cfg.budget - self.n_eval
        k = min(self.cfg.batch_size, remaining)
        
        print(f"{'─'*70}")
        print(f"🔄 Iter {iteration}: {self.n_eval}/{self.cfg.budget} | |H|={len(self.H_tasks)} |E|={len(self.E_tasks)}")
        print(f"{'─'*70}\n")
        
        # Generate candidates
        candidates = self.pool.sample(self.cfg.pool_size)
        print(f"Generated {len(candidates)} candidates")
        
        # Select top-K by utility
        selected, utilities = self._select_topk(candidates, k)
        print(f"Selected top-{k}: U ∈ [{utilities[0]:.3f}, {utilities[-1]:.3f}]\n")
        
        # Evaluate
        self._evaluate(selected, utilities, phase="dynamic")
        self._update()
        
        err_rate = len(self.E_tasks) / len(self.H_tasks)
        print(f"✓ Cumulative: |E|={len(self.E_tasks)} ({err_rate:.1%})\n")
    
    def _select_topk(
        self,
        candidates: List[Tuple[Task, np.ndarray]],
        k: int
    ) -> Tuple[List[Tuple[Task, np.ndarray]], List[float]]:
        """Select top-K by utility."""
        
        tasks = [t for t, _ in candidates]
        v_cand = self.embedder.encode(tasks)
        
        # U(t) = max(sim(t,E)) - λ·max(sim(t,H))
        scores = self.utility.compute(v_cand, self.H_embs, self.E_embs)
        
        # Top-K
        top_idx = np.argsort(scores)[-k:][::-1]
        
        selected = [candidates[i] for i in top_idx]
        selected_u = [scores[i] for i in top_idx]
        
        return selected, selected_u
    
    def _evaluate(
        self,
        batch: List[Tuple[Task, np.ndarray]],
        utilities: Optional[List[float]] = None,
        phase: str = "eval"
    ):
        """Evaluate a batch."""
        if utilities is None:
            utilities = [None] * len(batch)
        
        for (task, pc), u in tqdm(
            zip(batch, utilities),
            total=len(batch),
            desc=phase
        ):
            result = self._eval_single(task, pc, u)
            
            self.H_tasks.append(task)
            self.results.append(result)
            
            if not result.is_correct:
                self.E_tasks.append(task)
                self.E_point_clouds.append(pc)
                self.E_indices.append(self.n_eval)
            
            self.n_eval += 1
    
    def _eval_single(
        self,
        task: Task,
        pc: np.ndarray,
        u: Optional[float]
    ) -> TaskResult:
        """Evaluate single task."""
        result = self.model.multiple_choice_qa(
            data={'point_cloud': pc},
            question=task.question,
            choices=task.options,
            answer=task.answer
        )
        
        # Get instantiated layout description
        layout_desc = self.embedder._get_layout(task) if task.metadata else None
        
        # Infer category from metadata
        category = self._infer_category(task)
        
        return TaskResult(
            task_id=self.n_eval,
            question=task.question,
            answer=task.answer,
            model_answer=result['multiple_choice_answer'],
            is_correct=(result['accuracy'] == 1),
            utility=u,
            category=category,
            options=task.options,
            layout_description=layout_desc
        )
    
    def _update(self):
        """Update H and E embeddings."""
        if self.H_tasks:
            self.H_embs = self.embedder.encode(self.H_tasks)
        if self.E_tasks:
            self.E_embs = self.embedder.encode(self.E_tasks)
    
    def _save(self, output_dir: str) -> Dict[str, Any]:
        """Save results."""
        summary = {
            'config': self.cfg.to_dict(),
            'stats': {
                'total': self.n_eval,
                'errors': len(self.E_tasks),
                'error_rate': len(self.E_tasks) / self.n_eval,
                'error_indices': self.E_indices
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        path = os.path.join(output_dir, 'results.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save hard_data (error tasks in standard generator format)
        if self.E_tasks:
            self._save_hard_data(output_dir)
        
        print(f"\n📁 {path}")
        return summary
    
    def _save_hard_data(self, output_dir: str):
        """Save error tasks in standard generator format."""
        hard_dir = os.path.join(output_dir, 'hard_data')
        pcd_dir = os.path.join(hard_dir, 'pcd')
        os.makedirs(pcd_dir, exist_ok=True)
        
        task_records = []
        
        for i, (task, pc) in enumerate(zip(self.E_tasks, self.E_point_clouds)):
            # Save point cloud
            pcd_filename = f"{i:06d}.npy"
            np.save(os.path.join(pcd_dir, pcd_filename), pc)
            
            # Standard format task record
            record = {
                "question_id": i,
                "point": pcd_filename,
                "category": self._infer_category(task),
                "question": task.question,
                "options": task.options,
                "answer": task.answer
            }
            task_records.append(record)
        
        # Save tasks.jsonl
        tasks_file = os.path.join(hard_dir, "tasks.jsonl")
        with open(tasks_file, 'w', encoding='utf-8') as f:
            for record in task_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        # Save tasks_info.json (standard format)
        tasks_info = {
            "task_plan": {
                "generator_type": "mixed",
                "num_options": 4,
                "seed": self.cfg.seed
            },
            "generation_stats": {
                "num_tasks_requested": len(self.E_tasks),
                "num_tasks_generated": len(self.E_tasks),
                "output_directory": hard_dir
            }
        }
        
        info_file = os.path.join(hard_dir, "tasks_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_info, f, indent=2, ensure_ascii=False)
        
        print(f"📁 {hard_dir}/ ({len(self.E_tasks)} hard tasks)")
    
    def _infer_category(self, task: Task) -> str:
        """Get category from task metadata (generator_type + config)."""
        if task.metadata:
            gen_type = task.metadata.get('generator_type', '')
            config = task.metadata.get('generator_config', {})
            
            if gen_type:
                # Construct category same as generator.py L249
                dist_type = config.get('distance_type', '')
                return f"{gen_type}_{dist_type}" if dist_type else gen_type
        
        # Fallback for tasks without metadata (shouldn't happen)
        return "unknown"
    
    def _print_header(self):
        c = self.cfg
        print(f"\n{'='*70}")
        print(f"Dynamic Evaluation")
        print(f"  Budget: {c.budget} | Batch: {c.batch_size} | Pool: {c.pool_size} | λ: {c.lambda_explore}")
        print(f"{'='*70}\n")
    
    def _print_summary(self, summary: Dict):
        s = summary['stats']
        print(f"\n{'='*70}")
        print(f"🎉 Complete: {s['total']} evaluated, {s['errors']} errors ({s['error_rate']:.1%})")
        print(f"{'='*70}\n")

