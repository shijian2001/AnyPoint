"""Configuration for dynamic evaluation."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class EvalConfig:
    """
    Dynamic evaluation configuration.
    
    Algorithm parameters (see formulation in utility.py):
        budget: Total evaluation budget (B in paper)
        batch_size: Tasks per iteration (K in paper)
        pool_size: Total size of the pre-generated fixed candidate pool
        lambda_explore: Exploration weight (λ in paper, λ ∈ [0,1])
    """
    budget: int                         # B: Total evaluation budget
    batch_size: int                     # K: Batch size per iteration
    pool_size: int                      # N: Fixed candidate pool size
    lambda_explore: float = 0.2         # λ: Exploration weight
    seed: int = 42
    # Selection strategy: tradeoff (our method, λ) | exploit_only | explore_only
    #                     | random | acd_style | autobencher_style | sea_style
    strategy: str = "tradeoff"
    # Fixed-pool source: load candidates from this index instead of generating.
    pool_index_path: Optional[str] = None   # e.g. .../eval_pool_1m.jsonl (has layout)
    dataset_root: Optional[str] = None       # root that `point` paths are relative to
    emb_dir: Optional[str] = None            # precomputed embeddings dir (anypoint_2m_emb)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Single task evaluation result."""
    task_id: int
    question: str
    answer: str
    model_raw_output: str
    model_answer: str
    is_correct: bool
    utility: Optional[float] = None
    category: Optional[str] = None
    options: Optional[List[str]] = None
    layout_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
