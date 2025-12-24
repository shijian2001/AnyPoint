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
        pool_size: Candidate pool size (N in paper, N >> K)
        lambda_explore: Exploration weight (λ in paper, λ ∈ [0,1])
    """
    budget: int                         # B: Total evaluation budget
    batch_size: int                     # K: Batch size per iteration
    pool_size: int                      # N: Candidate pool size
    lambda_explore: float = 0.2         # λ: Exploration weight
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskResult:
    """Single task evaluation result."""
    task_id: int
    question: str
    answer: str
    model_answer: str
    is_correct: bool
    utility: Optional[float] = None
    category: Optional[str] = None
    options: Optional[List[str]] = None
    layout_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

