"""
Information utility calculator.

Formula:
    U(t) = max_{e∈E} ⟨v_t, v_e⟩ - λ·max_{h∈H} ⟨v_t, v_h⟩
"""

from typing import Optional
import numpy as np


class UtilityCalculator:
    """
    Compute information utility balancing exploitation vs exploration.
    
    Mathematical Formulation:
    ──────────────────────────────────────────────────────────────────
        U(t) = max_{e∈E} ⟨v_t, v_e⟩ - λ · max_{h∈H} ⟨v_t, v_h⟩
    
    where:
        • t: Candidate task
        • E: Error set (tasks where model failed)
        • H: History set (all tested tasks)
        • v_t, v_e, v_h: Task embeddings (normalized)
        • ⟨·,·⟩: Cosine similarity (dot product of normalized vectors)
        • λ ∈ [0,1]: Exploration weight
    
    Interpretation:
        • First term (Exploit): High if t is similar to known errors
        • Second term (Explore): Penalty if t is similar to tested tasks
        • λ=0: Pure exploitation
        • λ=1: Pure exploration
        • λ=0.2: Default (slight bias toward exploitation)
    ──────────────────────────────────────────────────────────────────
    """
    
    def __init__(self, lambda_explore: float = 0.2):
        self.lambda_explore = lambda_explore
    
    def compute(
        self,
        v_candidates: np.ndarray,        # (N, D)
        v_history: Optional[np.ndarray],  # (|H|, D)
        v_errors: Optional[np.ndarray]    # (|E|, D)
    ) -> np.ndarray:
        """
        Compute utility scores: U(t) = Exploit(t) - λ·Explore(t)
        
        Returns:
            (N,) utility scores
        """
        # Exploit: max similarity to errors
        exploit = self._max_sim(v_candidates, v_errors)
        
        # Explore penalty: max similarity to history
        explore = self._max_sim(v_candidates, v_history)
        
        # U(t) = Exploit - λ·Explore
        return exploit - self.lambda_explore * explore
    
    @staticmethod
    def _max_sim(v_candidates: np.ndarray, v_set: Optional[np.ndarray]) -> np.ndarray:
        """Compute max cosine similarity to a set."""
        if v_set is None or len(v_set) == 0:
            return np.zeros(len(v_candidates))
        
        # Matrix multiply: (N, D) @ (D, M) = (N, M)
        similarities = v_candidates @ v_set.T
        return similarities.max(axis=1)

