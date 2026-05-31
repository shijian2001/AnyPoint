"""
Information utility calculator.

Two utility forms are supported for ablation/comparison:
    sub:  U(t) = Affinity(t, E) - λ·Redundancy(t, C)
    mul:  U(t) = Affinity(t, E) · (1 - λ·Redundancy(t, C))
"""

from typing import Optional
import numpy as np


class UtilityCalculator:
    """
    Compute utility scores for dynamic evaluation.
    """

    def __init__(self, lambda_explore: float = 0.2, form: str = "sub"):
        self.lambda_explore = lambda_explore
        if form not in ("sub", "mul"):
            raise ValueError(f"Unknown utility form: {form}")
        self.form = form

    def _combine(self, affinity: np.ndarray, redundancy: np.ndarray) -> np.ndarray:
        if self.form == "sub":
            return affinity - self.lambda_explore * redundancy
        # mul
        return affinity * (1.0 - self.lambda_explore * redundancy)

    def compute(
        self,
        v_candidates: np.ndarray,         # (N, D)
        v_correct: Optional[np.ndarray],  # (|C|, D)
        v_errors: Optional[np.ndarray],   # (|E|, D)
    ) -> np.ndarray:
        affinity = np.clip(self._max_sim(v_candidates, v_errors), 0.0, 1.0)
        redundancy = np.clip(self._max_sim(v_candidates, v_correct), 0.0, 1.0)
        return self._combine(affinity, redundancy)

    def compute_strategy(
        self,
        strategy: str,
        v_candidates: np.ndarray,
        v_correct: Optional[np.ndarray],
        v_errors: Optional[np.ndarray],
    ) -> np.ndarray:
        """Compute utility scores for one comparison strategy."""
        affinity = np.clip(self._max_sim(v_candidates, v_errors), 0.0, 1.0)
        redundancy = np.clip(self._max_sim(v_candidates, v_correct), 0.0, 1.0)

        if strategy == "dynamic":
            return affinity - self.lambda_explore * redundancy
        if strategy == "dynamic_mul":
            return affinity * (1.0 - self.lambda_explore * redundancy)
        if strategy == "affinity_only":
            return affinity
        if strategy == "novelty_only":
            return 1.0 - self.lambda_explore * redundancy if self.form == "mul" else -redundancy

        raise ValueError(f"Unknown utility strategy: {strategy}")

    @staticmethod
    def _max_sim(v_candidates: np.ndarray, v_set: Optional[np.ndarray]) -> np.ndarray:
        """Compute max cosine similarity to a set."""
        if v_set is None or len(v_set) == 0:
            return np.zeros(len(v_candidates))
        similarities = v_candidates @ v_set.T
        return similarities.max(axis=1)
