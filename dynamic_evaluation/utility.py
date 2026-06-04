"""
Information utility calculator.

Four utility forms are supported (controlled per-strategy via UTILITY_STRATEGIES
in compare_eval_strategies.py):

    sub:  U(t) = Affinity(t, E) - λ·Redundancy(t, C)
        Linear penalty. Treats redundancy as an unconditional cost regardless
        of how relevant the candidate is. Can produce negative scores; λ has
        no probabilistic meaning.

    mul:  U(t) = Affinity(t, E) · (1 - λ·Redundancy(t, C))
        Multiplicative gating: redundancy is scaled by affinity, so an
        irrelevant candidate (A=0) is not penalised for being redundant.
        Empirically strong, but λ is still just a linear gain on R; (1-λR)
        is not a probability and λ>1 produces negative factors.

    geo:  U(t) = Affinity(t, E)^(1-λ) · (1 - Redundancy(t, C))^λ        (λ ∈ [0,1])
        Weighted geometric mean of affinity and novelty (1-R). λ is a proper
        convex mixing weight: λ=0 → pure affinity, λ=1 → pure novelty.
        Output is in [0,1].

    geo_log: log U(t) = (1-λ)·log A + λ·log(1-R)                       (λ ∈ [0,1])
        Same ranking as geo (strict monotone log transform of it). Useful for
        numerical stability when A or (1-R) is small, and for connecting to
        the log-linear / max-entropy family. Provided as a separate strategy
        so we can confirm empirically that the two forms produce identical
        top-k selections.
"""

from typing import Optional
import numpy as np


_EPS = 1e-8


class UtilityCalculator:
    """Compute utility scores for dynamic evaluation."""

    def __init__(self, lambda_explore: float = 0.2, form: str = "sub"):
        self.lambda_explore = float(lambda_explore)
        if form not in ("sub", "mul", "geo", "geo_log"):
            raise ValueError(f"Unknown utility form: {form}")
        self.form = form

    @staticmethod
    def _sub(affinity, redundancy, lam):
        return affinity - lam * redundancy

    @staticmethod
    def _mul(affinity, redundancy, lam):
        return affinity * (1.0 - lam * redundancy)

    @staticmethod
    def _geo(affinity, redundancy, lam):
        # Weighted geometric mean (exp form): U = A^(1-λ) * (1-R)^λ. Output in [0, 1].
        lam = float(np.clip(lam, 0.0, 1.0))
        a = np.clip(affinity, _EPS, 1.0)
        n = np.clip(1.0 - redundancy, _EPS, 1.0)
        return (a ** (1.0 - lam)) * (n ** lam)

    @staticmethod
    def _geo_log(affinity, redundancy, lam):
        # log U = (1-λ) log A + λ log(1-R). Argmax-equivalent to _geo (strict monotone
        # transform), more numerically stable when A or (1-R) is small. Output in (-inf, 0].
        lam = float(np.clip(lam, 0.0, 1.0))
        log_aff = np.log(np.clip(affinity, _EPS, 1.0))
        log_novel = np.log(np.clip(1.0 - redundancy, _EPS, 1.0))
        return (1.0 - lam) * log_aff + lam * log_novel

    def _combine(self, affinity, redundancy):
        if self.form == "sub":
            return self._sub(affinity, redundancy, self.lambda_explore)
        if self.form == "mul":
            return self._mul(affinity, redundancy, self.lambda_explore)
        if self.form == "geo_log":
            return self._geo_log(affinity, redundancy, self.lambda_explore)
        return self._geo(affinity, redundancy, self.lambda_explore)

    def compute(
        self,
        v_candidates: np.ndarray,
        v_correct: Optional[np.ndarray],
        v_errors: Optional[np.ndarray],
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
        """Score candidates under one named strategy."""
        affinity = np.clip(self._max_sim(v_candidates, v_errors), 0.0, 1.0)
        redundancy = np.clip(self._max_sim(v_candidates, v_correct), 0.0, 1.0)
        lam = self.lambda_explore

        if strategy == "dynamic":           # sub
            return self._sub(affinity, redundancy, lam)
        if strategy == "dynamic_mul":       # heuristic mul
            return self._mul(affinity, redundancy, lam)
        if strategy == "dynamic_geo":       # weighted geometric mean (exp form): A^(1-λ)·(1-R)^λ
            return self._geo(affinity, redundancy, lam)
        if strategy == "dynamic_geo_log":   # weighted geometric mean (log form): (1-λ)logA + λ log(1-R)
            return self._geo_log(affinity, redundancy, lam)
        if strategy == "affinity_only":
            return affinity
        if strategy == "novelty_only":
            return np.log(np.clip(1.0 - redundancy, _EPS, 1.0))

        raise ValueError(f"Unknown utility strategy: {strategy}")

    @staticmethod
    def _max_sim(v_candidates: np.ndarray, v_set: Optional[np.ndarray]) -> np.ndarray:
        if v_set is None or len(v_set) == 0:
            return np.zeros(len(v_candidates))
        similarities = v_candidates @ v_set.T
        return similarities.max(axis=1)
