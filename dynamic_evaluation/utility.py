"""
Information utility calculator.

Each candidate task t is scored from two cosine-similarity signals:

    error   = max_sim(t, errors)    # similarity to already-wrong tasks  (exploit)
    correct = max_sim(t, correct)   # similarity to already-right tasks
    (both clipped to [0, 1]; an empty set gives 0)

We want tasks that look like past errors but not like past correct answers.
Four combination forms are supported (controlled per-strategy via
UTILITY_STRATEGIES in compare_eval_strategies.py); λ ∈ [0, 1] is the
explore/exploit weight:

    tradeoff:      U(t) = error^(1-λ) · (1 - correct)^λ
        DEFAULT. Weighted geometric mean of "like errors" (exploit) and "unlike
        correct" (explore). λ=0 → pure exploit, λ=1 → pure explore. Output in [0,1].

    tradeoff_log:  log U(t) = (1-λ)·log(error) + λ·log(1 - correct)
        Log form of `tradeoff` — same ranking (strict monotone transform), more
        numerically stable when error or (1-correct) is small.

    linear:        U(t) = error - λ·correct
        Linear penalty. Can produce negative scores; λ has no probabilistic meaning.

    gated:         U(t) = error · (1 - λ·correct)
        Multiplicative gating: the penalty is scaled by error, so an irrelevant
        candidate (error=0) is not penalised for being redundant.
"""

from typing import Optional
import numpy as np


_EPS = 1e-8


class UtilityCalculator:
    """Compute utility scores for dynamic evaluation."""

    def __init__(self, lambda_explore: float = 0.2, form: str = "tradeoff"):
        self.lambda_explore = float(lambda_explore)
        if form not in ("linear", "gated", "tradeoff", "tradeoff_log"):
            raise ValueError(f"Unknown utility form: {form}")
        self.form = form

    # In all forms: `error` = max_sim(t, errors), `correct` = max_sim(t, correct);
    # the explore term is novelty = 1 - correct.
    @staticmethod
    def _linear(error, correct, lam):
        # U = error - λ·correct
        return error - lam * correct

    @staticmethod
    def _gated(error, correct, lam):
        # U = error · (1 - λ·correct)
        return error * (1.0 - lam * correct)

    @staticmethod
    def _tradeoff(error, correct, lam):
        # U = error^(1-λ) · (1-correct)^λ  (weighted geometric mean). Output in [0, 1].
        lam = float(np.clip(lam, 0.0, 1.0))
        exploit = np.clip(error, _EPS, 1.0)
        novelty = np.clip(1.0 - correct, _EPS, 1.0)
        return (exploit ** (1.0 - lam)) * (novelty ** lam)

    @staticmethod
    def _tradeoff_log(error, correct, lam):
        # log U = (1-λ)·log(error) + λ·log(1-correct). Same ranking as _tradeoff
        # (strict monotone transform), more stable when either term is small.
        lam = float(np.clip(lam, 0.0, 1.0))
        log_exploit = np.log(np.clip(error, _EPS, 1.0))
        log_novelty = np.log(np.clip(1.0 - correct, _EPS, 1.0))
        return (1.0 - lam) * log_exploit + lam * log_novelty

    def _combine(self, error, correct):
        if self.form == "linear":
            return self._linear(error, correct, self.lambda_explore)
        if self.form == "gated":
            return self._gated(error, correct, self.lambda_explore)
        if self.form == "tradeoff_log":
            return self._tradeoff_log(error, correct, self.lambda_explore)
        return self._tradeoff(error, correct, self.lambda_explore)

    def compute(
        self,
        v_candidates: np.ndarray,
        v_correct: Optional[np.ndarray],
        v_errors: Optional[np.ndarray],
    ) -> np.ndarray:
        error = np.clip(self._max_sim(v_candidates, v_errors), 0.0, 1.0)
        correct = np.clip(self._max_sim(v_candidates, v_correct), 0.0, 1.0)
        return self._combine(error, correct)

    def compute_strategy(
        self,
        strategy: str,
        v_candidates: np.ndarray,
        v_correct: Optional[np.ndarray],
        v_errors: Optional[np.ndarray],
    ) -> np.ndarray:
        """Score candidates under one named strategy (computes similarities first)."""
        error = self._max_sim(v_candidates, v_errors)
        correct = self._max_sim(v_candidates, v_correct)
        return self.score(strategy, error, correct)

    def score(self, strategy: str, error: np.ndarray, correct: np.ndarray) -> np.ndarray:
        """Apply a strategy's utility formula to PRECOMPUTED similarity arrays.

        ``error`` = max_sim(t, errors), ``correct`` = max_sim(t, correct). Lets callers
        that maintain similarities incrementally (e.g. a fixed 1M pool) reuse the exact
        same formulas without recomputing max-sim here.
        """
        error = np.clip(error, 0.0, 1.0)
        correct = np.clip(correct, 0.0, 1.0)
        lam = self.lambda_explore
        if strategy == "linear":
            return self._linear(error, correct, lam)
        if strategy == "gated":
            return self._gated(error, correct, lam)
        if strategy == "tradeoff":
            return self._tradeoff(error, correct, lam)
        if strategy == "tradeoff_log":
            return self._tradeoff_log(error, correct, lam)
        if strategy == "exploit_only":   # only chase errors: U = error  (== tradeoff λ=0)
            return error
        if strategy == "explore_only":   # only flee solved regions: U = 1 - correct  (== tradeoff λ=1, no log)
            return 1.0 - correct
        raise ValueError(f"Unknown utility strategy: {strategy}")

    @staticmethod
    def _max_sim(v_candidates: np.ndarray, v_set: Optional[np.ndarray]) -> np.ndarray:
        if v_set is None or len(v_set) == 0:
            return np.zeros(len(v_candidates))
        similarities = v_candidates @ v_set.T
        return similarities.max(axis=1)
