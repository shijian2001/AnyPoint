"""Stochastic Error Ascent (Song et al., 2025; arXiv:2503.23361) adapter.

Faithful port of SEA's Algorithm 1 to AnyPoint's fixed-pool 3D
multiple-choice QA setting. Preserves the three distinguishing ideas of SEA:

1. **Per-source Top-k error-similarity retrieval (Eq. 4).** The next batch is
   the union of Top-k candidates per active source, then random-subsampled to
   the caller's batch size (mirrors the paper's 50 -> 40 subsample).
2. **Hierarchical retrieval (document -> paragraph).** Instantiated here as
   layout -> task: stage 1 restricts candidates to those whose scene layout is
   Top-k similar to some source layout; stage 2 ranks the survivors by
   full-task similarity.
3. **Relation DAG + cumulative-error pruning (Eq. 5).** Each new source
   points to the source that attracted it. Once a source's average descendant
   error rate drops below ``pruning_threshold``, it is deactivated.

AnyPoint runs one shot per task (no rephrasing), so per-sample error is
binary. The paper's ``xi = gamma = 0.5`` therefore becomes: ``err == 1``
promotes a task to a source; ``average descendant error < 0.5`` prunes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dynamic_evaluation.embedder import TaskEmbedder
from dynamic_evaluation.task_pool import PoolItem
from point_qa_generator.base import Task


@dataclass
class _SourceNode:
    embedding: np.ndarray
    layout_embedding: np.ndarray
    parent_idx: Optional[int]
    active: bool = True
    descendant_errors: List[int] = field(default_factory=list)


class SEAState:
    """Maintains SEA's source set, relation DAG, and per-batch selection."""

    def __init__(
        self,
        embedder: TaskEmbedder,
        rng: np.random.RandomState,
        top_k_per_source: int = 50,
        layout_top_k: int = 16,
        pruning_threshold: float = 0.5,
        hierarchical: bool = True,
    ) -> None:
        self.embedder = embedder
        self.rng = rng
        self.top_k_per_source = top_k_per_source
        self.layout_top_k = layout_top_k
        self.pruning_threshold = pruning_threshold
        self.hierarchical = hierarchical

        self.sources: List[_SourceNode] = []
        self._last_parent_map: Dict[int, int] = {}
        self._last_full_embs: Optional[np.ndarray] = None
        self._last_layout_embs: Optional[np.ndarray] = None

    def seed(self, tasks: Sequence[Task]) -> None:
        """Initialise sources from a cold-start batch's error subset."""
        if not tasks:
            return
        full, layout = self._encode(list(tasks))
        for f, l in zip(full, layout):
            self.sources.append(
                _SourceNode(embedding=f, layout_embedding=l, parent_idx=None)
            )

    def select(self, remaining_items: Sequence[PoolItem], k: int) -> List[int]:
        """Return indices within ``remaining_items`` to evaluate next."""
        n = len(remaining_items)
        if k <= 0 or n == 0:
            return []

        active = [(i, s) for i, s in enumerate(self.sources) if s.active]
        if not active:
            return self._random_indices(n, k)

        tasks = [item.task for item in remaining_items]
        cand_full, cand_layout = self._encode(tasks)

        active_global_idx = [i for i, _ in active]
        src_full = np.stack([s.embedding for _, s in active])
        src_layout = np.stack([s.layout_embedding for _, s in active])

        mask = np.ones(n, dtype=bool)
        if self.hierarchical:
            mask = self._layout_prefilter(src_layout, cand_layout, n)
            if not mask.any():
                mask = np.ones(n, dtype=bool)

        sim = src_full @ cand_full.T
        sim_masked = np.where(mask[None, :], sim, -np.inf)

        per_source_best: Dict[int, Tuple[int, float]] = {}
        union: set = set()
        valid_total = int(mask.sum())
        k_per = min(self.top_k_per_source, valid_total)
        if k_per <= 0:
            return self._random_indices(n, k)

        for local_idx, row in enumerate(sim_masked):
            finite = np.isfinite(row).sum()
            if finite == 0:
                continue
            take = int(min(k_per, finite))
            if take <= 0:
                continue
            top = np.argpartition(-row, take - 1)[:take]
            for c in top:
                ci = int(c)
                if not np.isfinite(row[ci]):
                    continue
                union.add(ci)
                cur = per_source_best.get(ci)
                if cur is None or row[ci] > cur[1]:
                    per_source_best[ci] = (active_global_idx[local_idx], float(row[ci]))

        if not union:
            return self._random_indices(n, k)

        union_list = list(union)
        if len(union_list) > k:
            chosen = self.rng.choice(union_list, size=k, replace=False).tolist()
        elif len(union_list) < k:
            remaining_pool = [i for i in range(n) if i not in union]
            pad = min(k - len(union_list), len(remaining_pool))
            if pad > 0:
                extra = self.rng.choice(remaining_pool, size=pad, replace=False).tolist()
                chosen = list(union_list) + list(extra)
            else:
                chosen = list(union_list)
        else:
            chosen = list(union_list)

        self._last_parent_map = {
            int(ci): per_source_best[ci][0]
            for ci in chosen
            if ci in per_source_best
        }
        self._last_full_embs = cand_full
        self._last_layout_embs = cand_layout
        return [int(ci) for ci in chosen]

    def update(
        self,
        selected_indices: Sequence[int],
        is_correct_list: Sequence[bool],
    ) -> None:
        """Update the relation DAG and prune stale sources after evaluation."""
        if self._last_full_embs is None or self._last_layout_embs is None:
            return
        for cidx, is_correct in zip(selected_indices, is_correct_list):
            parent = self._last_parent_map.get(int(cidx))
            err = 0 if is_correct else 1
            if parent is not None:
                self._propagate_error(parent, err)
            if err == 1:
                self.sources.append(
                    _SourceNode(
                        embedding=self._last_full_embs[cidx],
                        layout_embedding=self._last_layout_embs[cidx],
                        parent_idx=parent,
                    )
                )

        for s in self.sources:
            if s.active and s.descendant_errors:
                if float(np.mean(s.descendant_errors)) < self.pruning_threshold:
                    s.active = False

        self._last_parent_map = {}
        self._last_full_embs = None
        self._last_layout_embs = None

    def stats(self) -> Dict[str, int]:
        return {
            "total_sources": len(self.sources),
            "active_sources": sum(1 for s in self.sources if s.active),
            "pruned_sources": sum(1 for s in self.sources if not s.active),
        }

    def _propagate_error(self, source_idx: int, err: int) -> None:
        seen: set = set()
        cur: Optional[int] = source_idx
        while cur is not None and cur not in seen:
            seen.add(cur)
            self.sources[cur].descendant_errors.append(err)
            cur = self.sources[cur].parent_idx

    def _encode(self, tasks: List[Task]) -> Tuple[np.ndarray, np.ndarray]:
        full = self.embedder.encode(tasks)
        layout_texts = [TaskEmbedder._get_layout(t) for t in tasks]
        layout = self.embedder._encode_texts(layout_texts, show_progress=False)
        return full, layout

    def _layout_prefilter(
        self,
        src_layout: np.ndarray,
        cand_layout: np.ndarray,
        n: int,
    ) -> np.ndarray:
        layout_sim = src_layout @ cand_layout.T
        keep = np.zeros(n, dtype=bool)
        k_doc = min(self.layout_top_k, n)
        if k_doc <= 0:
            keep[:] = True
            return keep
        for row in layout_sim:
            top = np.argpartition(-row, k_doc - 1)[:k_doc]
            keep[top] = True
        return keep

    def _random_indices(self, n: int, k: int) -> List[int]:
        size = min(k, n)
        if size <= 0:
            return []
        return self.rng.choice(n, size=size, replace=False).tolist()
