"""Dynamic evaluator — OUR utility-driven algorithm on a fixed, pre-built pool.

The candidate pool is loaded from an index (cfg.pool_index_path) and its task
vectors come PRECOMPUTED (cfg.emb_dir, Qwen3-Embedding-8B, 3 components). Each
round re-scores the remaining pool by utility, using incrementally-maintained
max-similarity to the error/correct sets (no per-round re-encoding).

Grading: an LLM judge (models.judger.Judger) extracts which option the model
chose from its free-form answer; correctness is then a deterministic compare to
ground truth.

cfg.strategy ∈ {tradeoff (our method, λ), exploit_only, explore_only}.
Baselines (random / acd_style / autobencher_style / sea_style) are separate
comparison methods with their own driver — intentionally not coupled here.
"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from tqdm import tqdm

from point_qa_generator.base import Task

from .config import EvalConfig, TaskResult
from .utility import UtilityCalculator
from .task_pool import PoolItem

_EPS = 1e-8
# tradeoff is our method (λ-swept; λ=0 == pure exploit, λ=1 == pure explore, so the
# two ablations are just its endpoints — no separate exploit_only/explore_only arm).
# random is the baseline lower bound (uniformly random selection each round).
OUR_STRATEGIES = ("tradeoff", "random")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def natural_prompt(question: str, options: List[str]) -> str:
    """Multiple-choice prompt with inline labeled options; free-form response (the LLM
    judge extracts the choice afterwards).

    NOTE on format: options are listed INLINE on a single line, not as a multi-line
    'Options:\\nA. ..\\nB. ..' block followed by 'Select exactly one option.'. That
    rigid block makes some models degenerate — MiniGPT3D emits only the stop sign
    '###' (-> empty answer) and OneLLM collapses to a constant letter regardless of the
    cloud. The inline phrasing keeps every model producing a real, cloud-grounded
    answer while still showing the option set so the judge can map it back."""
    labeled = "  ".join(f"{chr(65 + i)}) {o}" for i, o in enumerate(options))
    return ("Based on the provided point cloud, answer the question.\n"
            f"Question: {question}\nOptions: {labeled}\nYour answer:")


class DynamicEvaluator:
    def __init__(self, model, config: EvalConfig):
        if config.strategy not in OUR_STRATEGIES:
            raise ValueError(f"DynamicEvaluator runs {OUR_STRATEGIES}; '{config.strategy}' "
                             f"is a baseline — run it via the separate baseline driver.")
        if not config.pool_index_path or not config.emb_dir:
            raise ValueError("config needs pool_index_path and emb_dir.")
        self.model = model
        self.cfg = config
        self.batch_size = max(1, int(config.batch_size))

        import torch
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 candidate matrix, but accumulate dot products in fp32 (cosine precision)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

        from models.judger import Judger
        self.judge = Judger()

        # Fixed pool + precomputed candidate matrix (row == item_id) — built ONCE.
        # V is model- and strategy-independent (just the pool's embeddings), so a sweep
        # can reuse it (and the loaded model) across arms via reset().
        self.pool_items: List[PoolItem] = []
        self._load_pool()
        self.V = self._build_vectors()                       # (N, D) fp16 on GPU
        self.reset()                                         # per-run state

    def reset(self, strategy: Optional[str] = None, lambda_explore: Optional[float] = None,
              seed: Optional[int] = None) -> "DynamicEvaluator":
        """Reset per-run state for a fresh arm WITHOUT rebuilding V or reloading the model.
        Lets a sweep reuse the 24GB pool matrix + loaded model across (strategy, λ) arms.
        Unspecified args keep the current config value."""
        if strategy is not None:
            if strategy not in OUR_STRATEGIES:
                raise ValueError(f"DynamicEvaluator runs {OUR_STRATEGIES}; '{strategy}' is a baseline.")
            self.cfg.strategy = strategy
        if lambda_explore is not None:
            self.cfg.lambda_explore = float(lambda_explore)
        if seed is not None:
            self.cfg.seed = int(seed)
        self.strategy = self.cfg.strategy
        self.utility = UtilityCalculator(self.cfg.lambda_explore)
        self.rng = np.random.RandomState(self.cfg.seed)

        n = len(self.pool_items)
        self.remaining_ids: List[int] = list(range(n))
        self.max_sim_E = self.torch.zeros(n, device=self.device)  # running max-sim to errors
        self.max_sim_C = self.torch.zeros(n, device=self.device)  # running max-sim to correct
        self.H_tasks: List[Task] = []
        self.C_tasks: List[Task] = []
        self.E_tasks: List[Task] = []
        self.E_point_cloud_paths: List[str] = []
        self.E_indices: List[int] = []
        self.results: List[TaskResult] = []
        self.n_eval = 0
        self._pending_new: Tuple[List[int], List[int]] = ([], [])
        return self

    # ---------- pool + precomputed vectors ----------
    def _load_pool(self):
        root = self.cfg.dataset_root or ""
        print(f"[pool] loading {self.cfg.pool_index_path}", flush=True)
        self.pool_qids: List[int] = []
        with open(self.cfg.pool_index_path) as f:
            for i, line in enumerate(f):
                r = json.loads(line)
                task = Task(
                    point=os.path.join(root, r["point"]) if root else r["point"],
                    question=r["question"], options=list(r["options"]), answer=r["answer"],
                    metadata={"layout_description": r.get("layout", ""), "objects": [],
                              "category": r.get("category", ""), "generator_type": r.get("generator", "")},
                )
                self.pool_items.append(PoolItem(item_id=i, task=task, point_cloud=None))
                self.pool_qids.append(r["question_id"])
        self.remaining_ids = list(range(len(self.pool_items)))
        print(f"[pool] {len(self.pool_items)} candidates", flush=True)

    def _build_vectors(self):
        """Gather precomputed 3 components for the pool, concat + L2-normalize -> GPU fp16."""
        emb = self.cfg.emb_dir
        ids = json.load(open(os.path.join(emb, "ids.json")))
        id2row = {qid: row for row, qid in enumerate(ids)}
        rows = np.asarray([id2row[q] for q in self.pool_qids], dtype=np.int64)
        comps = [np.asarray(np.load(os.path.join(emb, f"{c}.fp16.npy"), mmap_mode="r")[rows],
                            dtype=np.float32) for c in ("layout", "question", "answer")]
        cat = np.concatenate(comps, axis=1)
        cat /= (np.linalg.norm(cat, axis=1, keepdims=True) + _EPS)
        V = self.torch.from_numpy(cat.astype(np.float16)).to(self.device)
        print(f"[pool] V={tuple(V.shape)} on {self.device}", flush=True)
        return V

    def _update_max_sim(self, new_ids: List[int], running):
        """running = max(running, max over new vectors of cos-sim to ALL pool vectors).
        V stays fp16 (no 2x full-matrix copy); the matmul accumulates in fp32."""
        if not new_ids:
            return running
        nv = self.V[self.torch.as_tensor(new_ids, device=self.device)]  # (m, D) fp16
        sims = (nv @ self.V.T).float()                                   # (m, N), fp32 accum
        return self.torch.maximum(running, sims.amax(dim=0))

    # ---------- main loop ----------
    def run(self, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        self._print_header()
        self._cold_start()
        it = 1
        while self.n_eval < self.cfg.budget:
            if not self._iterate(it):
                break
            it += 1
        summary = self._save(output_dir)
        self._print_summary(summary)
        return summary

    def _cold_start(self):
        print("🔥 Cold Start\n", flush=True)
        k = min(self.batch_size, len(self.remaining_ids))
        picked = [self.remaining_ids[p] for p in self.rng.choice(len(self.remaining_ids), size=k, replace=False)]
        self._pop(picked)
        self._evaluate([self.pool_items[i] for i in picked], phase="cold_start")
        self._update()

    def _iterate(self, iteration: int) -> bool:
        budget_left = self.cfg.budget - self.n_eval
        rem = [self.pool_items[i] for i in self.remaining_ids]
        k = min(self.batch_size, budget_left, len(rem))
        if k <= 0:
            return False
        print(f"{'─'*70}\n🔄 Iter {iteration}: {self.n_eval}/{self.cfg.budget} | "
              f"|H|={len(self.H_tasks)} |E|={len(self.E_tasks)} | rem={len(rem)}", flush=True)
        batch, utils = self._select(rem, k)
        self._evaluate(batch, utils, phase=self.strategy)
        self._update()
        return True

    def _select(self, rem: List[PoolItem], k: int) -> Tuple[List[PoolItem], List[float]]:
        if self.strategy == "random":                            # baseline: uniform random pick
            pos = self.rng.choice(len(rem), size=k, replace=False)
            batch = [rem[p] for p in pos]
            self._pop([it.item_id for it in batch])
            return batch, [None] * len(batch)
        ids = self.torch.as_tensor([it.item_id for it in rem], device=self.device)
        error = self.max_sim_E[ids].float().cpu().numpy()      # incrementally-maintained sims
        correct = self.max_sim_C[ids].float().cpu().numpy()
        scores = self.utility.score(self.strategy, error, correct)   # reuse utility.py formulas
        top = np.argpartition(scores, -k)[-k:]                       # O(N) top-k
        order = top[np.argsort(scores[top])[::-1]]                   # high → low
        batch = [rem[p] for p in order]
        self._pop([it.item_id for it in batch])
        return batch, [float(scores[p]) for p in order]

    def _pop(self, item_ids: List[int]):
        drop = set(item_ids)
        self.remaining_ids = [i for i in self.remaining_ids if i not in drop]

    def _evaluate(self, batch: List[PoolItem], utilities: Optional[List[float]] = None, phase: str = "eval"):
        if utilities is None:
            utilities = [None] * len(batch)
        new_E_ids, new_C_ids = [], []
        cs = self.batch_size
        for st in tqdm(range(0, len(batch), cs), desc=phase):
            chunk = list(zip(batch[st:st + cs], utilities[st:st + cs]))
            tasks = [it.task for it, _ in chunk]
            datas = [{"point_cloud_path": t.point} for t in tasks]   # index pool: .npy already exists
            questions = [t.question for t in tasks]
            options_list = [t.options for t in tasks]
            prompts = [natural_prompt(q, o) for q, o in zip(questions, options_list)]
            free = self.model._qa_batch(datas, prompts)                      # free-form text
            picks = self.judge.extract_batch(questions, options_list, free)  # option idx or None
            for (item, u), task, raw, pick in zip(chunk, tasks, free, picks):
                model_answer = task.options[pick] if pick is not None else ""
                is_correct = pick is not None and model_answer == task.answer
                self.results.append(TaskResult(
                    task_id=self.n_eval, question=task.question, answer=task.answer,
                    model_raw_output=raw, model_answer=model_answer, is_correct=is_correct,
                    utility=u, category=self._infer_category(task), options=task.options,
                    layout_description=(task.metadata or {}).get("layout_description")))
                self.H_tasks.append(task)
                if is_correct:
                    self.C_tasks.append(task)
                    new_C_ids.append(item.item_id)
                else:
                    self.E_tasks.append(task)
                    self.E_point_cloud_paths.append(task.point)
                    self.E_indices.append(self.n_eval)
                    new_E_ids.append(item.item_id)
                self.n_eval += 1
        self._pending_new = (new_E_ids, new_C_ids)

    def _update(self):
        new_E_ids, new_C_ids = self._pending_new
        self.max_sim_E = self._update_max_sim(new_E_ids, self.max_sim_E)
        self.max_sim_C = self._update_max_sim(new_C_ids, self.max_sim_C)
        self._pending_new = ([], [])

    # ---------- io / helpers ----------
    def _save(self, output_dir: str) -> Dict[str, Any]:
        summary = _to_jsonable({
            "config": self.cfg.to_dict(),
            "stats": {"total": self.n_eval, "errors": len(self.E_tasks),
                      "error_rate": len(self.E_tasks) / max(1, self.n_eval),
                      "error_indices": self.E_indices},
            "results": [r.to_dict() for r in self.results],
        })
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        if self.E_tasks:
            self._save_hard_data(output_dir)
        print(f"\n📁 {os.path.join(output_dir, 'results.json')}", flush=True)
        return summary

    def _save_hard_data(self, output_dir: str):
        hard_dir = os.path.join(output_dir, "hard_data")
        pcd_dir = os.path.join(hard_dir, "pcd")
        os.makedirs(pcd_dir, exist_ok=True)
        recs = []
        for i, (task, src) in enumerate(zip(self.E_tasks, self.E_point_cloud_paths)):
            if src and os.path.exists(src):
                try:
                    np.save(os.path.join(pcd_dir, f"{i:06d}.npy"), np.load(src))
                except Exception:  # noqa: BLE001
                    pass
            recs.append({"question_id": i, "point": f"pcd/{i:06d}.npy",
                         "category": self._infer_category(task), "question": task.question,
                         "options": task.options, "answer": task.answer,
                         "layout": (task.metadata or {}).get("layout_description", "")})
        with open(os.path.join(hard_dir, "tasks.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"📁 {hard_dir}/ ({len(self.E_tasks)} hard tasks)", flush=True)

    def _infer_category(self, task: Task) -> str:
        return (task.metadata or {}).get("category") or "unknown"

    def _print_header(self):
        c = self.cfg
        print(f"\n{'='*70}\nDynamic Evaluation | strategy={c.strategy}\n"
              f"  Budget: {c.budget} | Batch: {c.batch_size} | Pool: {len(self.pool_items)} | "
              f"λ: {c.lambda_explore}\n{'='*70}\n", flush=True)

    def _print_summary(self, summary: Dict):
        s = summary["stats"]
        print(f"\n{'='*70}\n🎉 {s['total']} evaluated, {s['errors']} errors "
              f"({s['error_rate']:.1%}) | strategy={self.cfg.strategy}\n{'='*70}\n", flush=True)
