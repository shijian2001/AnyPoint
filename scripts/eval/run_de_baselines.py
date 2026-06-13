"""Run the three EXTERNAL comparison baselines (ACD / AutoBencher / SEA) on the
SAME fixed 1M pool, the SAME model and LLM judge, and the SAME result schema as
our dynamic evaluation — but each via its OWN selection logic. This driver does
NOT go through DynamicEvaluator / the utility loop; it is the separate baseline
driver the evaluator docstring refers to.

All three see the identical 1M candidate pool (cfg.pool_index) and an identical
budget, so error counts are directly comparable to our tradeoff arms. The
selection is accelerated to the 1M pool WITHOUT changing each method's behavior:

  * ACD  — per-item complexity is static, so each fine category is pre-sorted by
           complexity ONCE; every round only the (30) category scores are
           recomputed via ACD's own scoring helpers, then round-robin popped.
           Behaviorally identical to calling select_acd_style_indices per round.
  * AutoBencher — exact MMR via an incrementally-maintained max-Jaccard to the
           selected set (the same max-similarity trick our evaluator uses), so
           no candidate cap / approximation is needed.
  * SEA  — the per-source <-> candidate similarity (and the layout pre-filter)
           run on the GPU against a precomputed fp16 matrix (full =
           layout|question|answer concat; layout = layout component only),
           instead of re-encoding the pool with SBERT each round. The DAG /
           promotion / pruning logic is a faithful port of sea.py.

The two adaptive baselines (ACD, SEA) get the SAME random cold-start batch as our
tradeoff/random arms (same seed, batch_size) so every method starts from the
same first observation; SEA in particular is *defined* to seed its sources from
a cold-start error subset. AutoBencher is static (no model feedback) and just
selects `budget` items up front.

Example (one model, all three baselines):
    JUDGE_API_KEY=... CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    .../python scripts/eval/run_de_baselines.py \
        --model shapellm --test-ckpt /path/ShapeLLM_7B_general_v1.0 \
        --recon-path ... --EVA-path ... \
        --baseline all \
        --pool-index   Experiments/data/eval_pool_1m.jsonl \
        --dataset-root .../anypoint_2m \
        --emb-dir      .../anypoint_2m_emb \
        --budget 1000 --batch-size 10 \
        --output Experiments/results/dyneval/shapellm
"""

import os
import sys
import json
import argparse
from collections import deque
from typing import List, Dict, Any, Tuple

import numpy as np

_ROOT = "/mnt/tidalfs-bdsz01/usr/wangshijian/AnyPoint"
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts/eval"))

from dynamic_evaluation.evaluator import natural_prompt                    # noqa: E402
from dynamic_evaluation.config import EvalConfig, TaskResult              # noqa: E402
from dynamic_evaluation.task_pool import PoolItem                          # noqa: E402
from point_qa_generator.base import Task                                   # noqa: E402
# NOTE: _build_model / Judger pull in the heavy model stack (sentence_transformers,
# model wrappers); imported lazily inside main() so this module's selection logic
# stays importable in any env for testing.

# Reuse each adapter's OWN scoring primitives so selection stays faithful to the
# published method — only the per-round bookkeeping is accelerated here.
from dynamic_evaluation.baselines._common import task_category, indices_by_category  # noqa: E402
from dynamic_evaluation.baselines.acd import (                            # noqa: E402
    _category_stats, _family_stats, _acd_category_score, _task_complexity, _task_family)
from dynamic_evaluation.baselines.autobencher import (                    # noqa: E402
    _category_quotas, _task_difficulty, _task_tokens, _jaccard)

_EPS = 1e-8


# ----------------------------- pool + eval -----------------------------
def load_pool(pool_index: str, dataset_root: str) -> Tuple[List[PoolItem], List[int]]:
    root = dataset_root or ""
    items: List[PoolItem] = []
    qids: List[int] = []
    print(f"[pool] loading {pool_index}", flush=True)
    with open(pool_index) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            task = Task(
                point=os.path.join(root, r["point"]) if root else r["point"],
                question=r["question"], options=list(r["options"]), answer=r["answer"],
                metadata={"layout_description": r.get("layout", ""), "objects": [],
                          "category": r.get("category", ""), "generator_type": r.get("generator", "")},
            )
            items.append(PoolItem(item_id=i, task=task, point_cloud=None))
            qids.append(r["question_id"])
    print(f"[pool] {len(items)} candidates", flush=True)
    return items, qids


class _Run:
    """Accumulates per-task results in the SAME shape DynamicEvaluator saves."""

    def __init__(self):
        self.results: List[TaskResult] = []
        self.C_tasks: List[Task] = []
        self.E_tasks: List[Task] = []
        self.E_paths: List[str] = []
        self.E_idx: List[int] = []
        self.n = 0

    def record(self, item: PoolItem, raw: str, pick, model_answer: str, is_correct: bool,
               utility=None):
        t = item.task
        self.results.append(TaskResult(
            task_id=self.n, question=t.question, answer=t.answer,
            model_raw_output=raw, model_answer=model_answer, is_correct=is_correct,
            utility=utility, category=(t.metadata or {}).get("category") or "unknown",
            options=t.options, layout_description=(t.metadata or {}).get("layout_description")))
        if is_correct:
            self.C_tasks.append(t)
        else:
            self.E_tasks.append(t)
            self.E_paths.append(t.point)
            self.E_idx.append(self.n)
        self.n += 1


def make_eval_fn(model, judge, batch_size: int):
    """Returns eval(items) -> list of (item, raw, pick, model_answer, is_correct)."""
    def _eval(items: List[PoolItem]):
        out = []
        for st in range(0, len(items), batch_size):
            chunk = items[st:st + batch_size]
            tasks = [it.task for it in chunk]
            datas = [{"point_cloud_path": t.point} for t in tasks]
            qs = [t.question for t in tasks]
            opts = [t.options for t in tasks]
            prompts = [natural_prompt(q, o) for q, o in zip(qs, opts)]
            free = model._qa_batch(datas, prompts)
            picks = judge.extract_batch(qs, opts, free)
            for it, t, raw, pick in zip(chunk, tasks, free, picks):
                ma = t.options[pick] if pick is not None else ""
                ok = pick is not None and ma == t.answer
                out.append((it, raw, pick, ma, ok))
        return out
    return _eval


# ----------------------------- ACD -----------------------------
def run_acd(pool_items, eval_fn, budget, batch_size, seed) -> _Run:
    run = _Run()
    rng = np.random.RandomState(seed)

    # Pre-sort each fine category by (-complexity, item_id) ONCE — complexity is
    # static, so this reproduces select_acd_style_indices' within-category order
    # without re-sorting the whole pool every round.
    groups: Dict[str, deque] = {}
    tmp: Dict[str, List[int]] = {}
    fam: Dict[str, str] = {}
    for it in pool_items:
        c = task_category(it.task)
        tmp.setdefault(c, []).append(it.item_id)
        if c not in fam:
            fam[c] = _task_family(it.task)
    for c, ids in tmp.items():
        ids.sort(key=lambda iid: (-_task_complexity(pool_items[iid].task), iid))
        groups[c] = deque(ids)

    # cold start: same random batch as our tradeoff/random arms
    k0 = min(batch_size, len(pool_items))
    cold = set(rng.choice(len(pool_items), size=k0, replace=False).tolist())
    for it, raw, pick, ma, ok in eval_fn([pool_items[i] for i in cold]):
        run.record(it, raw, pick, ma, ok)
    for c in groups:
        groups[c] = deque(iid for iid in groups[c] if iid not in cold)

    rnd = 1
    while run.n < budget:
        k = min(batch_size, budget - run.n)
        cat_stats = _category_stats(run.C_tasks, run.E_tasks)
        fam_stats = _family_stats(run.C_tasks, run.E_tasks)
        total_tested = max(1, len(run.C_tasks) + len(run.E_tasks))
        present = [c for c in groups if groups[c]]
        if not present:
            break
        scores = {c: _acd_category_score(
            cat_stats.get(c, {"tested": 0, "errors": 0}),
            fam_stats.get(fam[c], {"tested": 0, "errors": 0}),
            total_tested, 1.0) for c in present}
        ranked = sorted(present, key=lambda c: (-scores[c], c))
        avail = sum(len(groups[c]) for c in present)
        picked: List[int] = []
        while len(picked) < min(k, avail) and ranked:
            progressed = False
            for c in ranked:
                if len(picked) >= k:
                    break
                if groups[c]:
                    picked.append(groups[c].popleft())
                    progressed = True
            if not progressed:
                break
        if not picked:
            break
        print(f"  [acd] iter {rnd}: {run.n}/{budget} | errors={len(run.E_tasks)} | "
              f"top={ranked[0]}({scores[ranked[0]]:.3f})", flush=True)
        for it, raw, pick, ma, ok in eval_fn([pool_items[i] for i in picked]):
            run.record(it, raw, pick, ma, ok, utility=float(scores[task_category(it.task)]))
        rnd += 1
    return run


# ----------------------------- AutoBencher -----------------------------
def run_autobencher(pool_items, eval_fn, budget, batch_size) -> _Run:
    run = _Run()
    grouped = indices_by_category(pool_items)            # fine category -> [pos==item_id]
    quotas = _category_quotas(grouped, pool_items, budget)
    print(f"  [autobencher] {len(quotas)} categories, quotas sum="
          f"{sum(quotas.values())}", flush=True)

    selected: List[int] = []
    for ci, category in enumerate(sorted(quotas)):
        idxs = grouped[category]
        q = quotas[category]
        toks = {i: _task_tokens(pool_items[i].task) for i in idxs}
        diff = {i: _task_difficulty(pool_items[i].task) for i in idxs}
        max_j = {i: 0.0 for i in idxs}                   # running max-Jaccard to chosen set
        cand = list(idxs)
        chosen: List[int] = []
        while cand and len(chosen) < q:
            # exact MMR: difficulty - 5 * max_jaccard(cand, selected); tie -> smaller item_id
            best = max(cand, key=lambda i: (diff[i] - 5.0 * max_j[i], -pool_items[i].item_id))
            chosen.append(best)
            cand.remove(best)
            bt = toks[best]
            for i in cand:
                j = _jaccard(toks[i], bt)
                if j > max_j[i]:
                    max_j[i] = j
        selected.extend(chosen)
        print(f"  [autobencher] {ci+1}/{len(quotas)} {category}: picked {len(chosen)}/{q}",
              flush=True)

    # AutoBencher is static: evaluate the pre-selected set in batches.
    for st in range(0, len(selected), batch_size):
        items = [pool_items[i] for i in selected[st:st + batch_size]]
        for it, raw, pick, ma, ok in eval_fn(items):
            run.record(it, raw, pick, ma, ok,
                       utility=float(_task_difficulty(it.task)))
        print(f"  [autobencher] eval {run.n}/{len(selected)} | errors={len(run.E_tasks)}",
              flush=True)
    return run


# ----------------------------- SEA -----------------------------
class _GPUSea:
    """GPU port of sea.py's SEAState: per-source Top-k error-similarity retrieval
    with hierarchical layout pre-filter, plus the relation-DAG + cumulative-error
    pruning. Sources are stored as pool item ids and looked up in the precomputed
    fp16 matrices, so no per-round re-encoding is needed."""

    def __init__(self, Vf, Vl, torch, device, rng,
                 top_k_per_source=50, layout_top_k=16, pruning_threshold=0.5, hierarchical=True):
        self.Vf, self.Vl = Vf, Vl
        self.torch, self.device, self.rng = torch, device, rng
        self.kp, self.ld, self.pt, self.hier = top_k_per_source, layout_top_k, pruning_threshold, hierarchical
        self.sources: List[Dict[str, Any]] = []          # {id, parent, active, derr[]}
        self._last_parent: Dict[int, int] = {}
        self._last_ids: List[int] = None

    def seed(self, item_ids: List[int]):
        for iid in item_ids:
            self.sources.append({"id": int(iid), "parent": None, "active": True, "derr": []})

    def select(self, remaining_ids: List[int], k: int) -> List[int]:
        t = self.torch
        n = len(remaining_ids)
        if k <= 0 or n == 0:
            return []
        active = [(i, s) for i, s in enumerate(self.sources) if s["active"]]
        if not active:
            return self._rand(n, k)

        # CHUNKED over the candidate dimension: never gather the full (n, D) matrix —
        # that fancy-index copy is ~24GB(+8GB layout) at the 1M pool and OOMs on top of
        # the resident matrices + model. Instead stream `rem` in CHUNK-sized contiguous
        # blocks, keeping a running per-source Top-k merged across blocks. Peak memory is
        # one block gather + the (S, k) buffers, independent of pool size. Result is
        # identical to the dense version (global Top-k per source over the layout-keep set).
        rem = t.as_tensor(remaining_ids, device=self.device, dtype=t.long)
        src = t.as_tensor([s["id"] for _, s in active], device=self.device, dtype=t.long)
        src_global = [g for g, _ in active]
        S = len(active)
        src_f = self.Vf[src]
        CHUNK = 50000
        NEG = -float(np.finfo(np.float32).max)

        def _running_topk(src_mat, V, restrict_mask, kk):
            """Per-source Top-kk over `rem` (cos-sim vs V), merged across chunks.
            restrict_mask: (n,) bool over local positions or None. Returns (vals, idx)
            (S, kk); idx = local position in remaining_ids (-1 where unfilled)."""
            run_v = t.full((S, kk), NEG, device=self.device)
            run_i = t.full((S, kk), -1, device=self.device, dtype=t.long)
            for c0 in range(0, n, CHUNK):
                gids = rem[c0:c0 + CHUNK]
                cc = gids.shape[0]
                sim = (src_mat @ V[gids].T).float()                  # (S, cc)
                if restrict_mask is not None:
                    km = restrict_mask[c0:c0 + cc]
                    sim = t.where(km.unsqueeze(0), sim, t.full_like(sim, NEG))
                v, li = sim.topk(min(kk, cc), dim=1)
                av = t.cat([run_v, v], dim=1)
                ai = t.cat([run_i, li + c0], dim=1)
                run_v, sel = av.topk(min(kk, av.shape[1]), dim=1)
                run_i = t.gather(ai, 1, sel)
            return run_v, run_i

        # layout pre-filter (doc-level): union of per-source Top-ld layout-similar candidates
        if self.hier:
            kd = min(self.ld, n)
            _, lidx = _running_topk(self.Vl[src], self.Vl, None, kd)
            keep_local = lidx[lidx >= 0].unique()
            if keep_local.numel() == 0:
                keep_local = t.arange(n, device=self.device)
        else:
            keep_local = t.arange(n, device=self.device)

        valid_total = int(keep_local.numel())
        k_per = min(self.kp, valid_total)
        if k_per <= 0:
            return self._rand(n, k)
        keep_mask = t.zeros(n, dtype=t.bool, device=self.device)
        keep_mask[keep_local] = True

        # per-source Top-k_per by full-task similarity, restricted to the layout-keep set
        rv_t, ri_t = _running_topk(src_f, self.Vf, keep_mask, k_per)
        rv, ri = rv_t.cpu().numpy(), ri_t.cpu().numpy()

        union: set = set()
        per_best: Dict[int, Tuple[int, float]] = {}
        for si in range(S):
            g = src_global[si]
            for col in range(ri.shape[1]):
                ci, v = int(ri[si, col]), float(rv[si, col])
                if ci < 0 or v <= NEG / 2:
                    continue
                union.add(ci)
                cur = per_best.get(ci)
                if cur is None or v > cur[1]:
                    per_best[ci] = (g, v)
        if not union:
            return self._rand(n, k)

        ul = list(union)
        if len(ul) > k:
            chosen = self.rng.choice(ul, size=k, replace=False).tolist()
        elif len(ul) < k:
            pool = [i for i in range(n) if i not in union]
            pad = min(k - len(ul), len(pool))
            chosen = ul + (self.rng.choice(pool, size=pad, replace=False).tolist() if pad > 0 else [])
        else:
            chosen = ul
        self._last_parent = {int(ci): per_best[ci][0] for ci in chosen if ci in per_best}
        self._last_ids = remaining_ids
        return [int(c) for c in chosen]

    def update(self, local_idxs: List[int], corrects: List[bool]):
        if self._last_ids is None:
            return
        for ci, ok in zip(local_idxs, corrects):
            parent = self._last_parent.get(int(ci))
            err = 0 if ok else 1
            if parent is not None:
                self._propagate(parent, err)
            if err == 1:
                self.sources.append({"id": int(self._last_ids[ci]), "parent": parent,
                                     "active": True, "derr": []})
        for s in self.sources:
            if s["active"] and s["derr"] and float(np.mean(s["derr"])) < self.pt:
                s["active"] = False
        self._last_parent, self._last_ids = {}, None

    def _propagate(self, idx, err):
        seen, cur = set(), idx
        while cur is not None and cur not in seen:
            seen.add(cur)
            self.sources[cur]["derr"].append(err)
            cur = self.sources[cur]["parent"]

    def _rand(self, n, k):
        return self.rng.choice(n, size=min(k, n), replace=False).tolist()

    def stats(self):
        return {"total": len(self.sources),
                "active": sum(1 for s in self.sources if s["active"])}


def build_sea_matrices(emb_dir, pool_qids, torch, device):
    """full = L2-norm(layout|question|answer); layout = L2-norm(layout). fp16 on GPU."""
    ids = json.load(open(os.path.join(emb_dir, "ids.json")))
    id2row = {qid: r for r, qid in enumerate(ids)}
    rows = np.asarray([id2row[q] for q in pool_qids], dtype=np.int64)

    def load(c):
        return np.asarray(np.load(os.path.join(emb_dir, f"{c}.fp16.npy"), mmap_mode="r")[rows],
                          dtype=np.float32)

    lay, que, ans = load("layout"), load("question"), load("answer")
    full = np.concatenate([lay, que, ans], axis=1)
    del que, ans
    full /= (np.linalg.norm(full, axis=1, keepdims=True) + _EPS)
    Vf = torch.from_numpy(full.astype(np.float16)).to(device)
    del full
    lay /= (np.linalg.norm(lay, axis=1, keepdims=True) + _EPS)
    Vl = torch.from_numpy(lay.astype(np.float16)).to(device)
    del lay
    print(f"[sea] Vf={tuple(Vf.shape)} Vl={tuple(Vl.shape)} on {device}", flush=True)
    return Vf, Vl


def run_sea(pool_items, eval_fn, budget, batch_size, seed, Vf, Vl, torch, device) -> _Run:
    run = _Run()
    rng = np.random.RandomState(seed)
    sea = _GPUSea(Vf, Vl, torch, device, rng)

    remaining = list(range(len(pool_items)))
    k0 = min(batch_size, len(remaining))
    cold_ids = [remaining[p] for p in rng.choice(len(remaining), size=k0, replace=False)]
    res = eval_fn([pool_items[i] for i in cold_ids])
    for it, raw, pick, ma, ok in res:
        run.record(it, raw, pick, ma, ok)
    sea.seed([it.item_id for (it, _, _, _, ok) in res if not ok])   # seed from cold-start errors
    drop = set(cold_ids)
    remaining = [i for i in remaining if i not in drop]

    rnd = 1
    while run.n < budget and remaining:
        k = min(batch_size, budget - run.n, len(remaining))
        local = sea.select(remaining, k)
        if not local:
            break
        sel_ids = [remaining[l] for l in local]
        res = eval_fn([pool_items[i] for i in sel_ids])
        corrects = [ok for (_, _, _, _, ok) in res]
        for it, raw, pick, ma, ok in res:
            run.record(it, raw, pick, ma, ok)
        sea.update(local, corrects)
        drop = set(sel_ids)
        remaining = [i for i in remaining if i not in drop]
        st = sea.stats()
        print(f"  [sea] iter {rnd}: {run.n}/{budget} | errors={len(run.E_tasks)} | "
              f"sources={st['active']}/{st['total']}", flush=True)
        rnd += 1
    return run


# ----------------------------- io -----------------------------
def save_run(out_dir, run: _Run, cfg: EvalConfig):
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "config": cfg.to_dict(),
        "stats": {"total": run.n, "errors": len(run.E_tasks),
                  "error_rate": len(run.E_tasks) / max(1, run.n), "error_indices": run.E_idx},
        "results": [r.to_dict() for r in run.results],
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    if run.E_tasks:
        hd = os.path.join(out_dir, "hard_data")
        pcd = os.path.join(hd, "pcd")
        os.makedirs(pcd, exist_ok=True)
        recs = []
        for i, (t, src) in enumerate(zip(run.E_tasks, run.E_paths)):
            if src and os.path.exists(src):
                try:
                    np.save(os.path.join(pcd, f"{i:06d}.npy"), np.load(src))
                except Exception:  # noqa: BLE001
                    pass
            recs.append({"question_id": i, "point": f"pcd/{i:06d}.npy",
                         "category": (t.metadata or {}).get("category") or "unknown",
                         "question": t.question, "options": t.options, "answer": t.answer,
                         "layout": (t.metadata or {}).get("layout_description", "")})
        with open(os.path.join(hd, "tasks.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n📁 {out_dir}/results.json | total={run.n} errors={len(run.E_tasks)} "
          f"({len(run.E_tasks)/max(1,run.n):.1%})", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Run ACD / AutoBencher / SEA baselines on the fixed pool")
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--test-ckpt")
    ap.add_argument("--baseline", default="all",
                    help="comma list of {acd,autobencher,sea} or 'all'")
    ap.add_argument("--pool-index", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--emb-dir", required=True, help="precomputed Qwen3 embeddings (needed by SEA)")
    ap.add_argument("--output", required=True, help="<output>/<baseline>_style/ per method")
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg-path")
    from run_dynamic_eval import _build_model, _parse_unknown_args
    from models.judger import Judger

    args, unknown = ap.parse_known_args()
    extra = _parse_unknown_args(unknown)

    which = ["acd", "autobencher", "sea"] if args.baseline == "all" \
        else [b.strip() for b in args.baseline.split(",") if b.strip()]
    for b in which:
        if b not in ("acd", "autobencher", "sea"):
            raise ValueError(f"unknown baseline {b!r}")

    pool_items, pool_qids = load_pool(args.pool_index, args.dataset_root)
    model = _build_model(args.model, args.checkpoint or args.test_ckpt, args.output,
                         args.device, args.cfg_path, extra)
    judge = Judger()
    eval_fn = make_eval_fn(model, judge, args.batch_size)

    Vf = Vl = torch = None
    if "sea" in which:
        import torch  # noqa
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        Vf, Vl = build_sea_matrices(args.emb_dir, pool_qids, torch, device)

    def cfg_for(strategy):
        return EvalConfig(budget=args.budget, batch_size=args.batch_size,
                          pool_size=len(pool_items), lambda_explore=0.0, seed=args.seed,
                          strategy=strategy, pool_index_path=args.pool_index,
                          dataset_root=args.dataset_root, emb_dir=args.emb_dir)

    for b in which:
        print(f"\n{'='*70}\n# baseline: {b}_style | budget={args.budget} batch={args.batch_size}\n{'='*70}", flush=True)
        if b == "acd":
            run = run_acd(pool_items, eval_fn, args.budget, args.batch_size, args.seed)
        elif b == "autobencher":
            run = run_autobencher(pool_items, eval_fn, args.budget, args.batch_size)
        else:
            run = run_sea(pool_items, eval_fn, args.budget, args.batch_size, args.seed,
                          Vf, Vl, torch, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        save_run(os.path.join(args.output, f"{b}_style"), run, cfg_for(f"{b}_style"))


if __name__ == "__main__":
    main()
