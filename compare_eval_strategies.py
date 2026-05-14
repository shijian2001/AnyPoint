import argparse
import atexit
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from compare_eval_strategies_utils import (
    resolve_devices,
    split_evenly,
)
from dynamic_evaluation import (
    EvalConfig,
    TaskEmbedder,
    TaskPool,
    UtilityCalculator,
    SEAState,
    select_acd_style_indices,
    select_autobencher_style_indices,
)
from dynamic_evaluation.config import TaskResult
from dynamic_evaluation.task_pool import PoolItem
from models.base_qa_model import make_options
from models.point_qa_model import PointQAModel
from point_qa_generator.base import Task
from point_qa_generator.generator import PointQAGenerator

UTILITY_STRATEGIES = ("dynamic", "affinity_only", "novelty_only")
BASELINE_STRATEGIES = ("acd_style", "autobencher_style", "sea_style")
ADAPTIVE_STRATEGIES = (*UTILITY_STRATEGIES, "acd_style", "sea_style")


_WORKER_QA_GEN: Optional[PointQAGenerator] = None
_WORKER_MODEL: Optional[PointQAModel] = None


@dataclass(frozen=True)
class RuntimeConfig:
    metadata_file: str
    pcd_dir: str
    layouts_file: str
    background_dir: Optional[str]
    model_name: str
    checkpoint_path: Optional[str]
    output_dir: str
    seed: int
    cfg_path: Optional[str]
    prompt_template: Optional[str]
    model_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class EvalJob:
    task_id: int
    item: PoolItem
    utility: Optional[float]
    point_cloud_path: str


@dataclass
class EvalRecord:
    task_id: int
    task: Task
    result: TaskResult
    error_point_cloud_path: Optional[str]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _clone_pool_items(items: List[PoolItem]) -> List[PoolItem]:
    return [
        PoolItem(
            item_id=item.item_id,
            task=copy.deepcopy(item.task),
            point_cloud=None if item.point_cloud is None else np.copy(item.point_cloud),
        )
        for item in items
    ]


def _pop_random(remaining: List[PoolItem], rng: np.random.RandomState, size: int) -> List[PoolItem]:
    if size <= 0 or not remaining:
        return []
    size = min(size, len(remaining))
    indices = rng.choice(len(remaining), size=size, replace=False).tolist()
    return _pop_indices(remaining, indices)


def _pop_indices(remaining: List[PoolItem], indices: List[int]) -> List[PoolItem]:
    ordered_unique = list(dict.fromkeys(indices))
    removed_by_index: Dict[int, PoolItem] = {}
    for idx in sorted(ordered_unique, reverse=True):
        removed_by_index[idx] = remaining.pop(idx)
    return [removed_by_index[idx] for idx in ordered_unique]


def _chunk_items(items: List[PoolItem], chunk_size: int) -> List[List[PoolItem]]:
    if chunk_size <= 0:
        return [items]
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _infer_category(task: Task) -> str:
    if task.metadata:
        gen_type = task.metadata.get("generator_type", "")
        config = task.metadata.get("generator_config", {})
        if gen_type:
            dist_type = config.get("distance_type", "")
            return f"{gen_type}_{dist_type}" if dist_type else gen_type
    return "unknown"


def _save_hard_data(
    output_dir: str,
    error_tasks: List[Task],
    error_point_cloud_paths: List[str],
    cfg: EvalConfig,
) -> None:
    if not error_tasks:
        return

    hard_dir = os.path.join(output_dir, "hard_data")
    os.makedirs(hard_dir, exist_ok=True)

    task_records = []
    for i, (task, point_cloud_path) in enumerate(zip(error_tasks, error_point_cloud_paths)):
        task_records.append(
            {
                "question_id": i,
                "point": point_cloud_path,
                "category": _infer_category(task),
                "question": task.question,
                "options": task.options,
                "answer": task.answer,
            }
        )

    tasks_file = os.path.join(hard_dir, "tasks.jsonl")
    with open(tasks_file, "w", encoding="utf-8") as f:
        for record in task_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    tasks_info = {
        "task_plan": {
            "generator_type": "mixed",
            "num_options": 4,
            "seed": cfg.seed,
        },
        "generation_stats": {
            "num_tasks_requested": len(error_tasks),
            "num_tasks_generated": len(error_tasks),
            "output_directory": hard_dir,
        },
    }
    with open(os.path.join(hard_dir, "tasks_info.json"), "w", encoding="utf-8") as f:
        json.dump(tasks_info, f, indent=2, ensure_ascii=False)


def _save_results(
    output_dir: str,
    cfg: EvalConfig,
    results: List[TaskResult],
    error_indices: List[int],
    error_tasks: List[Task],
    error_point_cloud_paths: List[str],
) -> Dict[str, Any]:
    summary = {
        "config": cfg.to_dict(),
        "stats": {
            "total": len(results),
            "errors": len(error_indices),
            "error_rate": len(error_indices) / len(results) if results else 0.0,
            "error_indices": error_indices,
        },
        "results": [result.to_dict() for result in results],
    }
    summary = _to_jsonable(summary)

    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _save_hard_data(output_dir, error_tasks, error_point_cloud_paths, cfg)
    return summary


def _evaluate_single(
    model: PointQAModel,
    embedder: Optional[TaskEmbedder],
    task: Task,
    point_cloud_path: str,
    task_id: int,
    utility: Optional[float],
) -> TaskResult:
    _, _, formatted_options = make_options(task.options, model.format)
    result = model.multiple_choice_qa(
        data={"point_cloud_path": point_cloud_path},
        question=task.question,
        choices=task.options,
        answer=task.answer,
    )
    layout_desc = TaskEmbedder._get_layout(task) if task.metadata else None
    return TaskResult(
        task_id=task_id,
        question=task.question,
        answer=task.answer,
        model_raw_output=result["free_form_answer"],
        model_answer=result["multiple_choice_answer"],
        is_correct=(result["accuracy"] == 1),
        utility=utility,
        category=_infer_category(task),
        options=formatted_options,
        layout_description=layout_desc,
    )


def _init_eval_worker(runtime: RuntimeConfig, device: str) -> None:
    global _WORKER_QA_GEN, _WORKER_MODEL

    _WORKER_QA_GEN = PointQAGenerator(
        metadata_file=runtime.metadata_file,
        pcd_dir=runtime.pcd_dir,
        layouts_file=runtime.layouts_file,
        seed=runtime.seed,
        background_dir=runtime.background_dir,
    )
    _WORKER_MODEL = _build_model(
        model_name=runtime.model_name,
        checkpoint_path=runtime.checkpoint_path,
        output_dir=runtime.output_dir,
        device=device,
        cfg_path=runtime.cfg_path,
        prompt_template=runtime.prompt_template,
        model_kwargs=runtime.model_kwargs,
    )


def _evaluate_worker_jobs(jobs: Sequence[EvalJob]) -> List[EvalRecord]:
    if _WORKER_QA_GEN is None or _WORKER_MODEL is None:
        raise RuntimeError("Evaluation worker is not initialized")

    records: List[EvalRecord] = []
    for job in jobs:
        task = job.item.task
        point_cloud = job.item.point_cloud
        if point_cloud is None and not os.path.exists(job.point_cloud_path):
            point_cloud = _WORKER_QA_GEN.materialize_point_cloud(task)
        if point_cloud is not None:
            _save_eval_point_cloud(job.point_cloud_path, point_cloud)

        result = _evaluate_single(_WORKER_MODEL, None, task, job.point_cloud_path, job.task_id, job.utility)
        records.append(
            EvalRecord(
                task_id=job.task_id,
                task=task,
                result=result,
                error_point_cloud_path=None if result.is_correct else job.point_cloud_path,
            )
        )
    return records


def _eval_point_cloud_path(point_cloud_dir: str, item_id: int) -> str:
    return os.path.join(point_cloud_dir, f"item_{item_id:06d}.npy")


def _save_eval_point_cloud(path: str, point_cloud: np.ndarray) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, point_cloud)


class MultiGpuBatchEvaluator:
    def __init__(self, runtime: RuntimeConfig, devices: List[str]):
        if len(devices) < 2:
            raise ValueError("MultiGpuBatchEvaluator requires at least two devices")

        self.devices = devices
        context = mp.get_context("spawn")
        self._executors = [
            ProcessPoolExecutor(
                max_workers=1,
                mp_context=context,
                initializer=_init_eval_worker,
                initargs=(runtime, device),
            )
            for device in devices
        ]
        atexit.register(self.close)

    def evaluate(self, jobs: List[EvalJob], phase: str) -> List[EvalRecord]:
        chunks = split_evenly(jobs, len(self._executors))
        futures = [
            executor.submit(_evaluate_worker_jobs, chunk)
            for executor, chunk in zip(self._executors, chunks)
        ]

        records: List[EvalRecord] = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{phase}_workers"):
            records.extend(future.result())

        return sorted(records, key=lambda record: record.task_id)

    def close(self) -> None:
        while self._executors:
            executor = self._executors.pop()
            executor.shutdown(wait=True, cancel_futures=False)


def run_strategy(
    strategy: str,
    qa_gen: PointQAGenerator,
    model: Optional[PointQAModel],
    base_items: List[PoolItem],
    cfg: EvalConfig,
    output_dir: str,
    point_cloud_dir: str,
    parallel_evaluator: Optional[MultiGpuBatchEvaluator] = None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    remaining = _clone_pool_items(base_items)
    rng = np.random.RandomState(cfg.seed)
    needs_embedder = strategy in UTILITY_STRATEGIES or strategy == "sea_style"
    if needs_embedder:
        embedder = TaskEmbedder(device="cpu") if parallel_evaluator is not None else TaskEmbedder()
    else:
        embedder = None
    utility_calc = UtilityCalculator(cfg.lambda_explore) if strategy in UTILITY_STRATEGIES else None
    sea_state = (
        SEAState(embedder=embedder, rng=np.random.RandomState(cfg.seed + 1))
        if strategy == "sea_style"
        else None
    )

    c_tasks: List[Task] = []
    e_tasks: List[Task] = []
    e_point_cloud_paths: List[str] = []
    c_embs: Optional[np.ndarray] = None
    e_embs: Optional[np.ndarray] = None
    results: List[TaskResult] = []
    error_indices: List[int] = []
    n_eval = 0
    print(f"\n{'=' * 70}")
    if strategy == "random":
        print(
            f"Compare Run | Strategy: {strategy} | Budget: {cfg.budget} | Batch: {cfg.batch_size} | "
            f"Fixed Pool: {cfg.pool_size}"
        )
    else:
        print(
            f"Compare Run | Strategy: {strategy} | Budget: {cfg.budget} | Batch: {cfg.batch_size} | "
            f"Fixed Pool: {cfg.pool_size} | λ: {cfg.lambda_explore}"
        )
    print(f"{'=' * 70}\n")

    if strategy in ("random", "autobencher_style"):
        if strategy == "random":
            selected = _pop_random(remaining, rng, cfg.budget)
            print(f"Random baseline: selected {len(selected)} tasks from full pool once\n")
        else:
            selected_indices = select_autobencher_style_indices(remaining, cfg.budget)
            selected = _pop_indices(remaining, selected_indices)
            print(f"AutoBencher-style baseline: selected {len(selected)} balanced hard tasks once\n")
        batches = _chunk_items(selected, cfg.batch_size)
        for batch_idx, batch in enumerate(batches, start=1):
            print(f"{'─' * 70}")
            print(f"{strategy} Batch {batch_idx}/{len(batches)}: evaluating {len(batch)} tasks")
            print(f"{'─' * 70}\n")
            n_eval = _evaluate_batch(
                qa_gen,
                model,
                embedder,
                batch,
                results,
                c_tasks,
                e_tasks,
                e_point_cloud_paths,
                error_indices,
                n_eval,
                point_cloud_dir,
                phase=f"{strategy}_batch_{batch_idx}",
                parallel_evaluator=parallel_evaluator,
            )
            # Static baselines preselect the whole budget at once; drop reconstructed
            # point clouds after each batch so earlier batches do not accumulate in RAM.
            for item in batch:
                item.point_cloud = None
    elif strategy in ADAPTIVE_STRATEGIES:
        print("🔥 Cold Start\n")
        cold_batch = _pop_random(remaining, rng, cfg.batch_size)
        n_eval = _evaluate_batch(
            qa_gen,
            model,
            embedder,
            cold_batch,
            results,
            c_tasks,
            e_tasks,
            e_point_cloud_paths,
            error_indices,
            n_eval,
            point_cloud_dir,
            phase=f"{strategy}_cold_start",
            parallel_evaluator=parallel_evaluator,
        )
        if strategy in UTILITY_STRATEGIES:
            c_embs, e_embs = _update_embeddings(embedder, c_tasks, e_tasks)
        if sea_state is not None:
            sea_state.seed(list(e_tasks))
        history_count = len(c_tasks) + len(e_tasks)
        print(f"✓ Initial: |H|={history_count}, |E|={len(e_tasks)} ({len(e_tasks) / history_count:.1%})\n")

        iteration = 1
        while n_eval < cfg.budget:
            remaining_budget = cfg.budget - n_eval
            k = min(cfg.batch_size, remaining_budget, len(remaining))
            if k <= 0:
                break

            print(f"{'─' * 70}")
            history_count = len(c_tasks) + len(e_tasks)
            print(f"🔄 Iter {iteration}: {n_eval}/{cfg.budget} | |H|={history_count} |E|={len(e_tasks)}")
            print(f"{'─' * 70}\n")

            print(f"Remaining candidates: {len(remaining)}")
            if strategy == "acd_style":
                scores = None
                top_idx = select_acd_style_indices(remaining, c_tasks, e_tasks, k)
            elif strategy == "sea_style":
                scores = None
                if sea_state is None:
                    raise ValueError("SEA-style strategy requires an initialized SEA state")
                top_idx = sea_state.select(remaining, k)
                if not top_idx:
                    top_idx = rng.choice(len(remaining), size=min(k, len(remaining)), replace=False).tolist()
            else:
                tasks = [item.task for item in remaining]
                candidate_embs = embedder.encode(tasks)
                if utility_calc is None:
                    raise ValueError("Utility-guided strategy requires an initialized utility calculator")
                scores = utility_calc.compute_strategy(strategy, candidate_embs, c_embs, e_embs)
                top_idx = np.argsort(scores)[-k:][::-1].tolist()
            selected = _pop_indices(remaining, top_idx)
            utilities = [None] * len(selected) if scores is None else [scores[i] for i in top_idx]
            if scores is None:
                if strategy == "sea_style":
                    sea_stats = sea_state.stats() if sea_state is not None else {}
                    print(
                        f"Selected top-{len(selected)} by SEA-style retrieval "
                        f"(sources active={sea_stats.get('active_sources', 0)}, "
                        f"pruned={sea_stats.get('pruned_sources', 0)})\n"
                    )
                else:
                    print(f"Selected top-{k} by ACD-style category UCB\n")
            else:
                print(f"Selected top-{k}: U ∈ [{utilities[0]:.3f}, {utilities[-1]:.3f}]\n")

            pre_eval_len = len(results)
            n_eval = _evaluate_batch(
                qa_gen,
                model,
                embedder,
                selected,
                results,
                c_tasks,
                e_tasks,
                e_point_cloud_paths,
                error_indices,
                n_eval,
                point_cloud_dir,
                utilities=utilities,
                phase=strategy,
                parallel_evaluator=parallel_evaluator,
            )
            if strategy in UTILITY_STRATEGIES:
                c_embs, e_embs = _update_embeddings(embedder, c_tasks, e_tasks)
            if sea_state is not None:
                is_correct_list = [r.is_correct for r in results[pre_eval_len:]]
                sea_state.update(top_idx, is_correct_list)
            history_count = len(c_tasks) + len(e_tasks)
            print(f"✓ Cumulative: |E|={len(e_tasks)} ({len(e_tasks) / history_count:.1%})\n")
            iteration += 1
    else:
        raise ValueError(f"Unknown comparison strategy: {strategy}")

    summary = _save_results(output_dir, cfg, results, error_indices, e_tasks, e_point_cloud_paths)
    print(f"\n📁 {os.path.join(output_dir, 'results.json')}")
    print(f"\n{'=' * 70}")
    print(
        f"🎉 Complete: {summary['stats']['total']} evaluated, "
        f"{summary['stats']['errors']} errors ({summary['stats']['error_rate']:.1%})"
    )
    print(f"{'=' * 70}\n")
    return summary


def _evaluate_batch(
    qa_gen: PointQAGenerator,
    model: Optional[PointQAModel],
    embedder: Optional[TaskEmbedder],
    batch: List[PoolItem],
    results: List[TaskResult],
    c_tasks: List[Task],
    e_tasks: List[Task],
    e_point_cloud_paths: List[str],
    error_indices: List[int],
    n_eval: int,
    point_cloud_dir: str,
    utilities: Optional[List[Optional[float]]] = None,
    phase: str = "eval",
    parallel_evaluator: Optional[MultiGpuBatchEvaluator] = None,
) -> int:
    if utilities is None:
        utilities = [None] * len(batch)

    if parallel_evaluator is not None:
        jobs = [
            EvalJob(
                task_id=n_eval + offset,
                item=item,
                utility=utility,
                point_cloud_path=_eval_point_cloud_path(point_cloud_dir, item.item_id),
            )
            for offset, (item, utility) in enumerate(zip(batch, utilities))
        ]
        records = parallel_evaluator.evaluate(jobs, phase)
        for record in records:
            _append_eval_record(record, results, c_tasks, e_tasks, e_point_cloud_paths, error_indices)
        return n_eval + len(records)

    for item, utility in tqdm(zip(batch, utilities), total=len(batch), desc=phase):
        if model is None:
            raise ValueError("Single-process evaluation requires an initialized model")

        task = item.task
        point_cloud = item.point_cloud
        point_cloud_path = _eval_point_cloud_path(point_cloud_dir, item.item_id)
        if point_cloud is None and not os.path.exists(point_cloud_path):
            point_cloud = qa_gen.materialize_point_cloud(task)
            item.point_cloud = point_cloud
        if point_cloud is not None:
            _save_eval_point_cloud(point_cloud_path, point_cloud)

        task_result = _evaluate_single(model, embedder, task, point_cloud_path, n_eval, utility)
        _append_eval_record(
            EvalRecord(
                task_id=n_eval,
                task=task,
                result=task_result,
                error_point_cloud_path=None if task_result.is_correct else point_cloud_path,
            ),
            results,
            c_tasks,
            e_tasks,
            e_point_cloud_paths,
            error_indices,
        )

        n_eval += 1

    return n_eval


def _append_eval_record(
    record: EvalRecord,
    results: List[TaskResult],
    c_tasks: List[Task],
    e_tasks: List[Task],
    e_point_cloud_paths: List[str],
    error_indices: List[int],
) -> None:
    results.append(record.result)

    if not record.result.is_correct:
        if record.error_point_cloud_path is None:
            raise ValueError(f"Missing point cloud path for failed task {record.task_id}")
        e_tasks.append(record.task)
        e_point_cloud_paths.append(record.error_point_cloud_path)
        error_indices.append(record.task_id)
    else:
        c_tasks.append(record.task)


def _update_embeddings(
    embedder: TaskEmbedder,
    c_tasks: List[Task],
    e_tasks: List[Task],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    c_embs = embedder.encode(c_tasks) if c_tasks else None
    e_embs = embedder.encode(e_tasks) if e_tasks else None
    return c_embs, e_embs


def _build_model(
    model_name: str,
    checkpoint_path: str,
    output_dir: str,
    device: str,
    cfg_path: Optional[str],
    prompt_template: Optional[str],
    model_kwargs: Dict[str, Any],
) -> PointQAModel:
    runtime_kwargs = dict(model_kwargs)
    if cfg_path is not None:
        runtime_kwargs.setdefault("cfg_path", cfg_path)

    if prompt_template:
        def prompt_func(question: str, options: List[str] = None) -> str:
            if options:
                return prompt_template.format(question=question, choices="\n".join(options))
            return prompt_template.format(question=question)
    else:
        prompt_func = None

    return PointQAModel(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        prompt_func=prompt_func,
        cache_path=None,
        device=device,
        **runtime_kwargs,
    )


def _parse_unknown_args(unknown: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"无法解析额外参数: {token}")

        key = token[2:].replace("-", "_")
        value: Any = True

        if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
            value = _coerce_cli_value(unknown[i + 1])
            i += 1

        parsed[key] = value
        i += 1

    return parsed


def _coerce_cli_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluation strategies (random / utility-guided / ACD / AutoBencher / SEA) on a shared task pool")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pcd-dir", required=True)
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-ckpt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--pool-cache-dir", help="Optional existing task_pool_cache directory to reuse")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--lambda-explore", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background-dir", "--background_dir", dest="background_dir")
    parser.add_argument(
        "--devices",
        required=True,
        help="Comma-separated devices for evaluation, e.g. cuda:0 or cuda:0,cuda:1",
    )
    parser.add_argument("--cfg-path")
    parser.add_argument("--prompt-template")

    args, unknown = parser.parse_known_args()
    extra_kwargs = _parse_unknown_args(unknown)

    checkpoint_path = args.checkpoint or args.test_ckpt
    if args.model not in ["minigpt3d", "pointalign", "greenplm", "gpt4point"] and not checkpoint_path:
        raise ValueError("必须提供 --checkpoint 或 --test-ckpt")

    devices = resolve_devices(args.devices, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))

    qa_gen = PointQAGenerator(
        metadata_file=args.metadata,
        pcd_dir=args.pcd_dir,
        layouts_file=args.layouts,
        seed=args.seed,
        background_dir=args.background_dir,
    )

    model: Optional[PointQAModel] = None
    parallel_evaluator: Optional[MultiGpuBatchEvaluator] = None
    if len(devices) == 1:
        model = _build_model(
            model_name=args.model,
            checkpoint_path=checkpoint_path,
            output_dir=args.output,
            device=devices[0],
            cfg_path=args.cfg_path,
            prompt_template=args.prompt_template,
            model_kwargs=extra_kwargs,
        )
    else:
        print(f"[INFO] Multi-GPU evaluation enabled: {', '.join(devices)}")
        parallel_evaluator = MultiGpuBatchEvaluator(
            RuntimeConfig(
                metadata_file=args.metadata,
                pcd_dir=args.pcd_dir,
                layouts_file=args.layouts,
                background_dir=args.background_dir,
                model_name=args.model,
                checkpoint_path=checkpoint_path,
                output_dir=args.output,
                seed=args.seed,
                cfg_path=args.cfg_path,
                prompt_template=args.prompt_template,
                model_kwargs=extra_kwargs,
            ),
            devices,
        )

    pool = TaskPool(qa_gen, args.seed, args.pool_size)
    pool_cache_dir = args.pool_cache_dir or os.path.join(args.output, "task_pool_cache")
    print(f"[INFO] Using shared task pool cache: {pool_cache_dir}")
    pool.ensure_ready(pool_cache_dir)
    base_items = pool.remaining()

    common_cfg = {
        "budget": args.budget,
        "batch_size": args.batch_size,
        "pool_size": args.pool_size,
        "lambda_explore": args.lambda_explore,
        "seed": args.seed,
    }
    shared_point_cloud_dir = os.path.join(args.output, "eval_point_clouds")

    try:
        random_summary = run_strategy(
            "random",
            qa_gen,
            model,
            base_items,
            EvalConfig(**common_cfg),
            os.path.join(args.output, "random"),
            shared_point_cloud_dir,
            parallel_evaluator=parallel_evaluator,
        )
        strategy_summaries = {
            strategy: run_strategy(
                strategy,
                qa_gen,
                model,
                base_items,
                EvalConfig(**common_cfg),
                os.path.join(args.output, strategy),
                shared_point_cloud_dir,
                parallel_evaluator=parallel_evaluator,
            )
            for strategy in (*UTILITY_STRATEGIES, *BASELINE_STRATEGIES)
        }
    finally:
        if parallel_evaluator is not None:
            parallel_evaluator.close()

    compare_summary = {
        "random": random_summary["stats"],
        **{strategy: summary["stats"] for strategy, summary in strategy_summaries.items()},
        "delta": {
            "errors": strategy_summaries["dynamic"]["stats"]["errors"] - random_summary["stats"]["errors"],
            "error_rate": (
                strategy_summaries["dynamic"]["stats"]["error_rate"] - random_summary["stats"]["error_rate"]
            ),
        },
        "delta_vs_random": {
            strategy: {
                "errors": summary["stats"]["errors"] - random_summary["stats"]["errors"],
                "error_rate": summary["stats"]["error_rate"] - random_summary["stats"]["error_rate"],
            }
            for strategy, summary in strategy_summaries.items()
        },
    }
    compare_summary = _to_jsonable(compare_summary)
    compare_path = os.path.join(args.output, "compare_summary.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(compare_summary, f, indent=2, ensure_ascii=False)
    print(f"📁 {compare_path}")


if __name__ == "__main__":
    main()
