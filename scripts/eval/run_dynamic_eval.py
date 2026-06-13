"""
Run OUR dynamic evaluation (utility-driven) on a fixed, pre-built pool with
precomputed Qwen3-Embedding-8B vectors. Grading uses the LLM judge.

Example:
    JUDGE_API_KEY=... SENTENCE_TRANSFORMERS_HOME=.../_st_cache CUDA_VISIBLE_DEVICES=0 \\
    python run_dynamic_eval.py \\
        --model shapellm --test-ckpt /path/ShapeLLM_7B_general_v1.0 \\
        --recon-path /path/recon.pth --EVA-path /path/eva.safetensors \\
        --pool-index   .../Experiments/data/eval_pool_1m.jsonl \\
        --dataset-root .../3d_data/anypoint_2m \\
        --emb-dir      .../3d_data/anypoint_2m_emb \\
        --strategy tradeoff --lambda-explore 0.2 \\
        --budget 1000 --batch-size 10 --output results/exp01/shapellm/tradeoff_l0.2

Model-specific args (recon-path, EVA-path, cfg-path, weights-root, ...) are
forwarded to PointQAModel.
"""

import os
import argparse
import json
from typing import Dict, Any, List

from models.point_qa_model import PointQAModel
from dynamic_evaluation import DynamicEvaluator, EvalConfig


def _build_model(model_name, real_ckpt, output_dir, device, cfg_path, model_kwargs):
    if model_name not in ["minigpt3d", "pointalign", "greenplm", "gpt4point"] and not real_ckpt:
        raise ValueError("--checkpoint or --test-ckpt is required for this model")
    runtime_kwargs = dict(model_kwargs)
    if cfg_path is not None:
        runtime_kwargs.setdefault("cfg_path", cfg_path)
    # NOTE: we deliberately do NOT force a unified generation config here. Each model's
    # wrapper keeps its own official default decoding (max_new_tokens, num_beams,
    # min_length, temperature, ...). Forcing a single config broke models (e.g. MiniGPT3D
    # produced empty output, beam-search models were silently switched to greedy). The
    # benchmark's fairness comes from the same questions + same LLM judge, not from
    # identical decoding hyperparameters (which are part of each model itself).
    # A specific value can still be overridden per-model via CLI (e.g. --max-new-tokens).
    return PointQAModel(
        model_name=model_name,
        checkpoint_path=real_ckpt,
        cache_path=os.path.join(output_dir, "cache"),
        device=device,
        enable_choice_search=False,   # grading is done by the LLM judge, not SBERT choice_search
        **runtime_kwargs,
    )


def run_dynamic_eval(
    model_name: str,
    output_dir: str,
    pool_index: str,
    dataset_root: str,
    emb_dir: str,
    strategy: str = "tradeoff",
    model_checkpoint: str = None,
    budget: int = 1000,
    batch_size: int = 10,
    lambda_explore: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    cfg_path: str = None,
    test_ckpt: str = None,
    **model_kwargs,
) -> Dict[str, Any]:
    model = _build_model(model_name, model_checkpoint or test_ckpt, output_dir, device, cfg_path, model_kwargs)
    config = EvalConfig(
        budget=budget, batch_size=batch_size, pool_size=0,   # pool_size derived from the index
        lambda_explore=lambda_explore, seed=seed, strategy=strategy,
        pool_index_path=pool_index, dataset_root=dataset_root, emb_dir=emb_dir,
    )
    return DynamicEvaluator(model, config).run(output_dir)


def run_sweep(
    model_name: str,
    output_dir: str,
    pool_index: str,
    dataset_root: str,
    emb_dir: str,
    lambda_list: List[float],
    include_random: bool = True,
    model_checkpoint: str = None,
    budget: int = 1000,
    batch_size: int = 10,
    seed: int = 42,
    device: str = "cuda",
    cfg_path: str = None,
    test_ckpt: str = None,
    **model_kwargs,
) -> Dict[str, Any]:
    """Run all arms of ONE model sharing a single loaded model + 24GB pool matrix.

    Arms (per experiment plan 01_dynamic_evaluation.md): a `random` baseline + `tradeoff`
    at each λ in `lambda_list` (λ=0 is pure exploit, λ=1 is pure explore — the two
    ablations are just the endpoints, no separate exploit_only/explore_only arm). The
    model is loaded and V is built ONCE; each arm then just resets per-run state.
    Each arm writes to <output_dir>/<arm_name>/.
    """
    arms: List[Tuple[str, float]] = ([("random", None)] if include_random else []) \
        + [("tradeoff", float(l)) for l in lambda_list]
    if not arms:
        raise ValueError("sweep has no arms: pass --lambda-list and/or keep random on")

    model = _build_model(model_name, model_checkpoint or test_ckpt, output_dir, device, cfg_path, model_kwargs)
    first_strategy, first_lambda = arms[0][0], (arms[0][1] if arms[0][1] is not None else 0.2)
    evaluator = DynamicEvaluator(model, EvalConfig(
        budget=budget, batch_size=batch_size, pool_size=0,
        lambda_explore=first_lambda, seed=seed, strategy=first_strategy,
        pool_index_path=pool_index, dataset_root=dataset_root, emb_dir=emb_dir,
    ))

    summaries: Dict[str, Any] = {}
    for strategy, lam in arms:
        arm_name = f"tradeoff_l{lam}" if strategy == "tradeoff" else strategy
        print(f"\n{'#'*70}\n# arm: {arm_name}\n{'#'*70}", flush=True)
        evaluator.reset(strategy=strategy, lambda_explore=lam, seed=seed)
        summaries[arm_name] = evaluator.run(os.path.join(output_dir, arm_name))
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Dynamic evaluation (our utility-driven method)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-ckpt")
    parser.add_argument("--output", required=True)
    # fixed pool + precomputed embeddings
    parser.add_argument("--pool-index", required=True, help="eval_pool_1m.jsonl (has layout)")
    parser.add_argument("--dataset-root", required=True, help="root that index `point` paths resolve against")
    parser.add_argument("--emb-dir", required=True, help="precomputed embeddings dir (anypoint_2m_emb)")
    # algorithm
    parser.add_argument("--strategy", default="tradeoff", choices=["tradeoff", "random"])
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lambda-explore", type=float, default=0.2)
    # sweep mode: when set, run a `random` baseline + tradeoff at each λ here, sharing one
    # loaded model + pool matrix. Experiment-plan default: --lambda-list 0,0.2,0.5,0.8,1
    parser.add_argument("--lambda-list", help="comma-separated λ values -> run a multi-arm sweep")
    parser.add_argument("--no-random", action="store_true", help="skip the random baseline arm in a sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-path")

    args, unknown = parser.parse_known_args()
    extra = _parse_unknown_args(unknown)
    if args.lambda_list:
        lambda_list = [float(x) for x in args.lambda_list.split(",") if x.strip() != ""]
        run_sweep(
            model_name=args.model, output_dir=args.output,
            pool_index=args.pool_index, dataset_root=args.dataset_root, emb_dir=args.emb_dir,
            lambda_list=lambda_list, include_random=not args.no_random,
            model_checkpoint=args.checkpoint, test_ckpt=args.test_ckpt,
            budget=args.budget, batch_size=args.batch_size,
            seed=args.seed, device=args.device, cfg_path=args.cfg_path, **extra,
        )
    else:
        run_dynamic_eval(
            model_name=args.model, output_dir=args.output,
            pool_index=args.pool_index, dataset_root=args.dataset_root, emb_dir=args.emb_dir,
            strategy=args.strategy, model_checkpoint=args.checkpoint, test_ckpt=args.test_ckpt,
            budget=args.budget, batch_size=args.batch_size, lambda_explore=args.lambda_explore,
            seed=args.seed, device=args.device, cfg_path=args.cfg_path, **extra,
        )


def _parse_unknown_args(unknown: List[str]) -> Dict[str, Any]:
    """Parse ``--key value`` pairs and forward them to the model constructor."""
    parsed: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if not token.startswith("--"):
            raise ValueError(f"Failed to parse extra CLI argument: {token}")
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
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    main()
