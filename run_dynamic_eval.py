"""
Run dynamic evaluation.

Example:
    python run_dynamic_eval.py \\
        --metadata data/metadata.jsonl \\
        --pcd-dir data/point_clouds \\
        --layouts data/layout/outputs_gpt_oss/layouts.json \\
        --model pointllm \\
        --checkpoint /path/to/checkpoint \\
        --output results/eval \\
        --budget 100 \\
        --batch-size 10 \\
        --pool-size 1000 \\
        --lambda-explore 0.2

Model-specific args are forwarded to ``PointQAModel``:
    python run_dynamic_eval.py ... \\
        --model pointalign \\
        --checkpoint unused \\
        --cfg-path /path/to/config.yaml \\
        --weights-root /path/to/weights \\
        --qformer-pretrained-path /path/to/blip2.pth
"""

import os
import argparse
import json
from typing import Dict, Any, List

from point_qa_generator.generator import PointQAGenerator
from models.point_qa_model import PointQAModel
from dynamic_evaluation import DynamicEvaluator, EvalConfig


def run_dynamic_eval(
    metadata_file: str,
    pcd_dir: str,
    layouts_file: str,
    model_name: str,
    output_dir: str,
    model_checkpoint: str = None,
    budget: int = 100,
    batch_size: int = 10,
    pool_size: int = 1000,
    lambda_explore: float = 0.2,
    seed: int = 42,
    device: str = "cuda",
    cfg_path: str = None,
    test_ckpt: str = None,
    prompt_template: str = None,
    **model_kwargs
) -> Dict[str, Any]:
    """
    Run dynamic evaluation pipeline.
    
    Args:
        metadata_file: Object metadata JSONL
        pcd_dir: Point cloud directory
        layouts_file: Layouts JSON
        model_name: Model identifier
        model_checkpoint: Checkpoint path
        output_dir: Output directory
        budget: Total budget (B in algorithm)
        batch_size: Batch size per iteration (K in algorithm)
        pool_size: Candidate pool size (N in algorithm, N >> K)
        lambda_explore: Exploration weight (λ in algorithm, λ ∈ [0,1])
        seed: Random seed
        
    Returns:
        Evaluation summary
    """
    # Initialize
    qa_gen = PointQAGenerator(
        metadata_file=metadata_file,
        pcd_dir=pcd_dir,
        layouts_file=layouts_file,
        seed=seed
    )
    
    real_ckpt = model_checkpoint or test_ckpt
    if model_name not in ['minigpt3d', 'pointalign', 'greenplm', 'gpt4point'] and not real_ckpt:
        raise ValueError("必须提供 --checkpoint 或 --test-ckpt")

    runtime_kwargs = dict(model_kwargs)
    if cfg_path is not None:
        runtime_kwargs.setdefault('cfg_path', cfg_path)

    if prompt_template:
        def prompt_func(question: str, options: List[str] = None) -> str:
            if options:
                return prompt_template.format(question=question, choices="\n".join(options))
            return prompt_template.format(question=question)
    else:
        prompt_func = None

    model = PointQAModel(
        model_name=model_name,
        checkpoint_path=real_ckpt,
        prompt_func=prompt_func,
        cache_path=os.path.join(output_dir, 'cache'),
        device=device,
        **runtime_kwargs,
    )
    
    config = EvalConfig(
        budget=budget,
        batch_size=batch_size,
        pool_size=pool_size,
        lambda_explore=lambda_explore,
        seed=seed
    )
    
    # Run
    evaluator = DynamicEvaluator(qa_gen, model, config)
    results = evaluator.run(output_dir)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Dynamic evaluation")
    
    # Required
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--pcd-dir", required=True)
    parser.add_argument("--layouts", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--test-ckpt")
    parser.add_argument("--output", required=True)
    
    # Optional
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--lambda-explore", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg-path")
    parser.add_argument("--prompt-template")

    args, unknown = parser.parse_known_args()
    extra_kwargs = _parse_unknown_args(unknown)
    
    run_dynamic_eval(
        metadata_file=args.metadata,
        pcd_dir=args.pcd_dir,
        layouts_file=args.layouts,
        model_name=args.model,
        model_checkpoint=args.checkpoint,
        output_dir=args.output,
        budget=args.budget,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        lambda_explore=args.lambda_explore,
        seed=args.seed,
        device=args.device,
        cfg_path=args.cfg_path,
        test_ckpt=args.test_ckpt,
        prompt_template=args.prompt_template,
        **extra_kwargs,
    )


def _parse_unknown_args(unknown: List[str]) -> Dict[str, Any]:
    """Parse ``--key value`` pairs and forward them to the model constructor."""
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

        if key == "llava_model_base":
            key = "model_base"

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


if __name__ == "__main__":
    main()
