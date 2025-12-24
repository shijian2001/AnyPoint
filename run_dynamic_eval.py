"""
Run dynamic evaluation.

Example:
    python run_dynamic_eval.py \\
        --metadata data/metadata.jsonl \\
        --pcd-dir data/point_clouds \\
        --layouts data/layout/outputs_gpt_oss/layouts.json \\
        --model your_model \\
        --checkpoint checkpoints/model.pth \\
        --output results/eval \\
        --budget 100 \\
        --batch-size 10 \\
        --pool-size 1000 \\
        --lambda-explore 0.2
"""

import os
import argparse
from typing import Dict, Any

from point_qa_generator.generator import PointQAGenerator
from models.point_qa_model import PointQAModel
from dynamic_evaluation import DynamicEvaluator, EvalConfig


def run_dynamic_eval(
    metadata_file: str,
    pcd_dir: str,
    layouts_file: str,
    model_name: str,
    model_checkpoint: str,
    output_dir: str,
    budget: int = 100,
    batch_size: int = 10,
    pool_size: int = 1000,
    lambda_explore: float = 0.2,
    seed: int = 42
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
    
    model = PointQAModel(
        model_name=model_name,
        checkpoint_path=model_checkpoint,
        cache_path=os.path.join(output_dir, 'cache')
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    
    # Optional
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=1000)
    parser.add_argument("--lambda-explore", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
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
        seed=args.seed
    )


if __name__ == "__main__":
    main()

