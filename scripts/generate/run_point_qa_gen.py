#!/usr/bin/env python3
"""
Generate point cloud QA datasets.

Usage:
    # All 17 types, 1000 tasks, random options 4-6
    python run_point_qa_gen.py --num-tasks 1000 --output data/qa_output

    # Specific types
    python run_point_qa_gen.py --num-tasks 100 --types what_distance what_relation --output data/qa_output

    # Weighted types
    python run_point_qa_gen.py --num-tasks 500 --types what_distance:0.4 what_relation:0.3 multi_hop_relation:0.3 --output data/qa_output

    # Fixed options count
    python run_point_qa_gen.py --num-tasks 100 --num-options 4 --output data/qa_output

    # Specific config
    python run_point_qa_gen.py --num-tasks 100 --types what_distance --config distance_type=closest --output data/qa_output
"""

import argparse
import os
import time

from point_qa_generator import PointQAGenerator, TaskPlan


DEFAULT_METADATA = os.environ.get(
    "ANYPOINT_METADATA",
    "/mnt/tidalfs-bdsz01/dataset/llm_ckpt/task/3d_data/anypoint_metadata_to_date.jsonl",
)
DEFAULT_PCD_DIR = os.environ.get(
    "ANYPOINT_PCD_DIR",
    "/mnt/tidalfs-bdsz01/dataset/llm_ckpt/task/3d_data/anypoint_meta_obj_npys",
)
DEFAULT_LAYOUTS = os.environ.get(
    "ANYPOINT_LAYOUTS",
    "/mnt/tidalfs-bdsz01/dataset/llm_ckpt/task/3d_data/layout_10k/layouts.jsonl",
)
DEFAULT_BACKGROUND = os.environ.get(
    "ANYPOINT_BACKGROUND",
    "/mnt/tidalfs-bdsz01/dataset/llm_ckpt/task/3d_data/background",
)


def parse_types(type_args):
    """Parse type arguments into str, list, or dict.

    Formats:
        None -> all types (list of all 17)
        ["what_distance", "what_relation"] -> list
        ["what_distance:0.5", "what_relation:0.5"] -> dict with weights
    """
    if not type_args:
        return None

    has_weights = any(":" in t for t in type_args)
    if has_weights:
        weights = {}
        for t in type_args:
            parts = t.split(":")
            name = parts[0]
            weight = float(parts[1])
            weights[name] = weight
        return weights
    elif len(type_args) == 1:
        return type_args[0]
    else:
        return type_args


def parse_config(config_args):
    """Parse config key=value pairs into dict."""
    if not config_args:
        return {}
    config = {}
    for item in config_args:
        key, value = item.split("=", 1)
        config[key] = value
    return config


def main():
    parser = argparse.ArgumentParser(description="Generate point cloud QA datasets")
    parser.add_argument("--num-tasks", type=int, required=True, help="Number of tasks to generate")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--types", nargs="*", help="Generator types (optional weights via type:weight)")
    parser.add_argument("--num-options", type=str, default="4-6",
                        help="Number of options: fixed (e.g. '4') or range (e.g. '4-6')")
    parser.add_argument("--config", nargs="*", help="Generator config as key=value pairs")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--pcd-dir", type=str, default=DEFAULT_PCD_DIR)
    parser.add_argument("--layouts", type=str, default=DEFAULT_LAYOUTS)
    parser.add_argument("--background-dir", type=str, default=DEFAULT_BACKGROUND)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Parse num_options
    if "-" in args.num_options:
        lo, hi = args.num_options.split("-")
        num_options = (int(lo), int(hi))
    else:
        num_options = int(args.num_options)

    # Parse types
    generator_type = parse_types(args.types)
    if generator_type is None:
        gen_temp = PointQAGenerator(args.metadata, args.pcd_dir, args.layouts, seed=args.seed)
        generator_type = list(gen_temp.generators.keys())
        del gen_temp

    # Parse config
    generator_config = parse_config(args.config)

    # Build task plan
    task_plan = TaskPlan(
        generator_type=generator_type,
        num_options=num_options,
        seed=args.seed,
        generator_config=generator_config,
    )

    # Initialize and generate
    print(f"Initializing PointQAGenerator...")
    print(f"  Metadata: {args.metadata}")
    print(f"  PCD dir:  {args.pcd_dir}")
    print(f"  Layouts:  {args.layouts}")
    print()

    gen = PointQAGenerator(
        metadata_file=args.metadata,
        pcd_dir=args.pcd_dir,
        layouts_file=args.layouts,
        seed=args.seed,
        background_dir=args.background_dir,
    )

    print(f"Task plan:")
    print(f"  Types: {generator_type}")
    print(f"  Num options: {num_options}")
    print(f"  Config: {generator_config or 'random'}")
    print(f"  Seed: {args.seed}")
    print(f"  Num tasks: {args.num_tasks}")
    print()

    t0 = time.time()
    info = gen.generate(task_plan, args.num_tasks, args.output)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s ({args.num_tasks / elapsed:.1f} tasks/s)")


if __name__ == "__main__":
    main()
