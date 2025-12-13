#!/usr/bin/env python3
"""
Layout Generator Script

Generate layouts: 1 object list → 1 template (LLM) → 1 layout (Solver)

Usage:
    python layout_gen.py --num-layouts 1000 --keys-file configs/keys.yaml
"""

import asyncio
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

from layout_generator import LayoutGenerator, sample_object_names


# 10 objects, choose 2-9 → 1000+ unique combinations
OBJECTS = ["table", "chair", "lamp", "book", "cup", "sphere", "cube", "cylinder", "cone", "pyramid"]


async def generate(api_keys: list, num_layouts: int, model: str):
    """Generate layouts: each with unique object list."""
    generator = LayoutGenerator(
        model_name=model,
        api_keys=api_keys,
        max_concurrent_per_key=10,
        max_retries=10,
        solver_threads=16,
        seed=42
    )
    
    # Each layout gets a unique object list
    object_lists = [
        sample_object_names(OBJECTS, seed=i)
        for i in range(num_layouts)
    ]
    
    print(f"Generating {num_layouts} layouts (1 template each)...")
    
    templates, layouts = await generator.generate_batch(
        object_lists=object_lists,
        layouts_per_template=1
    )
    
    return [t.to_dict() for t in templates], [l.to_dict() for l in layouts]


def main():
    parser = argparse.ArgumentParser(description="Generate 3D scene layouts")
    parser.add_argument("--num-layouts", type=int, default=10, help="Number of layouts")
    parser.add_argument("--output", type=str, default="data/layout/outputs", help="Output directory")
    parser.add_argument("--keys-file", type=str, required=True, help="YAML file with API keys")
    parser.add_argument("--model", type=str, default="DeepSeek-V3", help="LLM model")
    args = parser.parse_args()
    
    api_keys = yaml.safe_load(open(args.keys_file))["keys"]
    
    print(f"Target: {args.num_layouts} layouts")
    print(f"Model: {args.model}, API keys: {len(api_keys)}")
    
    templates, layouts = asyncio.run(generate(api_keys, args.num_layouts, args.model))
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "templates.json", 'w') as f:
        json.dump(templates, f, indent=2)
    
    with open(output_dir / "layouts.json", 'w') as f:
        json.dump(layouts, f, indent=2)
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "num_templates": len(templates),
            "num_layouts": len(layouts),
            "model": args.model,
        }, f, indent=2)
    
    print(f"Done! {len(layouts)} layouts -> {output_dir}")


if __name__ == "__main__":
    main()
