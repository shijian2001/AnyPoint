#!/usr/bin/env python3
"""
Generate high-quality 3D scene layouts at scale.

Usage:
    python layout_gen.py --num-layouts 10000 --output data/layout/outputs_10k --concurrency 30
"""

import os
import asyncio
import argparse
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter
from dotenv import load_dotenv

from layout_generator import LayoutGenerator, sample_object_names
from layout_generator.constants import MIN_OBJECTS, MAX_OBJECTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_API_KEY = os.getenv("API_KEY", "")
DEFAULT_BASE_URL = os.getenv("BASE_URL", "")
DEFAULT_MODEL = os.getenv("MODEL", "deepseek-v4-pro")
DEFAULT_METADATA = os.getenv("METADATA_PATH", "")


def load_object_names(path: str) -> List[str]:
    objects = []
    with open(path) as f:
        for line in f:
            data = json.loads(line)
            for v in data.values():
                objects.append(v['object'])
    return objects


def sample_unique_object_lists(
    all_objects: List[str], num: int, seed: int
) -> List[List[str]]:
    """Sample unique object lists (no duplicate sets)."""
    rng = np.random.RandomState(seed)
    seen = set()
    result = []

    while len(result) < num:
        count = rng.randint(MIN_OBJECTS, MAX_OBJECTS + 1)
        names = list(rng.choice(all_objects, size=count, replace=False))
        key = tuple(sorted(names))
        if key not in seen:
            seen.add(key)
            result.append(names)

    return result


async def generate(args) -> Dict[str, Any]:
    all_objects = load_object_names(args.metadata)
    logger.info(f"Loaded {len(all_objects)} objects from metadata")

    # Patch API base_url into the wrapper
    import layout_generator.api.wrapper as wrapper_mod
    from openai import AsyncOpenAI
    orig_init = wrapper_mod.QAWrapper.__init__

    def patched_init(self, model_name, api_key, max_retries=5):
        orig_init(self, model_name, api_key, max_retries)
        self.client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)

    wrapper_mod.QAWrapper.__init__ = patched_init

    # Sample unique object lists
    object_lists = sample_unique_object_lists(all_objects, args.num_layouts, args.seed)
    logger.info(f"Sampled {len(object_lists)} unique object lists")

    # Generate in batches for progress reporting
    generator = LayoutGenerator(
        model_name=args.model,
        api_keys=[args.api_key],
        max_concurrent_per_key=args.concurrency,
        max_retries=args.max_retries,
        solver_threads=8,
        seed=args.seed,
    )

    batch_size = args.concurrency * 3
    all_templates = []
    all_layouts = []
    start_time = time.time()

    for batch_start in range(0, len(object_lists), batch_size):
        batch = object_lists[batch_start:batch_start + batch_size]

        templates, layouts = await generator.generate_batch(batch)
        all_templates.extend(templates)
        all_layouts.extend(layouts)

        elapsed = time.time() - start_time
        total_attempted = batch_start + len(batch)
        rate = len(all_layouts) / elapsed if elapsed > 0 else 0
        eta = (len(object_lists) - total_attempted) / rate if rate > 0 else 0
        logger.info(
            f"Progress: {total_attempted}/{len(object_lists)} batched, "
            f"{len(all_layouts)} layouts produced ({len(all_layouts)/total_attempted:.0%} yield), "
            f"{rate:.1f}/sec, ETA: {eta:.0f}s"
        )

    return all_templates, all_layouts


def save_results(templates, layouts, output_dir: Path, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "layouts.json", 'w') as f:
        json.dump([l.to_dict() for l in layouts], f, indent=2)

    with open(output_dir / "templates.json", 'w') as f:
        json.dump([t.to_dict() for t in templates], f, indent=2)

    # Quality summary
    rel_counter = Counter()
    for layout in layouts:
        for rel in layout.relations:
            rel_counter[rel.relation] += 1

    n_objs = [len(l.objects) for l in layouts]
    n_rels = [len(l.relations) for l in layouts]

    summary = {
        "generated_at": datetime.now().isoformat(),
        "num_layouts": len(layouts),
        "num_templates": len(templates),
        "model": args.model,
        "seed": args.seed,
        "quality": {
            "relation_distribution": dict(rel_counter.most_common()),
            "avg_objects_per_layout": float(np.mean(n_objs)) if n_objs else 0,
            "avg_relations_per_layout": float(np.mean(n_rels)) if n_rels else 0,
            "total_relation_types_used": len(rel_counter),
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved {len(layouts)} layouts to {output_dir}")
    logger.info(f"Quality: {len(rel_counter)} relation types, "
                f"avg {summary['quality']['avg_objects_per_layout']:.1f} obj/layout, "
                f"avg {summary['quality']['avg_relations_per_layout']:.1f} rel/layout")


def main():
    parser = argparse.ArgumentParser(description="Generate high-quality 3D scene layouts")
    parser.add_argument("--num-layouts", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/layout/outputs")
    parser.add_argument("--metadata", type=str, default=DEFAULT_METADATA)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info(f"Target: {args.num_layouts} layouts | Model: {args.model} | Concurrency: {args.concurrency}")

    templates, layouts = asyncio.run(generate(args))
    save_results(templates, layouts, Path(args.output), args)


if __name__ == "__main__":
    main()
