#!/usr/bin/env python3
"""Sample a QA subset from a generated dataset, by flexible per-generator quotas.

The 2M dataset is stored organized by generator:
    <dataset>/index.jsonl                      # every task, with `generator` + `point`
    <dataset>/<generator>/shard_*/pcd/*.npy    # scene point clouds
    <dataset>/<generator>/shard_*/metadata.jsonl

This script reads index.jsonl and produces a subset index (e.g. 10k) with a
fresh contiguous question_id (0..N-1). It does NOT move or rename any .npy --
each output record's `point` still points at the original file (relative to the
dataset root), so training/eval just reads dataset_root/<point>.

Composition is flexible and composable:

  # 10k, evenly across all generators
  python sample_qa_subset.py --dataset <d> --total 10000 --output train_10k.jsonl

  # 10k with explicit per-generator weights (unlisted generators get 0)
  python sample_qa_subset.py --dataset <d> --total 10000 \
      --weights what_attribute=0.5 count_object=0.3 what_relation=0.2 \
      --output train_10k.jsonl

  # only certain generators, evenly
  python sample_qa_subset.py --dataset <d> --total 10000 \
      --only what_attribute count_object --output sub.jsonl

  # hierarchical: split 1:1 into two groups, sample within each
  #   (run twice with --only + --total, or use --group)
  python sample_qa_subset.py --dataset <d> --total 5000 \
      --group attr=what_attribute,list_attribute,count_attribute \
      --group spatial=what_relation,what_distance_closest,where_distance_closest \
      --group-weights attr=0.5 spatial=0.5 --output mix_5k.jsonl

Notes:
- Sampling is reproducible via --seed.
- Categories (closest/farthest/etc.) are sampled randomly inside each generator
  unless you target a specific category name via --only/--group (index has both
  `generator` and `category`).
- Splits: use --exclude <file> to avoid reusing question_ids already taken by a
  previous split (e.g. build train, then val excluding train) for leakage-free
  train/val/test.
"""
import argparse
import json
import os
import random
from collections import defaultdict


def load_index(dataset):
    path = os.path.join(dataset, "index.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"index.jsonl not found in {dataset}")
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def parse_kv(items, cast=str):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = cast(v)
    return out


def allocate(total, keys, weights=None):
    """Split total across keys by weights (default even). Returns {key: count}."""
    if weights:
        s = sum(weights.get(k, 0) for k in keys)
        raw = {k: total * weights.get(k, 0) / s for k in keys} if s > 0 else {}
        alloc = {k: int(v) for k, v in raw.items()}
        rem = total - sum(alloc.values())
        for k in sorted(keys, key=lambda k: raw.get(k, 0) - alloc.get(k, 0), reverse=True)[:rem]:
            alloc[k] = alloc.get(k, 0) + 1
        return alloc
    base, rem = divmod(total, len(keys))
    return {k: base + (1 if i < rem else 0) for i, k in enumerate(keys)}


def sample_from_pool(pool, n, rng):
    if n >= len(pool):
        if n > len(pool):
            print(f"  WARNING: requested {n} but pool only has {len(pool)}; taking all")
        return list(pool)
    return rng.sample(pool, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="generated dataset root (has index.jsonl)")
    ap.add_argument("--total", type=int, required=True)
    ap.add_argument("--output", required=True, help="output subset .jsonl")
    ap.add_argument("--key", choices=["generator", "category"], default="generator",
                    help="balance/group by generator (17) or category (30)")
    ap.add_argument("--only", nargs="*", help="restrict to these generator/category values")
    ap.add_argument("--weights", nargs="*", help="per-key weights, e.g. what_attribute=0.5")
    ap.add_argument("--group", action="append", default=[],
                    help="named group: name=key1,key2,... (repeatable)")
    ap.add_argument("--group-weights", nargs="*", help="per-group weights, e.g. attr=0.5")
    ap.add_argument("--exclude", help="another subset .jsonl whose question_ids to exclude")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = load_index(args.dataset)

    # Exclusion for leakage-free splits (match on original question_id + point).
    excluded = set()
    if args.exclude and os.path.exists(args.exclude):
        with open(args.exclude, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    excluded.add(r.get("point"))
        recs = [r for r in recs if r.get("point") not in excluded]
        print(f"Excluded {len(excluded)} already-used items from pool")

    # Build pools keyed by generator or category.
    pools = defaultdict(list)
    for r in recs:
        pools[r.get(args.key)].append(r)

    if args.only:
        pools = {k: v for k, v in pools.items() if k in args.only}

    # Decide allocation.
    chosen = []
    if args.group:
        # Hierarchical: allocate total across groups, then evenly within group keys.
        groups = {}
        for g in args.group:
            name, members = g.split("=", 1)
            groups[name] = [m.strip() for m in members.split(",") if m.strip()]
        gweights = parse_kv(args.group_weights, float) if args.group_weights else None
        group_alloc = allocate(args.total, list(groups.keys()), gweights)
        print("Group allocation:", group_alloc)
        for gname, members in groups.items():
            member_alloc = allocate(group_alloc[gname], members)
            for k, n in member_alloc.items():
                got = sample_from_pool(pools.get(k, []), n, rng)
                chosen.extend(got)
                print(f"  [{gname}] {k}: {len(got)}")
    else:
        weights = parse_kv(args.weights, float) if args.weights else None
        keys = list(pools.keys())
        alloc = allocate(args.total, keys, weights)
        for k in sorted(keys):
            got = sample_from_pool(pools[k], alloc[k], rng)
            chosen.extend(got)
            print(f"  {k}: {len(got)}")

    # Shuffle and reassign contiguous question_id.
    rng.shuffle(chosen)
    with open(args.output, "w", encoding="utf-8") as out:
        for i, r in enumerate(chosen):
            r = dict(r)
            r["question_id"] = i
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(chosen):,} records -> {args.output}")
    print(f"Each record's `point` is relative to {args.dataset} "
          f"(load with os.path.join(dataset, rec['point'])).")


if __name__ == "__main__":
    main()
