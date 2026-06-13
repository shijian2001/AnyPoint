#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(question: str, options: list[str]) -> str:
    option_block = "\n".join(options)
    return (
        "<point>\n"
        "Answer the question based on the provided point cloud.\n"
        f"Question: {question}\n"
        f"Options:\n{option_block}\n"
        "Output only the answer option, such as: Answer: A."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Dataset root containing tasks.jsonl and pcd/*.npy",
    )
    parser.add_argument(
        "--pointllm-root",
        type=Path,
        help="PointLLM repository root",
    )
    parser.add_argument("--pointnum", type=int, default=8192)
    args = parser.parse_args()

    tasks_path = args.dataset_root / "tasks.jsonl"
    pcd_dir = args.dataset_root / "pcd"

    point_data_dir = args.pointllm_root / "data" / "pointllm_train_data"
    anno_dir = args.pointllm_root / "data" / "anno_data"
    anno_dir.mkdir(parents=True, exist_ok=True)
    point_data_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(tasks_path)

    stage1 = []
    stage2 = []

    for row in rows:
        point_name = row["point"]
        object_id = Path(point_name).stem
        src = pcd_dir / point_name
        dst = point_data_dir / f"{object_id}_{args.pointnum}.npy"

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

        prompt = build_prompt(row["question"], row["options"])
        answer = row["answer"]

        stage1.append(
            {
                "object_id": object_id,
                "conversation_type": "simple_description",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

        stage2.append(
            {
                "object_id": object_id,
                "conversation_type": "single_round",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

    stage1_path = anno_dir / "pointllm_train_stage1.json"
    stage2_path = anno_dir / "pointllm_train_stage2.json"
    stage1_path.write_text(json.dumps(stage1, ensure_ascii=False, indent=2), encoding="utf-8")
    stage2_path.write_text(json.dumps(stage2, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared {len(rows)} samples.")
    print(f"Point cloud dir: {point_data_dir}")
    print(f"Stage-1 anno: {stage1_path}")
    print(f"Stage-2 anno: {stage2_path}")


if __name__ == "__main__":
    main()
