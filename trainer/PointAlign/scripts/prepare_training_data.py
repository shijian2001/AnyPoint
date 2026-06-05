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
        default=Path("/path/to/dataset_root"),
        help="Dataset root containing tasks.jsonl and pcd/*.npy",
    )
    parser.add_argument(
        "--pointalign-root",
        type=Path,
        default=Path("/path/to/PointQA_Eval/trainer/PointAlign"),
        help="PointAlign repository root",
    )
    parser.add_argument("--pointnum", type=int, default=8192)
    args = parser.parse_args()

    tasks_path = args.dataset_root / "tasks.jsonl"
    pcd_dir = args.dataset_root / "pcd"

    point_data_dir = args.pointalign_root / "data" / "pointalign_train_data"
    anno_dir = args.pointalign_root / "data" / "anno_data"
    anno_dir.mkdir(parents=True, exist_ok=True)
    point_data_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(tasks_path)

    single_round = []
    linked = 0

    for row in rows:
        object_id = Path(row["point"]).stem
        src = pcd_dir / row["point"]
        dst = point_data_dir / f"{object_id}_{args.pointnum}.npy"

        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())
        linked += 1

        prompt = build_prompt(row["question"], row["options"])
        answer = row["answer"]

        single_round.append(
            {
                "object_id": object_id,
                "conversation_type": "single_round",
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

    anno_path = anno_dir / "pointalign_train_single_round.json"
    anno_path.write_text(json.dumps(single_round, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared {len(rows)} samples.")
    print(f"Linked point clouds: {linked} -> {point_data_dir}")
    print(f"Annotation: {anno_path}")


if __name__ == "__main__":
    main()
