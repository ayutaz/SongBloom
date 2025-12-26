import argparse
import json
import os
import random
from typing import List, Tuple


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def split_items(items: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if val_ratio <= 0:
        return items, []
    if val_ratio >= 1:
        return [], items

    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)
    val_count = int(len(items) * val_ratio)
    val_idx = set(indices[:val_count])
    train_items = [items[i] for i in range(len(items)) if i not in val_idx]
    val_items = [items[i] for i in range(len(items)) if i in val_idx]
    return train_items, val_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--val-jsonl", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    items = load_jsonl(args.input_jsonl)
    train_items, val_items = split_items(items, args.val_ratio, args.seed)

    os.makedirs(os.path.dirname(args.train_jsonl), exist_ok=True)
    os.makedirs(os.path.dirname(args.val_jsonl), exist_ok=True)

    write_jsonl(args.train_jsonl, train_items)
    write_jsonl(args.val_jsonl, val_items)

    print(f"train: {len(train_items)} samples -> {args.train_jsonl}")
    print(f"val:   {len(val_items)} samples -> {args.val_jsonl}")


if __name__ == "__main__":
    main()
