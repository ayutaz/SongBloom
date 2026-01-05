"""Combine multiple JSONL datasets for SongBloom training.

Usage:
    uv run python -m SongBloom.training.combine_datasets \
        --input-jsonls data/jacappella_prepared_ja/jacappella.jsonl \
                       data/kiritan_prepared/kiritan.jsonl \
        --output-jsonl data/japanese_combined/combined.jsonl
"""

import argparse
import json
import os
from typing import List


def combine_datasets(input_jsonls: List[str], output_jsonl: str) -> int:
    """Combine multiple JSONL files into one.

    Args:
        input_jsonls: List of input JSONL file paths
        output_jsonl: Output JSONL file path

    Returns:
        Total number of samples
    """
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    all_items = []
    seen_idx = set()

    for jsonl_path in input_jsonls:
        if not os.path.exists(jsonl_path):
            print(f"Warning: File not found: {jsonl_path}")
            continue

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in {jsonl_path}: {e}")
                    continue

                # Ensure unique idx
                idx = item.get("idx", "")
                if idx in seen_idx:
                    # Make unique by appending suffix
                    suffix = 1
                    while f"{idx}_{suffix}" in seen_idx:
                        suffix += 1
                    item["idx"] = f"{idx}_{suffix}"

                seen_idx.add(item["idx"])
                all_items.append(item)

        print(f"Loaded {len([i for i in all_items if i.get('idx', '').startswith(os.path.basename(jsonl_path).split('.')[0])])} items from {jsonl_path}")

    # Write combined JSONL
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(all_items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine multiple JSONL datasets for SongBloom training"
    )
    parser.add_argument(
        "--input-jsonls",
        type=str,
        nargs="+",
        required=True,
        help="Input JSONL files to combine",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output combined JSONL file",
    )

    args = parser.parse_args()

    total = combine_datasets(
        input_jsonls=args.input_jsonls,
        output_jsonl=args.output_jsonl,
    )
    print(f"Combined {total} samples -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
