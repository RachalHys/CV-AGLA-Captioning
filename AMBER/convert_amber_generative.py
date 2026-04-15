import argparse
import json
import random
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AMBER to AGLA format.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--num", type=int, default=None, help="Number of samples")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data")

    args = parser.parse_args()

    # Load data
    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Shuffle if requested
    if args.shuffle:
        random.shuffle(data)

    # Slice data if num_samples is specified
    if args.num is not None:
        data = data[:args.num]

    # Write to JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for item in data:
            record = {
                "question_id": item["id"],
                "image": item["image"],
                "text": item["query"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Processed {len(data)} items -> {args.output}")

if __name__ == "__main__":
    main()