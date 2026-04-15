import json
import argparse
from pathlib import Path

def convert_to_amber(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    processed_data = []
    seen_ids = set()

    with input_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue

            try:
                item = json.loads(line)
                
                # Get ID from multiple possible keys
                raw_id = item.get("question_id") or item.get("image_id") or item.get("id")
                if raw_id is None: continue
                
                item_id = int(raw_id)
                if item_id in seen_ids: continue
                seen_ids.add(item_id)

                # Get and clean response text
                raw_text = item.get("text") or item.get("answer") or item.get("response") or item.get("caption") or ""
                clean_text = raw_text.replace("<s>", "").replace("</s>", "").strip()

                processed_data.append({"id": item_id, "response": clean_text})
            
            except (json.JSONDecodeError, ValueError):
                print(f"Skip line {i}: Invalid format.")

    # Sort by ID ascending
    processed_data.sort(key=lambda x: x["id"])

    # Save as formatted JSON array
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print(f"Done. Processed {len(processed_data)} items -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean AMBER output.")
    parser.add_argument("--input", type=Path, default="amber_generative_7b_output.jsonl")
    parser.add_argument("--output", type=Path, default="amber_eval_ready.json")
    args = parser.parse_args()

    convert_to_amber(args.input, args.output)