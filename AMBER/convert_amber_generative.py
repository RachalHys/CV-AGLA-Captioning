import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configure logging for cleaner console output
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def convert_to_amber_format(input_path: Path, output_path: Path) -> None:
    """
    Converts LLaVA-generated JSONL output into AMBER's expected JSON array format.
    """
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    processed_data: List[Dict[str, Any]] = []
    seen_ids = set()

    with input_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                
                # 1. Extract ID
                if "id" in item:
                    item_id = int(item["id"])
                elif "question_id" in item:
                    item_id = int(item["question_id"])
                else:
                    continue
                
                # 2. Deduplicate: Keep only the first occurrence of an ID
                if item_id in seen_ids: 
                    continue
                seen_ids.add(item_id)

                # 3. Extract and clean the response text
                raw_text = item.get("response", "")
                clean_text = str(raw_text).replace("<s>", "").replace("</s>", "").strip()

                # Append to the results payload
                processed_data.append({
                    "id": item_id, 
                    "response": clean_text
                })
            
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Skipping line {line_num} due to parsing error: {e}")

    # Sort sequentially by ID (standard requirement for benchmark evaluation scripts)
    processed_data.sort(key=lambda x: x["id"])

    # Dump the final list to a formatted JSON array file
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    logging.info(f"Successfully processed {len(processed_data)} unique items.")
    logging.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA JSONL to AMBER evaluation format.")
    
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True, 
        help="Path to the generated JSONL file from LLaVA inference."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("amber_eval.json"), 
        help="Path to save the AMBER-compatible JSON array."
    )
    
    args = parser.parse_args()
    convert_to_amber_format(args.input, args.output)