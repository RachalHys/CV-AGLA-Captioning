import torch
import re
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Detect objects in an image
class BLIPInventoryDetector:
    def __init__(self, model_id="Salesforce/blip-vqa-base", device="cuda:1", max_objects=10):
        self.device      = device
        self.max_objects = max_objects # Cap the number of detected items to avoid overwhelming YOLO-World
        print(f"Loading BLIP VQA on {device}...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        print("BLIP Inventory Detector Ready!")

    def _ask_blip_batch(self, image: Image.Image, questions: list[str]) -> list[str]:
        """Batch multiple questions about the same image into ONE forward pass."""
        images_batch = [image] * len(questions)
        inputs = self.processor(
            images_batch, questions,
            return_tensors="pt",
            padding=True,
        ).to(self.device, torch.float16)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=15)

        return [
            self.processor.decode(out[i], skip_special_tokens=True).lower().strip()
            for i in range(len(questions))
        ]
    
    def _clean_text(self, text: str) -> list:
        # Remove Common Phrases
        junk_patterns = [
            r"\bi don'?t know\b", r"\bnot sure\b", r"\bunknown\b",
            r"\bmore than\b",     r"\byes\b",       r"\bno\b",
            r"\bnone\b",          r"\bnothing\b",
            r"\bimage\b",         r"\bpicture\b",   r"\bphoto\b",
        ]
        for pattern in junk_patterns:
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

        # Replace connectors with commas
        text = re.sub(
            r"\b(and|with|next to|near|on top of|in front of|behind|"
            r"on|in|standing|sitting|holding|wearing)\b",
            ",", text, flags=re.IGNORECASE,
        )
        # Remove prepositions
        text = re.sub(
            r"\b(a|an|the|some|many|several|few|one|two|three|four|five|"
            r"there are|there is|i see|any)\b",
            " ", text, flags=re.IGNORECASE,
        )
        raw_items = [item.strip() for item in text.split(",")]
        final_items = []

        # Standardize nouns to describe people
        word_map = {
            "people": "person", "humans": "person", "human": "person",
            "men":    "man",    "women":  "woman",  "guys":  "man",
            "children": "child", "kids": "child",
        }

        for item in raw_items:
            item = re.sub(r"[^\w\s]", "", item).strip()
            item = re.sub(r"\s+", " ", item)
            if len(item) < 2:
                continue
            # NOTE: No singularization — YOLO-World handles plurals fine,
            # and naive item[:-1] creates non-words ("dress"→"dres").
            item = word_map.get(item, item)
            final_items.append(item)

        # Deduplicate preserving insertion order
        seen, unique = set(), []
        for x in final_items:
            if x not in seen:
                seen.add(x)
                unique.append(x)
        return unique

    def get_inventory(self, image: Image.Image) -> list:
        # Ask BLIP multiple questions about the same image in one batch to save time.
        questions = [
                    "What animals are in this image?",
                    "Are there any people in this image?",
                    "What objects are in this image?",
                    "What man-made objects are in this image?"
                ]

        answers = self._ask_blip_batch(image, questions)

        # Extract and clean the answers
        ans_anim = answers[0]

        person_nouns  = ["person", "people", "man", "woman", "boy", "girl", "child", "human", "men", "women", "children", "guy", "lady", "gentleman"]
        ans_ppl  = answers[1].lower().strip()
        if re.search(r'\bno\b|\bnot\b|\bnone\b', ans_ppl):
            ans_ppl = ""
        elif "yes" in ans_ppl or any(w in ans_ppl for w in person_nouns):
            ans_ppl = "person"
        else:
            ans_ppl = ""
                           
        ans_misc = answers[2]
        ans_elec = answers[3]
        
        combined = f"{ans_anim}, {ans_ppl}, {ans_misc}, {ans_elec}"
        items    = self._clean_text(combined)

        # Cap the number of detected items to avoid overwhelming YOLO-World
        return items[: self.max_objects]