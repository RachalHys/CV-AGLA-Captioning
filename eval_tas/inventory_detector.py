import torch
import re
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Detect objects in an image
class BLIPInventoryDetector:
    def __init__(self, model_id="Salesforce/blip-vqa-base", device="cuda:1"):
        self.device = device
        print(f"Loading BLIP VQA on {device}...")
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_id, torch_dtype=torch.float16
        ).to(device).eval()
        print("BLIP Inventory Detector Ready!")

    def _ask_blip(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20)
        return self.processor.decode(out[0], skip_special_tokens=True).lower()

    def _clean_text(self, text: str) -> list:
        # Remove Common Phrases
        junk_patterns = [
            r"\bi don't know\b", r"\bdont know\b", r"\bnot sure\b", r"\bunknown\b", 
            r"\bmore than\b", r"\byes\b", r"\bno\b", r"\bnone\b", r"\bnothing\b", 
            r"\bimage\b", r"\bpicture\b", r"\bphoto\b"
        ]
        for pattern in junk_patterns:
            text = re.sub(pattern, ' ', text)

        # Replace connectors with commas
        text = re.sub(r'\b(and|with|next to|on|in|standing|sitting|holding)\b', ',', text)
        
        # Remove prepositions
        stopwords = r'\b(a|an|the|some|many|several|few|one|two|three|four|there are|there is|i see|any)\b'
        text = re.sub(stopwords, '', text)
        
        raw_items = [item.strip() for item in text.split(',')]
        final_items = []
        
        # Standardize nouns to describe people
        word_map = {'people': 'person', 'humans': 'person', 'men': 'man', 'women': 'woman', 'guys': 'man', 'children': 'child'}
        
        for item in raw_items:
            item = re.sub(r'[^\w\s]', '', item).strip()
            # Switch to singular nouns
            if len(item) > 3 and item.endswith('s') and item not in ['grass', 'glass', 'bus', 'gas']:
                item = item[:-1] 
                
            item = word_map.get(item, item)
            if len(item) > 1: 
                final_items.append(item)
                
        return list(set(final_items))

    def get_inventory(self, image: Image.Image) -> list:
        ans_anim = self._ask_blip(image, "What animals are in this image?")
        ans_ppl  = "person" if "yes" in self._ask_blip(image, "Are there any people in this image?") else ""
        ans_misc = self._ask_blip(image, "What objects are in this image?")
        ans_elec = self._ask_blip(image, "What man-made objects are in this image?")
        
        combined = f"{ans_anim}, {ans_ppl}, {ans_elec}, {ans_misc}"
        return self._clean_text(combined)