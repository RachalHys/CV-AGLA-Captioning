# InstructBLIP 7B – Setup & Evaluation Guide

## Requirements

* Python 3.10
* GPU with CUDA support (recommended)
---

## 1. Setup Environment
### 1.1 Install PyTorch (match your CUDA version)

Example (CUDA 12.8):

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
--extra-index-url https://download.pytorch.org/whl/cu128
```

---

### 1.2 Install dependencies

```bash
pip install -r test_requirements.txt
```

---

### 1.3 Install LAVIS (IMPORTANT)

```bash
pip install salesforce-lavis==1.0.2 --no-deps
```

---

## 2. Prepare for AMBER Evaluation

### 2.1 Download spaCy model

```bash
python -m spacy download en_core_web_lg
```

### 2.2 Download NLTK data

```python
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
```

---

## 3. Convert Amber query to Model format (ONLY IF NEEDED. THE FILE IS ALREADY CREATED)
* You should go to AMBER folder first before running this CLI
```bash
python convert.py --input data/query/query_generative.json --output amber_generative.jsonl
```

---

## 4. Run inference baseline LLavis + AGLA
```bash
python eval/run_llava.py \
    --image-folder AMBER/image \
    --question-file AMBER/amber_generative.jsonl \
    --answers-file AMBER/amber_llava_small_output.jsonl \
    --use_agla \
    --precision fp16 \
    2>&1 | tee run_log.txt
```
---

## 4. AMBER evaluation
* You should go to AMBER folder first before running this CLI

### 4.1 Convert Output to AMBER Format
```bash
python convert_amber_eval.py \
    --input amber_llava_base_output.jsonl \
    --output amber_eval.json
```

### 4.2 Run AMBER Evaluation

```bash
python inference.py \
    --inference_data amber_eval.json \
    --evaluation_type g \
    --top_k 20
```

---

## 6. Use the visualize.ipynb in eval/ to perform quantitative evaluation.

## Notes

* Ensure CUDA version matches PyTorch installation
* Do NOT install LAVIS with dependencies (use `--no-deps`)
* NLP resources (spaCy, NLTK) are required for AMBER evaluation
