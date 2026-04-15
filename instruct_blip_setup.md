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
pip install -r requirements.txt
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

## 3. Configuration

* Inference mode (FP16, BF16, INT8, INT4):
  `lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml`

* Model checkpoint path:
  `lavis/configs/default.yaml`

---

## 4. Run Inference (InstructBLIP 7B)

```bash
python eval/run_instructblip.py \
    --model-type vicuna7b \
    --image-folder AMBER/image \
    --question-file AMBER/amber_generative.jsonl \
    --answers-file AMBER/amber_generative_7b_output.jsonl \
    --max-new-tokens 512 \
    --num-beams 1 \
    --no-one-word-answer \
    --use_agla \
    --agla-size base \
    2>&1 | tee AMBER/run_log.txt
```

---

## 5. Convert Output to AMBER Format

```bash
python AMBER/convert_amber_eval.py \
    --input AMBER/amber_generative_7b_output.jsonl \
    --output AMBER/amber_eval.json
```

---

## 6. Run AMBER Evaluation

```bash
python AMBER/inference.py \
    --inference_data AMBER/amber_eval.json \
    --evaluation_type g \
    --top_k 20
```

---

## Notes

* Ensure CUDA version matches PyTorch installation
* Do NOT install LAVIS with dependencies (use `--no-deps`)
* NLP resources (spaCy, NLTK) are required for AMBER evaluation
