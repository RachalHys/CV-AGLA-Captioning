# 🚀 LLaVA-SAM Integration: Setup & Evaluation Guide

---

## 🟢 PART 1: Running on Kaggle (Notebook)

**Prerequisites:** 
* Turn on **Internet**.
* Set Accelerator to **GPU T4 x2**.

### Cell 1: Environment Setup
```python
import os
%cd /kaggle/working
!rm -rf CV-AGLA-Captioning

# 1. Clone the integration branch
!git clone -b llava-SAM-integration https://github.com/RachalHys/CV-AGLA-Captioning.git
%cd CV-AGLA-Captioning

# 2. Install dependencies
!pip install -r test_requirements.txt
!pip install salesforce-lavis==1.0.2 --no-deps  # MUST be installed without deps

# 3. Download Evaluation Resources
!python -m spacy download en_core_web_lg
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
```

### Cell 2: Run Inference
```python
%cd /kaggle/working/CV-AGLA-Captioning

# Set paths (Update IMAGE_FOLDER if using a different Kaggle dataset)
IMAGE_FOLDER = "ENTER YOUR IMAGE FOLDER"
QUESTION_FILE = "AMBER/amber_generative.jsonl"
OUTPUT_FILE = "AMBER/amber_llava_sam_output.jsonl"

!python eval/run_llava.py \
    --image-folder {IMAGE_FOLDER} \
    --question-file {QUESTION_FILE} \
    --answers-file {OUTPUT_FILE} \
    --precision fp16 \
    --num-gpus 2 \
    --use_agla \
    --max-new-tokens 180 \
    --alpha 2.0 \
    --beta 0.5 \
    --yolo-conf 0.2 \
    --expansion-ratio 0.0 \
    2>&1 | tee run_log_llava.txt
```

### Cell 3: Format & Evaluate
```python
%cd /kaggle/working/CV-AGLA-Captioning/AMBER

# 1. Convert output to AMBER format
!python convert_amber_eval.py \
    --input amber_llava_sam_output.jsonl \
    --output amber_eval.json

# 2. Run Benchmark
!python inference.py \
    --inference_data amber_eval.json \
    --evaluation_type g \
    --top_k 30
```

---

## 🖥️ PART 2: Running via Command Line (Local/Server)

**Prerequisites:** Ubuntu/Linux, Python 3.10+, and NVIDIA GPU(s).

### Step 1: Clone & Install
Open your terminal and run:
```bash
git clone -b llava-SAM-integration https://github.com/RachalHys/CV-AGLA-Captioning.git
cd CV-AGLA-Captioning

# Install libraries
pip install -r test_requirements.txt
pip install salesforce-lavis==1.0.2 --no-deps

# Download NLP models for AMBER
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet')"
```

### Step 2: Run Inference
*(Assuming your AMBER images are located in `AMBER/image`)*
```bash
python eval/run_llava.py \
    --image-folder AMBER/image \
    --question-file AMBER/amber_generative.jsonl \
    --answers-file AMBER/amber_llava_sam_output.jsonl \
    --precision fp16 \
    --num-gpus 2 \
    --use_agla \
    --max-new-tokens 180 \
    --alpha 2.0 \
    --beta 0.5 \
    --yolo-conf 0.2 \
    --expansion-ratio 0.0 \
    2>&1 | tee run_log_llava.txt
```
*(Note: Change `--num-gpus 2` to `1` and `--precision fp16` to `int8` if you only have a single GPU with limit VRAM).*

### Step 3: Format & Evaluate
```bash
cd AMBER

# Convert format
python convert_amber_eval.py --input amber_llava_sam_output.jsonl --output amber_eval.json

# Evaluate
python inference.py --inference_data amber_eval.json --evaluation_type g --top_k 30
```
