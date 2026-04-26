# Tag-and-Segment AGLA: Mitigating Object Hallucinations in Open-Ended Image Captioning

## Overview
This project introduces an improved architecture based on the **AGLA** (Assembly of Global and Local Attention) decoding strategy. While Large Vision-Language Models (LVLMs) demonstrate remarkable capabilities, they frequently suffer from *object hallucinations*—generating objects in text that do not exist in the provided images. 

Our work specifically addresses the limitations of the original AGLA framework in **open-ended image captioning** tasks, offering a highly robust, prompt-independent pipeline that strictly grounds the model's generation to verified visual entities.

## Motivation: The Limitation of Original AGLA
The original AGLA mitigates hallucination by fusing logits from the original image and an "augmented view." However, it relies on **GradCAM** and the user's text prompt to generate this augmented view. 

While this works well for specific Visual Question Answering (VQA) prompts (e.g., *"Is there a dog?"*), it fails during open-ended generative tasks. When given a generic prompt like *"Describe this image"*, GradCAM lacks specific object keywords to anchor its attention, resulting in noisy visual masks that fail to suppress background distractions.

## Our Approach: The Tag-and-Segment Pipeline
To solve this, we shift from a *prompt-based* to an *object-based* augmentation strategy. Our pipeline proactively discovers and isolates real objects before the LVLM begins text generation. The architecture consists of three main stages:

1. **Inventory Detection (VQA + Language Filtering):**
   Instead of relying on the user's prompt, we utilize a general VQA model (e.g., BLIP) to interrogate the image (e.g., *"What objects are in this image?"*). The output is filtered to create a clean, standardized list of verified entities ($O$).
   
2. **Precision Segmentation (YOLO-World + MobileSAM):**
   Using the inventory list ($O$), an open-vocabulary object detector (YOLO-World) locates the bounding boxes of the objects. MobileSAM then extracts pixel-perfect segmentation masks. We apply a strict **Hard Masking** technique—blacking out the background entirely without mask dilation—to completely remove hallucination-inducing environmental contexts.

3. **Logit Fusion (Assembly of Attention):**
   During the decoding phase, the LVLM processes the original image (to maintain natural language fluency) and our Tag-and-Segment augmented image (to enforce local accuracy) in parallel. The logits are fused using Adaptive Plausibility Constraints to guarantee grammatically sound and hallucination-free text generation.

## Experimental Results
We evaluated our proposed method on the **AMBER Generative Benchmark** (1,004 high-quality images) using **LLaVA-1.5 (7B)**. Our architecture achieves state-of-the-art hallucination suppression while maintaining competitive computational efficiency.

| Method | CHAIR ($\downarrow$) | Cover ($\uparrow$) | Hal ($\downarrow$) | Cog ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| Regular (LLaVA-1.5 Baseline) | 7.8 | **51.0** | 36.4 | 4.2 |
| VCD | 8.0 | 49.3 | 34.4 | 3.8 |
| Original AGLA | 7.3 | 51.3 | 34.5 | 3.9 |
| **Tag-and-Segment AGLA (Ours)** | **6.9** | 48.4 | **30.7** | **3.2** |

*Note: The slight reduction in the Cover metric is an expected trade-off of our strict "Hard Masking" approach, which sacrifices minor background details to achieve the lowest possible hallucination rates (CHAIR, Hal, Cog).*

## Setup and Reproduction
*(Setup, installation, and inference instructions are detailed in a separate documentation file. Please refer to `setup_inference.md` for full guidelines on how to run this pipeline.)*

## Acknowledgements
* The logit adjustment framework and original AGLA methodology are based on the [AGLA repository](https://github.com/Lackel/AGLA).
* We utilize [YOLO-World](https://github.com/AILab-CVC/YOLO-World) for open-vocabulary detection and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) for lightweight segmentation.
* Evaluation was conducted using the [AMBER Benchmark](https://github.com/junyangwang0410/AMBER).