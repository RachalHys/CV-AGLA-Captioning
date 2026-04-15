import argparse
import torch
import os
import json
from tqdm import tqdm
import sys
# os.environ['http_proxy'] = 'http://202.117.43.244:10007'
# os.environ['https_proxy'] = 'http://202.117.43.244:10007'

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "huggingface")
os.environ["TORCH_HOME"] = os.path.join(PROJECT_ROOT, ".cache", "torch")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVAL_DIR)
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
if EVAL_DIR in sys.path:
    sys.path.remove(EVAL_DIR)
sys.path.insert(0, EVAL_DIR)

from transformers import set_seed
from llava.utils import disable_torch_init
from PIL import Image
import math
from lavis.models import load_model_and_preprocess
from sample import evolve_agla_sampling
from torchvision import transforms
from lavis.common.registry import registry
from augmentation import augmentation 
evolve_agla_sampling()

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    disable_torch_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type=args.model_type, is_eval=True, device=device)
    generation_max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else args.max_length

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # model_class = registry.get_model_class('blip_image_text_matching')
    # model_class.PRETRAINED_MODEL_CONFIG_DICT['large'] = '/workspace/model/blip_itm_large/blip_itm_large.yaml' 
    # 'Large' for normal AGLA, 'base' for small AGLA. Both have the same architecture but different pretrained weights.
    model_itm, image_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", args.agla_size, device=device, is_eval=True)
    runtime_dtype_itm = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model_itm.to(runtime_dtype_itm)
    
    # GradCAM needs activation gradients (captured by forward hooks), NOT
    # weight gradients.  Setting requires_grad=False for all parameters means
    model_itm.requires_grad_(False)
 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    loader = transforms.Compose([transforms.ToTensor()])


    for i, line in enumerate(tqdm(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        question = line["text"]
        # POPE expects one-word answers; generative datasets should disable this.
        prompt = question + " Please answer this question with one word." if args.one_word_answer else question
        # For generative tasks and MME
        # conv.append_message(conv.roles[0],  qs)
        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # Cast directly to runtime dtype so Visual Encoder can skip some redundant casts.
        image_tensor = vis_processors["eval"](raw_image).unsqueeze(0).to(device, dtype=model.runtime_dtype)


        if args.use_agla:
             # Pre-load tensor image ONCE on CPU (float32)
            tensor_image_fp32 = loader(raw_image.resize((384, 384))).float()
            image = image_processors["eval"](raw_image).unsqueeze(0).to(device, dtype=runtime_dtype_itm)
            image.requires_grad_(True) # GradCAM needs gradients with respect to the input image, so we set requires_grad=True for the image tensor.

            question = text_processors["eval"](question)
            tokenized_text = model_itm.tokenizer(question, padding='longest', truncation=True, return_tensors="pt").to(device)
            augmented_image = augmentation(image, question, tensor_image_fp32, model_itm, tokenized_text, raw_image)
            image_tensor_cd = vis_processors["eval"](augmented_image).unsqueeze(0).to(device, dtype=model.runtime_dtype)
        else:
            image_tensor_cd = None      

        with torch.inference_mode():
            outputs = model.generate({"image": image_tensor, "prompt": prompt},
                use_nucleus_sampling=True, num_beams=args.num_beams,
                max_new_tokens=generation_max_new_tokens,
                top_p = args.top_p, repetition_penalty=1,
                images_cd=image_tensor_cd, cd_alpha = args.alpha, cd_beta = args.beta, temperature=args.temperature)


        outputs = outputs[0]
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "model_id": "instruct_blip",
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        CYCLE = 50 # flush the file every 50 lines to avoid data loss in case of crashes
        if (i + 1) % CYCLE == 0:
            ans_file.flush()
        

    ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-folder", type=str, default="/workspace/data/val2014")
    parser.add_argument("--question-file", type=str, default="/workspace/data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers-file", type=str, default="/workspace/eval/output/test.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--use_agla", action='store_true', default=True)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--model-type", type=str, default="vicuna7b")
    parser.add_argument("--agla-size", type=str, default="large", choices=["base", "large"], help="'large'=AGLA full (307M), 'base'=AGLA-small (120M)")
    parser.add_argument("--one-word-answer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
