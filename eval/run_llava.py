import argparse
import json
from tqdm import tqdm
import sys
import os
import torch
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EVAL_DIR)

# 2. Setup Cache
os.environ.setdefault("HF_HOME",    os.path.join(PROJECT_ROOT, ".cache", "huggingface"))
os.environ.setdefault("TORCH_HOME", os.path.join(PROJECT_ROOT, ".cache", "torch"))

# 3. Setup sys.path
for path in [PROJECT_ROOT, EVAL_DIR]:
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria    
from PIL import Image
from transformers import set_seed
from sample import evolve_agla_sampling
evolve_agla_sampling()
from augmentation import augmentation
from lavis.models import load_model_and_preprocess
from torchvision import transforms

def eval_model(args):
    disable_torch_init()
    # Global PyTorch acceleration flags
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Hardware allocation option
    if args.num_gpus == 2:
        print("Using Dual-GPU Mode")
        llava_device_map = 'auto' 
        device_blip = 'cuda:1'    
    else:
        print("Using Single-GPU Mode")
        llava_device_map = 'cuda:0'
        device_blip = 'cuda:0'

    # Handle data precision option
    load_8bit = args.precision == "int8"
    load_4bit = args.precision == "int4"
    has_bf16  = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if args.precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    print(f"⚙️ Loading model with precision: {args.precision.upper()}")

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, 
        load_8bit=load_8bit, load_4bit=load_4bit,
        torch_dtype=compute_dtype, device_map=llava_device_map
    )

    # With device_map='auto', the vision tower can be on any GPU.
    vision_tower = model.get_vision_tower()
    vt_device = next(vision_tower.parameters()).device
    runtime_dtype_itm = torch.bfloat16 if has_bf16 else torch.float16
    print("Loading BLIP-ITM model...")
    model_itm, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", args.agla_size, device=device_blip, is_eval=True,)
    model_itm.to(runtime_dtype_itm) # Cast model_itm to correct dtype

    model_itm.requires_grad_(False) # Disable weight gradients — GradCAM only needs activation grad

    # Defragment CUDA allocator after loading two large models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(os.path.expanduser(args.question_file), "r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    
    # Handle max question option
    if args.max_questions is not None:
        print(f"⚠️ Limiting evaluation to the first {args.max_questions} questions.")
        questions = questions[:args.max_questions]
    else:
        print(f"📊 Running full evaluation on all {len(questions)} questions.")

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    # File I/O with explicit UTF-8 encoding
    ans_file = open(answers_file, "w", encoding="utf-8")
    CYCLE = 25 # Flush the file every 25 lines to avoid data loss

    loader = transforms.Compose([transforms.ToTensor()])
    for i, line in enumerate(tqdm(questions)):
        idx = line.get("id") or line.get("question_id")
        image_file = line["image"]
        raw_query = line.get("query") or line.get("text") or ""
        
        # Use --prompt-suffix to add an optional suffix when needed.
        final_query = raw_query + (" " + args.prompt_suffix if args.prompt_suffix else "")

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + final_query
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + final_query

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs) 
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Text inputs go to LLM's base device
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)
        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert("RGB")
        # Main image → vision tower device dynamically casted to compute_dtype
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors="pt")["pixel_values"][0].to(dtype=torch.float16, device=vt_device)
        
        if args.use_agla:
            tensor_image = loader(raw_image.resize((384,384))).float()
            # Cast to correct dtype for BLIP-ITM
            image_itm_input = vis_processors["eval"](raw_image).unsqueeze(0).to(device=device_blip, dtype=runtime_dtype_itm)
            image_itm_input.requires_grad_(True)

            itm_text = text_processors["eval"](raw_query)
            tokenized_text = model_itm.tokenizer(itm_text, padding='longest', truncation=True, return_tensors="pt").to(device_blip)
            
            augmented_image = augmentation(image_itm_input, itm_text, tensor_image.float(), model_itm, tokenized_text, raw_image)
            # Send CD image to vt_device and cast to compute_dtype
            image_tensor_cd = image_processor.preprocess(augmented_image, return_tensors="pt")["pixel_values"][0].to(dtype=torch.float16, device=vt_device)
        else:
            image_tensor_cd = None

        # Stopping criteria
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0),
                images_cd=(image_tensor_cd.unsqueeze(0) if image_tensor_cd is not None else None),
                cd_alpha=args.alpha,
                cd_beta=args.beta,
                do_sample=True,
                temperature=args.temperature, 
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        ans_file.write(json.dumps({
            "id": idx,
            "prompt": prompt,
            "response": outputs,
            "image": image_file,
        }, ensure_ascii=False) + "\n")
        # FLUSH block
        if (i + 1) % CYCLE == 0:
            ans_file.flush()
        
    ans_file.close()
    print(f"✅ Finished! Results saved to {ans_file.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use_agla", action='store_true', default=False)
    parser.add_argument("--prompt-suffix", type=str, default="", help="Optional suffix appended to every query.")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs: 1 or 2.")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "int8", "int4"], default="fp16", help="Model precision format.")
    parser.add_argument("--agla-size", type=str, default="large", choices=["base", "large"], help="'large'=AGLA full (307M), 'base'=AGLA-small (120M)")
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to process. Default is ALL.")
    
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)