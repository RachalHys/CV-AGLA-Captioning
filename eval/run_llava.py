import argparse
import json
from tqdm import tqdm
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    
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
    
    if args.precision == "bf16":
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16

    print(f"⚙️ Loading model with precision: {args.precision.upper()}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, 
        load_8bit=load_8bit, load_4bit=load_4bit,
        torch_dtype=compute_dtype, device_map=llava_device_map
    )

    model_itm, vis_processors, text_processors = load_model_and_preprocess(
        "blip_image_text_matching", "large", device=device_blip, is_eval=True
    )
    loader = transforms.Compose([transforms.ToTensor()])


    with open(os.path.expanduser(args.question_file), "r") as f:
        questions = json.load(f)
    
    # Handle max question option
    if args.max_questions is not None:
        print(f"⚠️ Limiting evaluation to the first {args.max_questions} questions.")
        questions = questions[:args.max_questions]
    else:
        print(f"📊 Running full evaluation on all {len(questions)} questions.")

    ans_list = [] 

    for line in tqdm(questions):
        idx = line.get("id") or line.get("question_id")
        image_file = line["image"]
        raw_query = line.get("query") or line.get("text") or ""
        
        custom_prompt = f"{raw_query} Describe this image."

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + custom_prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + custom_prompt

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs) 
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        raw_image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        raw_image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_agla:
            tensor_image = loader(raw_image.resize((384,384)))
            image_itm_input = vis_processors["eval"](raw_image).unsqueeze(0).to(device_blip)
            itm_text = text_processors["eval"](raw_query)
            tokenized_text = model_itm.tokenizer(itm_text, padding='longest', truncation=True, return_tensors="pt").to(device_blip)
            
            augmented_image = augmentation(image_itm_input, itm_text, tensor_image, model_itm, tokenized_text, raw_image)
            image_tensor = image_processor.preprocess(augmented_image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=raw_image_tensor.unsqueeze(0).to(dtype=compute_dtype, device='cuda'),
                images_cd=(image_tensor.unsqueeze(0).to(dtype=compute_dtype, device=device_blip) if image_tensor is not None else None),
                cd_alpha = args.alpha,
                cd_beta = args.beta,
                do_sample=True,
                temperature=1.0, 
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                use_cache=True)
            
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        ans_list.append({
            "id": idx,
            "response": outputs,
            "image": image_file
        })

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w") as f:
        json.dump(ans_list, f, indent=4)
        
    print(f"✅ Finished! Results saved to {answers_file}")

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
    
    parser.add_argument("--use_agla", action='store_true', default=False)
    
    parser.add_argument("--num-gpus", type=int, default=2, help="Number of GPUs: 1 or 2. Default is 2 (Kaggle T4x2).")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16", "int8", "int4"], default="fp16", help="Model precision format.")
    
    parser.add_argument("--max-questions", type=int, default=None, help="Maximum number of questions to process. Default is ALL.")
    
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)