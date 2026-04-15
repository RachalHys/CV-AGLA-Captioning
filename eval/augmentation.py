import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
import matplotlib.pyplot as plt
import os
import time

# Creating a transformation to convert tensor images to PIL images
_to_pil = transforms.ToPILImage()

# Generating augmentated images based on the input prompt
def augmentation(image, question, tensor_image, model, tokenized_text, raw_image, vis = False, vis_dir = "./AMBER/vis_results"):
    device = image.device

    # wrap ITC with torch.no_grad() and compute it before the GradCAM pass
    # so the GPU memory for the GradCAM graph is allocated on a clean slate.
    with torch.no_grad():
        itc_score = model({"image": image, "text_input": question}, match_head="itc")

    # use tensor .clamp() (stays on GPU) then .item() once for the scalar.
    ratio: float = (1.0 - itc_score / 2.0).clamp(max=1.0 - 1e-5).item()

    with torch.set_grad_enabled(True):
        gradcams, _ = compute_gradcam(model=model,
                                      visual_input=image,
                                      text_input=question,
                                      tokenized_text=tokenized_text,
                                      block_num=6)

    gradcams = [gradcam_[1] for gradcam_ in gradcams]
    gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
    gradcam = gradcams1.reshape(24, 24)

    # Pure GPU Pipeline: Replaces CPU-bound getAttMap (scipy & skimage)
    gradcam_raw = gradcam.float().to(device=device)  # [24, 24] on GPU
    g_min, g_max = gradcam_raw.min(), gradcam_raw.max()
    gradcam_norm = (gradcam_raw - g_min) / (g_max - g_min + 1e-8)

    gradcam_up = F.interpolate(
        gradcam_norm.unsqueeze(0).unsqueeze(0),
        size=(384, 384),
        mode="bicubic",
        align_corners=False
    )
    
    avg_gradcam_gpu = TF.gaussian_blur(
        gradcam_up,
        kernel_size=[63, 63],
        sigma=[7.68, 7.68]
    ).squeeze()

    # quantile value at (1 - ratio): pixels ABOVE this threshold are kept
    threshold = torch.quantile(avg_gradcam_gpu.reshape(-1), 1.0 - ratio)

    # Build binary mask on GPU: 1 = keep, 0 = mask
    # Shape: [384, 384] → broadcast to [384, 384, 3] for RGB multiplication
    mask_hw  = (avg_gradcam_gpu >= threshold).to(dtype=torch.float32)          # [H, W]
    mask_hwc = mask_hw.unsqueeze(2).expand(-1, -1, 3)                          # [H, W, 3]

    # tensor_image is CPU [3, H, W]; permute to [H, W, 3] and move to GPU
    new_image_gpu = tensor_image.to(device=device).permute(1, 2, 0) * mask_hwc  # [H, W, 3]

    # Convert back to PIL (requires CPU)
    new_image_cpu = new_image_gpu.cpu()
    imag = _to_pil(new_image_cpu.permute(2, 0, 1).clamp(0.0, 1.0))
    
    # Heatmap Visualization (Optional): Overlay GradCAM heatmap on original image and save
    # When running model, comment out this section. Only for debugging/visualization.

    if vis:
        save_dir = vis_dir
        os.makedirs(save_dir, exist_ok=True)
        
        img_bg = tensor_image.permute(1, 2, 0).cpu().numpy()
        heatmap = avg_gradcam_gpu.cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(img_bg)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.axis('off')
        
        save_path = f"{save_dir}/heatmap_{int(time.time()*100)}.png"   
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved heatmap at: {save_path}")
        plt.close()

    return imag