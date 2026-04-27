import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

# Creating a transformation to convert tensor images to PIL images
_to_pil = transforms.ToPILImage()

# Generating augmentated images based on the input prompt
def augmentation(image, question, tensor_image, model, tokenized_text, raw_image, kernel_size=49, vis = False):
    device = image.device

    # 1. Compute similarity score using the ITM head (no grad needed)
    with torch.no_grad():
        itm_logits = model({"image": image, "text_input": question}, match_head="itm")
        sim = F.softmax(itm_logits.float(), dim=-1)[:, 1]
 
    # Masking ratio: clamp both sides so quantile is always in (0, 1)
    ratio: float = (1.0 - sim / 2.0).clamp(min=1e-5, max=1.0 - 1e-5).item()

    with torch.enable_grad():
        gradcams, _ = compute_gradcam(model=model,
                                      visual_input=image,
                                      text_input=question,
                                      tokenized_text=tokenized_text,
                                      block_num=6)

    model.text_encoder.base_model.base_model.encoder.layer[6].crossattention.self.save_attention = False
    if image.grad is not None:
        image.grad = None

    gradcams = [gradcam_[1] for gradcam_ in gradcams]
    gradcams1 = torch.stack(gradcams).reshape(image.size(0), -1)
    gradcam = gradcams1.reshape(24, 24)

    # Pure GPU Pipeline: Replaces CPU-bound getAttMap (scipy & skimage)
    # TF.gaussian_blur and interpolate might not support FP16, so we ensure gradcam is float32 on GPU
    attMap = gradcam.detach().float()
    attMap -= attMap.min()
    attMap_max = attMap.max()
    if attMap_max > 0:
        attMap /= attMap_max

    attMap_up = F.interpolate(
        attMap.unsqueeze(0).unsqueeze(0),      # [1, 1, 24, 24]
        size=(384, 384),
        mode="bicubic",                     
        align_corners=False
    )                               # [384, 384]

    sigma = 0.02 * 384.0                       # = 7.68

    blurred = TF.gaussian_blur(
        attMap_up,   # [1, 1, 384, 384]
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma]
    ).squeeze()                                # [384, 384]

    # 3.4 Min-max normalize
    blurred -= blurred.min()
    blurred_max = blurred.max()
    if blurred_max > 0:
        blurred /= blurred_max

    blurred = blurred.float().to(device)
    # quantile value at (1 - ratio): pixels ABOVE this threshold are kept
    threshold = torch.quantile(blurred.reshape(-1), 1.0 - ratio)

    # Build binary mask on GPU: 1 = keep, 0 = mask
    # Shape: [384, 384] → broadcast to [384, 384, 3] for RGB multiplication
    mask_hw  = (blurred >= threshold).to(dtype=torch.float32)          # [H, W]
    mask_hwc = mask_hw.unsqueeze(2).expand(-1, -1, 3)                          # [H, W, 3]

    # tensor_image is CPU [3, H, W]; permute to [H, W, 3] and move to GPU
    new_image_gpu = tensor_image.to(device=device, non_blocking=True).permute(1, 2, 0) * mask_hwc  # [H, W, 3]

    # Convert back to PIL
    imag = _to_pil(new_image_gpu.cpu().permute(2, 0, 1).clamp(0.0, 1.0))

    if vis:
        heatmap_np = blurred.cpu().numpy()
        return imag, heatmap_np

    return imag