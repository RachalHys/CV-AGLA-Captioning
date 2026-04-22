import os
import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld
from mobile_sam import sam_model_registry, SamPredictor
_MAX_DILATION_PX = 50

class YoloSamAugmenter:
    def __init__(self, yolo_id="yolov8s-world.pt", sam_checkpoint="mobile_sam.pt", device="cuda:1"):
        self.device = device
        self._last_classes = [] # Cache the last set of classes to avoid redundant YOLO model updates
        # Yolo for grounding
        print(f"Loading YOLO-World on {device}...")
        self.yolo_model = YOLOWorld(yolo_id).to(device)

        # Mobile SAM for augmentation
        print(f"Loading MobileSAM on {device}...")
        if not os.path.exists(sam_checkpoint):
            os.system(
                "wget -q https://raw.githubusercontent.com/ChaoningZhang/"
                "MobileSAM/master/weights/mobile_sam.pt"
            )

        sam = sam_model_registry["vit_t"](checkpoint=sam_checkpoint).to(device=device)
        self.sam_predictor = SamPredictor(sam)
        print("YoloSam Augmenter Ready!")

    # create the augmented view
    # input: original image and list of objects
    # output: masked image or None if no objects found
    def augmentation(self, raw_image: Image.Image, object_list: list, conf_threshold=0.2, expansion_ratio=0.0) -> "Image.Image | None":
        if not object_list:
            return None

        # Find bounding boxes
        search_classes = object_list + ["object"]
        if search_classes != self._last_classes:
            self.yolo_model.set_classes(search_classes)
            self._last_classes = search_classes

        results = self.yolo_model.predict(raw_image, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return None

        # Segment using mobile SAM
        image_np = np.array(raw_image)
        self.sam_predictor.set_image(image_np)
        h, w = image_np.shape[:2]
        combined_mask = np.zeros((h, w), dtype=bool)

        for box in boxes:
            masks, _, _ = self.sam_predictor.predict(box=box[None, :], multimask_output=False)
            combined_mask = np.logical_or(combined_mask, masks[0])

        # Dilation (Expand mask section)
        if expansion_ratio > 0:
            dilation_size = int(max(h, w) * expansion_ratio)
            # Cap kernel size — cv2.dilate is O(k^2) and >50px kernels are slow
            dilation_size = min(dilation_size, _MAX_DILATION_PX)
            if dilation_size > 0:
                kernel = np.ones((dilation_size, dilation_size), np.uint8)
                combined_mask = cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Apply mask on the original image
        masked_image = image_np.copy()
        masked_image[~combined_mask] = 0
        
        return Image.fromarray(masked_image)