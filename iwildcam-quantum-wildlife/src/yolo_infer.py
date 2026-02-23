"""
Stable YOLOv8 inference helper for cropping ROIs.
Uses ultralytics YOLO API. Loads model once for reuse.
"""
from ultralytics import YOLO
import os
import cv2
from PIL import Image
import numpy as np

# Provide a loader that reuses model instance.
class YOLOCropper:
    def __init__(self, weights_path="detectors/yolov8/yolov8n.pt", conf_thresh=0.25, device=0):
        # ultralytics will download default weights if provided name, or use local file
        self.model = YOLO(weights_path)
        self.conf_thresh = conf_thresh
        self.device = device

    def crop_from_image_path(self, image_path, output_dir="web/crops"):
        os.makedirs(output_dir, exist_ok=True)
        results = self.model.predict(source=image_path, conf=self.conf_thresh, device=self.device, show=False)
        crops = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1,y1,x2,y2 = [int(float(v)) for v in box[:4]]
                crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
                if crop.size == 0:
                    continue
                crop_path = os.path.join(output_dir, f"{os.path.basename(image_path)}__{i}.jpg")
                cv2.imwrite(crop_path, crop)
                crops.append(crop_path)
        return crops

    def crop_from_array(self, img_array, output_dir="web/crops", prefix="arr"):
        """
        img_array: HxWxC uint8 (0-255) or float 0-1.
        Saves a temp image and runs crop on it.
        """
        if img_array.dtype != "uint8":
            img_array = (img_array * 255).astype("uint8")
        temp_path = os.path.join(output_dir, f"{prefix}_temp.jpg")
        os.makedirs(output_dir, exist_ok=True)
        Image.fromarray(img_array).save(temp_path)
        return self.crop_from_image_path(temp_path, output_dir=output_dir)
