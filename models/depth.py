# models/depth.py

import torch
import cv2
import numpy as np
from PIL import Image
from models.nutrition import get_standard_portion

DEPTH_INFLUENCE = 0.4


class DepthEstimator:
    def __init__(self, model_name="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas  = torch.hub.load("intel-isl/MiDaS", model_name).to(self.device)
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        self._orig_w = None
        self._orig_h = None
        self._depth_values: list[float] = []

    def infer(self, img: Image.Image) -> np.ndarray:
        self._orig_w, self._orig_h = img.size
        self._depth_values = []
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        x = self.transform(img_cv).to(self.device)
        with torch.no_grad():
            depth = self.midas(x).squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    def estimate(self, image_path: str) -> np.ndarray:
        return self.infer(Image.open(image_path).convert("RGB"))

    def _scale_bbox(self, bbox, dH, dW):
        x1, y1, x2, y2 = bbox
        if self._orig_w and self._orig_h:
            sx = dW / self._orig_w; sy = dH / self._orig_h
            x1 = int(x1*sx); x2 = int(x2*sx)
            y1 = int(y1*sy); y2 = int(y2*sy)
        return (max(0, min(x1, dW-1)), max(0, min(y1, dH-1)),
                max(0, min(x2, dW)),   max(0, min(y2, dH)))

    def estimate_grams(self, bbox, depth_map: np.ndarray, label: str = None) -> float:
        dH, dW = depth_map.shape[:2]
        x1, y1, x2, y2 = self._scale_bbox(bbox, dH, dW)
        crop = depth_map[y1:y2, x1:x2]

        base_g = get_standard_portion(label) if label else 200

        if crop.size == 0:
            return float(base_g)

        avg_depth = float(np.mean(crop))
        self._depth_values.append(avg_depth)

        if len(self._depth_values) > 1:
            mean_d = float(np.mean(self._depth_values))
            std_d  = float(np.std(self._depth_values)) + 1e-8
            scale  = 1.0 + DEPTH_INFLUENCE * np.tanh((avg_depth - mean_d) / std_d)
        else:
            scale = 1.0

        return round(float(base_g * scale), 1)