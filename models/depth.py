# models/depth.py
import torch
import cv2
import numpy as np
from PIL import Image

class DepthEstimator:
    def __init__(self, model_name="DPT_Large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", model_name).to(self.device)
        self.midas.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform

    def infer(self, img: Image.Image) -> np.ndarray:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        x = self.transform(img_cv).to(self.device)
        with torch.no_grad():
            pred = self.midas(x)
            depth = pred.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth

    # --- FIX: Analyzer expects this method ---
    def estimate(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        return self.infer(img)

    # --- FIX: Analyzer expects this method too ---
    def estimate_grams(self, bbox, depth_map) -> float:
        # bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox
        crop = depth_map[y1:y2, x1:x2]

        if crop.size == 0:
            return 0.0

        avg_depth = float(np.mean(crop))
        grams = max(10.0, avg_depth * 300.0)  # simple placeholder formula
        return grams
