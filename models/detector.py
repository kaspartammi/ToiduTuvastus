# models/detector.py
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict
from config import YOLO_WEIGHTS

class FoodDetector:
    def __init__(self):
        self.model = YOLO(str(YOLO_WEIGHTS))

    def detect(self, img: Image.Image) -> List[Dict]:
        results = self.model.predict(img, verbose=False)[0]
        items = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            items.append({
                "bbox": [x1, y1, x2, y2],
                "conf": conf,
                "cls_id": cls_id,
            })
        return items
