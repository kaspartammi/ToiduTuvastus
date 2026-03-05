# models/detector.py
from ultralytics import YOLO
from PIL import Image
import numpy as np

class FoodDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        results = self.model(image_path)[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])

            # lõikame pildi välja
            crop = img.crop((x1, y1, x2, y2))

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop": crop,
                "confidence": conf
            })

        return detections
