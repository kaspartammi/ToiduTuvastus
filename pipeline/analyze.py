# pipeline/analyze.py
from typing import Dict, Any, List
from PIL import Image

from models.detector import FoodDetector
from models.classifier import FoodClassifier
from models.depth import DepthEstimator
from pipeline.portion import estimate_volume_cm3, volume_to_grams
from pipeline.calories import NutritionDB

class Analyzer:
    def __init__(self, clf_weights, class_names):
        self.detector = FoodDetector()
        self.classifier = FoodClassifier(clf_weights, class_names)
        self.depth_estimator = DepthEstimator()
        self.nutrition = NutritionDB()

    def analyze_image(self, img: Image.Image) -> Dict[str, Any]:
        detections = self.detector.detect(img)
        depth = self.depth_estimator.infer(img)

        items: List[Dict[str, Any]] = []
        for det in detections:
            bbox = det["bbox"]
            crop = img.crop(bbox)
            name, cls_conf = self.classifier.classify(crop)

            volume = estimate_volume_cm3(bbox, depth)
            grams = volume_to_grams(name, volume)
            calories = self.nutrition.grams_to_calories(name, grams)

            items.append({
                "name": name,
                "bbox": bbox,
                "cls_conf": cls_conf,
                "volume_cm3": volume,
                "grams": grams,
                "calories": calories,
            })

        total_calories = sum(i["calories"] for i in items if i["calories"] is not None)
        return {"items": items, "total_calories": total_calories}
