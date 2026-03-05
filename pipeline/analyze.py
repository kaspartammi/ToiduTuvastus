# pipeline/analyze.py
from models.classifier import FoodClassifier
from models.depth import DepthEstimator
from models.detector import FoodDetector

class Analyzer:
    def __init__(self, clf_weights, class_names):
        self.classifier = FoodClassifier(clf_weights, class_names)
        self.detector = FoodDetector()
        self.depth_estimator = DepthEstimator()

        # --- FIX: depth estimator fallback ---
        if not hasattr(self.depth_estimator, "estimate"):
            # Try to map to another likely method name
            if hasattr(self.depth_estimator, "predict"):
                self.depth_estimator.estimate = self.depth_estimator.predict
            elif hasattr(self.depth_estimator, "estimate_depth"):
                self.depth_estimator.estimate = self.depth_estimator.estimate_depth
            else:
                raise AttributeError("DepthEstimator: puudub meetod 'estimate'")

    def analyze(self, image_path):
        detections = self.detector.detect(image_path)
        depth_map = self.depth_estimator.estimate(image_path)

        results = []
        for det in detections:
            crop = det["crop"]
            bbox = det["bbox"]

            name, conf = self.classifier.classify(crop)
            grams = self.depth_estimator.estimate_grams(bbox, depth_map)

            results.append({
                "name": name,
                "confidence": conf,
                "grams": grams
            })

        return results

    def analyze_image(self, image_path):
        return self.analyze(image_path)
