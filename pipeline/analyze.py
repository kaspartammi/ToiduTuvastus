# pipeline/analyze.py

import numpy as np
from PIL import Image as PILImage
from models.classifier import FoodClassifier
from models.depth import DepthEstimator
from models.detector import FoodDetector
from models.nutrition import get_calories_per_100g

# ── thresholds ────────────────────────────────────────────────────────────────
VIT_CONFIDENCE_THRESHOLD = 0.50
IOU_DEDUP_THRESHOLD      = 0.15
SAME_LABEL_DIST_RATIO    = 0.60

# Mixed-dish ingredients (appear as chunks inside other foods — lower threshold)
MIXED_DISH_LABELS    = {"steak", "pork_chop", "prime_rib", "grilled_salmon",
                        "scallops", "shrimp_and_grits"}
MIXED_DISH_THRESHOLD = 0.25

MAX_PER_LABEL = 1   # default — most dishes appear once per image
MAX_PER_LABEL_EXCEPTIONS = {
    "spring_rolls": 3,
    "gyoza":        6,
    "tacos":        3,
    "sushi":        6,
    "donuts":       4,
    "cup_cakes":    4,
    "macarons":     6,
    "chicken_wings":8,
}

PLATE_BRIGHTNESS_MIN = 200
PLATE_SATURATION_MAX = 25
DARK_BACKGROUND_MAX  = 60


def _looks_like_background(crop: PILImage.Image) -> bool:
    thumb = crop.convert("RGB").resize((64, 64))
    arr   = np.array(thumb, dtype=np.float32)
    brightness = arr.mean()
    r, g, b = arr[:,:,0]/255, arr[:,:,1]/255, arr[:,:,2]/255
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    sat  = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-8), 0).mean() * 255

    is_white_plate = brightness > PLATE_BRIGHTNESS_MIN and sat < PLATE_SATURATION_MAX
    is_dark_bg     = brightness < DARK_BACKGROUND_MAX  and sat < 30

    if is_white_plate or is_dark_bg:
        print(f"[Analyzer] Background crop (brightness={brightness:.0f}, sat={sat:.0f}) — skipping")
        return True
    return False


def _iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def _center_dist(a, b, w, h):
    ca=((a[0]+a[2])/2,(a[1]+a[3])/2)
    cb=((b[0]+b[2])/2,(b[1]+b[3])/2)
    return (((ca[0]-cb[0])/w)**2+((ca[1]-cb[1])/h)**2)**0.5


def _deduplicate(results, img_w, img_h):
    results = sorted(results, key=lambda r: r["confidence"], reverse=True)
    kept = []
    for candidate in results:
        dup = False
        for existing in kept:
            if _iou(candidate["bbox"], existing["bbox"]) > IOU_DEDUP_THRESHOLD:
                dup = True; break
            if (candidate["name"] == existing["name"] and
                    _center_dist(candidate["bbox"], existing["bbox"], img_w, img_h)
                    < SAME_LABEL_DIST_RATIO):
                dup = True; break
        if not dup:
            kept.append(candidate)
    return kept


def _estimate_calories(label, grams):
    cal = get_calories_per_100g(label)
    if grams <= 0: return None
    return round((grams / 100) * cal, 1)


class Analyzer:
    def __init__(self, clf_weights, class_names):
        self.classifier      = FoodClassifier(clf_weights, class_names)
        self.detector        = FoodDetector()
        self.depth_estimator = DepthEstimator()

        if not hasattr(self.depth_estimator, "estimate"):
            for m in ("predict", "estimate_depth"):
                if hasattr(self.depth_estimator, m):
                    self.depth_estimator.estimate = getattr(self.depth_estimator, m)
                    break
            else:
                raise AttributeError("DepthEstimator: puudub meetod 'estimate'")

    def analyze(self, image_path: str) -> list[dict]:
        img_w, img_h = PILImage.open(image_path).size
        detections   = self.detector.detect(image_path)
        depth_map    = self.depth_estimator.estimate(image_path)

        raw_results = []
        for det in detections:
            crop       = det["crop"]
            name, conf = self.classifier.classify(crop)

            threshold = MIXED_DISH_THRESHOLD if name in MIXED_DISH_LABELS else VIT_CONFIDENCE_THRESHOLD
            if conf < threshold:
                print(f"[Analyzer] Skipping '{name}' (conf={conf:.2f} < {threshold})")
                continue

            if _looks_like_background(crop):
                continue

            grams    = self.depth_estimator.estimate_grams(det["bbox"], depth_map, label=name)
            calories = _estimate_calories(name, grams)

            raw_results.append({
                "name":       name,
                "confidence": conf,
                "grams":      grams,
                "calories":   calories,
                "bbox":       det["bbox"],
            })

        results = _deduplicate(raw_results, img_w, img_h)

        label_counts = {}
        capped = []
        for r in results:
            n = r["name"]
            label_counts[n] = label_counts.get(n, 0) + 1
            limit = MAX_PER_LABEL_EXCEPTIONS.get(n, MAX_PER_LABEL)
            if label_counts[n] <= limit:
                capped.append(r)
            else:
                print(f"[Analyzer] Dropping extra '{n}'")
        results = capped

        for r in results:
            r.pop("bbox", None)

        return results

    def analyze_image(self, image_path: str) -> list[dict]:
        return self.analyze(image_path)