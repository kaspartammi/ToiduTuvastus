# models/detector.py
#
# Drop-in replacement for the YOLO-based FoodDetector.
# Uses SAM (Segment Anything Model) to find regions, then filters them
# to plausible food segments — no food-specific training required.
#
# Install dependencies:
#   pip install segment-anything opencv-python numpy pillow
#   # Download SAM checkpoint (once):
#   wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
#
# SAM checkpoint options (trade speed vs accuracy):
#   sam_vit_b_01ec64.pth  ~375 MB  fastest, good enough for most cases
#   sam_vit_l_0b3195.pth  ~1.2 GB  better quality
#   sam_vit_h_4b8939.pth  ~2.4 GB  best quality, slow on CPU

import threading
import numpy as np
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

# ── tuneable knobs ────────────────────────────────────────────────────────────
SAM_CHECKPOINT   = "sam_vit_b_01ec64.pth"   # path to downloaded weights
SAM_MODEL_TYPE   = "vit_b"                  # must match checkpoint

# Segment filtering thresholds
MIN_AREA_RATIO   = 0.01   # segment must cover ≥1 % of image (removes dust/noise)
MAX_AREA_RATIO   = 0.90   # segment must cover ≤90 % of image (removes background)
MIN_STABILITY    = 0.80   # SAM stability score filter
CROP_PADDING     = 0.08   # expand bbox by 8 % on each side (recovers cut-off edges)
MAX_SEGMENTS     = 12     # cap so ViT isn't called hundreds of times
# ─────────────────────────────────────────────────────────────────────────────


def _pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio=CROP_PADDING):
    """Expand a bounding box by pad_ratio, clamped to image bounds."""
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = max(0,     int(x1 - pw))
    y1 = max(0,     int(y1 - ph))
    x2 = min(img_w, int(x2 + pw))
    y2 = min(img_h, int(y2 + ph))
    return x1, y1, x2, y2


def _is_plausible_food_segment(mask, img_area):
    """
    Heuristic filter: keep segments that look like individual food items.
    Rejects tiny noise, full-image background, and very thin/elongated shapes
    (knives, forks, table edges).
    """
    area = int(mask["area"])
    ratio = area / img_area

    if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
        return False

    if mask.get("stability_score", 1.0) < MIN_STABILITY:
        return False

    # Aspect-ratio check: skip very elongated segments (likely cutlery/edges)
    bbox = mask["bbox"]           # [x, y, w, h]  SAM format
    w, h = bbox[2], bbox[3]
    if w == 0 or h == 0:
        return False
    aspect = max(w, h) / min(w, h)
    if aspect > 5.0:              # more than 5:1 → probably not food
        return False

    return True


class FoodDetector:
    """
    SAM-based detector.  Public interface is identical to the old YOLO version:

        detector = FoodDetector()
        detections = detector.detect("path/to/image.jpg")

    Each detection dict contains:
        "bbox"       : [x1, y1, x2, y2]  (padded, image-clamped)
        "crop"       : PIL.Image
        "confidence" : float  (SAM stability score, 0-1)
    """

    def __init__(self, checkpoint: str = SAM_CHECKPOINT,
                 model_type: str = SAM_MODEL_TYPE):

        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything is not installed.\n"
                "Run:  pip install segment-anything"
            )

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device)

        # SamAutomaticMaskGenerator finds ALL segments automatically —
        # no prompts, no food-specific training needed.
        self._mask_gen = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,          # lower = faster, less fine-grained
            pred_iou_thresh=0.86,
            stability_score_thresh=MIN_STABILITY,
            min_mask_region_area=100,    # pixel area — removes tiny artefacts
        )

        # SAM's internal predictor is stateful — concurrent Flask threads will
        # corrupt each other's set_image() state without this lock.
        self._lock = threading.Lock()

        print(f"[FoodDetector] SAM loaded on {device} ({model_type})")

    # ------------------------------------------------------------------
    def detect(self, image_path: str) -> list[dict]:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        img_area = img_np.shape[0] * img_np.shape[1]

        # Acquire lock — only one thread may run SAM generate() at a time
        with self._lock:
            masks = self._mask_gen.generate(img_np)

        # Sort largest-first so prominent food items come first
        masks.sort(key=lambda m: m["area"], reverse=True)

        detections = []
        for mask in masks:
            if not _is_plausible_food_segment(mask, img_area):
                continue

            # SAM bbox is [x, y, w, h] → convert to [x1,y1,x2,y2]
            x, y, w, h = mask["bbox"]
            x1, y1, x2, y2 = _pad_bbox(
                x, y, x + w, y + h,
                img_np.shape[1], img_np.shape[0]
            )

            crop = img.crop((x1, y1, x2, y2))
            stability = float(mask.get("stability_score", 0.9))

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "crop":       crop,
                "confidence": stability,
            })

            if len(detections) >= MAX_SEGMENTS:
                break

        print(f"[FoodDetector] {len(detections)} candidate segments found")
        return detections