# pipeline/portion.py
import json
import numpy as np
from config import DENSITIES_JSON

with open(DENSITIES_JSON, "r", encoding="utf-8") as f:
    DENSITIES = json.load(f)  # e.g. {"rice": 0.72, "pasta": 0.60, ...}

def estimate_volume_cm3(bbox, depth_map: np.ndarray) -> float:
    x1, y1, x2, y2 = map(int, bbox)
    crop = depth_map[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    mean_depth = float(crop.mean())
    area_px = crop.size
    # crude heuristic: volume ~ area * depth
    volume = area_px * mean_depth
    return volume  # arbitrary units, you’ll calibrate later

def volume_to_grams(food_name: str, volume_cm3: float) -> float:
    density = DENSITIES.get(food_name.lower(), 1.0)
    return volume_cm3 * density
