# pipeline/portion.py
import json
import numpy as np
from config import DENSITIES_JSON

with open(DENSITIES_JSON, "r", encoding="utf-8") as f:
    DENSITIES = json.load(f)  # e.g. {"rice": 0.72, "pasta": 0.60, ...}

def estimate_volume_cm3(bbox, depth_map):
    x1, y1, x2, y2 = map(int, bbox)
    area_px = (x2 - x1) * (y2 - y1)

    # Temporary heuristic: assume 1 pixel ≈ 0.002 cm²
    area_cm2 = area_px * 0.002

    # Assume average food height ≈ 3 cm
    height_cm = 3.0

    return area_cm2 * height_cm


def volume_to_grams(food_name: str, volume_cm3: float) -> float:
    density = DENSITIES.get(food_name.lower(), 1.0)
    return volume_cm3 * density
