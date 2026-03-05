# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
NUTRITION_DB = BASE_DIR / "data" / "nutrition.db"

# Models
YOLO_WEIGHTS = BASE_DIR / "weights" / "yolov8s.pt"
YOLO_WEIGHTS = BASE_DIR / "weights" / "yolov8_fooddet150.pt"
VIT_WEIGHTS = BASE_DIR / "weights" / "vit_food101.pth"



with open(BASE_DIR / "weights" / "food101_classes.txt", "r", encoding="utf-8") as f:
    CLASS_NAMES = [line.strip() for line in f if line.strip()]

MIDAS_MODEL_NAME = "DPT_Large"  # from torch.hub or timm

# Data
NUTRITION_DB = BASE_DIR / "data" / "nutrition.db"
DENSITIES_JSON = BASE_DIR / "data" / "densities.json"

# API
HOST = "0.0.0.0"
PORT = 8000
