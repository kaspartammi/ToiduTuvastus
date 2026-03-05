# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Models
YOLO_WEIGHTS = BASE_DIR / "weights" / "yolov8_food.pt"
EFFNET_WEIGHTS = BASE_DIR / "weights" / "efficientnet_food101.pth"
MIDAS_MODEL_NAME = "DPT_Large"  # from torch.hub or timm

# Data
NUTRITION_DB = BASE_DIR / "data" / "nutrition.db"
DENSITIES_JSON = BASE_DIR / "data" / "densities.json"

# API
HOST = "0.0.0.0"
PORT = 8000
