# models/classifier.py
#
# Swapped from Kaludi/food-category-classification-v2.0 (12 categories)
#         to   prithivMLmods/Food-101-93M (101 specific dishes, SigLIP2 backbone)
#
# Interface identical: classify(img) → (label, confidence)

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image

MODEL_ID = "prithivMLmods/Food-101-93M"


class FoodClassifier:
    def __init__(self, weights_path=None, class_names=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[FoodClassifier] Loading {MODEL_ID} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model     = AutoModelForImageClassification.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        print(f"[FoodClassifier] Ready — {len(self.id2label)} classes")

    def classify(self, img: Image.Image) -> tuple[str, float]:
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs     = self.model(**inputs).logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)
        return self.id2label[idx.item()], conf.item()