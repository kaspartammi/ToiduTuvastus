from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

class FoodClassifier:
    def __init__(self, weights_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ViTForImageClassification.from_pretrained(
            "nateraw/food"


        ).to(self.device)

        self.processor = ViTImageProcessor.from_pretrained(
            "nateraw/food"


        )

        self.class_names = class_names

    def classify(self, img: Image.Image):
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)
        return self.class_names[idx.item()], conf.item()
