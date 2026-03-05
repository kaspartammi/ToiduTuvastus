# models/classifier.py
import torch
from torchvision import transforms
from PIL import Image
from typing import Tuple

class FoodClassifier:
    def __init__(self, weights_path, class_names):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(weights_path, map_location=self.device)
        self.model.eval()
        self.class_names = class_names
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def classify(self, img: Image.Image) -> Tuple[str, float]:
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return self.class_names[idx.item()], conf.item()
