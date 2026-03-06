YOLOv8n (COCO) + Vision Transformer (ViT)

This project is an early‑stage food recognition system designed to detect food items in images and classify them into specific food categories. The current version uses YOLOv8n.pt (trained on COCO) for object detection and a Vision Transformer (ViT) for classification.

Because COCO is not a food‑specific dataset, detection accuracy is limited — but the pipeline is fully functional and ready for future upgrades.

📌 Current Status (Realistic)
✔️ Implemented

    YOLOv8n detector (COCO)

    ViT classifier for food categories

    Image cropping based on YOLO detections

    Basic inference pipeline

    Grams estimation (experimental)

    Calorie estimation (placeholder)

❗ Limitations (Current Version)

    YOLOv8n detects forks, tables, pizza, bowls, etc.

    Bounding boxes are often inaccurate

    Some foods are misclassified due to poor crops

    No food‑specific detection model yet

    No custom training yet

🎯 Next Major Upgrade (Planned)

    Train YOLOv8‑L on UEC‑Food256 for 256 food classes

    Replace COCO detector with a food‑specific model

    Improve grams + calorie estimation

    Project Architecture
    Input Image
     │
     ▼
YOLOv8n (COCO) Detector → Bounding Boxes (not food‑specific)
     │
     ▼
Crop Each Detected Region
     │
     ▼
ViT Classifier → Food Label
     │
     ▼
(Optional) Nutrition Lookup → Calories, Macros

Repository Structure
project/
 ├── weights/
 │    ├── yolov8n.pt              # current detector (COCO)
 │    └── vit_classifier.pt       # classifier
 ├── src/
 │    ├── detector.py             # YOLO wrapper
 │    ├── classifier.py           # ViT wrapper
 │    ├── pipeline.py             # full inference pipeline
 │    └── utils.py
 ├── data/
 │    └── labels.txt              # classifier labels
 ├── notebooks/
 │    └── future_train_uec256.ipynb  # placeholder for future training
 ├── README.md
 └── requirements.txt

 How to Run Inference
 pip install ultralytics torch torchvision pillow opencv-python numpy

 from src.pipeline import FoodPipeline

pipeline = FoodPipeline(
    detector_path="weights/yolov8n.pt",
    classifier_path="weights/vit_classifier.pt"
)

results = pipeline("example.jpg")
print(results)

Output example
[
  {
    "label": "steak",
    "confidence": 0.82,
    "grams": 210.3,
    "calories": 540
  },
  {
    "label": "asparagus",
    "confidence": 0.77,
    "grams": 45.7,
    "calories": 20
  }
]


Known Issues

    YOLOv8n detects non‑food objects (forks, tables, bowls)

    Bounding boxes may cut off food items

    Misclassifications occur due to poor crops

    Calorie estimation is experimental

These issues will be resolved once the project switches to a food‑specific YOLO model.

Roadmap
Phase 1 — Current (Done)

    Basic pipeline with YOLOv8n + ViT

    Working inference

    Basic grams estimation

Phase 2 — Next (In Progress)

    Train YOLOv8‑L on UEC‑Food256

    Replace COCO detector

    Improve bounding box quality

    Improve classifier accuracy

Phase 3 — Future

    Nutrition database integration

    Portion size estimation

    Multi‑food calorie estimation

    Mobile app version

    Why Upgrade to UEC‑Food256?

COCO has 0 real food classes except:

    pizza

    cake

    hot dog

    sandwich

    bowl

    fork

    spoon

UEC‑Food256 has 256 real food categories, including:

    steak

    asparagus

    ramen

    sushi

    pasta

    soups

    desserts

    vegetables

This upgrade will dramatically improve detection accuracy.
License

MIT License.
