# ToiduTuvastus — Food Recognition & Calorie Estimator

A computer vision pipeline that detects food items in photos, classifies them, and estimates calories. Built as a learning project with a fully local inference stack — no cloud APIs required.

---

## How It Works

```
Input Image
    │
    ▼
SAM (Segment Anything Model)
    │  Finds all distinct regions in the image
    │  Filters out background, plates, cutlery
    ▼
Food Classifier (prithivMLmods/Food-101-93M)
    │  Classifies each crop into one of 101 food categories
    │  Low-confidence detections are skipped
    ▼
MiDaS Depth Estimator
    │  Estimates relative portion size using depth map
    │  Scaled against standard portion sizes per food type
    ▼
Nutrition Lookup
    │  Maps food label → kcal/100g
    ▼
Output: food name, estimated grams, estimated calories
```

---

## Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Detection | [SAM vit_b](https://github.com/facebookresearch/segment-anything) | Segment food regions — no food-specific training needed |
| Classification | [prithivMLmods/Food-101-93M](https://huggingface.co/prithivMLmods/Food-101-93M) | 101-class food classifier (SigLIP2 backbone) |
| Depth | [MiDaS DPT_Large](https://github.com/isl-org/MiDaS) | Relative portion size estimation |

---

## Project Structure

```
ToiduTuvastus/
├── models/
│   ├── detector.py        # SAM-based food region segmentation
│   ├── classifier.py      # Food-101 ViT classifier wrapper
│   ├── depth.py           # MiDaS depth estimator + grams estimation
│   └── nutrition.py       # Calorie & standard portion database (101 foods)
├── pipeline/
│   └── analyze.py         # Full inference pipeline with dedup & filtering
├── data/
│   └── labels.txt
├── weights/
│   └── sam_vit_b_01ec64.pth   # SAM weights (download separately)
├── main.py
├── config.py
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install ultralytics torch torchvision pillow opencv-python numpy
pip install transformers segment-anything
```

### 2. Download SAM weights

```bash
# ~375 MB — place in project root or update SAM_CHECKPOINT in models/detector.py
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Classifier and depth model weights are downloaded automatically from HuggingFace/PyTorch Hub on first run.

### 3. Run

```bash
python main.py path/to/food_image.jpg
```

---

## Example Output

```
=== ANALYSIS RESULT ===
- omelette
  grams:      200.0g  (estimated)
  calories:   310.0 kcal
  confidence: 0.98

- steak
  grams:      220.0g  (estimated)
  calories:   550.0 kcal
  confidence: 0.91

TOTAL CALORIES: 860.0 kcal  (estimated)
```

---

## Accuracy & Limitations

### What works well
- Common restaurant-style dishes (omelette, steak, pizza, sushi, ramen, fried chicken)
- Multi-food images — SAM segments individual items cleanly
- Background/plate suppression — white plates, dark countertops, cutlery are filtered out
- Duplicate detection suppression via IoU + center-distance deduplication

### Known limitations

| Issue | Cause | Status |
|-------|-------|--------|
| Home-cooked foods misclassified | Food-101 is restaurant/US-cuisine biased — no mashed potato, plov, porridge etc. | Known limitation |
| Portion estimates are rough | Single-image depth estimation without a reference object | Estimated ±40% accuracy |
| Mixed dishes lose secondary ingredients | SAM can't isolate meat chunks inside rice/soup | Partially mitigated via lower confidence threshold for mixed-dish labels |
| Drinks classified as food | Tea/coffee sometimes detected as miso soup or broth | Low impact on calorie total |

### Why SAM instead of YOLO?

The original version used YOLOv8n trained on COCO, which detected non-food objects (forks, tables, bowls) and had inaccurate bounding boxes for food items. SAM replaced it because it requires no food-specific training — it finds all distinct regions in any image, and the food classifier handles the food/non-food distinction downstream.

---

## Roadmap

- [ ] Fine-tune classifier on a broader food dataset (UEC-Food256 or OpenFoodFacts images)
- [ ] Improve portion estimation with reference object detection (coin, hand, plate diameter)
- [ ] Add food diary / history tracking
- [ ] Mobile-friendly interface

---

## License

MIT License.