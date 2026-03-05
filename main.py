# main.py
import sys
from pathlib import Path
from PIL import Image

from pipeline.analyze import Analyzer
from config import EFFNET_WEIGHTS, CLASS_NAMES

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    analyzer = Analyzer(EFFNET_WEIGHTS, CLASS_NAMES)

    img = Image.open(img_path).convert("RGB")
    result = analyzer.analyze_image(img)

    print("\n=== ANALYSIS RESULT ===")
    for item in result["items"]:
        print(f"- {item['name']}")
        print(f"  grams: {item['grams']:.1f}")
        print(f"  calories: {item['calories']:.1f}")
        print(f"  confidence: {item['cls_conf']:.2f}")
        print()

    print(f"TOTAL CALORIES: {result['total_calories']:.1f}")

if __name__ == "__main__":
    main()
