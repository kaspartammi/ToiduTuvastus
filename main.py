# main.py
import sys
from pathlib import Path
from pipeline.analyze import Analyzer
from config import VIT_WEIGHTS, CLASS_NAMES

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        sys.exit(1)

    analyzer = Analyzer(VIT_WEIGHTS, CLASS_NAMES)
    result   = analyzer.analyze_image(str(img_path))

    print("\n=== ANALYSIS RESULT ===")
    if not result:
        print("No food items detected with sufficient confidence.")
        return

    total_calories = 0.0
    for item in result:
        name     = item["name"]
        grams    = item["grams"]
        conf     = item["confidence"]
        calories = item.get("calories")

        print(f"- {name}")
        print(f"  grams:      {grams:.1f}g  (estimated)")
        print(f"  calories:   {f'{calories:.1f} kcal' if calories is not None else 'unknown'}")
        print(f"  confidence: {conf:.2f}")
        print()

        if calories is not None:
            total_calories += calories

    print(f"TOTAL CALORIES: {total_calories:.1f} kcal  (estimated)")

if __name__ == "__main__":
    main()