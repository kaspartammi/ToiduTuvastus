import sys
from pathlib import Path
from PIL import Image

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

    # FIX: args.image ei eksisteeri → kasutame img_path
    result = analyzer.analyze_image(str(img_path))

    print("\n=== ANALYSIS RESULT ===")

    # FIX: Analyzer tagastab listi, mitte dicti
    total_calories = 0.0

    for item in result:
        name = item["name"]
        grams = item["grams"]
        conf = item["confidence"]

        # kalorid puuduvad sinu pipeline'is → paneme None
        calories = None

        print(f"- {name}")
        print(f"  grams: {grams:.1f}")
        if calories is None:
            print("  calories: unknown")
        else:
            print(f"  calories: {calories:.1f}")
        print(f"  confidence: {conf:.2f}")
        print()

        if calories is not None:
            total_calories += calories

    print(f"TOTAL CALORIES: {total_calories:.1f}")


if __name__ == "__main__":
    main()
