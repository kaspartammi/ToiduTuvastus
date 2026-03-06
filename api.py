# api.py
from flask import Flask, request, jsonify
from pathlib import Path
import tempfile, os, traceback, sqlite3, json
from datetime import datetime

from pipeline.analyze import Analyzer
from models.nutrition import get_calories_per_100g, get_standard_portion, FOOD_DATA
from config import VIT_WEIGHTS, CLASS_NAMES

app = Flask(__name__)
DB_PATH = "toidutuvastus.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT UNIQUE NOT NULL,
            created   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meals (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            username       TEXT NOT NULL,
            timestamp      TEXT NOT NULL,
            total_calories REAL,
            items_json     TEXT,
            corrected      INTEGER DEFAULT 0,
            correction_raw TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("[DB] Database ready")


print("[API] Loading models...")
analyzer = Analyzer(VIT_WEIGHTS, CLASS_NAMES)
init_db()
print("[API] Models ready. Server starting...")


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_correction(text: str) -> list[dict]:
    """
    Turns free text into food items with grams and calories.
    Supports Estonian (via ET_ALIASES) and English food names.
    Separators: comma, and, ja, &, +, with
    """
    import re
    from difflib import get_close_matches
    from models.nutrition import DEFAULT_CAL, DEFAULT_PORTION, ET_ALIASES

    # Split on separators — supports Estonian "ja" and English "and"
    parts = re.split(r'\s*(?:,|and|ja|&|\+|with)\s*', text.lower().strip())
    parts = [p.strip() for p in parts if p.strip()]

    all_keys  = list(FOOD_DATA.keys())
    et_keys   = list(ET_ALIASES.keys())
    results   = []

    for part in parts:
        key = None

        # 1. Exact Estonian alias match
        if part in ET_ALIASES:
            key = ET_ALIASES[part]

        # 2. Exact English key match (underscored)
        if not key:
            normalised = part.replace(" ", "_").replace("-", "_")
            if normalised in FOOD_DATA:
                key = normalised

        # 3. Fuzzy English match
        if not key:
            normalised = part.replace(" ", "_").replace("-", "_")
            matches = get_close_matches(normalised, all_keys, n=1, cutoff=0.6)
            if matches:
                key = matches[0]

        # 4. Fuzzy Estonian alias match
        if not key:
            et_matches = get_close_matches(part, et_keys, n=1, cutoff=0.65)
            if et_matches:
                key = ET_ALIASES[et_matches[0]]

        if key and key in FOOD_DATA:
            cal_per_100g = FOOD_DATA[key]["cal"]
            portion      = FOOD_DATA[key]["portion"]
            label        = key.replace("_", " ")
        else:
            cal_per_100g = DEFAULT_CAL
            portion      = DEFAULT_PORTION
            label        = part

        calories = round((portion / 100) * cal_per_100g, 1)
        results.append({
            "name":       label,
            "grams":      float(portion),
            "calories":   calories,
            "confidence": 1.0,
        })

    return results


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or "username" not in data:
        return jsonify({"error": "username required"}), 400
    username = data["username"].strip()
    if not username:
        return jsonify({"error": "username cannot be empty"}), 400
    conn = get_db()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users (username, created) VALUES (?, ?)",
            (username, datetime.now().isoformat())
        )
        conn.commit()
        return jsonify({"status": "ok", "username": username})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image file in request"}), 400

    username           = request.form.get("username", "unknown").strip()
    food_hint          = request.form.get("food_hint", "").strip()
    original_timestamp = request.form.get("original_timestamp", "").strip()
    file      = request.files["image"]
    suffix    = Path(file.filename).suffix or ".jpg"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        if food_hint:
            # User told us what the food is — skip the classifier entirely
            print(f"[API] Food hint provided: '{food_hint}' — skipping classifier")
            results = _parse_correction(food_hint)
        else:
            results = analyzer.analyze_image(tmp_path)
        total_calories = sum(r["calories"] for r in results if r.get("calories"))

        timestamp = original_timestamp if original_timestamp else datetime.now().isoformat()
        conn = get_db()
        cur  = conn.execute(
            "INSERT INTO meals (username, timestamp, total_calories, items_json) VALUES (?, ?, ?, ?)",
            (username, timestamp, round(total_calories, 1), json.dumps(results))
        )
        meal_id = cur.lastrowid
        conn.commit()
        conn.close()
        print(f"[DB] Saved meal {meal_id} for '{username}': {total_calories:.1f} kcal")

        return jsonify({
            "meal_id":        meal_id,
            "items":          results,
            "total_calories": round(total_calories, 1)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)


@app.route("/correct", methods=["POST"])
def correct():
    """
    User submits a free-text correction for a meal.
    Re-estimates grams+calories from the corrected label and updates the DB.
    """
    data = request.get_json()
    if not data or "meal_id" not in data or "correction" not in data:
        return jsonify({"error": "meal_id and correction required"}), 400

    meal_id    = data["meal_id"]
    correction = data["correction"].strip()

    if not correction:
        return jsonify({"error": "correction text cannot be empty"}), 400

    items          = _parse_correction(correction)
    total_calories = sum(i["calories"] for i in items if i.get("calories"))

    conn = get_db()
    conn.execute(
        """UPDATE meals
           SET corrected=1, correction_raw=?, items_json=?, total_calories=?
           WHERE id=?""",
        (correction, json.dumps(items), round(total_calories, 1), meal_id)
    )
    conn.commit()
    conn.close()
    print(f"[DB] Corrected meal {meal_id}: '{correction}' → {total_calories:.1f} kcal")

    return jsonify({
        "meal_id":        meal_id,
        "items":          items,
        "total_calories": round(total_calories, 1),
        "corrected":      True
    })


@app.route("/history/<username>", methods=["GET"])
def history(username):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, timestamp, total_calories, items_json, corrected FROM meals WHERE username=? ORDER BY timestamp DESC LIMIT 50",
        (username,)
    ).fetchall()
    conn.close()
    meals = [{
        "meal_id":        row["id"],
        "timestamp":      row["timestamp"],
        "total_calories": row["total_calories"],
        "corrected":      bool(row["corrected"]),
        "items":          json.loads(row["items_json"])
    } for row in rows]
    return jsonify({"username": username, "meals": meals})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)