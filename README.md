# ToiduTuvastus — Server

Flask-based backend for the ToiduTuvastus food recognition app. Receives images from the Android app, runs them through a SAM + ViT pipeline to detect and classify food, estimates calories, and stores results in a SQLite database.

---

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- ~2 GB disk space (models)
- A SAM checkpoint file (see below)

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone https://github.com/kaspartammi/ToiduTuvastus
cd ToiduTuvastus
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install flask torch torchvision pillow numpy transformers segment-anything difflib
```

### 3. Download the SAM checkpoint

```bash
# ~375 MB, place it in the project root
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The checkpoint path is set in `models/detector.py`:
```python
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
```

### 4. Start the server

```bash
python api.py
```

Server starts on `http://0.0.0.0:5000`. First startup takes ~30 seconds while models load.

---

## Exposing to the internet (ngrok)

The Android app connects over the internet. Use ngrok to create a public tunnel:

```bash
# Install ngrok, then:
ngrok http 5000
```

Copy the `https://....ngrok-free.app` URL and paste it into `ApiClient.kt` in the Android project:

```kotlin
var BASE_URL = "https://YOUR-NGROK-URL.ngrok-free.app"
```

> **Note:** The free ngrok tier changes URL every restart. Update `ApiClient.kt` and rebuild the app each time. All requests include the `ngrok-skip-browser-warning: true` header to bypass ngrok's browser warning page.

---

## Project structure

```
ToiduTuvastus/
├── api.py                  # Flask app, all endpoints
├── toidutuvastus.db        # SQLite database (auto-created)
├── sam_vit_b_01ec64.pth    # SAM weights (download manually)
├── models/
│   ├── detector.py         # SAM-based food region detector
│   ├── classifier.py       # ViT food classifier (Food-101-93M)
│   ├── depth.py            # MiDaS depth estimator (portion size)
│   └── nutrition.py        # Calorie database + Estonian aliases
└── pipeline/
    └── analyze.py          # Full analysis pipeline (detector → classifier → depth)
```

---

## API endpoints

### `POST /register`
Register a new user.

**Body (JSON):**
```json
{ "username": "Mikk" }
```

**Response:**
```json
{ "status": "ok" }
```

---

### `POST /analyze`
Analyze a food image. Accepts multipart form data.

**Form fields:**
| Field | Required | Description |
|---|---|---|
| `image` | Yes | JPEG image file |
| `username` | Yes | Registered username |
| `food_hint` | No | Text description of food (e.g. `"kartulipuder ja praetud kana"`) — skips the classifier entirely |
| `original_timestamp` | No | ISO-8601 timestamp from when the photo was taken (used by offline queue) |

If `food_hint` is provided, the SAM + ViT pipeline is skipped and the hint is parsed directly — response time drops from ~30s to ~1s.

**Response:**
```json
{
  "meal_id": 42,
  "items": [
    { "name": "mashed_potato", "grams": 200, "calories": 175.0, "confidence": 1.0 }
  ],
  "total_calories": 175.0
}
```

---

### `POST /correct`
Submit a correction for a previously saved meal.

**Body (JSON):**
```json
{ "meal_id": 42, "correction": "kartulipuder ja praetud kana" }
```

Accepts Estonian or English food names. Returns the corrected result in the same format as `/analyze`.

---

### `GET /history/<username>`
Get meal history for a user.

**Response:**
```json
{
  "meals": [
    {
      "id": 42,
      "timestamp": "2026-03-06T14:01:00",
      "total_calories": 610.0,
      "corrected": 0,
      "items": [...]
    }
  ]
}
```

---

## Analysis pipeline

```
Image
  └─► SAM (Segment Anything)     — finds food regions automatically
        └─► ViT classifier        — identifies each region (Food-101, 101 classes)
              └─► MiDaS depth     — estimates portion size from depth map
                    └─► nutrition.py  — looks up kcal/100g, calculates calories
```

If the image analysis fails or confidence is low, the user can type a correction in the app which re-runs through the nutrition lookup only.

---

## Food hint / correction parsing

The `_parse_correction()` function in `api.py` handles both the `food_hint` field and the `/correct` endpoint. It splits input on `,`, `and`, `ja`, `&`, `+`, `with` and for each token:

1. Checks Estonian alias table (`ET_ALIASES` in `nutrition.py`) — 247 entries
2. Exact English key match
3. Fuzzy English match (cutoff 0.6)
4. Fuzzy Estonian match (cutoff 0.65)
5. Falls back to 200 kcal / 200g default if nothing matches

---

## Nutrition database

`models/nutrition.py` contains:
- 101 original Food-101 classes
- 100+ Eastern European and everyday Estonian foods (soups, potatoes, meat, grains, dairy, fruit, drinks)
- 247 Estonian aliases covering all entries

---

## Database schema

```sql
CREATE TABLE users (
    id       INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL
);

CREATE TABLE meals (
    id             INTEGER PRIMARY KEY,
    username       TEXT,
    timestamp      TEXT,
    total_calories REAL,
    items_json     TEXT,
    corrected      INTEGER DEFAULT 0,
    correction_raw TEXT
);
```

---

## Known limitations

- SAM is stateful — concurrent requests are serialized with a `threading.Lock` in `detector.py`. Requests queue rather than run in parallel.
- The ViT classifier is trained on Food-101 which is biased toward restaurant / American food. Estonian home cooking is better handled via the food hint or correction flow.
- Running on CPU is slow (~25–35s per image). GPU reduces this to ~3–5s.
- ngrok free tier URL changes on every restart — the Android app must be rebuilt each time unless you set a static domain.
