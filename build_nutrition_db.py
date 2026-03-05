import csv
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "nutrition.csv"
DB_PATH = BASE_DIR / "data" / "nutrition.db"

def build_db():
    DB_PATH.parent.mkdir(exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop old table if it exists
    cur.execute("DROP TABLE IF EXISTS foods")

    # Create table
    cur.execute("""
        CREATE TABLE foods (
            name TEXT PRIMARY KEY,
            calories_per_100g REAL
        )
    """)

    # Read CSV and insert rows
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            name = row["name"].strip().lower()
            cal = float(row["calories_per_100g"])
            rows.append((name, cal))

    cur.executemany(
        "INSERT INTO foods (name, calories_per_100g) VALUES (?, ?)",
        rows,
    )

    conn.commit()
    conn.close()
    print(f"Built {DB_PATH} with {len(rows)} foods.")

if __name__ == "__main__":
    build_db()
