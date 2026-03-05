import sqlite3
from pathlib import Path

DB_PATH = Path("data") / "nutrition.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name, calories_per_100g FROM foods ORDER BY name")
rows = cur.fetchall()

for name, cal in rows:
    print(f"{name}: {cal} kcal/100g")

conn.close()
