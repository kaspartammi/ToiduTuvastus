# pipeline/calories.py
import sqlite3
from typing import Optional
from config import NUTRITION_DB

class NutritionDB:
    def __init__(self):
        self.conn = sqlite3.connect(NUTRITION_DB)

    def get_cal_per_100g(self, food_name: str) -> Optional[float]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT calories_per_100g FROM foods WHERE name = ?",
            (food_name.lower(),),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def grams_to_calories(self, food_name: str, grams: float) -> Optional[float]:
        cal100 = self.get_cal_per_100g(food_name)
        if cal100 is None:
            return None
        return grams * cal100 / 100.0
