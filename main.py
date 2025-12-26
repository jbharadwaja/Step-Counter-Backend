from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime
import sqlite3 

app = FastAPI()

# --- DATABASE SETUP ---
DB_FILE = "step_history.db"

def init_db():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            steps INTEGER,
            calories REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- ML MODEL SETUP ---
# We check for your specific model file name
if os.path.exists("calorie_model.pkl"):
    model = joblib.load("calorie_model.pkl")
    print("✅ Smart ML Model Loaded!")
else:
    model = None
    print("⚠️ No ML model found. Using fallback math.")

# --- INPUT DATA MODEL ---
class ActivityData(BaseModel):
    steps: int
    weight: float
    hour: int

# --- API ENDPOINTS ---

@app.post("/calculate_calories")
async def calculate_calories(data: ActivityData):
    # ---------------------------------------------------------
    # 🧠 PART 1: PREDICT CALORIES
    # ---------------------------------------------------------
    if model:
        # We need 2D array for sklearn: [[steps, weight, hour]]
        features = [[data.steps, data.weight, data.hour]]
        prediction = model.predict(features)
        result = max(0, prediction[0]) # No negative calories
        method = "Smart Random Forest"
    else:
        # Fallback: Simple multiplier
        result = data.steps * 0.04
        method = "Fallback Math"
    
    # ---------------------------------------------------------
    # 🗣️ PART 2: GENERATE THE ROAST (The New Logic)
    # ---------------------------------------------------------
    # Calculate how many steps they "should" have by this hour (assuming 6am wake up)
    # Goal: 10,000 steps over 16 hours = ~625 steps per hour
    active_hours = max(1, data.hour - 6)
    expected_steps = active_hours * 625 
    
    # Calculate performance ratio (Actual / Expected)
    ratio = data.steps / expected_steps if expected_steps > 0 else 0
    
    # Generate the message
    if data.steps == 0:
        message = "Zero steps? Is your phone in the fridge? 🧊"
    elif ratio >= 1.2:
        message = f"🚀 Crushing it! You're {int((ratio-1)*100)}% ahead of schedule."
    elif ratio >= 0.8:
        message = "Solid pace. Consistency is key. 🎯"
    elif ratio >= 0.5:
        message = "You're lagging a bit. Time for a quick walk? 🚶"
    else:
        message = "My prediction model says you are currently... sitting on the couch. 🛋️"

    # ---------------------------------------------------------
    # 💾 PART 3: SAVE TO DATABASE
    # ---------------------------------------------------------
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO logs (time, steps, calories)
        VALUES (?, ?, ?)
    ''', (current_time, data.steps, float(result)))
    
    conn.commit()
    conn.close()

    # ---------------------------------------------------------
    # 📤 PART 4: RETURN JSON
    # ---------------------------------------------------------
    return {
        "calories_burned": round(result, 2),
        "method": method,
        "message": message  # <--- Swift needs this!
    }

@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM logs ORDER BY id DESC LIMIT 20')
    rows = cursor.fetchall()
    conn.close()
    
    clean_history = []
    for row in rows:
        clean_history.append({
            "id": row["id"],
            "time": row["time"],
            "steps": row["steps"],
            "calories": round(row["calories"], 2)
        })
        
    return {"recent_scans": clean_history}

@app.get("/")
def home():
    return {"status": "Online", "framework": "FastAPI + SQLite"}