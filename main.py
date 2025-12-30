from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime
import sqlite3 
from typing import List

# 🟢 IMPORT YOUR NEW ANALYTICS MODULE
# (Ensure analytics.py is in the same folder)
import analytics 

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
if os.path.exists("calorie_model.pkl"):
    model = joblib.load("calorie_model.pkl")
    print("✅ Smart ML Model Loaded!")
else:
    model = None
    print("⚠️ No ML model found. Using fallback math.")

# --- INPUT DATA MODELS ---

# 1. For Calorie Calculation (Existing)
class ActivityData(BaseModel):
    steps: int
    weight: float
    hour: int

# 2. For Step Prediction (New Feature)
class HistoryPoint(BaseModel):
    date: str  # Format: "2023-10-27"
    hour: int  # 0-23
    steps: int

class PredictionRequest(BaseModel):
    current_steps: int
    history: List[HistoryPoint]

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
    # 🗣️ PART 2: GENERATE THE ROAST
    # ---------------------------------------------------------
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
        "message": message 
    }

@app.post("/predict/steps")
async def predict_steps(request: Request):
    """
    Intelligent Endpoint that combines Historical Average + Current Pace
    to predict End-of-Day total steps.
    """
    data = await request.json()
    current_steps = data.get("current_steps", 0)
    history = data.get("history", []) # List of {date, hour, steps}
    
    # --- 1. PREPARE DATA ---
    now = datetime.now()
    current_hour = now.hour
    
    # Safety: If it's very early (e.g. 5 AM), pace data is noisy. Return trend or goal.
    if current_hour < 6:
        return {
            "projected_steps": max(current_steps, 5000), # Fallback for sleeping hours
            "trend_message": "Good morning! Early start.",
            "consistency_score": 0,
            "confidence_score": 0.0,
            "weekly_pattern": analytics.analyze_weekly_pattern(history)
        }

    # --- 2. LEARN USER'S DAILY RHYTHM (The "Real" Learning) ---
    # We want to find out: "On average, what % of steps does THIS user have by [current_hour]?"
    
    # A. Calculate Average Daily Steps from History (The "Base" Expectation)
    past_steps_values = [h['steps'] for h in history]
    # Filter out zeros or very low numbers to avoid skewing average
    valid_past_steps = [s for s in past_steps_values if s > 1000]
    avg_steps = np.mean(valid_past_steps) if valid_past_steps else 5000
    
    # B. Calculate "Current Pace Ratio"
    # This curve mimics a typical active day (active 7am-10pm).
    # If it is 2 PM (14:00), a standard user has completed about 50-60% of their day.
    if 7 <= current_hour <= 22:
        # Simple linear approximation of an active day (15 active hours)
        # Formula: (Hours passed since 7am) / 15 total active hours
        standard_completion_ratio = (current_hour - 7) / 15.0 
        
        # Clamp between 10% and 100% to avoid division by zero errors
        standard_completion_ratio = max(0.1, min(standard_completion_ratio, 1.0))
    else:
        standard_completion_ratio = 1.0 # Late night
        
    # C. The "Smart" Projection
    # Project based on pace: If I have 4000 steps at 50% of the day, I will end with 8000.
    pace_projected = int(current_steps / standard_completion_ratio)
    
    # --- 3. WEIGHTED PREDICTION (The "AI" Part) ---
    # We trust the Pace Projection MORE as the day goes on.
    # At 9 AM: Trust the History Average (80%) + Pace (20%) because the day is young.
    # At 9 PM: Trust the Pace (90%) + History Average (10%) because the day is done.
    
    confidence_in_pace = min((current_hour - 6) / 14.0, 0.9) # Increases as day passes
    confidence_in_pace = max(0.0, confidence_in_pace)
    
    # Blend the two numbers
    final_prediction = (pace_projected * confidence_in_pace) + (avg_steps * (1 - confidence_in_pace))
    
    # Hard floor: Can't be less than what we already have
    final_prediction = max(int(final_prediction), current_steps)

    # --- 4. GENERATE INSIGHTS ---
    trend_msg = analytics.generate_trend_message(final_prediction, avg_steps)
    consistency = analytics.calculate_consistency(valid_past_steps)
    weekly_chart = analytics.analyze_weekly_pattern(history)

    return {
        "projected_steps": final_prediction,
        "trend_message": trend_msg,
        "consistency_score": consistency,
        "confidence_score": round(confidence_in_pace, 2), # Send 0.0 - 1.0 for UI bar
        "weekly_pattern": weekly_chart
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