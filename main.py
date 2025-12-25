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
    # SQL Command: Create a table with columns for Time, Steps, and Calories
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            steps INTEGER,
            calories REAL
        )
    ''')
    conn.commit() # Save changes
    conn.close()

# Initialize DB immediately when server starts
init_db()

# --- ML MODEL SETUP ---
if os.path.exists("calorie_model.pkl"):
    model = joblib.load("calorie_model.pkl")
    print("✅ Smart ML Model Loaded!")
else:
    model = None

# --- INPUT DATA MODEL ---
class ActivityData(BaseModel):
    steps: int
    weight: float
    hour: int

# --- API ENDPOINTS ---

@app.post("/calculate_calories")
async def calculate_calories(data: ActivityData):
    # 1. Run the Prediction
    if model:
        features = [[data.steps, data.weight, data.hour]]
        prediction = model.predict(features)
        result = max(0, prediction[0])
        method = "Smart Random Forest"
    else:
        result = data.steps * 0.04
        method = "Fallback Math"
    
    # 2. SQL INSERT: Save to Database
    # We open a connection, write the data, and close it.
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO logs (time, steps, calories)
        VALUES (?, ?, ?)
    ''', (current_time, data.steps, float(result)))
    
    conn.commit() # IMPORTANT: Save the transaction
    conn.close()

    return {
        "calories_burned": round(result, 2),
        "method": method
    }

@app.get("/history")
def get_history():
    # 3. SQL SELECT: Read from Database
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row # This lets us access columns by name
    cursor = conn.cursor()
    
    # Get the last 20 entries, newest first
    cursor.execute('SELECT * FROM logs ORDER BY id DESC LIMIT 20')
    rows = cursor.fetchall()
    conn.close()
    
    # Convert SQL rows back to a Python list
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
    return {"status": "Online", "database": "SQLite"}