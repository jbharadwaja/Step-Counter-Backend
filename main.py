from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
from datetime import datetime # <--- 1. Import datetime

app = FastAPI()

# 2. Create a simple in-memory database (List)
history_log = [] 

# Load Model
if os.path.exists("calorie_model.pkl"):
    model = joblib.load("calorie_model.pkl")
    print("✅ Smart ML Model Loaded!")
else:
    model = None

class ActivityData(BaseModel):
    steps: int
    weight: float
    hour: int

@app.post("/calculate_calories")
async def calculate_calories(data: ActivityData):
    if model:
        features = [[data.steps, data.weight, data.hour]]
        prediction = model.predict(features)
        result = max(0, prediction[0])
        method = "Smart Random Forest"
    else:
        result = data.steps * 0.04
        method = "Fallback Math"
    
    # 3. Save to History Log
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "steps": data.steps,
        "calories": round(result, 2)
    }
    history_log.append(log_entry)
    
    # Keep only last 10 entries to keep it clean
    if len(history_log) > 10:
        history_log.pop(0)

    return {
        "calories_burned": round(result, 2),
        "method": method
    }

# 4. New Endpoint: View History
@app.get("/history")
def get_history():
    return {"recent_scans": history_log} # Returns the list

@app.get("/")
def home():
    return {"status": "Online", "model_loaded": model is not None}