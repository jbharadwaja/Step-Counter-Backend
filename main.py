from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Load the Smart Model
if os.path.exists("calorie_model.pkl"):
    model = joblib.load("calorie_model.pkl")
    print("✅ Smart ML Model Loaded!")
else:
    model = None

# 2. Update Input to include 'hour'
class ActivityData(BaseModel):
    steps: int
    weight: float
    hour: int   # <--- New Feature!

@app.post("/calculate_calories")
async def calculate_calories(data: ActivityData):
    
    if model:
        # Scikit-learn expects [[steps, weight, hour]]
        features = [[data.steps, data.weight, data.hour]]
        
        prediction = model.predict(features)
        result = max(0, prediction[0]) # Safety clamp (no negatives)
        
        print(f"🤖 Smart AI says: {result:.2f} (Time: {data.hour}:00)")
        method = "Smart Random Forest"
        
    else:
        # Fallback if model missing
        result = data.steps * 0.04
        method = "Fallback Math"

    return {
        "calories_burned": round(result, 2),
        "method": method
    }

@app.get("/")
def home():
    return {"status": "Online", "model_loaded": model is not None}