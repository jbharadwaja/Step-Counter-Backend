import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🤖 Generatring smart training data...")

np.random.seed(42)
n_samples = 5000 

# 1. Generate Basic Features
steps = np.random.randint(100, 20000, n_samples)
weight = np.random.uniform(50, 120, n_samples)
hours = np.random.randint(0, 24, n_samples) # New Feature: 0 to 23 hours

# 2. Define "Intensity" based on Time
# If hour is between 6am and 10am, we assume high intensity (1.2x burn)
# If hour is between 10pm and 5am, we assume low intensity (0.8x burn)
# Otherwise normal (1.0)
intensity_multiplier = []

for h in hours:
    if 6 <= h <= 10:
        intensity_multiplier.append(1.2) # Morning Workout
    elif 22 <= h or h <= 5:
        intensity_multiplier.append(0.8) # Late night shuffle
    else:
        intensity_multiplier.append(1.0) # Normal walking

intensity_multiplier = np.array(intensity_multiplier)

# 3. Calculate Calories with the new "Smart" formula
# Base Formula: Steps * 0.04 * WeightFactor
base_burn = (steps * 0.04) * (weight / 70)

# Apply the Time Intelligence
calories = base_burn * intensity_multiplier

# Add some noise (randomness) so the model has to learn the pattern
noise = np.random.normal(0, 5, n_samples)
calories = calories + noise
calories = np.maximum(calories, 0) # No negatives

# 4. Train the Model
df = pd.DataFrame({
    'steps': steps,
    'weight': weight,
    'hour': hours,       # <--- We feed this to the model now
    'calories': calories
})

X = df[['steps', 'weight', 'hour']]
y = df['calories']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

print("✅ Smart Model Trained!")

# Test: Compare 5000 steps at 8 AM vs 8 PM
morning_pred = model.predict([[5000, 70, 8]])[0]  # 8 AM
night_pred =   model.predict([[5000, 70, 20]])[0] # 8 PM (20:00)

print(f"☀️ Morning Walk (5000 steps): {morning_pred:.2f} kcal")
print(f"🌙 Night Walk   (5000 steps): {night_pred:.2f} kcal")

joblib.dump(model, "calorie_model.pkl")