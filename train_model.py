import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

print("🤖 Generatring optimized training data...")

np.random.seed(42)
n_samples = 10000

# 1. Generate Data
steps = np.random.randint(100, 25000, n_samples)
weight = np.random.uniform(50, 120, n_samples)
hours = np.random.randint(0, 24, n_samples)

df = pd.DataFrame({'steps': steps, 'weight': weight, 'hour': hours})

# 2. Intensity Logic
intensity_multiplier = []
for h in hours:
    if 6 <= h <= 9: intensity_multiplier.append(1.25)
    elif 17 <= h <= 20: intensity_multiplier.append(1.15)
    elif 23 <= h or h <= 5: intensity_multiplier.append(0.85)
    else: intensity_multiplier.append(1.0)
intensity_multiplier = np.array(intensity_multiplier)

# 3. Calculate Target
base_burn = (steps * 0.045) * (weight / 70) 
calories = base_burn * intensity_multiplier
noise = np.random.normal(0, 8, n_samples)
calories = np.maximum(calories + noise, 0)

# 4. Train Smaller Model
print("🧠 Training Optimized Model...")
# 🟢 CHANGE 1: n_estimators=50 (was 200), max_depth=10 (was 15)
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(df[['steps', 'weight', 'hour']], calories)

# 5. Save with Compression
print("💾 Saving with compression...")
# 🟢 CHANGE 2: compress=3 drastically reduces file size
joblib.dump(model, "calorie_model.pkl", compress=3)

# Check File Size
size_mb = os.path.getsize("calorie_model.pkl") / (1024 * 1024)
print(f"✅ Success! New model size: {size_mb:.2f} MB")