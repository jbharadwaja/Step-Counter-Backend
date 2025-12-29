import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def predict_end_of_day_steps(current_steps, current_time, historical_data):
    """
    Production-Ready Step Prediction Engine (v2 - Dampened).
    Uses Linear Regression but blends it with historical averages to prevent wild outliers.
    """
    
    # 1. SAFETY CHECKS
    # If not enough data, use simple fallback
    if not historical_data or len(historical_data) < 5:
        return fallback_prediction(current_steps, current_time)

    # 2. PREPARE DATA
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate "Final Total" for every day in history
    daily_totals = df.groupby('date')['steps'].sum().reset_index()
    daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)
    
    # Calculate User's "Global Average" (The Anchor)
    global_avg = daily_totals['actual_final_total'].mean()
    if pd.isna(global_avg) or global_avg < 1: global_avg = 5000 # Default safety
    
    # Calculate "Cumulative Steps" up to the CURRENT HOUR for every past day
    current_hour = current_time.hour
    df_filtered = df[df['hour'] <= current_hour].copy()
    steps_at_current_hour = df_filtered.groupby('date')['steps'].sum().reset_index()
    steps_at_current_hour.rename(columns={'steps': 'steps_by_now'}, inplace=True)
    
    # Merge X (steps by now) and Y (final total)
    training_data = pd.merge(steps_at_current_hour, daily_totals, on='date')
    
    # If rarely active at this hour, training data might be empty
    if len(training_data) < 5:
        return fallback_prediction(current_steps, current_time)
        
    # 3. TRAIN MODEL
    X = training_data[['steps_by_now']]
    y = training_data['actual_final_total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. RAW PREDICTION
    raw_prediction = model.predict([[current_steps]])[0]
    
    # 5. DAMPING LOGIC (The Fix 🛠️)
    # We blend the "AI Guess" (70%) with the "Historical Average" (30%)
    # This prevents one active morning from predicting a 20k day.
    weighted_prediction = (0.7 * raw_prediction) + (0.3 * global_avg)
    
    # Safety Cap: The prediction cannot be less than what you already walked!
    final_prediction = max(int(weighted_prediction), current_steps)
    
    # 6. GENERATE INTELLIGENT CONTEXT
    deviation = final_prediction - global_avg
    
    if deviation > (global_avg * 0.2): # Trending 20% above average
        msg = f"🚀 Great pace! Trending ~{int(deviation)} steps above normal."
        score = 0.9
    elif deviation < -(global_avg * 0.2): # Trending 20% below
        msg = f"📉 A bit quiet. Trending ~{abs(int(deviation))} steps below normal."
        score = 0.4
    else:
        msg = "🎯 You are right on track with your usual habits."
        score = 0.75

    # 7. CONSISTENCY SCORE
    if not daily_totals.empty:
        std_dev = daily_totals['actual_final_total'].std()
        mean = daily_totals['actual_final_total'].mean()
        if mean > 0:
            cv = std_dev / mean
            consistency = max(0, 100 - (cv * 100))
        else:
            consistency = 0
    else:
        consistency = 0

    # 8. WEEKLY PATTERN
    daily_totals['weekday'] = daily_totals['date'].dt.dayofweek
    weekday_avgs = daily_totals.groupby('weekday')['actual_final_total'].mean()
    weekday_avgs = weekday_avgs.reindex(range(7), fill_value=0)
    weekly_pattern = weekday_avgs.tolist()

    return {
        "projected_steps": int(final_prediction),
        "completion_rate_at_this_hour": 0.0,
        "confidence_score": score,
        "trend_message": msg,
        "consistency_score": int(consistency),
        "weekly_pattern": weekly_pattern
    }

def fallback_prediction(current_steps, current_time):
    """Simple multiplier fallback if AI fails or data is missing."""
    hour = current_time.hour
    if hour == 0: hour = 1
    
    # Conservative Estimate: Assume steady walking over 16 active hours
    fraction_of_day_passed = min((hour - 6) / 16.0, 1.0) 
    if fraction_of_day_passed <= 0: fraction_of_day_passed = 0.05
    
    projected = current_steps / fraction_of_day_passed
    
    return {
        "projected_steps": int(projected),
        "completion_rate_at_this_hour": 0.0,
        "confidence_score": 0.2,
        "trend_message": "Gathering more data for AI predictions...",
        "consistency_score": 0,
        "weekly_pattern": [0,0,0,0,0,0,0]
    }