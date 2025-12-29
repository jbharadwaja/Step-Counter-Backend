import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def predict_end_of_day_steps(current_steps, current_time, historical_data):
    """
    Smarter Model v3 (Time-Weighted & Cleaned).
    - Removes 'lazy/sick' days (outliers) from training.
    - Trusts 'Average' in the morning.
    - Trusts 'AI Projection' in the evening.
    """
    
    # 1. SAFETY & DATA VALIDATION
    if not historical_data or len(historical_data) < 5:
        return fallback_prediction(current_steps, current_time)

    # 2. PREPARE & CLEAN DATA
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date to get daily totals
    daily_totals = df.groupby('date')['steps'].sum().reset_index()
    daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)

    # --- 🧹 CLEANING STEP: Remove Outliers ---
    # Ignore days with < 1000 steps (Assume phone was left at home or sick day)
    # This prevents one "0 step" day from ruining your average.
    valid_days = daily_totals[daily_totals['actual_final_total'] > 1000]
    
    if valid_days.empty:
        # If all history is bad, use raw data
        valid_days = daily_totals

    # Calculate User's "True Potential Average"
    global_avg = valid_days['actual_final_total'].mean()
    if pd.isna(global_avg) or global_avg < 1: global_avg = 5000 
    
    # 3. PREPARE REGRESSION DATA
    current_hour = current_time.hour
    
    # Filter only logs up to the current hour
    df_filtered = df[df['hour'] <= current_hour].copy()
    steps_at_current_hour = df_filtered.groupby('date')['steps'].sum().reset_index()
    steps_at_current_hour.rename(columns={'steps': 'steps_by_now'}, inplace=True)
    
    # Merge X (steps by now) and Y (final total)
    training_data = pd.merge(steps_at_current_hour, valid_days, on='date')
    
    if len(training_data) < 5:
        return fallback_prediction(current_steps, current_time)
        
    # 4. TRAIN AI MODEL
    X = training_data[['steps_by_now']]
    y = training_data['actual_final_total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Raw AI Guess (Pure Math)
    raw_prediction = model.predict([[current_steps]])[0]
    
    # 5. 🧠 DYNAMIC CONFIDENCE WEIGHTING (The Fix)
    # We calculate how much we trust the AI vs the Average based on time of day.
    
    # Assume "Active Day" is 6 AM to 10 PM (16 hours)
    # At 6 AM: 0% progress -> Trust Average 100%
    # At 2 PM: 50% progress -> Trust AI 50%, Average 50%
    # At 10 PM: 100% progress -> Trust AI 100%
    
    active_hour_index = max(0, current_hour - 6) # Normalize 6am to 0
    day_progress = min(active_hour_index / 16.0, 1.0) # 0.0 to 1.0
    
    # We always keep at least 20% weight on history to prevent explosion
    ai_weight = min(day_progress, 0.8) 
    history_weight = 1.0 - ai_weight
    
    # Calculate Final Blend
    final_prediction = (raw_prediction * ai_weight) + (global_avg * history_weight)
    
    # Logic Check: Prediction can never be LESS than what you currently have
    final_prediction = max(int(final_prediction), current_steps)
    
    # 6. GENERATE CONTEXT MESSAGE
    deviation = final_prediction - global_avg
    
    # We require a significant deviation (25%) to trigger a "Hype" message now
    threshold = global_avg * 0.25 
    
    if deviation > threshold:
        msg = f"🚀 Great pace! Trending ~{int(deviation)} steps above normal."
        score = 0.9
    elif deviation < -threshold:
        msg = f"📉 Taking it easy? Trending ~{abs(int(deviation))} below avg."
        score = 0.4
    else:
        msg = "🎯 You are right on track with your usual habits."
        score = 0.75

    # 7. CALCULATE CONSISTENCY (CV)
    if not valid_days.empty:
        std_dev = valid_days['actual_final_total'].std()
        mean_val = valid_days['actual_final_total'].mean()
        if mean_val > 0:
            cv = std_dev / mean_val
            consistency = max(0, 100 - (cv * 100))
        else:
            consistency = 0
    else:
        consistency = 0

    # 8. WEEKLY PATTERN
    # Ensure we look at the full valid dataset
    valid_days['weekday'] = valid_days['date'].dt.dayofweek
    weekday_avgs = valid_days.groupby('weekday')['actual_final_total'].mean()
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
    """Conservative fallback using remaining hours."""
    hour = current_time.hour
    if hour == 0: hour = 1
    
    # Assume 10k goal roughly
    fraction_passed = min((hour - 6) / 16.0, 1.0)
    if fraction_passed <= 0.1: 
        # Very early morning? Just return a safe generic number or current * small multiplier
        projected = max(current_steps * 2, 4000) 
    else:
        projected = current_steps / fraction_passed
    
    return {
        "projected_steps": int(projected),
        "completion_rate_at_this_hour": 0.0,
        "confidence_score": 0.3,
        "trend_message": "Gathering data...",
        "consistency_score": 0,
        "weekly_pattern": [0,0,0,0,0,0,0]
    }