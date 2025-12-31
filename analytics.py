import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ==========================================
# 🧠 CORE PREDICTION LOGIC (Your New Code)
# ==========================================

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
    try:
        df = pd.DataFrame(historical_data)
        # Ensure date parsing handles strings
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"Data formatting error: {e}")
        return fallback_prediction(current_steps, current_time)
    
    # Group by date to get daily totals
    daily_totals = df.groupby('date')['steps'].sum().reset_index()
    daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)

    # --- 🧹 CLEANING STEP: Remove Outliers ---
    # Ignore days with < 1000 steps (Assume phone was left at home or sick day)
    valid_days = daily_totals[daily_totals['actual_final_total'] > 1000]
    
    if valid_days.empty:
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
    # Predicted Total = (Slope * CurrentSteps) + Intercept
    raw_prediction = model.predict([[current_steps]])[0]
    
    # 5. 🧠 DYNAMIC CONFIDENCE WEIGHTING
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

    # 7. CALCULATE METRICS
    consistency = calculate_consistency_from_df(valid_days)
    weekly_pattern = analyze_weekly_pattern_from_df(valid_days)

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
    
    fraction_passed = min((hour - 6) / 16.0, 1.0)
    if fraction_passed <= 0.1: 
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

# ==========================================
# 🛠️ HELPER FUNCTIONS (Used internally & by main.py)
# ==========================================

def calculate_consistency(past_steps_list: list) -> int:
    """Wrapper for main.py to call consistency logic with a simple list."""
    if not past_steps_list: return 0
    df = pd.DataFrame({'actual_final_total': past_steps_list})
    return calculate_consistency_from_df(df)

def calculate_consistency_from_df(valid_days_df):
    """Internal logic using Pandas"""
    if valid_days_df.empty: return 0
    
    std_dev = valid_days_df['actual_final_total'].std()
    mean_val = valid_days_df['actual_final_total'].mean()
    
    if mean_val > 0:
        cv = std_dev / mean_val
        # CV of 0 = 100 Score. CV of 1.0 = 0 Score.
        score = max(0, 100 - (cv * 100))
        return int(score)
    return 0

def analyze_weekly_pattern(history: list) -> list:
    """Wrapper for main.py to call weekly pattern logic."""
    if not history: return [0]*7
    try:
        df = pd.DataFrame(history)
        df['date'] = pd.to_datetime(df['date'])
        daily_totals = df.groupby('date')['steps'].sum().reset_index()
        daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)
        return analyze_weekly_pattern_from_df(daily_totals)
    except:
        return [0]*7

def analyze_weekly_pattern_from_df(valid_days_df):
    """Internal logic using Pandas"""
    if valid_days_df.empty: return [0]*7
    
    # 0=Mon, 6=Sun
    valid_days_df['weekday'] = valid_days_df['date'].dt.dayofweek
    weekday_avgs = valid_days_df.groupby('weekday')['actual_final_total'].mean()
    
    # Ensure all 7 days exist (fill missing with 0)
    weekday_avgs = weekday_avgs.reindex(range(7), fill_value=0)
    return weekday_avgs.astype(int).tolist()

def generate_trend_message(predicted: int, average: float) -> str:
    """Standalone helper for basic message generation."""
    if average == 0: return "Building history..."
    ratio = predicted / average
    if ratio >= 1.25: return "You're on fire! 🔥 Way above average."
    if ratio >= 1.1: return "Great momentum! Beating your average."
    if ratio >= 0.9: return "Solid consistency. Right on track."
    if ratio >= 0.7: return "A bit quiet today, keep moving!"
    return "Rest day? Activity is lower than usual."

def suggest_daily_goal(history, current_weekday_index):
    # Filter history for only "Mondays" (if today is Monday)
    relevant_days = [h['steps'] for h in history if h['weekday'] == current_weekday_index]
    
    if len(relevant_days) < 3:
        return 10000 # Default if not enough data
    
    # Calculate the 75th percentile (The "Push" Goal)
    smart_goal = np.percentile(relevant_days, 75)
    
    # Round to nearest 100
    return int(round(smart_goal / 100.0) * 100)

