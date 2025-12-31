import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

# ==========================================
# 🧠 CORE PREDICTION LOGIC
# ==========================================

def predict_end_of_day_steps(current_steps, current_time, historical_data):
    """
    Smarter Model v4 (Robust Fallback).
    """
    
    # 1. SAFETY: If no history, assume Day 1 behavior
    if not historical_data or len(historical_data) < 1:
        return fallback_prediction(current_steps, current_time, [])

    # 2. PREPARE DATA
    try:
        df = pd.DataFrame(historical_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Daily Totals
        daily_totals = df.groupby('date')['steps'].sum().reset_index()
        daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)
        
        # Valid Days (Filter out tiny data, e.g. < 100 steps)
        valid_days = daily_totals[daily_totals['actual_final_total'] > 100]
        if valid_days.empty: valid_days = daily_totals

        # Global Average
        global_avg = valid_days['actual_final_total'].mean()
        if pd.isna(global_avg) or global_avg < 1: global_avg = 5000 
        
        # 3. REGRESSION DATA
        current_hour = current_time.hour
        df_filtered = df[df['hour'] <= current_hour].copy()
        steps_at_current_hour = df_filtered.groupby('date')['steps'].sum().reset_index()
        steps_at_current_hour.rename(columns={'steps': 'steps_by_now'}, inplace=True)
        
        training_data = pd.merge(steps_at_current_hour, valid_days, on='date')
        
        # 🟢 CHECK: If regression data is insufficient, use SMART Fallback
        if len(training_data) < 2:
            return fallback_prediction(current_steps, current_time, daily_totals)
            
        # 4. TRAIN AI
        X = training_data[['steps_by_now']]
        y = training_data['actual_final_total']
        model = LinearRegression()
        model.fit(X, y)
        raw_prediction = model.predict([[current_steps]])[0]
        
        # 5. WEIGHTING
        active_hour_index = max(0, current_hour - 6)
        day_progress = min(active_hour_index / 16.0, 1.0)
        ai_weight = min(day_progress, 0.8) 
        final_prediction = (raw_prediction * ai_weight) + (global_avg * (1.0 - ai_weight))
        final_prediction = max(int(final_prediction), current_steps)
        
        # 6. SCORE CALCULATION
        deviation = final_prediction - global_avg
        if deviation > (global_avg * 0.15):
            msg = f"🚀 Crushing it! Trending ~{int(deviation)} above average."
            score = 0.95
        elif deviation < -(global_avg * 0.15):
            msg = "📉 A bit quiet today? Trending below average."
            score = 0.6
        else:
            msg = "🎯 Right on track with your usual habits."
            score = 0.85

        return {
            "projected_steps": int(final_prediction),
            "trend_message": msg,
            "confidence_score": score,
            "consistency_score": calculate_consistency_from_df(valid_days),
            "weekly_pattern": analyze_weekly_pattern_from_df(valid_days),
            "walker_type": determine_walker_type(df)
        }

    except Exception as e:
        print(f"Prediction Error: {e}")
        return fallback_prediction(current_steps, current_time, [])

def fallback_prediction(current_steps, current_time, daily_totals_df):
    """
    🟢 SMART FALLBACK: 
    If the AI fails (or goal is met), we trust 'Current Steps' more as the day progresses.
    """
    hour = current_time.hour
    if hour == 0: hour = 1
    
    # 1. Calculate Confidence based on Time
    # At 8 AM (2/16 hours): Confidence 40%
    # At 8 PM (14/16 hours): Confidence 95% (Because the day is done!)
    day_progress = max(0.1, min((hour - 6) / 16.0, 1.0))
    smart_confidence = 0.3 + (day_progress * 0.65) # Scale from 0.3 to 0.95
    
    # 2. Simple Projection
    if day_progress < 0.2:
        projected = max(current_steps * 2, 4000)
    else:
        projected = current_steps / day_progress

    # 3. Handle Consistency Data safely
    consistency = 0
    weekly_pattern = [0]*7
    walker_type = "Newbie 🥚"
    
    if not isinstance(daily_totals_df, list) and not daily_totals_df.empty:
        consistency = calculate_consistency_from_df(daily_totals_df)
        weekly_pattern = analyze_weekly_pattern_from_df(daily_totals_df)
        walker_type = "Determining..."

    # 🟢 Special Case: If user has 0 history, Consistency is perfect (100) for Day 1
    if consistency == 0 and current_steps > 500:
        consistency = 100

    return {
        "projected_steps": int(max(projected, current_steps)),
        "trend_message": "Keep going! Building your history profile.",
        "confidence_score": round(smart_confidence, 2), # Now dynamic!
        "consistency_score": int(consistency),
        "weekly_pattern": weekly_pattern,
        "walker_type": walker_type
    }

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def calculate_consistency_from_df(valid_days_df):
    """
    Calculates consistency score. 
    Ignores 'Today' if it drags the average down, and handles new users gently.
    """
    # 1. NEW USER CHECK
    # If we have less than 3 days of history, calculating Standard Deviation is unfair.
    # We return a high default score to be encouraging.
    if valid_days_df.empty or len(valid_days_df) < 3:
        return 90 # "Great Start / Very Stable"
    
    # 2. FILTER "TODAY" (Rough Heuristic)
    # If the last day in the list is significantly lower than the average of the rest,
    # assume it's "Today" (incomplete) and drop it from the math.
    steps_list = valid_days_df['actual_final_total'].tolist()
    
    if len(steps_list) > 1:
        last_day = steps_list[-1]
        previous_days = steps_list[:-1]
        avg_prev = sum(previous_days) / len(previous_days)
        
        # If last entry is less than 50% of previous average, ignore it (it's likely incomplete)
        if last_day < (avg_prev * 0.5):
            steps_list = previous_days

    # Re-check length after filtering
    if len(steps_list) < 2: return 90

    # 3. CALCULATE SCORE (Coefficient of Variation)
    mean_val = np.mean(steps_list)
    std_dev = np.std(steps_list)
    
    if mean_val > 0:
        cv = std_dev / mean_val
        
        # 4. GRADING CURVE
        # A CV of 0.3 is decent. A CV of 1.0 is chaotic.
        # We map this to a 0-100 score more gently.
        # Formula: Start at 100, subtract (CV * 80). Floor at 20.
        score = 100 - (cv * 80)
        score = max(20, min(score, 100)) # Clamp between 20 and 100
        
        return int(score)
        
    return 0

def analyze_weekly_pattern_from_df(valid_days_df):
    if valid_days_df.empty: return [0]*7
    valid_days_df['weekday'] = valid_days_df['date'].dt.dayofweek
    weekday_avgs = valid_days_df.groupby('weekday')['actual_final_total'].mean()
    weekday_avgs = weekday_avgs.reindex(range(7), fill_value=0)
    return weekday_avgs.astype(int).tolist()

def determine_walker_type(df):
    if df.empty: return "Newbie 🥚"
    try:
        morning = df[(df['hour'] >= 5) & (df['hour'] < 12)]['steps'].sum()
        noon = df[(df['hour'] >= 12) & (df['hour'] < 17)]['steps'].sum()
        evening = df[(df['hour'] >= 17) & (df['hour'] < 23)]['steps'].sum()
        
        total = morning + noon + evening
        if total == 0: return "Newbie 🥚"
        
        if morning > noon and morning > evening: return "Morning Lark 🌅"
        elif noon > morning and noon > evening: return "Lunchtime Stroller ☀️"
        else: return "Night Owl 🌙"
    except:
        return "Newbie 🥚"

# Stub functions for main.py compatibility
def calculate_consistency(data): return 0
def analyze_weekly_pattern(data): return [0]*7
def generate_trend_message(p, a): return ""