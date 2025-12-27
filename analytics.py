import pandas as pd
import numpy as np
from datetime import datetime

def predict_end_of_day_steps(current_steps, current_time, historical_data):
    """
    Predicts the end-of-day step count based on intraday completion rates.
    
    Args:
        current_steps (int): Steps taken so far today.
        current_time (datetime): The current timestamp.
        historical_data (list of dicts): A list of past hourly logs.
            Format: [{'date': '2023-10-01', 'hour': 14, 'steps': 500}, ...]
            
    Returns:
        dict: {
            "projected_steps": int,
            "confidence_score": float (0.0 - 1.0),
            "trend_message": str
        }
    """
    
    # 1. Convert Data to DataFrame
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # 2. Feature Engineering: Segment by Weekday vs. Weekend
    #    (People walk differently on weekends)
    is_weekend = current_time.weekday() >= 5
    if is_weekend:
        df = df[df['date'].dt.dayofweek >= 5]
    else:
        df = df[df['date'].dt.dayofweek < 5]
    
    # 3. Calculate Daily Totals for every historical date
    daily_totals = df.groupby('date')['steps'].sum().reset_index()
    daily_totals.rename(columns={'steps': 'total_daily_steps'}, inplace=True)
    
    # 4. Calculate Cumulative Steps by Hour for every historical date
    #    Sort by date and hour first
    df = df.sort_values(['date', 'hour'])
    df['cumulative_steps'] = df.groupby('date')['steps'].cumsum()
    
    # 5. Merge to get "Completion Rate" (Cumulative / Total)
    df = df.merge(daily_totals, on='date')
    df['completion_rate'] = df['cumulative_steps'] / df['total_daily_steps']
    
    # 6. Get the "Typical Completion Rate" for the current hour
    #    We use MEDIAN (not Mean) to be robust against outlier days
    current_hour = current_time.hour
    
    hourly_stats = df[df['hour'] == current_hour]['completion_rate']
    
    if hourly_stats.empty:
        # Fallback if no history exists for this specific hour
        return {
            "projected_steps": current_steps, 
            "confidence_score": 0.0,
            "trend_message": "Not enough data"
        }
        
    median_completion_rate = hourly_stats.median()
    
    # 7. The Prediction Math
    #    Guard against division by zero (e.g., 3 AM with 0 steps)
    if median_completion_rate < 0.05:
        # If we are < 5% through the day's activity, prediction is too volatile.
        # Fallback to the historical median daily total.
        projected = daily_totals['total_daily_steps'].median()
        confidence = 0.2 # Low confidence
        message = "Early day estimation"
    else:
        projected = current_steps / median_completion_rate
        confidence = 0.85 # High confidence
        message = "Based on your daily patterns"

    return {
        "projected_steps": int(projected),
        "completion_rate_at_this_hour": round(median_completion_rate, 2),
        "confidence_score": confidence,
        "trend_message": message
    }

# --- EXAMPLE USAGE ---
# Let's say it's 2 PM (Hour 14) on a Tuesday, and I have 4,500 steps.
# My history shows I'm usually 45% done by 2 PM.

# mock_history = [
#    {'date': '2025-12-01', 'hour': 9, 'steps': 1000},
#    {'date': '2025-12-01', 'hour': 14, 'steps': 3500}, ... 
# ]
# 
# result = predict_end_of_day_steps(4500, datetime.now(), mock_history)
# print(result) 
# Output: {'projected_steps': 10000, ...}