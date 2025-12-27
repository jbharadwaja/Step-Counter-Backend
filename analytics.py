import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

def predict_end_of_day_steps(current_steps, current_time, historical_data):
    """
    Production-Ready Step Prediction Engine.
    Uses Linear Regression to correlate 'steps by hour X' with 'final daily total'.
    """
    
    # 1. SAFETY CHECKS
    if not historical_data or len(historical_data) < 50:
        # Not enough data for AI? Fallback to simple multiplier.
        return fallback_prediction(current_steps, current_time)

    # 2. PREPARE DATA (ETL Pipeline)
    df = pd.DataFrame(historical_data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate "Final Total" for every day in history
    daily_totals = df.groupby('date')['steps'].sum().reset_index()
    daily_totals.rename(columns={'steps': 'actual_final_total'}, inplace=True)
    
    # Calculate "Cumulative Steps" up to the CURRENT HOUR for every past day
    current_hour = current_time.hour
    
    # Filter only logs that happened BEFORE or AT the current hour
    df_filtered = df[df['hour'] <= current_hour].copy()
    
    # Group by date to see what the count was at this specific time in the past
    steps_at_current_hour = df_filtered.groupby('date')['steps'].sum().reset_index()
    steps_at_current_hour.rename(columns={'steps': 'steps_by_now'}, inplace=True)
    
    # Merge: We now have X (steps by 2pm) and Y (final steps) for every day
    training_data = pd.merge(steps_at_current_hour, daily_totals, on='date')
    
    # 3. TRAIN THE MODEL (Linear Regression)
    # We only train if we have at least 5 days of history for this specific hour
    if len(training_data) < 5:
        return fallback_prediction(current_steps, current_time)
        
    X = training_data[['steps_by_now']] # Feature
    y = training_data['actual_final_total'] # Target
    
    # Initialize and Fit
    model = LinearRegression()
    model.fit(X, y)
    
    # 4. PREDICT FOR TODAY
    # We wrap current_steps in a 2D array because sklearn expects it
    prediction = model.predict([[current_steps]])[0]
    
    # 5. GENERATE INTELLIGENT CONTEXT
    predicted_final = int(max(prediction, current_steps)) # Can't be less than now
    
    # Compare against their typical average to generate the "Roast/Hype"
    avg_total = daily_totals['actual_final_total'].mean()
    deviation = predicted_final - avg_total
    
    if deviation > 2000:
        msg = f"🚀 Amazing pace! You're trending {int(deviation)} steps above your average."
        score = 0.95
    elif deviation < -2000:
        msg = f"📉 A bit slow today. You're trending {abs(int(deviation))} steps below normal."
        score = 0.3
    else:
        msg = "🎯 You are right on track with your usual habits."
        score = 0.75

    return {
        "projected_steps": predicted_final,
        "completion_rate_at_this_hour": 0.0, # Deprecated but kept for safety
        "confidence_score": score,
        "trend_message": msg
    }

def fallback_prediction(current_steps, current_time):
    """Simple multiplier fallback if AI fails or data is missing."""
    hour = current_time.hour
    if hour == 0: hour = 1
    
    # Rough estimate: Assume steady walking over 16 active hours
    fraction_of_day_passed = min((hour - 6) / 16.0, 1.0) 
    if fraction_of_day_passed <= 0: fraction_of_day_passed = 0.05
    
    projected = current_steps / fraction_of_day_passed
    
    return {
        "projected_steps": int(projected),
        "completion_rate_at_this_hour": 0.0,
        "confidence_score": 0.2,
        "trend_message": "Gathering more data for AI predictions..."
    }