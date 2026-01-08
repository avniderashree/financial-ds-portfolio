# src/predict.py
import xgboost as xgb
import pandas as pd
import sys

def predict_risk(current_data):
    # 1. Load Model
    model = xgb.XGBClassifier()
    
    # --- ADD THIS LINE TO FIX THE ERROR ---
    model._estimator_type = "classifier" 
    # --------------------------------------

    model.load_model('models/xgb_risk_model.json')
    
    # 2. Convert input to DataFrame (Ensure feature order matches training!)
    # It is safer to explicitly order your columns here
    feature_order = ['SPY_Log_Ret', 'SPY_Vol_30d', 'RSI', 'BB_Width', 'Trend_Signal']
    df = pd.DataFrame([current_data])[feature_order]
    
    # 3. Predict
    prob = model.predict_proba(df)[:, 1][0]
    is_crash = model.predict(df)[0]
    
    return prob, is_crash

if __name__ == "__main__":
    # Simulate live data (e.g., today's market close stats)
    # These are random numbers just to test the plumbing
    todays_market = {
        'SPY_Log_Ret': -0.005,   # Market fell 0.5%
        'SPY_Vol_30d': 0.25,     # Volatility is high (25%)
        'RSI': 35,               # RSI is low (Oversold)
        'BB_Width': 0.08,        # Bands are wide
        'Trend_Signal': 0        # Trend is Negative
    }
    
    try:
        probability, crash_pred = predict_risk(todays_market)
        
        print("--------------------------------")
        print("🔮 LIVE RISK REPORT")
        print("--------------------------------")
        print(f"Crash Probability: {probability:.2%}")
        
        if crash_pred == 1:
            print("🚨 ALERT: HIGH RISK DETECTED. Increase Collateral.")
        else:
            print("✅ STATUS: MARKET NORMAL.")
            
    except Exception as e:
        print(f"Error: {e}")