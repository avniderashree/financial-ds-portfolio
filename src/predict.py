# src/predict.py
import xgboost as xgb
import pandas as pd
import sys

def predict_risk(current_data):
    """
    Loads the saved model and predicts risk for new data.
    current_data: Dict or DataFrame containing the 5 features.
    """
    # 1. Load Model
    model = xgb.XGBClassifier()
    model.load_model('models/xgb_risk_model.json')
    
    # 2. Convert input to DataFrame
    df = pd.DataFrame([current_data])
    
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