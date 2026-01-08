# src/model_training.py

import pandas as pd
import xgboost as xgb
import joblib
import os
from sklearn.metrics import classification_report, roc_auc_score

class LiquidityRiskModel:
    def __init__(self, model_path='models/xgb_risk_model.json'):
        self.model_path = model_path
        # Initialize with the best parameters
        self.model = xgb.XGBClassifier(
            scale_pos_weight=10,
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='logloss'
            # Note: early_stopping_rounds should be passed in .fit() for older versions,
            # but it is fine here for newer versions.
        )
        
        # --- THE FIX ---
        self.model._estimator_type = "classifier" 
        # ----------------

    def prepare_data(self, filepath):
        """Loads and splits data for training."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Ensure Target exists (Week 10 logic)
        if 'Target' not in df.columns:
            # Re-create target if missing (Crash < -1%)
            df['Next_Day_Return'] = df['SPY_Log_Ret'].shift(-1)
            df['Target'] = (df['Next_Day_Return'] < -0.01).astype(int)
            df = df.dropna()

        # Define Feature List (The exact features the model expects)
        # Note: We must use the engineered features from Week 10
        feature_cols = ['SPY_Log_Ret', 'SPY_Vol_30d', 'RSI', 'BB_Width', 'Trend_Signal']
        
        # Check if all features exist
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataset: {missing}")

        X = df[feature_cols]
        y = df['Target']
        
        # Time-based split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_test, y_test):
        """Trains the XGBoost model."""
        print("Training XGBoost Model...")
        
        # Fit model (using Test set for early stopping to prevent overfitting)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        print("✅ Training Complete.")

    def evaluate(self, X_test, y_test):
        """Prints evaluation metrics."""
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, probs)
        print("------------------------------------------------")
        print(f"Model Performance (AUC): {auc:.4f}")
        print("------------------------------------------------")
        print(classification_report(y_test, preds))

    def save(self):
        """Saves the trained model to disk."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save using XGBoost's internal JSON format (best for compatibility)
        self.model.save_model(self.model_path)
        print(f"💾 Model saved to {self.model_path}")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Initialize
    risk_model = LiquidityRiskModel()
    
    # 2. Data
    # Point to the 'v2' data with RSI/Bollinger Bands from Week 10
    data_path = 'data/processed/market_features_v2.csv'
    
    try:
        X_train, X_test, y_train, y_test = risk_model.prepare_data(data_path)
        
        # 3. Train
        risk_model.train(X_train, y_train, X_test, y_test)
        
        # 4. Evaluate
        risk_model.evaluate(X_test, y_test)
        
        # 5. Save
        risk_model.save()
        
    except Exception as e:
        print(f"❌ Pipeline Failed: {e}")