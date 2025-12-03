# src/data_processing.py

import pandas as pd
import numpy as np
import os

def load_and_process_data(filepath):
    """
    Loads raw market data, fixes headers, and calculates financial features.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return None

    print(f"Processing file: {filepath}...")

    # 1. Load Data (Handling the specific Multi-Index format of yfinance)
    # header=[0, 1] tells pandas the first two rows are headers (Ticker, Price Type)
    df = pd.read_csv(filepath, header=[0, 1], index_col=0, parse_dates=True)
    
    # 2. Flatten Columns (Technique: List Comprehension)
    # Converts ('SPY', 'Close') -> 'SPY_Close'
    # This makes the dataframe much easier to query
    df.columns = [f"{col[0]}_{col[1]}" for col in df.columns]
    
    # 3. Filter only for 'Close' prices to keep it clean
    # We use 'Adj Close' if available, but let's stick to 'Close' for simplicity unless specified
    close_cols = [c for c in df.columns if 'Close' in c]
    df_clean = df[close_cols].copy()
    
    # 4. Handle Missing Data (The Finance Way)
    # 'ffill': If data is missing (holiday), assume price hasn't changed since yesterday.
    df_clean = df_clean.ffill()
    # Drop any remaining NaNs (usually the first few rows)
    df_clean = df_clean.dropna()

    return df_clean

def add_features(df):
    """
    Adds Log Returns and Rolling Volatility to the dataframe.
    """
    # Identify the SPY column (Market Proxy)
    spy_col = 'SPY_Close' 
    
    if spy_col not in df.columns:
        print(f"⚠️ Warning: {spy_col} not found. Skipping feature generation.")
        return df

    # --- FEATURE ENGINEERING ---
    
    # 1. Log Returns (Daily Performance)
    # np.log(Price_t / Price_t-1)
    df['SPY_Log_Ret'] = np.log(df[spy_col] / df[spy_col].shift(1))
    
    # 2. Realized Volatility (30-Day Window)
    # Standard Deviation of returns * sqrt(252) to annualize it
    # This tells us: "How risky has the market been over the last month?"
    df['SPY_Vol_30d'] = df['SPY_Log_Ret'].rolling(window=30).std() * np.sqrt(252)
    
    # 3. Simple Moving Average (Trend Indicator)
    df['SPY_SMA_50'] = df[spy_col].rolling(window=50).mean()
    
    # Drop the NaN rows created by the rolling windows (first 50 rows)
    df_final = df.dropna()
    
    return df_final

if __name__ == "__main__":
    # Update this filename to match the actual file you downloaded in Week 1
    # Hint: Check your data/raw folder for the exact date string
    input_file = 'data/raw/market_data_2020-01-01_2024-01-01.csv'
    output_file = 'data/processed/market_features.csv'
    
    # Step 1: Clean
    df_clean = load_and_process_data(input_file)
    
    if df_clean is not None:
        # Step 2: Engineer Features
        df_features = add_features(df_clean)
        
        # Step 3: Save to Processed folder
        df_features.to_csv(output_file)
        
        print("------------------------------------------------")
        print(f"✅ Feature Engineering Complete.")
        print(f"Saved processed data to: {output_file}")
        print("------------------------------------------------")
        print(df_features.head())