# src/data_ingestion.py

import yfinance as yf
import pandas as pd
import os

def fetch_market_data(tickers, start_date, end_date, output_folder='data/raw'):
    """
    Fetches historical market data from Yahoo Finance and saves it to a CSV.
    
    Parameters:
    - tickers (list): List of ticker symbols (e.g., ['SPY', 'VIX'])
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format
    - output_folder (str): Directory to save the CSV file
    
    Returns:
    - None
    """
    
    # 1. Ensure the output directory exists (Best Practice: Error Prevention)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"Fetching data for: {tickers}...")

    try:
        # 2. Download data using yfinance
        # group_by='ticker' allows us to handle multiple stocks easily
        raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
        
        # 3. Check if data is empty (Best Practice: Validation)
        if raw_data.empty:
            print("⚠️ Warning: No data fetched. Check your tickers or date range.")
            return

        # 4. Save to CSV
        # We use a dynamic filename so we don't overwrite old data accidentally
        filename = f"{output_folder}/market_data_{start_date}_{end_date}.csv"
        raw_data.to_csv(filename)
        
        print(f"✅ Success! Data saved to {filename}")
        print(f"Shape of data: {raw_data.shape}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")

# --- EXECUTION BLOCK ---
# This block only runs if you run this script directly (not if you import it elsewhere)
if __name__ == "__main__":
    # We are pulling the S&P 500 (SPY) and Volatility Index (VIX)
    # These are crucial for your future Liquidity Model.
    my_tickers = ['SPY', 'VIX'] 
    fetch_market_data(my_tickers, '2020-01-01', '2024-01-01')