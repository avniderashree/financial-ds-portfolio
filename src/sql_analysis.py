# src/sql_analysis.py

import pandas as pd
import sqlite3

def run_quant_queries():
    # 1. Load your Processed Data
    # We are loading the CSV from Week 2 into a Pandas DataFrame
    df = pd.read_csv('data/processed/market_features.csv')
    
    # 2. Create an In-Memory SQL Database
    # This creates a temporary SQL server in your RAM
    conn = sqlite3.connect(':memory:')
    
    # 3. Dump Data into SQL
    # We name the table 'market_data'
    df.to_sql('market_data', conn, index=False, if_exists='replace')
    
    print("✅ Data loaded into SQL Database.")
    print("---------------------------------------------------")

    # --- QUERY 1: The "Volatility Spike" Detector ---
    # Concept: Use LAG() to compare Today vs. Yesterday
    print("\n📊 Query 1: Top 5 Volatility Spikes (Day-over-Day Change)")
    
    query_1 = """
    WITH VolatilityCalc AS (
        SELECT 
            Date,
            SPY_Close,
            SPY_Vol_30d,
            -- LAG looks at the previous row (Yesterday)
            LAG(SPY_Vol_30d, 1) OVER (ORDER BY Date) as Prev_Vol
        FROM market_data
    )
    SELECT 
        Date, 
        SPY_Vol_30d, 
        Prev_Vol,
        (SPY_Vol_30d - Prev_Vol) as Vol_Change
    FROM VolatilityCalc
    WHERE Prev_Vol IS NOT NULL
    ORDER BY Vol_Change DESC
    LIMIT 5;
    """
    
    result_1 = pd.read_sql(query_1, conn)
    print(result_1)
    
    # --- QUERY 2: Rolling Averages in SQL ---
    # Concept: Calculate a 5-day Moving Average inside the DB
    print("\n📊 Query 2: Comparing Price vs. 5-Day Moving Average")
    
    query_2 = """
    SELECT 
        Date,
        SPY_Close,
        -- ROWS BETWEEN 4 PRECEDING AND CURRENT ROW = 5 Day Window
        AVG(SPY_Close) OVER (
            ORDER BY Date 
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) as SMA_5_Day
    FROM market_data
    ORDER BY Date DESC
    LIMIT 5;
    """
    
    result_2 = pd.read_sql(query_2, conn)
    print(result_2)

    # --- QUERY 3: Risk Ranking ---
    # Concept: Rank days by "Riskiness" (Volatility)
    print("\n📊 Query 3: Ranking the Riskiest Days")
    
    query_3 = """
    SELECT 
        Date,
        SPY_Vol_30d,
        -- Rank 1 = Highest Volatility
        RANK() OVER (ORDER BY SPY_Vol_30d DESC) as Risk_Rank
    FROM market_data
    LIMIT 5;
    """
    
    result_3 = pd.read_sql(query_3, conn)
    print(result_3)

if __name__ == "__main__":
    run_quant_queries()