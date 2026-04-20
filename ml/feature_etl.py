"""Feature store ETL: Load engineered features into PostgreSQL."""

import pandas as pd
from sqlalchemy import text
from ml.feature_engineering import (
    engineer_technical_features,
    engineer_macro_features,
    engineer_fundamentals_features
)


def load_technical_features(stock_df: pd.DataFrame, tickers: list, engine):
    """Compute and load technical indicators to feat_technical_indicators table."""
    print("Loading technical indicators...")
    
    for ticker in tickers:
        tech_features = engineer_technical_features(stock_df, ticker)
        
        if tech_features.empty:
            print(f"  Skipping {ticker}: no data")
            continue
        
        # Convert date to string for SQL
        tech_features['date'] = pd.to_datetime(tech_features['date']).dt.date
        
        # Delete existing records for this ticker (to allow re-computation)
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM feat_technical_indicators WHERE ticker = :ticker"),
                        {"ticker": ticker})
            conn.commit()
        
        # Load new data
        tech_features.to_sql(
            'feat_technical_indicators',
            engine,
            if_exists='append',
            index=False
        )
        print(f"  ✓ Loaded {len(tech_features)} technical feature records for {ticker}")


def load_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame, tickers: list, engine):
    """Compute and load macro aggregation features to feat_macro_ratios table."""
    print("Loading macro aggregation features...")
    
    for ticker in tickers:
        macro_features = engineer_macro_features(stock_df, macro_df, ticker)
        
        if macro_features.empty:
            print(f"  Skipping {ticker}: no data")
            continue
        
        macro_features['date'] = pd.to_datetime(macro_features['date']).dt.date
        
        # Delete existing records
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM feat_macro_ratios WHERE ticker = :ticker"),
                        {"ticker": ticker})
            conn.commit()
        
        macro_features.to_sql(
            'feat_macro_ratios',
            engine,
            if_exists='append',
            index=False
        )
        print(f"  ✓ Loaded {len(macro_features)} macro feature records for {ticker}")


def load_fundamentals_features(fin_df: pd.DataFrame, tickers: list, engine):
    """Compute and load company fundamentals features to feat_company_fundamentals table."""
    print("Loading company fundamentals features...")
    
    for ticker in tickers:
        fund_features = engineer_fundamentals_features(fin_df, ticker)
        
        if fund_features.empty:
            print(f"  Skipping {ticker}: no data")
            continue
        
        fund_features['date'] = pd.to_datetime(fund_features['date']).dt.date
        
        # Delete existing records
        with engine.connect() as conn:
            conn.execute(text("DELETE FROM feat_company_fundamentals WHERE ticker = :ticker"),
                        {"ticker": ticker})
            conn.commit()
        
        fund_features.to_sql(
            'feat_company_fundamentals',
            engine,
            if_exists='append',
            index=False
        )
        print(f"  ✓ Loaded {len(fund_features)} fundamentals feature records for {ticker}")


def populate_feature_store(stock_df: pd.DataFrame, macro_df: pd.DataFrame, 
                          fin_df: pd.DataFrame, tickers: list, engine):
    """Main ETL function: Populate all feature store tables."""
    print("\n=== Populating Feature Store ===")
    
    load_technical_features(stock_df, tickers, engine)
    load_macro_features(stock_df, macro_df, tickers, engine)
    load_fundamentals_features(fin_df, tickers, engine)
    
    print("✓ Feature store population complete")


if __name__ == "__main__":
    from db import get_engine
    engine = get_engine()
    
    # Example usage (would be called from main.py)
    print("Feature ETL module loaded. Call populate_feature_store() from main.py")
