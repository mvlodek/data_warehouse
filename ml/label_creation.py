"""Label creation: Generate training targets based on forward-looking returns."""

import pandas as pd
import numpy as np
from sqlalchemy import text


# Label thresholds (30-day forward return)
BUY_THRESHOLD = 0.05      # > 5% = BUY
SELL_THRESHOLD = -0.02    # < -2% = SELL
# -2% to 5% = HOLD


def create_labels_from_stock_data(stock_df: pd.DataFrame, tickers: list, horizon_days: int = 30) -> pd.DataFrame:
    """
    Create classification labels based on forward-looking returns.
    
    Args:
        stock_df: DataFrame with [date, ticker, close]
        tickers: List of tickers to process
        horizon_days: Number of days ahead to look for returns
    
    Returns:
        DataFrame with columns [date, ticker, forward_return_30d, label, is_valid]
    """
    labels_list = []
    
    for ticker in tickers:
        ticker_data = stock_df[stock_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').reset_index(drop=True)
        
        if len(ticker_data) < horizon_days + 1:
            print(f"  Skipping {ticker}: insufficient data (need {horizon_days + 1}, have {len(ticker_data)})")
            continue
        
        # Calculate forward return
        ticker_data['forward_close'] = ticker_data['close'].shift(-horizon_days)
        ticker_data['forward_return'] = (ticker_data['forward_close'] - ticker_data['close']) / ticker_data['close']
        
        # Create labels based on thresholds
        def label_fn(ret):
            if pd.isna(ret):
                return 'HOLD'  # Default
            elif ret > BUY_THRESHOLD:
                return 'BUY'
            elif ret < SELL_THRESHOLD:
                return 'SELL'
            else:
                return 'HOLD'
        
        ticker_data['label'] = ticker_data['forward_return'].apply(label_fn)
        
        # is_valid = True only if we have future data
        ticker_data['is_valid'] = ticker_data['forward_return'].notna()
        
        # Select output columns
        labels_list.append(ticker_data[['date', 'ticker', 'forward_return', 'label', 'is_valid']])
    
    labels_df = pd.concat(labels_list, ignore_index=True)
    labels_df.rename(columns={'forward_return': 'forward_return_30d'}, inplace=True)
    
    return labels_df


def load_labels_to_db(labels_df: pd.DataFrame, engine):
    """Load labels to feat_labels table in PostgreSQL."""
    print("Loading labels...")
    
    # Delete existing labels (allow re-computation)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM feat_labels"))
        conn.commit()
    
    # Convert dates
    labels_df['date'] = pd.to_datetime(labels_df['date']).dt.date
    
    # Load
    labels_df.to_sql(
        'feat_labels',
        engine,
        if_exists='append',
        index=False
    )
    
    print(f"✓ Loaded {len(labels_df)} labels")
    print(f"  BUY:  {(labels_df['label'] == 'BUY').sum()}")
    print(f"  HOLD: {(labels_df['label'] == 'HOLD').sum()}")
    print(f"  SELL: {(labels_df['label'] == 'SELL').sum()}")


def create_and_load_labels(stock_df: pd.DataFrame, tickers: list, engine):
    """Main function: Create labels and load to DB."""
    print("\n=== Creating Training Labels ===")
    labels_df = create_labels_from_stock_data(stock_df, tickers, horizon_days=30)
    load_labels_to_db(labels_df, engine)
    return labels_df


if __name__ == "__main__":
    from db import get_engine
    print("Label creation module loaded.")
