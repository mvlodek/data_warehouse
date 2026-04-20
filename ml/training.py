"""Training dataset assembly: Join features + labels for model training."""

import pandas as pd
from sqlalchemy import text


def assemble_training_dataset(engine, tickers: list, test_split_date: str = '2024-01-01'):
    """
    Assemble complete training dataset by joining features + labels.
    
    Args:
        engine: SQLAlchemy engine
        tickers: List of tickers to include
        test_split_date: Date to split train/test (before = train, after = test)
    
    Returns:
        (train_df, test_df) - Training and test datasets
    """
    print("Assembling training dataset...")
    
    # Query: Join technical features + macro features + fundamentals + labels
    query = """
        SELECT 
            t.ticker,
            t.date,
            -- Technical features
            t.close,
            t.sma_20, t.sma_50, t.sma_200,
            t.rsi_14,
            t.macd, t.macd_signal, t.macd_diff,
            t.atr_14,
            t.momentum_10, t.momentum_20,
            t.volume_sma_20,
            t.volatility_20,
            -- Macro features
            m.interest_rate,
            m.interest_rate_momentum_1m,
            m.interest_rate_momentum_3m,
            m.inflation,
            m.inflation_momentum_1m,
            m.inflation_momentum_3m,
            m.interest_rate_volatility_20d,
            m.inflation_volatility_20d,
            -- Fundamentals
            f.revenue_growth_yoy,
            f.eps_growth_yoy,
            f.net_income_growth_yoy,
            f.revenue_momentum_ttm,
            f.earnings_momentum_ttm,
            -- Label
            l.label,
            l.forward_return_30d,
            l.is_valid
        FROM feat_technical_indicators t
        LEFT JOIN feat_macro_ratios m ON t.ticker = m.ticker AND t.date = m.date
        LEFT JOIN feat_company_fundamentals f ON t.ticker = f.ticker AND DATE_TRUNC('month', t.date)::date = DATE_TRUNC('month', f.date)::date
        LEFT JOIN feat_labels l ON t.ticker = l.ticker AND t.date = l.date
        WHERE t.ticker IN ({})
        AND l.label IS NOT NULL
        AND l.is_valid = TRUE
        ORDER BY t.date;
    """.format(','.join([f"'{t}'" for t in tickers]))
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        print("⚠ Warning: Empty dataset. Check that feature store is populated.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"✓ Loaded {len(df)} total records")
    
    # Forward-fill missing fundamentals (sparse annual data)
    # We keep rows even if fundamentals are missing, as long as technical/macro/label are present
    fundamental_cols = [
        'revenue_growth_yoy', 'eps_growth_yoy', 'net_income_growth_yoy',
        'revenue_momentum_ttm', 'earnings_momentum_ttm'
    ]
    
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df.groupby('ticker')[col].ffill()
    
    # Critical features that cannot be missing
    critical_cols = ['close', 'sma_20', 'rsi_14', 'interest_rate', 'inflation', 'label']
    
    # Drop only rows with missing CRITICAL features (not fundamentals)
    initial_rows = len(df)
    df = df.dropna(subset=critical_cols)
    dropped = initial_rows - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with missing critical features")
    
    # Fill remaining fundamental NaN with 0 (they contribute less to predictions anyway)
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Split into train/test
    df['date'] = pd.to_datetime(df['date'])
    test_date = pd.to_datetime(test_split_date)
    
    train_df = df[df['date'] < test_date].copy()
    test_df = df[df['date'] >= test_date].copy()
    
    print(f"  Train set: {len(train_df)} records (before {test_split_date})")
    print(f"  Test set:  {len(test_df)} records (on/after {test_split_date})")
    
    # Label distribution
    print(f"  Label distribution (train):")
    print(f"    BUY:  {(train_df['label'] == 'BUY').sum()}")
    print(f"    HOLD: {(train_df['label'] == 'HOLD').sum()}")
    print(f"    SELL: {(train_df['label'] == 'SELL').sum()}")
    
    return train_df, test_df


def get_feature_names() -> list:
    """Return list of feature column names (all numeric features)."""
    return [
        'close', 'sma_20', 'sma_50', 'sma_200',
        'rsi_14',
        'macd', 'macd_signal', 'macd_diff',
        'atr_14',
        'momentum_10', 'momentum_20',
        'volume_sma_20',
        'volatility_20',
        'interest_rate',
        'interest_rate_momentum_1m',
        'interest_rate_momentum_3m',
        'inflation',
        'inflation_momentum_1m',
        'inflation_momentum_3m',
        'interest_rate_volatility_20d',
        'inflation_volatility_20d',
        'revenue_growth_yoy',
        'eps_growth_yoy',
        'net_income_growth_yoy',
        'revenue_momentum_ttm',
        'earnings_momentum_ttm',
    ]


if __name__ == "__main__":
    from db import get_engine
    print("Training data assembly module loaded.")
