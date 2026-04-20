"""
Enhanced feature engineering with macro indicators, FMP data, and stock APIs.
Optimized for multi-database data warehouse.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
from db_multi import DatabaseFactory, load_db_env


# =============================================================================
# Technical Indicators (Enhanced)
# =============================================================================

def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=1).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd = ema_fast - ema_slow
    signal_line = compute_ema(macd, signal)
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=1).mean()


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = compute_sma(series, period)
    std = series.rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=3).mean()
    return k.fillna(50), d.fillna(50)


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period).fillna(0)


def compute_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    return returns.rolling(window=period, min_periods=1).std().fillna(0)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def compute_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


# =============================================================================
# Feature Engineering Functions
# =============================================================================

def engineer_technical_features(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Generate comprehensive technical indicators for a stock.
    """
    df = stock_df[stock_df['ticker'] == ticker].copy()
    if df.empty:
        return pd.DataFrame()
    
    df = df.sort_values('date').reset_index(drop=True)
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Price-based indicators
    df['sma_5'] = compute_sma(df['close'], 5)
    df['sma_20'] = compute_sma(df['close'], 20)
    df['sma_50'] = compute_sma(df['close'], 50)
    df['sma_200'] = compute_sma(df['close'], 200)
    
    df['ema_12'] = compute_ema(df['close'], 12)
    df['ema_26'] = compute_ema(df['close'], 26)
    
    # RSI
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['rsi_7'] = compute_rsi(df['close'], 7)
    
    # MACD
    df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(df['close'])
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = compute_bollinger_bands(df['close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = compute_stochastic(df['high'], df['low'], df['close'])
    
    # ATR
    df['atr_14'] = compute_atr(df['high'], df['low'], df['close'], 14)
    df['atr_pct'] = df['atr_14'] / df['close'] * 100
    
    # Momentum
    df['momentum_5'] = compute_momentum(df['close'], 5)
    df['momentum_10'] = compute_momentum(df['close'], 10)
    df['momentum_20'] = compute_momentum(df['close'], 20)
    
    # Volatility
    returns = df['close'].pct_change()
    df['volatility_10'] = compute_volatility(returns, 10)
    df['volatility_20'] = compute_volatility(returns, 20)
    df['volatility_60'] = compute_volatility(returns, 60)
    
    # Volume indicators
    df['volume_sma_20'] = compute_sma(df['volume'], 20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['obv'] = compute_obv(df['close'], df['volume'])
    df['obv_ma'] = compute_sma(df['obv'], 10)
    
    # VWAP
    df['vwap'] = compute_vwap(df['high'], df['low'], df['close'], df['volume'])
    
    # Price position
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Returns
    df['return_1d'] = returns
    df['return_5d'] = df['close'].pct_change(5)
    df['return_20d'] = df['close'].pct_change(20)
    
    return df


def engineer_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge macroeconomic indicators with stock data.
    """
    dates = stock_df[stock_df['ticker'] == ticker]['date'].sort_values().reset_index(drop=True)
    output = pd.DataFrame({'date': dates, 'ticker': ticker})
    
    if macro_df.empty:
        return output
    
    # Interest Rate features
    ir_df = macro_df[macro_df['indicator'] == 'INTEREST_RATE'][['date', 'value']].copy()
    if not ir_df.empty:
        ir_df.columns = ['date', 'interest_rate']
        ir_df['date'] = pd.to_datetime(ir_df['date']).dt.normalize()
        output = output.merge(ir_df, on='date', how='left')
        output['interest_rate_ma_3'] = output['interest_rate'].rolling(3).mean()
        output['interest_rate_momentum_1m'] = output['interest_rate'].pct_change(1)
        output['interest_rate_momentum_3m'] = output['interest_rate'].pct_change(3)
    
    # Inflation features
    inf_df = macro_df[macro_df['indicator'] == 'INFLATION'][['date', 'value']].copy()
    if not inf_df.empty:
        inf_df.columns = ['date', 'inflation']
        inf_df['date'] = pd.to_datetime(inf_df['date']).dt.normalize()
        output = output.merge(inf_df, on='date', how='left')
        output['inflation_ma_3'] = output['inflation'].rolling(3).mean()
        output['inflation_momentum_1m'] = output['inflation'].pct_change(1)
        output['inflation_momentum_3m'] = output['inflation'].pct_change(3)
    
    # GDP features
    gdp_df = macro_df[macro_df['indicator'] == 'GDP_GROWTH'][['date', 'value']].copy()
    if not gdp_df.empty:
        gdp_df.columns = ['date', 'gdp_growth']
        gdp_df['date'] = pd.to_datetime(gdp_df['date']).dt.normalize()
        output = output.merge(gdp_df, on='date', how='left')
    
    # Unemployment features
    unemp_df = macro_df[macro_df['indicator'] == 'UNEMPLOYMENT'][['date', 'value']].copy()
    if not unemp_df.empty:
        unemp_df.columns = ['date', 'unemployment']
        unemp_df['date'] = pd.to_datetime(unemp_df['date']).dt.normalize()
        output = output.merge(unemp_df, on='date', how='left')
    
    # Forward fill missing macro data
    output = output.sort_values('date')
    for col in output.columns:
        if col not in ['date', 'ticker']:
            output[col] = output[col].ffill()
    
    return output


def engineer_fmp_features(stock_df: pd.DataFrame, fmp_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge FMP (Financial Modeling Prep) data with stock data.
    """
    dates = stock_df[stock_df['ticker'] == ticker]['date'].sort_values().reset_index(drop=True)
    output = pd.DataFrame({'date': dates, 'ticker': ticker})
    
    if fmp_df.empty or 'ticker' not in fmp_df.columns:
        return output
    
    fmp_ticker = fmp_df[fmp_df['ticker'] == ticker].copy()
    if fmp_ticker.empty:
        return output
    
    # Key metrics to merge
    fmp_cols = ['date', 'pe_ratio', 'eps', 'market_cap', 'dividend_yield', 
                'beta', '52_week_high', '52_week_low']
    available_cols = [c for c in fmp_cols if c in fmp_ticker.columns]
    
    if available_cols:
        fmp_subset = fmp_ticker[available_cols].copy()
        fmp_subset['date'] = pd.to_datetime(fmp_subset['date']).dt.normalize()
        output = output.merge(fmp_subset, on='date', how='left')
    
    return output


def create_all_features(stock_df: pd.DataFrame, 
                        macro_df: pd.DataFrame = None,
                        fmp_df: pd.DataFrame = None,
                        tickers: List[str] = None) -> pd.DataFrame:
    """
    Create comprehensive feature set from all data sources.
    """
    if tickers is None:
        tickers = stock_df['ticker'].unique().tolist()
    
    all_features = []
    
    for ticker in tickers:
        # Technical features
        tech_df = engineer_technical_features(stock_df, ticker)
        
        # Macro features
        if macro_df is not None and not macro_df.empty:
            tech_df = tech_df.merge(
                engineer_macro_features(stock_df, macro_df, ticker),
                on=['date', 'ticker'],
                how='left'
            )
        
        # FMP features
        if fmp_df is not None and not fmp_df.empty:
            tech_df = tech_df.merge(
                engineer_fmp_features(stock_df, fmp_df, ticker),
                on=['date', 'ticker'],
                how='left'
            )
        
        all_features.append(tech_df)
    
    return pd.concat(all_features, ignore_index=True)


# =============================================================================
# Feature Selection & Importance
# =============================================================================

def get_feature_importance(model) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        return pd.DataFrame({
            'feature': range(len(importance)),
            'importance': importance
        }).sort_values('importance', ascending=False)
    return pd.DataFrame()


def select_top_features(X: pd.DataFrame, y: pd.Series, n_features: int = 20) -> list:
    """
    Select top features using correlation and importance.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    
    selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
    selector.fit(X.fillna(0), y)
    
    selected_indices = selector.get_support(indices=True)
    return X.columns[selected_indices].tolist()


# =============================================================================
# Data Loading from Multi-Database
# =============================================================================

def load_data_from_databases(db_type: str = 'postgresql') -> dict:
    """
    Load data from configured database.
    """
    load_db_env()
    db = DatabaseFactory.get_database(db_type)
    db.connect()
    
    data = {}
    
    try:
        # Load stock data
        data['stock'] = db.execute_query("SELECT * FROM fact_stock_prices ORDER BY date")
    except Exception as e:
        print(f"Could not load stock data: {e}")
        data['stock'] = pd.DataFrame()
    
    try:
        # Load macro data
        data['macro'] = db.execute_query("SELECT * FROM feat_macro_ratios")
    except Exception as e:
        print(f"Could not load macro data: {e}")
        data['macro'] = pd.DataFrame()
    
    try:
        # Load FMP data
        data['fmp'] = db.execute_query("SELECT * FROM feat_fmp_metrics")
    except Exception as e:
        print(f"Could not load FMP data: {e}")
        data['fmp'] = pd.DataFrame()
    
    db.close()
    return data


if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering module...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'date': dates,
        'ticker': ['AAPL'] * 100,
        'open': np.random.uniform(150, 180, 100),
        'high': np.random.uniform(155, 185, 100),
        'low': np.random.uniform(145, 175, 100),
        'close': np.random.uniform(150, 180, 100),
        'volume': np.random.uniform(1e6, 5e6, 100)
    })
    
    features = engineer_technical_features(sample_data, 'AAPL')
    print(f"Generated {len(features.columns)} features for {len(features)} rows")
    print(f"Features: {list(features.columns[:10])}...")