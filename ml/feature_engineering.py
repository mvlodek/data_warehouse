"""Feature engineering module: Technical indicators, macro ratios, and fundamentals."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return series.pct_change(periods=period)


def compute_volatility(returns: pd.Series, period: int = 20) -> pd.Series:
    return returns.rolling(window=period).std()


def engineer_technical_features(stock_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = stock_df[stock_df['ticker'] == ticker].copy().sort_values('date')
    df = df.reset_index(drop=True)

    df['sma_20']  = compute_sma(df['close'], 20)
    df['sma_50']  = compute_sma(df['close'], 50)
    df['sma_200'] = compute_sma(df['close'], 200)

    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_diff'] = compute_macd(df['close'])

    returns = df['close'].pct_change()
    df['atr_14']       = compute_atr(df['high'], df['low'], df['close'], 14)
    df['volatility_20'] = compute_volatility(returns, 20)

    df['momentum_10'] = compute_momentum(df['close'], 10)
    df['momentum_20'] = compute_momentum(df['close'], 20)

    df['volume_sma_20'] = compute_sma(df['volume'], 20)

    output_cols = ['date', 'ticker', 'close', 'sma_20', 'sma_50', 'sma_200',
                   'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'atr_14',
                   'momentum_10', 'momentum_20', 'volume_sma_20', 'volatility_20']

    return df[output_cols].copy()


def engineer_macro_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dates  = stock_df[stock_df['ticker'] == ticker]['date'].sort_values().reset_index(drop=True)
    output = pd.DataFrame({'date': dates, 'ticker': ticker})

    ir_df  = macro_df[macro_df['indicator'] == 'INTEREST_RATE'][['date', 'value']].copy()
    ir_df.columns = ['date', 'interest_rate']
    ir_df['date']  = pd.to_datetime(ir_df['date']).dt.normalize()

    inf_df = macro_df[macro_df['indicator'] == 'CPI'][['date', 'value']].copy()
    inf_df.columns = ['date', 'inflation']
    inf_df['date'] = pd.to_datetime(inf_df['date']).dt.normalize()

    ir_df  = ir_df.sort_values('date').set_index('date').asfreq('D').ffill().reset_index()
    inf_df = inf_df.sort_values('date').set_index('date').asfreq('D').ffill().reset_index()

    output['date'] = pd.to_datetime(output['date']).dt.normalize()
    output = output.merge(ir_df,  on='date', how='left')
    output = output.merge(inf_df, on='date', how='left')

    output['interest_rate_momentum_1m']  = output['interest_rate'].pct_change(periods=21)
    output['interest_rate_momentum_3m']  = output['interest_rate'].pct_change(periods=63)
    output['inflation_momentum_1m']      = output['inflation'].pct_change(periods=21)
    output['inflation_momentum_3m']      = output['inflation'].pct_change(periods=63)
    output['interest_rate_volatility_20d'] = output['interest_rate'].rolling(window=20).std()
    output['inflation_volatility_20d']     = output['inflation'].rolling(window=20).std()

    output_cols = ['date', 'ticker', 'interest_rate', 'interest_rate_momentum_1m',
                   'interest_rate_momentum_3m', 'inflation', 'inflation_momentum_1m',
                   'inflation_momentum_3m', 'interest_rate_volatility_20d',
                   'inflation_volatility_20d']

    return output[output_cols].copy()


def engineer_fundamentals_features(fin_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = fin_df[fin_df['ticker'] == ticker].copy().sort_values('date')

    if df.empty:
        return pd.DataFrame()

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    # FIX: groupby+transform+pct_change produces object dtype when a group has
    # only one row (NaN result keeps pandas in object mode). Use a plain
    # pct_change() on the sorted series instead — same result, always float.
    df['revenue_growth_yoy']    = df['revenue'].pct_change()
    df['eps_growth_yoy']        = df['eps'].pct_change()
    df['net_income_growth_yoy'] = df['net_income'].pct_change()
    df['revenue_momentum_ttm']  = df['revenue'].pct_change(periods=4)
    df['earnings_momentum_ttm'] = df['eps'].pct_change(periods=4)

    # Force all five columns to float and replace inf/-inf with NaN so
    # XGBoost never receives object or infinite values.
    growth_cols = [
        'revenue_growth_yoy', 'eps_growth_yoy', 'net_income_growth_yoy',
        'revenue_momentum_ttm', 'earnings_momentum_ttm',
    ]
    for col in growth_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    output_cols = ['date', 'ticker'] + growth_cols
    return df[output_cols].copy()


if __name__ == "__main__":
    print("Feature engineering module loaded.")