import pandas as pd
import matplotlib.pyplot as plt
from db import get_engine

# Interest Rate Impact on Stock Prices
def interest_rate_impact(ticker="AAPL"):
    engine = get_engine()
    query = f"""
        SELECT s.date, s.close, m.value as interest_rate
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m
        ON DATE_TRUNC('month', s.date) = DATE_TRUNC('month', m.date)
        WHERE s.ticker = '{ticker}'
        AND m.indicator = 'INTEREST_RATE'
        ORDER BY s.date;
    """

    df = pd.read_sql(query, engine)
    df = df.dropna()

    corr = df["close"].corr(df["interest_rate"])
    print(f"Correlation (Interest Rate vs {ticker}):", corr)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(df["date"], df["close"], label="Stock Price")
    ax.plot(df["date"], df["interest_rate"], label="Interest Rate")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Interest Rate vs {ticker} (Correlation: {corr:.4f})")
    ax.grid(True)

    return fig

# Inflation vs Stock Volatility
def inflation_vs_volatility(ticker="AAPL"):
    engine = get_engine()
    query = f"""
        SELECT s.date, s.close, m.value as inflation
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m
        ON DATE_TRUNC('month', s.date) = DATE_TRUNC('month', m.date)
        WHERE s.ticker = '{ticker}'
        AND m.indicator = 'CPI'
        ORDER BY s.date;
    """

    df = pd.read_sql(query, engine)

    # Daily returns
    df["returns"] = df["close"].pct_change()

    # Rolling volatility (standard deviation)
    df["volatility"] = df["returns"].rolling(window=10).std()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(df["date"], df["volatility"], label="Volatility")
    ax.plot(df["date"], df["inflation"], label="Inflation")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_title("Inflation vs Stock Volatility")
    ax.grid(True)

    return fig

# Earnings vs Stock Returns
def earnings_vs_returns(ticker="AAPL"):
    engine = get_engine()

    stock_df = pd.read_sql(f"""
        SELECT date, close
        FROM fact_stock_prices
        WHERE ticker = '{ticker}';
    """, engine)

    fin_df = pd.read_sql(f"""
        SELECT date, revenue
        FROM fact_financials
        WHERE ticker = '{ticker}';
    """, engine)

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    fin_df["date"] = pd.to_datetime(fin_df["date"])

    # Convert stock to yearly data
    stock_df = stock_df.set_index("date").resample("YE").last().reset_index()

    # FMP fiscal year-end don't align with Dec. 31 --> merge on year instead
    # so fiscal-year financials match current stock year
    stock_df["year"] = stock_df["date"].dt.year
    fin_df["year"] = fin_df["date"].dt.year

    df = stock_df.merge(fin_df[["year", "revenue"]], on="year", how="inner")

    if df.empty or len(df) < 2:
        print(f"Not enough overlapping data for {ticker} earnings vs returns analysis.")
        return None
    
    df["returns"] = df["close"].pct_change()
    df["revenue_growth"] = df["revenue"].pct_change()
    df = df.dropna()

    if df.empty:
        return None

    corr = df["returns"].corr(df["revenue_growth"])
    print(f"Correlation (Earnings vs Returns): {corr:.4f}")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df["revenue_growth"], df["returns"])
    for _, row in df.iterrows():
        ax.annotate(row["year"], (row["revenue_growth"], row["returns"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    ax.set_xlabel("Revenue Growth")
    ax.set_ylabel("Stock Returns")
    ax.set_title(f"Earnings Growth vs Stock Returns - {ticker} (Correlation: {corr:.4f})")
    ax.grid(True)

    return fig

# Can we predict stock returns based on macro indicators?
def macro_prediction_signal(ticker="AAPL"):
    engine = get_engine()
    query = f"""
        SELECT s.date, s.close,
               m1.value as interest_rate,
               m2.value as inflation
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m1 
            ON DATE_TRUNC('month', s.date) = DATE_TRUNC('month', m1.date) AND m1.indicator = 'INTEREST_RATE'
        JOIN fact_macro_indicators m2 
            ON DATE_TRUNC('month', s.date) = DATE_TRUNC('month', m2.date) AND m2.indicator = 'CPI'
        WHERE s.ticker = '{ticker}';
    """

    df = pd.read_sql(query, engine)

    df["returns"] = df["close"].pct_change()
    df = df.dropna()

    corr1 = df["returns"].corr(df["interest_rate"])
    corr2 = df["returns"].corr(df["inflation"])

    print("Interest vs Returns:", f"{corr1:.4f}")
    print("Inflation vs Returns:", f"{corr2:.4f}")

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(df["interest_rate"], df["returns"], label="Interest Rate")
    ax.scatter(df["inflation"], df["returns"], label="Inflation")
    ax.set_xlabel("Macro Indicator")
    ax.set_ylabel("Daily Stock Returns")
    ax.set_title(f"Macro Indicators vs Stock Returns - {ticker}")
    ax.legend()
    ax.grid(True)

    return fig, corr1, corr2

if __name__ == "__main__":
    interest_rate_impact()
    inflation_vs_volatility()
    earnings_vs_returns()
    macro_prediction_signal()