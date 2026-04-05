import pandas as pd
import matplotlib.pyplot as plt
from db import get_engine


engine = get_engine()

# Interest Rate Impact on Stock Prices
def interest_rate_impact(ticker="AAPL"):
    query = f"""
        SELECT s.date, s.close, m.value as interest_rate
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m
        ON s.date = m.date
        WHERE s.ticker = '{ticker}'
        AND m.indicator = 'INTEREST_RATE'
        ORDER BY s.date;
    """

    df = pd.read_sql(query, engine)

    corr = df["close"].corr(df["interest_rate"])
    print(f"Correlation (Interest Rate vs {ticker}):", corr)

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["close"], label="Stock Price")
    ax.plot(df["date"], df["interest_rate"], label="Interest Rate")
    ax.legend()
    ax.title("Interest Rates vs Stock Price")
    ax.xticks(rotation=45)
    ax.set_title(f"Correlation: {corr:.2f}")
    
    return fig, corr

# Inflation vs Stock Volatility
def inflation_vs_volatility(ticker="AAPL"):
    query = f"""
        SELECT s.date, s.close, m.value as inflation
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m
        ON s.date = m.date
        WHERE s.ticker = '{ticker}'
        AND m.indicator = 'CPI'
        ORDER BY s.date;
    """

    df = pd.read_sql(query, engine)

    # Daily returns
    df["returns"] = df["close"].pct_change()

    # Rolling volatility (standard deviation)
    df["volatility"] = df["returns"].rolling(window=10).std()

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["volatility"], label="Volatility")
    ax.plot(df["date"], df["inflation"], label="Inflation")
    ax.legend()
    ax.title("Inflation vs Stock Volatility")
    ax.xticks(rotation=45)
    return fig

# Earnings vs Stock Returns
def earnings_vs_returns(ticker="AAPL"):
    stock_query = f"""
        SELECT date, close
        FROM fact_stock_prices
        WHERE ticker = '{ticker}';
    """

    fin_query = f"""
        SELECT date, revenue
        FROM fact_financials
        WHERE ticker = '{ticker}';
    """

    stock_df = pd.read_sql(stock_query, engine)
    fin_df = pd.read_sql(fin_query, engine)

    df = stock_df.merge(fin_df, on="date", how="inner")

    df["returns"] = df["close"].pct_change()
    df["revenue_growth"] = df["revenue"].pct_change()

    corr = df["returns"].corr(df["revenue_growth"])
    print("Correlation (Earnings vs Returns):", corr)

# Can we predict stock returns based on macro indicators?
def macro_prediction_signal(ticker="AAPL"):
    query = f"""
        SELECT s.date, s.close,
               m1.value as interest_rate,
               m2.value as inflation
        FROM fact_stock_prices s
        JOIN fact_macro_indicators m1 ON s.date = m1.date AND m1.indicator = 'INTEREST_RATE'
        JOIN fact_macro_indicators m2 ON s.date = m2.date AND m2.indicator = 'CPI'
        WHERE s.ticker = '{ticker}';
    """

    df = pd.read_sql(query, engine)

    df["returns"] = df["close"].pct_change()

    print("Interest vs Returns:", df["returns"].corr(df["interest_rate"]))
    print("Inflation vs Returns:", df["returns"].corr(df["inflation"]))


if __name__ == "__main__":
    interest_rate_impact()
    inflation_vs_volatility()
    earnings_vs_returns()
    macro_prediction_signal()