import pandas as pd

def clean_financials(df, ticker):
    print("Cleaning financial data...")

    if df is None or df.empty:
        print("No financial data to clean.")
        return None

    # Keep only what we need
    df = df[["date","revenue","netIncome","eps"]].copy()

    # Rename columns
    df.columns = ["date","revenue","net_income","eps"]

    # Convert date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Add ticker
    df["ticker"] = ticker

    # Drop nulls
    df = df.dropna()

    return df