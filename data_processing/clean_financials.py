import pandas as pd

def clean_financials(df, ticker):
    print("Cleaning financial data...")

    if df is None or df.empty:
        print("No financial data to clean.")
        return None

    # Keep only what we need
    df = df[["date","revenue","net_income","eps"]].copy()

    # Convert date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Ensure ticker is set correctly
    df["ticker"] = ticker

    # Drop nulls
    df = df.dropna()

    print(f"Cleaned {len(df)} financial records for {ticker}.")
    return df