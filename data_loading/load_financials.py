import pandas as pd

def load_financials(financials_df, engine):
    print("Loading financial data...")

    if financials_df is None or financials_df.empty:
        print("No financial data to load.")
        return
    
    # Update symbol column to ticker for consistency
    financials_df.rename(columns={"symbol": "ticker"}, inplace=True)

    # Convert date columns to datetime for accurate merging
    financials_df['date'] = pd.to_datetime(financials_df['date'])
    
    # Filter to only include companies that exist in dim_company
    existing_companies = pd.read_sql("SELECT ticker FROM dim_company", engine)

    financials_df = financials_df[financials_df['ticker'].isin(existing_companies['ticker'])]

    # Remove duplicates based on ticker + date
    try:
        existing = pd.read_sql("SELECT ticker, date FROM fact_financials", engine)

        existing['date'] = pd.to_datetime(existing['date'])

        financials_df = financials_df.merge(existing,on=["ticker", "date"],how="left",indicator=True)

        financials_df = financials_df[financials_df["_merge"] == "left_only"].drop(columns=["_merge"])

    except:
        print("fact_financials table does not exist yet. Loading all data.")

    financials_df.to_sql("fact_financials",engine,if_exists="append",index=False)

    print("Financial data loaded successfully.")