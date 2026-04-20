import pandas as pd

def load_fact_table(stock_df, engine):
    print("Loading fact table...")

    existing_companies = pd.read_sql("SELECT ticker FROM dim_company", engine)

    # Merge to find new records
    stock_df = stock_df[stock_df['ticker'].isin(existing_companies['ticker'])]


    if stock_df.empty:
        print("No stock data to load.")
        return
    
     # Deduplicate against existing fact rows to avoid re-inserting on re-runs
    try:
        existing_facts = pd.read_sql("SELECT ticker, date FROM fact_stock_prices", engine)
        existing_facts["date"] = pd.to_datetime(existing_facts["date"])
        stock_df["date"] = pd.to_datetime(stock_df["date"])
 
        stock_df = stock_df.merge(existing_facts, on=["ticker", "date"], how="left", indicator=True)
        stock_df = stock_df[stock_df["_merge"] == "left_only"].drop(columns=["_merge"])

    except Exception:
        print("fact_stock_prices table does not exist yet. Loading all data.")

    print(f"Inserting {len(stock_df)} new rows...")

    if not stock_df.empty:
        stock_df.to_sql(
            "fact_stock_prices",
            engine,
            if_exists="append",
            index=False
        )
    else:
        print("No new rows to insert.")