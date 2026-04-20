import pandas as pd

def load_to_postgres(stock_df, engine):
    print("Loading data into PostgreSQL...")

    # Check for existing records to avoid duplicates
    try:
        existing = pd.read_sql("SELECT ticker, date FROM raw_stock_prices", engine)
        existing['date'] = pd.to_datetime(existing['date'])
        stock_df['date']  = pd.to_datetime(stock_df['date'])

        stock_df = stock_df.merge(existing, on=["ticker", "date"], how="left", indicator=True)
        stock_df = stock_df[stock_df["_merge"] == "left_only"].drop(columns=["_merge"])
    except Exception:
        # Table doesn't exist yet — first load, just insert everything
        print("raw_stock_prices table does not exist yet. Loading all data.")

    if stock_df.empty:
        print("No new stock data to load.")
        return

    stock_df.to_sql(
        "raw_stock_prices",
        engine,
        if_exists="append",   # safe: never drops existing rows
        index=False,
    )

    print("Data successfully loaded into PostgreSQL.")