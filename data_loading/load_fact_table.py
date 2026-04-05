import pandas as pd

def load_fact_table(stock_df, engine):
    print("Loading fact table...")

    existing_companies = pd.read_sql("SELECT ticker FROM dim_company", engine)

    # Merge to find new records
    merged_df = stock_df.merge(existing_companies, on="ticker", how="left", indicator=True)
    # Keep the new rows
    new_records = merged_df[merged_df["_merge"] == "left_only"].drop(columns=["_merge"])

    # stock_df = stock_df[stock_df['ticker'].isin(existing_companies['ticker'])]

    print(f"Inserting {len(new_records)} new rows...")

    if not new_records.empty:
        new_records.to_sql(
            "fact_stock_prices",
            engine,
            if_exists="append",
            index=False
        )
    else:
        print("No new rows to insert.")