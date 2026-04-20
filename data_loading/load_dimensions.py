import pandas as pd

def load_dimensions(data_df, engine):
    print("Loading dimension tables...")

     # Normalize incoming dates
    data_df['date'] = pd.to_datetime(data_df['date']).dt.date

    # Get existing dates
    existing_dates = pd.read_sql("SELECT date FROM dim_date", engine)
    existing_dates['date'] = pd.to_datetime(existing_dates['date']).dt.date

    # Create date dimension
    date_df = data_df[['date']].drop_duplicates().copy()

    # Remove existing dates properly
    date_df = date_df[~date_df['date'].isin(existing_dates['date'])]

    # Add attributes
    date_df['year'] = pd.to_datetime(date_df['date']).dt.year
    date_df['month'] = pd.to_datetime(date_df['date']).dt.month
    date_df['day'] = pd.to_datetime(date_df['date']).dt.day

    # Insert only new rows
    if not date_df.empty:
        date_df.to_sql("dim_date", engine, if_exists="append", index=False)
        print(f"Inserted {len(date_df)} new dates into dim_date.")

    # Get existing companies
    if 'ticker' not in data_df.columns:
        return
    
    existing_companies = pd.read_sql("SELECT ticker FROM dim_company", engine)

    new_tickers = data_df['ticker'].drop_duplicates().copy()
    new_tickers = new_tickers[~new_tickers.isin(existing_companies['ticker'])]

    if not new_tickers.empty:
        new_tickers.to_sql("dim_company", engine, if_exists="append", index=False)
        print(f"Inserted {len(new_tickers)} new tickers into dim_company: {new_tickers.tolist()}")

    else:
        print("No new tickers to insert into dim_company.")