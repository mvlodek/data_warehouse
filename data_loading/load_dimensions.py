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
