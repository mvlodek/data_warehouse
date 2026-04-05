import pandas as pd
from sqlalchemy import inspect

def load_macro_data(df, engine):
    print("Loading macroeconomic fact table...")

    inspector = inspect(engine)

    # Ensure df date is in datetime
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Remove duplicates already in DB
    existing = pd.read_sql("SELECT date, indicator FROM fact_macro_indicators", engine)

    existing['date'] = pd.to_datetime(existing['date']).dt.date

    df = df.merge(
        existing,
        on=["date", "indicator"],
        how="left",
        indicator=True
    )

    df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

    df.to_sql(
        "fact_macro_indicators",
        engine,
        if_exists="append",
        index=False
    )

    print("Macro data loaded successfully.")