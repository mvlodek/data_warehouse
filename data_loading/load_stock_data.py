
def load_to_postgres(stock_df, engine):
    print("Loading data into PostgreSQL...")

    stock_df.to_sql(
        "raw_stock_prices",
        engine,
        if_exists="replace",
        index=False
    )

    print("Data successfully loaded into PostgreSQL.")