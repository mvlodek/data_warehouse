def clean_stock_data(stock_data, ticker):
    print("Cleaning stock data...")
    # Drop rows with missing values
    stock_data = stock_data.dropna() 
    # Date will become a column
    stock_df = stock_data.reset_index() 
    # Flatten columns and convert to lowercase
    stock_df.columns = [
        col[0].lower() if isinstance(col, tuple) else str(col).lower()
        for col in stock_df.columns
    ]
    # Add ticker column
    stock_df['ticker'] = ticker

    return stock_df