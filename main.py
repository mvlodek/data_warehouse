from db import get_engine
import pandas as pd

from data_ingestion.stock_data import fetch_stock_data
from data_ingestion.fred_data import fetch_fred_data

from data_processing.clean_stock_data import clean_stock_data

from data_loading.load_stock_data import load_to_postgres
from data_loading.load_dimensions import load_dimensions
from data_loading.load_fact_table import load_fact_table
from data_loading.load_macro_data import load_macro_data

from data_ingestion.fred_data import fetch_fred_data
from data_processing.clean_fred_data import clean_macro_data
from data_loading.load_macro_data import load_macro_data

from data_ingestion.fmp_data import fetch_financials
from data_processing.clean_financials import clean_financials
from data_loading.load_financials import load_financials

API_KEY = "4d2c22a44448b31bf6fa1a1d252895cb"

def main():
    engine = get_engine()

    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2021-01-01"

    # STOCK PIPELINE
    stock_data = fetch_stock_data(ticker, start_date, end_date)
    cleaned_data = clean_stock_data(stock_data, ticker)

    load_to_postgres(cleaned_data, engine)
    load_dimensions(cleaned_data, engine)
    load_fact_table(cleaned_data, engine)

    # MACRO PIPELINE
    macro_df = fetch_fred_data(API_KEY, start_date=start_date, end_date=end_date)
    macro_df = clean_macro_data(macro_df)
    load_dimensions(macro_df, engine)
    load_macro_data(macro_df, engine)

    # FINANCIALS PIPELINE
    FMP_API_KEY = "w2zgYMCrNZ4jy27hNYtwAx6bFNc44mR8"
    financials_raw = fetch_financials(ticker, FMP_API_KEY)
    financials_clean = clean_financials(financials_raw, ticker)

    load_dimensions(financials_clean, engine)  # Ensure companies are in dim_company
    
    if financials_clean is not None:
        load_financials(financials_clean, engine)
    else:
        print("No financial data to load.")


    print("\nPreview:")
    print(cleaned_data.head())


if __name__ == "__main__":
    main()