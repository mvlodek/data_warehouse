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

from ml.feature_store_schema import create_feature_store_schema
from ml.feature_etl import populate_feature_store
from ml.label_creation import create_and_load_labels
from ml.training import assemble_training_dataset
from ml.model_training import train_model, save_model
from ml.inference import score_latest_data, save_predictions_to_db

FRED_API_KEY = "4d2c22a44448b31bf6fa1a1d252895cb"
FMP_API_KEY = "w2zgYMCrNZ4jy27hNYtwAx6bFNc44mR8"

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
start_date = "2020-01-01"
end_date = "2024-01-01"

def main():
    engine = get_engine()

    # MACRO PIPELINE
    print("\n=== Loading macro data ===")
    macro_df = fetch_fred_data(FRED_API_KEY, start_date=start_date, end_date=end_date)
    macro_df = clean_macro_data(macro_df)
    load_macro_data(macro_df, engine)
    
    # Per ticker processing
    for ticker in tickers:
        print(f"\n=== Processing {ticker} ===")

        # STOCK PIPELINE
        stock_data = fetch_stock_data(ticker, start_date, end_date)
        cleaned_data = clean_stock_data(stock_data, ticker)

        load_to_postgres(cleaned_data, engine)
        load_dimensions(cleaned_data, engine)
        load_fact_table(cleaned_data, engine)


    # FINANCIALS PIPELINE
    financials_raw = fetch_financials(ticker, FMP_API_KEY)
    financials_clean = clean_financials(financials_raw, ticker)

    if financials_clean is not None:
        load_dimensions(financials_clean, engine)  # Ensure companies are in dim_company
        load_financials(financials_clean, engine)
    else:
        print("No financial data to load.")

    # ML PIPELINE: Feature Store + Training
    print("\n=== ML Pipeline ===")
    create_feature_store_schema(engine)
    
    # Fetch data for feature engineering
    stock_df = pd.read_sql("SELECT * FROM raw_stock_prices", engine)
    macro_df_full = pd.read_sql("SELECT * FROM fact_macro_indicators", engine)
    fin_df = pd.read_sql("SELECT * FROM fact_financials", engine)
    
    populate_feature_store(stock_df, macro_df_full, fin_df, tickers, engine)
    create_and_load_labels(stock_df, tickers, engine)
    
    # Train model
    train_df, test_df = assemble_training_dataset(engine, tickers, test_split_date='2024-01-01')
    
    if not train_df.empty:
        model, le, feature_names, results = train_model(train_df, test_df)
        model_name = results['model_name']
        save_model(model, le, feature_names, model_name)
        
        # Score latest data
        scores_df = score_latest_data(engine, tickers, model_name)
        if not scores_df.empty:
            save_predictions_to_db(scores_df, engine, model_name)
    else:
        print("⚠ Insufficient data to train model")

    print("\n✓ ML pipeline complete")


if __name__ == "__main__":
    main()