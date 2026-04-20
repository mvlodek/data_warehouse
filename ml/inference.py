"""Inference pipeline: Score new data and generate predictions."""

import pandas as pd
import numpy as np
from sqlalchemy import text
from ml.model_training import load_model


def generate_scores_for_date(engine, ticker: str, prediction_date: str, model_name: str):
    """
    Generate prediction scores for a specific ticker on a specific date.
    
    Args:
        engine: SQLAlchemy engine
        ticker: Stock ticker
        prediction_date: Date to make prediction for (YYYY-MM-DD)
        model_name: Name of saved model to use
    
    Returns:
        dict with prediction results
    """
    # Load model
    model, le, feature_names = load_model(model_name)
    
    # Query features for this ticker on this date
    query = """
        SELECT 
            t.ticker,
            t.date,
            {}
        FROM feat_technical_indicators t
        LEFT JOIN feat_macro_ratios m ON t.ticker = m.ticker AND t.date = m.date
        LEFT JOIN feat_company_fundamentals f ON t.ticker = f.ticker AND DATE_TRUNC('month', t.date)::date = DATE_TRUNC('month', f.date)::date
        WHERE t.ticker = %s AND t.date = %s
    """.format(','.join([f't.{col}' if col in ['close', 'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal', 'macd_diff', 'atr_14', 'momentum_10', 'momentum_20', 'volume_sma_20', 'volatility_20']
                       else f'm.{col}' if col in ['interest_rate', 'interest_rate_momentum_1m', 'interest_rate_momentum_3m', 'inflation', 'inflation_momentum_1m', 'inflation_momentum_3m', 'interest_rate_volatility_20d', 'inflation_volatility_20d']
                       else f'f.{col}'
                       for col in feature_names]))
    
    df = pd.read_sql(query, engine, params=(ticker, prediction_date))
    
    if df.empty:
        return {
            'ticker': ticker,
            'prediction_date': prediction_date,
            'error': 'No features found for this date'
        }
    
    # Prepare features
    X = df[feature_names].values.reshape(1, -1)
    
    # Generate predictions
    pred_label_encoded = model.predict(X)[0]
    pred_probs = model.predict_proba(X)[0]
    pred_label = le.inverse_transform([pred_label_encoded])[0]
    confidence = pred_probs.max()
    
    result = {
        'ticker': ticker,
        'prediction_date': prediction_date,
        'predicted_label': pred_label,
        'confidence': float(confidence),
        'predicted_probs_buy': float(pred_probs[list(le.classes_).index('BUY')] if 'BUY' in le.classes_ else 0),
        'predicted_probs_hold': float(pred_probs[list(le.classes_).index('HOLD')] if 'HOLD' in le.classes_ else 0),
        'predicted_probs_sell': float(pred_probs[list(le.classes_).index('SELL')] if 'SELL' in le.classes_ else 0),
    }
    
    return result


def score_latest_data(engine, tickers: list, model_name: str):
    """
    Score the most recent date for all tickers.
    
    Returns:
        DataFrame with predictions
    """
    print(f"\n=== Scoring Latest Data (Model: {model_name}) ===")
    
    # Get latest date from feature store
    query = "SELECT MAX(date) as latest_date FROM feat_technical_indicators"
    latest_date = pd.read_sql(query, engine).iloc[0, 0]
    
    if latest_date is None:
        print("No features found in feature store")
        return pd.DataFrame()
    
    print(f"Latest date in feature store: {latest_date}")
    
    # Score all tickers
    results = []
    for ticker in tickers:
        result = generate_scores_for_date(engine, ticker, str(latest_date), model_name)
        results.append(result)
    
    scores_df = pd.DataFrame(results)
    
    # Filter out errors
    scores_df = scores_df[~scores_df.get('error', pd.Series()).notna()].reset_index(drop=True)
    
    print(f"✓ Generated {len(scores_df)} predictions")
    
    return scores_df


def save_predictions_to_db(scores_df: pd.DataFrame, engine, model_name: str):
    """Save prediction scores to fact_predictions table."""
    print("Saving predictions to database...")
    
    # Prepare data
    predictions_df = scores_df.copy()
    predictions_df['model_version'] = model_name
    predictions_df['prediction_horizon_days'] = 30
    
    # Convert date
    predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date']).dt.date
    
    # Select columns for insert
    insert_cols = ['ticker', 'prediction_date', 'prediction_horizon_days', 'predicted_label',
                   'predicted_probs_buy', 'predicted_probs_hold', 'predicted_probs_sell',
                   'confidence', 'model_version']
    
    predictions_df = predictions_df[insert_cols]
    
    # Insert
    predictions_df.to_sql(
        'fact_predictions',
        engine,
        if_exists='append',
        index=False
    )
    
    print(f"✓ Saved {len(predictions_df)} predictions")


def get_latest_predictions(engine, ticker: str = None):
    """
    Get latest predictions for all tickers (or specific ticker).
    
    Returns:
        DataFrame with most recent predictions
    """
    query = """
        SELECT 
            ticker,
            prediction_date,
            predicted_label,
            confidence,
            predicted_probs_buy,
            predicted_probs_hold,
            predicted_probs_sell,
            model_version
        FROM fact_predictions
        WHERE prediction_date = (SELECT MAX(prediction_date) FROM fact_predictions)
    """
    
    if ticker:
        query += f" AND ticker = '{ticker}'"
    
    query += " ORDER BY confidence DESC"
    
    return pd.read_sql(query, engine)


if __name__ == "__main__":
    from db import get_engine
    print("Inference pipeline module loaded.")
