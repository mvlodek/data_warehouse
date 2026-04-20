"""FastAPI scoring server for model predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from ml.inference import generate_scores_for_date, get_latest_predictions
from ml.model_training import load_model
from db import get_engine
import pandas as pd

app = FastAPI(
    title="Stock Prediction API",
    description="ML-powered stock buy/hold/sell prediction endpoint",
    version="1.0.0"
)

# Load model on startup
MODEL_NAME = None  # Will be set based on latest available model


class PredictionResponse(BaseModel):
    ticker: str
    prediction_date: str
    predicted_label: str
    confidence: float
    predicted_probs_buy: float
    predicted_probs_hold: float
    predicted_probs_sell: float
    model_version: str


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "stock-prediction-api"}


@app.get("/api/predict/{ticker}", response_model=PredictionResponse)
def predict_ticker(ticker: str, prediction_date: Optional[str] = None):
    """
    Get prediction for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., AAPL)
        prediction_date: Optional prediction date (default: latest)
    
    Returns:
        PredictionResponse with prediction details
    """
    engine = get_engine()
    
    # Get latest model if not specified
    global MODEL_NAME
    if MODEL_NAME is None:
        try:
            latest = get_latest_predictions(engine, ticker=ticker)
            if latest.empty:
                raise HTTPException(status_code=404, detail=f"No predictions for {ticker}")
            MODEL_NAME = latest.iloc[0]['model_version']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
    # Generate prediction
    try:
        result = generate_scores_for_date(engine, ticker, prediction_date or "latest", MODEL_NAME)
        
        if 'error' in result:
            raise HTTPException(status_code=404, detail=result['error'])
        
        return PredictionResponse(**result, model_version=MODEL_NAME)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predictions/{ticker}")
def get_ticker_predictions(ticker: str, limit: int = 30):
    """Get recent prediction history for a ticker."""
    engine = get_engine()
    
    query = f"""
        SELECT 
            prediction_date,
            predicted_label,
            confidence,
            predicted_probs_buy,
            predicted_probs_hold,
            predicted_probs_sell
        FROM fact_predictions
        WHERE ticker = '{ticker}'
        ORDER BY prediction_date DESC
        LIMIT {limit}
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No predictions found for {ticker}")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions")
def get_all_latest_predictions():
    """Get latest predictions for all tickers."""
    engine = get_engine()
    
    query = """
        SELECT 
            ticker,
            prediction_date,
            predicted_label,
            confidence,
            model_version
        FROM fact_predictions
        WHERE prediction_date = (SELECT MAX(prediction_date) FROM fact_predictions)
        ORDER BY confidence DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise HTTPException(status_code=404, detail="No predictions available")
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting Stock Prediction API...")
    print("Docs available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
