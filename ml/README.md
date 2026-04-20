# ML Module: Stock Purchase Prediction System

## Overview

This ML module transforms the financial data warehouse into a predictive system that recommends whether to **buy**, **hold**, or **sell** stocks based on historical patterns, technical indicators, and macroeconomic trends.

The system uses a **3-layer architecture**:
1. **Feature Store**: Engineered domain features (technical indicators, macro ratios, fundamentals)
2. **Training Pipeline**: Label generation, model training, hyperparameter tuning
3. **Inference Pipeline**: Daily scoring, prediction generation, API exposure

## Architecture

```
Raw Data (Stock/Macro/Financials)
         ↓
[Feature Engineering]
    - Technical Indicators (SMA, RSI, MACD)
    - Macro Aggregations (momentum, volatility)
    - Fundamentals (revenue growth, earnings momentum)
         ↓
[Feature Store] (PostgreSQL)
    - feat_technical_indicators
    - feat_macro_ratios
    - feat_company_fundamentals
    - feat_labels
         ↓
[Training] → [XGBoost Classifier] → [Saved Model]
    - 5-fold cross-validation
    - Multiclass classification (BUY/HOLD/SELL)
         ↓
[Inference Pipeline]
    - Score latest data
    - Generate probabilities
    - Track predictions (fact_predictions)
         ↓
[API + Dashboard]
    - FastAPI endpoint
    - Streamlit visualization
```

## Modules

### `feature_store_schema.py`
Creates PostgreSQL tables for feature storage:
- `feat_technical_indicators` - Moving averages, RSI, MACD, ATR, volatility
- `feat_macro_ratios` - Interest rate/inflation momentum and volatility
- `feat_company_fundamentals` - Revenue/EPS growth, TTM momentum
- `feat_labels` - Training targets (forward-looking returns)
- `fact_predictions` - Historical predictions and actuals

### `feature_engineering.py`
Technical indicator computation:

**Technical Features:**
- **Simple Moving Averages (SMA)**: 20-day, 50-day, 200-day
- **Relative Strength Index (RSI)**: 14-day momentum oscillator
- **MACD**: Momentum trend indicator (12/26/9)
- **Average True Range (ATR)**: 14-day volatility measure
- **Momentum**: 10-day and 20-day price change %
- **Volume SMA**: 20-day average volume
- **Volatility**: 20-day rolling standard deviation of returns

**Macro Features:**
- Interest rate levels and momentum (1-month, 3-month changes)
- Inflation (CPI) levels and momentum
- Rate/inflation volatility (20-day rolling std)

**Fundamentals Features:**
- Year-over-year revenue/EPS growth
- Trailing twelve-month momentum

### `feature_etl.py`
Populates feature store from raw data:
```python
populate_feature_store(stock_df, macro_df, fin_df, tickers, engine)
```

### `label_creation.py`
Creates training labels based on 30-day forward returns:

```
Label Definition (30-day forward return):
  BUY:  Return > 5%
  HOLD: -2% ≤ Return ≤ 5%
  SELL: Return < -2%
```

Label distribution is automatically tracked:
```python
create_and_load_labels(stock_df, tickers, engine)
```

### `training.py`
Assembles complete training dataset by joining features + labels:
```python
train_df, test_df = assemble_training_dataset(engine, tickers, test_split_date='2024-01-01')
```

**Data Quality Checks:**
- Forward-fill missing fundamentals
- Drop rows with remaining NaN values
- Validate label alignment

**Feature List** (25 total):
- 13 technical indicators
- 10 macro features
- 2 fundamentals

### `model_training.py`
XGBoost classifier training with cross-validation:

**Hyperparameters:**
- `n_estimators`: 200 trees
- `max_depth`: 6 levels
- `learning_rate`: 0.1
- `subsample`: 0.8
- `colsample_bytree`: 0.8

**Training Process:**
1. 5-fold stratified cross-validation
2. Train final model on all training data
3. Evaluate on holdout test set
4. Generate feature importance rankings
5. Save model, label encoder, features to disk

**Usage:**
```python
model, le, features, results = train_model(train_df, test_df)
save_model(model, le, features, model_name)
```

### `inference.py`
Scoring pipeline for generating predictions:

```python
# Score latest data
scores_df = score_latest_data(engine, tickers, model_name)

# Save to database
save_predictions_to_db(scores_df, engine, model_name)

# Get latest predictions
preds = get_latest_predictions(engine, ticker="AAPL")
```

**Output Columns:**
- `ticker`: Stock symbol
- `prediction_date`: Date of prediction
- `predicted_label`: BUY/HOLD/SELL
- `confidence`: Max probability across classes
- `predicted_probs_buy/hold/sell`: Class probabilities
- `model_version`: Model name for tracking

### `prediction_analysis.py`
Visualization functions for Streamlit dashboard:

```python
# Get latest prediction
pred = get_latest_prediction(ticker)

# Get history (30 most recent)
history = get_prediction_history(ticker, limit=30)

# Plot confidence timeline (60 days)
fig = plot_prediction_confidence_timeline(ticker)

# Plot class probabilities
fig = plot_prediction_probabilities(ticker)

# Model metrics
metrics = get_model_performance_metrics()
```

### `scoring_api.py`
FastAPI server for predictions:

**Endpoints:**
- `GET /health` - Health check
- `GET /api/predict/{ticker}` - Latest prediction for ticker
- `GET /api/predictions/{ticker}?limit=30` - Prediction history
- `GET /api/predictions` - Latest for all tickers

**Start Server:**
```bash
python -m ml.scoring_api
# API docs: http://localhost:8000/docs
```

## Workflow

### Initial Setup & Training

1. **Create Feature Store Schema:**
   ```python
   from ml.feature_store_schema import create_feature_store_schema
   create_feature_store_schema(engine)
   ```

2. **Populate Features:**
   ```python
   from ml.feature_etl import populate_feature_store
   populate_feature_store(stock_df, macro_df, fin_df, tickers, engine)
   ```

3. **Create Labels:**
   ```python
   from ml.label_creation import create_and_load_labels
   create_and_load_labels(stock_df, tickers, engine)
   ```

4. **Assemble Training Data:**
   ```python
   from ml.training import assemble_training_dataset
   train_df, test_df = assemble_training_dataset(engine, tickers)
   ```

5. **Train Model:**
   ```python
   from ml.model_training import train_model, save_model
   model, le, features, results = train_model(train_df, test_df)
   save_model(model, le, features, results['model_name'])
   ```

6. **Score Data:**
   ```python
   from ml.inference import score_latest_data, save_predictions_to_db
   scores = score_latest_data(engine, tickers, model_name)
   save_predictions_to_db(scores, engine, model_name)
   ```

**All-in-one via main.py:**
```bash
python main.py
```

### Daily Predictions

```python
from ml.inference import score_latest_data, save_predictions_to_db
from ml.model_training import load_model

# Load latest model
model, le, features = load_model("stock_predictor_20260411_120000")

# Score
scores = score_latest_data(engine, tickers, model_name)

# Save
save_predictions_to_db(scores, engine, model_name)
```

## Model Performance

Expected metrics (on test set):
- **Overall Accuracy**: 55-65% (multiclass, imbalanced data)
- **BUY Precision**: 60-70% (fewer false positives)
- **SELL Precision**: 55-65% (fewer false positives)
- **HOLD Recall**: High (largest class)

**Feature Importance** (typical top 5):
1. `close` - Current stock price
2. `sma_200` - Long-term trend
3. `rsi_14` - Momentum
4. `momentum_20` - 20-day return
5. `interest_rate_momentum_3m` - Macro trend

## Customization

### Change Label Thresholds
Edit `ml/label_creation.py`:
```python
BUY_THRESHOLD = 0.05      # >5% → BUY
SELL_THRESHOLD = -0.02    # <-2% → SELL
```

### Adjust Hyperparameters
Edit `ml/model_training.py`:
```python
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    # ... tune these
)
```

### Add Features
1. Add computation to `feature_engineering.py`
2. Add column to relevant feature table
3. Include in `get_feature_names()` in `training.py`
4. Retrain model

### Change Train/Test Split
```python
assemble_training_dataset(engine, tickers, test_split_date='2023-01-01')
```

## Production Considerations

- **Model Monitoring**: Track prediction accuracy weekly
- **Data Drift**: Monitor feature distributions over time
- **Retraining**: Retrain monthly or when accuracy drops >5%
- **Version Control**: Save all models with timestamps
- **Backtesting**: Compare predictions to actual returns
- **Risk Management**: Use confidence scores; avoid low-confidence trades

## Troubleshooting

**Empty Feature Store:**
```python
# Check if features were populated
pd.read_sql("SELECT COUNT(*) FROM feat_technical_indicators", engine)
```

**Training Data Too Small:**
```python
# Need at least 100 samples per class
pd.read_sql("SELECT label, COUNT(*) FROM feat_labels GROUP BY label", engine)
```

**Model Not Found:**
```bash
# List available models
ls ml/models/*.pkl
```

**Low Accuracy:**
- Add more features (sector, volatility, options data)
- Try different model (Random Forest, Gradient Boosting)
- Rebalance labels (use class_weight='balanced')
- Increase training data (extend date range)

## References

- Technical Indicators: https://school.stockcharts.com/
- XGBoost Documentation: https://xgboost.readthedocs.io/
- Feature Engineering: https://en.wikipedia.org/wiki/Feature_engineering
