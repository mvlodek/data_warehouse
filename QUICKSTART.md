# Smart Data Warehouse - Quick Start Guide

## What's New

Your financial data warehouse has been transformed into a **smart prediction system** that:
- 📊 Engineers 25+ domain features (technical, macro, fundamentals)
- 🤖 Trains an XGBoost ML model to predict buy/hold/sell decisions
- 📈 Scores new data daily with confidence scores
- 🎯 Provides actionable recommendations via dashboard + API

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Verify PostgreSQL Connection
Ensure `db.py` has correct credentials:
```python
create_engine("postgresql+psycopg2://postgres:87654321@localhost:5432/financial_data_warehouse")
```

### 3. Run Full Pipeline
```bash
python main.py
```

This will:
1. Fetch stock, macro, and financial data
2. Create feature store tables
3. Engineer 25+ features
4. Generate training labels (buy/hold/sell based on 30-day returns)
5. Train XGBoost classifier with 5-fold CV
6. Score latest data
7. Populate predictions table

**Expected Runtime**: 10-15 minutes

## Components

### Feature Store
**Location**: PostgreSQL tables
- `feat_technical_indicators` - SMA, RSI, MACD, ATR, volatility (+ more)
- `feat_macro_ratios` - Interest rate/inflation momentum
- `feat_company_fundamentals` - Revenue/EPS growth
- `feat_labels` - Training targets
- `fact_predictions` - Historical predictions

### Model Training
**File**: `ml/model_training.py`
- **Algorithm**: XGBoost Classifier (multiclass)
- **Classes**: BUY (>5% return), HOLD (-2% to +5%), SELL (<-2%)
- **Features**: 25 engineered features
- **Validation**: 5-fold stratified cross-validation

### Dashboard
**File**: `app.py`
```bash
streamlit run app.py
```

Shows:
- Traditional research Q1-Q5 (macro correlations)
- **NEW**: ML Predictions section with:
  - Latest recommendation + confidence
  - Probability distribution (BUY/HOLD/SELL)
  - 60-day confidence timeline
  - Recent prediction history table

### API Server
**File**: `ml/scoring_api.py`
```bash
python -m ml.scoring_api
# Navigate to http://localhost:8000/docs
```

Endpoints:
- `GET /api/predict/{ticker}` - Get latest prediction
- `GET /api/predictions/{ticker}?limit=30` - History
- `GET /api/predictions` - All latest predictions

## Daily Workflow

### Score Latest Data
```python
from ml.inference import score_latest_data, save_predictions_to_db
from ml.model_training import load_model
from db import get_engine

engine = get_engine()

# Load model
model, le, features = load_model("stock_predictor_20260411_120000")

# Score today's data
scores = score_latest_data(engine, ["AAPL", "MSFT", "GOOGL", "AMZN", "META"], model_name)

# Save to DB
save_predictions_to_db(scores, engine, model_name)
```

### Monitor Model Performance
```python
from ml.prediction_analysis import get_model_performance_metrics
metrics = get_model_performance_metrics()
# {'accuracy': 0.62, 'total_predictions': 245, ...}
```

## Key Files

| File | Purpose |
|------|---------|
| `ml/feature_engineering.py` | Compute technical, macro, fundamental features |
| `ml/feature_etl.py` | Populate feature store |
| `ml/label_creation.py` | Generate training labels |
| `ml/training.py` | Assemble training dataset |
| `ml/model_training.py` | Train/save XGBoost model |
| `ml/inference.py` | Score new data, generate predictions |
| `ml/prediction_analysis.py` | Dashboard visualization functions |
| `ml/scoring_api.py` | FastAPI prediction server |
| `ml/README.md` | Detailed technical documentation |
| `main.py` | Master pipeline script |
| `app.py` | Streamlit dashboard (updated) |

## Customization

### Change Label Thresholds
Edit `ml/label_creation.py`:
```python
BUY_THRESHOLD = 0.05      # Adjust buy threshold
SELL_THRESHOLD = -0.02    # Adjust sell threshold
```

### Add New Features
1. Implement in `feature_engineering.py`
2. Add to feature table
3. Include in `get_feature_names()` in `training.py`
4. Retrain model: `python main.py`

### Tune Hyperparameters
Edit `ml/model_training.py`:
```python
model = XGBClassifier(
    n_estimators=200,     # Try 300-500
    max_depth=6,          # Try 5-8
    learning_rate=0.1,    # Try 0.05-0.2
    # ... more parameters
)
```

### Change Train/Test Split Date
```python
train_df, test_df = assemble_training_dataset(
    engine, tickers, test_split_date='2023-01-01'
)
```

## Monitoring

### Check Feature Store Health
```sql
SELECT COUNT(*) FROM feat_technical_indicators;
SELECT label, COUNT(*) FROM feat_labels GROUP BY label;
SELECT ticker, COUNT(*) FROM feat_predictions GROUP BY ticker;
```

### Model Accuracy
```sql
SELECT 
    predicted_label,
    ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
FROM fact_predictions
WHERE actual_label IS NOT NULL
GROUP BY predicted_label;
```

### Latest Predictions
```sql
SELECT * FROM fact_predictions 
WHERE prediction_date = CURRENT_DATE 
ORDER BY confidence DESC;
```

## Troubleshooting

**Q: Why is accuracy low (~50%)?**
- Label distribution may be imbalanced (try class_weight='balanced')
- 30-day horizon may be too short (try 60 or 90 days)
- Add more predictive features

**Q: How often should I retrain?**
- Monthly or when accuracy drops >5%
- Always after adding new features

**Q: Can I use this for live trading?**
- Backtest thoroughly first
- Use confidence scores; skip predictions <60%
- Position size proportional to confidence
- Never trade on single indicator

**Q: Model training is slow**
- Reduce `n_estimators` in `model_training.py`
- Reduce date range in `main.py`
- Use GPU with `tree_method='gpu_hist'`

## Next Steps

1. **Verify setup**: `python main.py` (should complete in 10-15 min)
2. **View dashboard**: `streamlit run app.py`
3. **Check API**: `python -m ml.scoring_api` → http://localhost:8000/docs
4. **Analyze results**: Review prediction accuracy and feature importance
5. **Customize**: Adjust thresholds, add features, tune hyperparameters
6. **Deploy**: Schedule daily scoring job (cron/Task Scheduler)

## Resources

- **ML Module Docs**: `ml/README.md`
- **Feature Definitions**: `ml/feature_engineering.py`
- **Model Architecture**: `ml/model_training.py`
- **Prediction Functions**: `ml/inference.py`

---

**Questions?** Check `ml/README.md` for detailed technical documentation.
