# ML System - Quick Reference Card

## 🚀 Getting Started (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify connection to PostgreSQL
# Edit db.py if needed

# 3. Train model + score data (10-15 min)
python main.py

# 4. View dashboard
streamlit run app.py
# Open: http://localhost:8501

# 5. Try API
python -m ml.scoring_api
# Navigate to: http://localhost:8000/docs
```

---

## 📊 Core Workflow

### Full Pipeline (from main.py)
```python
from ml.feature_store_schema import create_feature_store_schema
from ml.feature_etl import populate_feature_store
from ml.label_creation import create_and_load_labels
from ml.training import assemble_training_dataset
from ml.model_training import train_model, save_model
from ml.inference import score_latest_data, save_predictions_to_db

# 1. Create tables
create_feature_store_schema(engine)

# 2. Engineer & load features
populate_feature_store(stock_df, macro_df, fin_df, tickers, engine)

# 3. Create labels
create_and_load_labels(stock_df, tickers, engine)

# 4. Assemble training data
train_df, test_df = assemble_training_dataset(engine, tickers)

# 5. Train model
model, le, features, results = train_model(train_df, test_df)
save_model(model, le, features, results['model_name'])

# 6. Score & save predictions
scores = score_latest_data(engine, tickers, model_name)
save_predictions_to_db(scores, engine, model_name)
```

---

## 🎯 Daily Predictions (60 seconds)

```python
from ml.inference import score_latest_data, save_predictions_to_db
from ml.model_training import load_model
from db import get_engine

engine = get_engine()
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Load latest model
model, le, features = load_model("stock_predictor_20260411_120000")

# Score today's data
scores = score_latest_data(engine, tickers, model_name)

# Save predictions
save_predictions_to_db(scores, engine, model_name)

print("✓ Daily predictions saved")
```

---

## 📈 Dashboard Integration

**Streamlit automatically displays:**
- Latest prediction card (color-coded: BUY/HOLD/SELL)
- Class probability chart
- 60-day confidence timeline
- Recent prediction history table

**No additional code needed** - runs when you open the app!

---

## 🔌 REST API Endpoints

```bash
# Get latest prediction
curl http://localhost:8000/api/predict/AAPL

# Get prediction history (last 10)
curl http://localhost:8000/api/predictions/AAPL?limit=10

# Get all latest predictions
curl http://localhost:8000/api/predictions

# Health check
curl http://localhost:8000/health
```

---

## 🛠️ Customization

### Change Label Thresholds
```python
# ml/label_creation.py
BUY_THRESHOLD = 0.05       # >5% = BUY (increase to 10% for fewer buy signals)
SELL_THRESHOLD = -0.02     # <-2% = SELL (increase to 0% for more sell signals)
```

### Tune Model Hyperparameters
```python
# ml/model_training.py
model = XGBClassifier(
    n_estimators=200,      # 100-500 (more = better fit, slower)
    max_depth=6,           # 4-8 (deeper = more complex)
    learning_rate=0.1,     # 0.01-0.3 (lower = slower learning)
    subsample=0.8,         # 0.5-1.0 (data sampling per tree)
    colsample_bytree=0.8,  # 0.5-1.0 (feature sampling)
)
```

### Add New Features
```python
# 1. In feature_engineering.py, add computation function
def compute_my_feature(series):
    return series.rolling(20).mean()

# 2. Update engineer_technical_features()
df['my_feature'] = compute_my_feature(df['close'])

# 3. In training.py, add to feature_names
return [
    'close', 'sma_20', 'my_feature',  # ← NEW
    # ... rest of features
]

# 4. Retrain: python main.py
```

### Change Train/Test Split Date
```python
# ml/training.py
train_df, test_df = assemble_training_dataset(
    engine, tickers, test_split_date='2023-06-01'  # ← Change this
)
```

---

## 📊 Monitoring & Diagnostics

### Check Feature Store Health
```sql
-- Features available?
SELECT COUNT(*) as technical_features 
FROM feat_technical_indicators;

-- Labels balanced?
SELECT label, COUNT(*) as count 
FROM feat_labels 
GROUP BY label;

-- Latest predictions?
SELECT * FROM fact_predictions 
WHERE prediction_date = CURRENT_DATE 
ORDER BY confidence DESC;
```

### Model Performance
```sql
-- Overall accuracy
SELECT 
    ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
FROM fact_predictions
WHERE actual_label IS NOT NULL;

-- Accuracy by class
SELECT 
    predicted_label,
    COUNT(*) as predictions,
    ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) as accuracy
FROM fact_predictions
WHERE actual_label IS NOT NULL
GROUP BY predicted_label;
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| **Connection refused** | Check PostgreSQL running, verify credentials in db.py |
| **Empty feature store** | Run `python main.py` to populate |
| **No predictions** | Ensure model trained first, check `ml/models/` directory |
| **API won't start** | Check port 8000 not in use, verify dependencies installed |
| **Low accuracy (~50%)** | Add features, change label thresholds, increase training data |
| **Streamlit crashes** | Install: `pip install streamlit --upgrade` |

---

## 📁 Key Files Cheat Sheet

| File | Purpose | Key Function |
|------|---------|--------------|
| `feature_engineering.py` | Compute features | `engineer_technical_features()` |
| `feature_etl.py` | Load to DB | `populate_feature_store()` |
| `label_creation.py` | Create targets | `create_and_load_labels()` |
| `training.py` | Assemble data | `assemble_training_dataset()` |
| `model_training.py` | Train XGBoost | `train_model()`, `save_model()` |
| `inference.py` | Score data | `score_latest_data()` |
| `prediction_analysis.py` | Visualize | `get_latest_prediction()` |
| `scoring_api.py` | REST API | `/api/predict/{ticker}` |

---

## 📚 Documentation

- **QUICKSTART.md** - Setup & daily usage (6,000 words)
- **ml/README.md** - Technical deep-dive (9,500 words)
- **IMPLEMENTATION_SUMMARY.md** - Full project overview

---

## 💡 Tips

- Use confidence scores to filter predictions (only trade >60% confidence)
- Backtest predictions before live trading
- Retrain monthly or when accuracy drops >5%
- Monitor feature store - empty data = empty predictions
- Version all models with timestamps
- Use `SELECT * FROM fact_predictions LIMIT 5;` to see latest

---

## 🎯 Common Tasks

### Retrain Model
```bash
python main.py  # Full pipeline, or:
```

```python
from ml.training import assemble_training_dataset
from ml.model_training import train_model, save_model

train_df, test_df = assemble_training_dataset(engine, tickers)
model, le, features, results = train_model(train_df, test_df)
save_model(model, le, features, results['model_name'])
```

### Get Prediction for Specific Ticker
```python
from ml.prediction_analysis import get_latest_prediction

pred = get_latest_prediction("AAPL")
print(f"{pred['predicted_label']} ({pred['confidence']:.0%})")
```

### View Recent Predictions
```python
from ml.prediction_analysis import get_prediction_history

history = get_prediction_history("AAPL", limit=30)
print(history)
```

### Check Model Accuracy
```python
from ml.prediction_analysis import get_model_performance_metrics

metrics = get_model_performance_metrics()
print(f"Overall Accuracy: {metrics['accuracy']:.1%}")
```

---

## ⚡ Performance Specs

- **Full pipeline**: 10-15 minutes
- **Daily scoring**: <1 minute
- **Model training**: 2-3 minutes (5-fold CV)
- **Feature computation**: 3-5 minutes
- **Expected accuracy**: 55-65%

---

**Ready to predict! Start with:** `python main.py` 🚀
