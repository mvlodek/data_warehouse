#!/usr/bin/env python
"""Quick verification script - tests all imports and basic functionality."""

import sys
print("=" * 80)
print("Financial Data Warehouse - Dependency Verification")
print("=" * 80)

# Test 1: Core dependencies
print("\nTesting core dependencies...")
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("      ✓ Data processing (pandas, numpy, matplotlib, seaborn)")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 2: ML dependencies
print("\nTesting ML dependencies...")
try:
    import sklearn
    import xgboost
    import joblib
    print("      ✓ ML framework (scikit-learn, xgboost, joblib)")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 3: Database dependencies
print("\nTesting database dependencies...")
try:
    from sqlalchemy import create_engine
    import psycopg2
    print("      ✓ Database (sqlalchemy, psycopg2)")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 4: API dependencies
print("\nTesting API dependencies...")
try:
    import fastapi
    import uvicorn
    import pydantic
    print("      ✓ API framework (fastapi, uvicorn, pydantic)")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 5: Dashboard dependencies
print("\nTesting dashboard dependencies...")
try:
    import streamlit as st
    print("      ✓ Dashboard (streamlit)")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

# Test 6: ML module imports
print("\nTesting ML module imports...")
try:
    from ml.feature_engineering import engineer_technical_features
    from ml.feature_etl import populate_feature_store
    from ml.label_creation import create_and_load_labels
    from ml.training import assemble_training_dataset
    from ml.model_training import train_model, save_model, load_model
    from ml.inference import score_latest_data, save_predictions_to_db
    from ml.prediction_analysis import get_latest_prediction
    print("      ✓ All ML modules")
except ImportError as e:
    print(f"      ✗ Error: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL DEPENDENCIES INSTALLED SUCCESSFULLY")
print("=" * 80)
print("\nYou're ready to run: python main.py")
print("\n" + "=" * 80)
