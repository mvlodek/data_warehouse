"""
Enhanced model training with hyperparameter optimization and multi-database support.
Optimized for stock price prediction using macro indicators, FMP data, and stock APIs.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, 
    RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

from ml.feature_engineering_v2 import (
    create_all_features, select_top_features, get_feature_importance
)
from db_multi import DatabaseFactory, load_db_env


MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_features(df: pd.DataFrame, feature_names: list = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and labels for training.
    """
    if feature_names is None:
        # Auto-detect feature columns (exclude non-feature columns)
        exclude_cols = ['date', 'ticker', 'label', 'forward_return_30d', 'is_valid', 
                        'predicted_label', 'prediction_date', 'model_version']
        feature_names = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_names].copy()
    y = df['label'].copy() if 'label' in df.columns else None
    
    # Clean features
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle infinities
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))
    
    return X, y, feature_names


def encode_labels(y: pd.Series) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode labels to numeric format.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


def split_train_test(df: pd.DataFrame, test_date: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by date (time-series aware split).
    """
    df['date'] = pd.to_datetime(df['date'])
    test_date = pd.to_datetime(test_date)
    
    train_df = df[df['date'] < test_date].copy()
    test_df = df[df['date'] >= test_date].copy()
    
    return train_df, test_df


# =============================================================================
# Model Training
# =============================================================================

def get_base_models() -> Dict[str, Any]:
    """
    Get base models for ensemble.
    """
    return {
        'xgboost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    }


def train_single_model(X_train: pd.DataFrame, y_train: np.ndarray, 
                       model_name: str = 'xgboost') -> Tuple[Any, Dict]:
    """
    Train a single model with cross-validation.
    """
    print(f"\n=== Training {model_name.upper()} ===")
    
    models = get_base_models()
    model = models.get(model_name, models['xgboost'])
    
    # Time-series aware cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full data
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    results = {
        'model_name': model_name,
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': train_acc
    }
    
    return model, results


def train_ensemble(X_train: pd.DataFrame, y_train: np.ndarray,
                   voting: str = 'soft') -> Tuple[Any, Dict]:
    """
    Train ensemble model using voting classifier.
    """
    print("\n=== Training Ensemble Model ===")
    
    models = get_base_models()
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', models['xgboost']),
            ('rf', models['random_forest']),
            ('gb', models['gradient_boosting'])
        ],
        voting=voting,
        n_jobs=-1
    )
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring='accuracy')
    
    print(f"Ensemble CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train on full data
    ensemble.fit(X_train, y_train)
    
    train_pred = ensemble.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    
    results = {
        'model_name': 'ensemble',
        'cv_scores': cv_scores.tolist(),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_accuracy': train_acc,
        'voting_type': voting
    }
    
    return ensemble, results


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: np.ndarray,
                          n_iter: int = 20) -> Tuple[Any, Dict]:
    """
    Optimize XGBoost hyperparameters using randomized search.
    """
    print("\n=== Hyperparameter Tuning ===")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }
    
    base_model = XGBClassifier(
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {random_search.best_params_}")
    print(f"Best CV Score: {random_search.best_score_:.4f}")
    
    results = {
        'model_name': 'xgboost_tuned',
        'best_params': random_search.best_params_,
        'best_cv_score': random_search.best_score_
    }
    
    return random_search.best_estimator_, results


def train_model(train_df: pd.DataFrame, 
                test_df: pd.DataFrame = None,
                model_type: str = 'ensemble',
                tune_hyperparams: bool = False,
                feature_names: list = None) -> Tuple[Any, Dict]:
    """
    Main training function with multiple options.
    
    Args:
        train_df: Training data with features and labels
        test_df: Optional test data for evaluation
        model_type: 'xgboost', 'random_forest', 'gradient_boosting', 'ensemble', or 'tuned'
        tune_hyperparams: Whether to perform hyperparameter tuning
        feature_names: List of feature column names
    
    Returns:
        (model, label_encoder, feature_names, results_dict)
    """
    print("\n" + "="*60)
    print("MODEL TRAINING PIPELINE")
    print("="*60)
    
    # Prepare features
    X_train, y_train, feature_names = prepare_features(train_df, feature_names)
    y_train_encoded, le = encode_labels(y_train)
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Features: {len(feature_names)}")
    print(f"Classes: {le.classes_}")
    
    # Train model based on type
    if tune_hyperparams:
        model, results = hyperparameter_tuning(X_train, y_train_encoded)
    elif model_type == 'ensemble':
        model, results = train_ensemble(X_train, y_train_encoded)
    elif model_type == 'xgboost':
        model, results = train_single_model(X_train, y_train_encoded, 'xgboost')
    elif model_type == 'random_forest':
        model, results = train_single_model(X_train, y_train_encoded, 'random_forest')
    elif model_type == 'gradient_boosting':
        model, results = train_single_model(X_train, y_train_encoded, 'gradient_boosting')
    else:
        model, results = train_single_model(X_train, y_train_encoded, 'xgboost')
    
    # Evaluate on test set if provided
    if test_df is not None and not test_df.empty:
        X_test, y_test, _ = prepare_features(test_df, feature_names)
        y_test_encoded, _ = encode_labels(y_test)
        
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test_encoded, y_pred)
        
        results['test_accuracy'] = test_acc
        results['test_predictions'] = y_pred.tolist()
        
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    # Add metadata
    results['label_encoder_classes'] = le.classes_.tolist()
    results['feature_names'] = feature_names
    results['training_date'] = datetime.now().isoformat()
    
    return model, le, feature_names, results


# =============================================================================
# Model Persistence
# =============================================================================

def save_model(model, le: LabelEncoder, feature_names: list, 
               results: Dict, model_name: str = None) -> str:
    """
    Save trained model and metadata.
    """
    if model_name is None:
        model_name = f"stock_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.joblib")
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'label_encoder': le,
        'feature_names': feature_names,
        'results': results,
        'model_name': model_name
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"\n✓ Model saved to {model_path}")
    print(f"✓ Metadata saved to {metadata_path}")
    
    return model_name


def load_model(model_name: str) -> Tuple[Any, LabelEncoder, list, Dict]:
    """
    Load trained model and metadata.
    """
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.joblib")
    
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    return (
        model,
        metadata['label_encoder'],
        metadata['feature_names'],
        metadata['results']
    )


# =============================================================================
# Prediction
# =============================================================================

def predict(model, X: pd.DataFrame, le: LabelEncoder) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with confidence scores.
    """
    predictions = model.predict(X)
    probas = model.predict_proba(X)
    
    # Get confidence (max probability)
    confidence = np.max(probas, axis=1)
    
    # Decode predictions
    pred_labels = le.inverse_transform(predictions)
    
    return pred_labels, confidence


def predict_for_ticker(model, le: LabelEncoder, feature_names: list,
                       ticker: str, db_type: str = 'postgresql') -> Dict:
    """
    Make prediction for a specific ticker using database data.
    """
    load_db_env()
    db = DatabaseFactory.get_database(db_type)
    db.connect()
    
    # Get latest data
    query = f"""
        SELECT * FROM feat_technical_indicators 
        WHERE ticker = '{ticker}'
        ORDER BY date DESC
        LIMIT 1
    """
    
    df = db.execute_query(query)
    db.close()
    
    if df.empty:
        return {'error': f'No data for ticker {ticker}'}
    
    # Prepare features
    X = df[feature_names].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(X.median())
    
    # Predict
    pred_labels, confidence = predict(model, X, le)
    
    return {
        'ticker': ticker,
        'prediction': pred_labels[0],
        'confidence': float(confidence[0]),
        'date': df['date'].iloc[0].isoformat() if hasattr(df['date'].iloc[0], 'isoformat') else str(df['date'].iloc[0])
    }


# =============================================================================
# Model Evaluation & Visualization
# =============================================================================

def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """
    Plot feature importance.
    """
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'), dpi=150)
        plt.close()
        print("✓ Feature importance plot saved")


def plot_confusion_matrix(y_true, y_pred, le: LabelEncoder):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("✓ Confusion matrix plot saved")


# =============================================================================
# Main Training Pipeline
# =============================================================================

def run_training_pipeline(db_type: str = 'postgresql',
                          tickers: list = None,
                          model_type: str = 'ensemble',
                          tune_hyperparams: bool = False):
    """
    Complete training pipeline from database to saved model.
    """
    print("="*60)
    print("RUNNING TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    load_db_env()
    db = DatabaseFactory.get_database(db_type)
    db.connect()
    
    if tickers is None:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Query training data
    ticker_list = ','.join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT 
            t.*, m.*, l.label, l.forward_return_30d
        FROM feat_technical_indicators t
        LEFT JOIN feat_macro_ratios m ON t.ticker = m.ticker AND t.date = m.date
        LEFT JOIN feat_labels l ON t.ticker = l.ticker AND t.date = l.date
        WHERE t.ticker IN ({ticker_list})
        AND l.label IS NOT NULL
        ORDER BY t.date
    """
    
    train_df = db.execute_query(query)
    db.close()
    
    if train_df.empty:
        print("⚠ No training data found. Please populate the feature store first.")
        return None
    
    print(f"Loaded {len(train_df)} training samples")
    
    # Split train/test
    train_data, test_data = split_train_test(train_df, '2024-01-01')
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Train model
    model, le, feature_names, results = train_model(
        train_data,
        test_data,
        model_type=model_type,
        tune_hyperparams=tune_hyperparams
    )
    
    # Save model
    model_name = save_model(model, le, feature_names, results)
    
    # Generate plots
    if not test_data.empty:
        X_test, y_test, _ = prepare_features(test_data, feature_names)
        y_test_encoded, _ = encode_labels(y_test)
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test_encoded, y_pred, le)
    
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return model_name


if __name__ == "__main__":
    # Run training
    model_name = run_training_pipeline(model_type='ensemble')
    print(f"\nTrained model: {model_name}")