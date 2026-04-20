"""Model training: XGBoost classifier for buy/hold/sell predictions."""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from ml.training import get_feature_names


MODEL_DIR = "ml/models"
os.makedirs(MODEL_DIR, exist_ok=True)


def _prepare_X(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Extract feature matrix and ensure all columns are float.
    Replaces inf/-inf with NaN, then fills remaining NaN with column median
    so XGBoost never receives non-numeric or infinite values.
    """
    X = df[feature_names].copy()

    # Force every column to numeric — catches any lingering object columns
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # Replace inf/-inf
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median (safer than 0 for financial ratios)
    X = X.fillna(X.median(numeric_only=True))

    return X


def train_model(train_df, test_df=None, model_name: str = None):
    """
    Train XGBoost classifier on training data.

    Returns:
        (model, le, feature_names, results_dict)
    """
    print("\n=== Training XGBoost Model ===")

    if model_name is None:
        model_name = f"stock_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    feature_names = get_feature_names()

    X_train = _prepare_X(train_df, feature_names)
    y_train = train_df['label'].copy()

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    print(f"Classes: {le.classes_}")
    print(f"Training samples: {len(X_train)}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
    )

    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train_encoded,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy',
    )

    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nTraining final model on full training set...")
    model.fit(X_train, y_train_encoded)

    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train_encoded, y_train_pred)
    print(f"Training Accuracy: {train_acc:.4f}")

    results = {
        'model_name':    model_name,
        'cv_scores':     cv_scores.tolist(),
        'cv_mean':       cv_scores.mean(),
        'cv_std':        cv_scores.std(),
        'train_accuracy': train_acc,
        'label_encoder': le,
        'feature_names': feature_names,
    }

    if test_df is not None and not test_df.empty:
        X_test = _prepare_X(test_df, feature_names)
        y_test = test_df['label'].copy()
        y_test_encoded = le.transform(y_test)

        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test_encoded, y_test_pred)

        print(f"\n=== Test Set Evaluation ===")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test_encoded, y_test_pred,
                                    target_names=le.classes_))

        results['test_accuracy'] = test_acc
        results['y_test_true']   = y_test_encoded
        results['y_test_pred']   = y_test_pred

    return model, le, feature_names, results


def save_model(model, le, feature_names, model_name: str):
    model_path    = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    le_path       = os.path.join(MODEL_DIR, f"{model_name}_le.pkl")
    features_path = os.path.join(MODEL_DIR, f"{model_name}_features.pkl")

    joblib.dump(model,         model_path)
    joblib.dump(le,            le_path)
    joblib.dump(feature_names, features_path)

    print(f"\n✓ Model saved:         {model_path}")
    print(f"✓ Label encoder saved: {le_path}")
    print(f"✓ Features saved:      {features_path}")

    return model_path, le_path, features_path


def load_model(model_name: str):
    model_path    = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    le_path       = os.path.join(MODEL_DIR, f"{model_name}_le.pkl")
    features_path = os.path.join(MODEL_DIR, f"{model_name}_features.pkl")

    model    = joblib.load(model_path)
    le       = joblib.load(le_path)
    features = joblib.load(features_path)

    return model, le, features


def plot_feature_importance(model, feature_names, top_k: int = 20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_k} Feature Importances')
    ax.invert_yaxis()
    return fig


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    return fig