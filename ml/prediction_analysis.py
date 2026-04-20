"""Prediction analysis functions for dashboard visualization."""

import pandas as pd
import matplotlib.pyplot as plt
from db import get_engine


def get_latest_prediction(ticker: str):
    """
    Get the most recent prediction for a ticker.
    
    Returns:
        dict with prediction details or None if no prediction exists
    """
    engine = get_engine()
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
        WHERE ticker = %s
        ORDER BY prediction_date DESC
        LIMIT 1
    """
    
    df = pd.read_sql(query, engine, params=(ticker,))
    
    if df.empty:
        return None
    
    return df.iloc[0].to_dict()


def get_prediction_history(ticker: str, limit: int = 30):
    """
    Get prediction history for a ticker.
    
    Returns:
        DataFrame with recent predictions
    """
    engine = get_engine()
    query = """
        SELECT 
            prediction_date,
            predicted_label,
            confidence,
            predicted_probs_buy,
            predicted_probs_hold,
            predicted_probs_sell
        FROM fact_predictions
        WHERE ticker = %s
        ORDER BY prediction_date DESC
        LIMIT %s
    """
    
    return pd.read_sql(query, engine, params=(ticker, limit))


def plot_prediction_confidence_timeline(ticker: str, days: int = 60):
    """
    Plot confidence scores over time.
    
    Returns:
        matplotlib figure
    """
    engine = get_engine()
    query = """
        SELECT 
            prediction_date,
            predicted_label,
            confidence
        FROM fact_predictions
        WHERE ticker = %s
        AND prediction_date >= CURRENT_DATE - INTERVAL '%d days'
        ORDER BY prediction_date
    """ % (f"'{ticker}'", days)
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No prediction history for {ticker}", ha='center')
        return fig
    
    df['prediction_date'] = pd.to_datetime(df['prediction_date'])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot confidence with color by label
    colors = {'BUY': '#2ecc71', 'HOLD': '#f39c12', 'SELL': '#e74c3c'}
    for label in df['predicted_label'].unique():
        mask = df['predicted_label'] == label
        ax.scatter(df[mask]['prediction_date'], df[mask]['confidence'],
                  label=label, color=colors.get(label, '#95a5a6'), s=100, alpha=0.7)
    
    ax.plot(df['prediction_date'], df['confidence'], alpha=0.3, color='#34495e')
    ax.set_xlabel('Date')
    ax.set_ylabel('Confidence')
    ax.set_title(f'Prediction Confidence Timeline - {ticker}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    return fig


def plot_prediction_probabilities(ticker: str):
    """
    Plot probability distribution for latest prediction.
    
    Returns:
        matplotlib figure
    """
    pred = get_latest_prediction(ticker)
    
    if not pred:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No prediction for {ticker}", ha='center')
        return fig
    
    labels = ['BUY', 'HOLD', 'SELL']
    probs = [
        pred['predicted_probs_buy'],
        pred['predicted_probs_hold'],
        pred['predicted_probs_sell']
    ]
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Probability')
    ax.set_title(f'Latest Prediction Probabilities - {ticker}\n(Date: {pred["prediction_date"]})')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    return fig


def get_model_performance_metrics():
    """
    Get overall model performance metrics.
    
    Returns:
        dict with metrics
    """
    engine = get_engine()
    
    # Get predictions with actuals
    query = """
        SELECT 
            predicted_label,
            actual_label,
            is_correct
        FROM fact_predictions
        WHERE actual_label IS NOT NULL
    """
    
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return None
    
    total = len(df)
    correct = df['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0
    
    return {
        'total_predictions': total,
        'correct_predictions': correct,
        'accuracy': accuracy,
        'by_label': df.groupby('predicted_label')['is_correct'].agg(['sum', 'count', 'mean']).to_dict()
    }


if __name__ == "__main__":
    print("Prediction analysis module loaded.")
