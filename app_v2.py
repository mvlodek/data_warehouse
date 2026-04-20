"""
Enhanced Streamlit Dashboard - Data Warehouse Frontend
Supports PostgreSQL, MySQL, and MongoDB for unified data access.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Database imports
from db_multi import DatabaseFactory, load_db_env
from ml.model_training_v2 import load_model, predict_for_ticker
from ml.feature_engineering_v2 import engineer_technical_features

# Page configuration
st.set_page_config(
    page_title="Financial Data Warehouse",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }
    
    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d2137 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #f0f6fc;
        margin: 0;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .db-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        margin-right: 8px;
    }
    
    .db-postgresql { background: #336791; color: white; }
    .db-mysql { background: #00758F; color: white; }
    .db-mongodb { background: #47A248; color: white; }
    
    .chart-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.4rem;
    }
    
    .prediction-buy { color: #22c55e; }
    .prediction-hold { color: #eab308; }
    .prediction-sell { color: #ef4444; }
    
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Database Connection Management
# =============================================================================

@st.cache_resource
def get_database(db_type: str):
    """Get database connection."""
    load_db_env()
    db = DatabaseFactory.get_database(db_type)
    try:
        db.connect()
        return db
    except Exception as e:
        st.error(f"Failed to connect to {db_type}: {e}")
        return None


def get_available_databases():
    """Check which databases are available."""
    available = {}
    
    for db_type in ['postgresql', 'mysql', 'mongodb']:
        try:
            db = get_database(db_type)
            if db:
                available[db_type] = True
                db.close()
            else:
                available[db_type] = False
        except:
            available[db_type] = False
    
    return available


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_stock_data(db, ticker: str = None, limit: int = 100):
    """Load stock price data."""
    try:
        if ticker:
            query = f"SELECT * FROM fact_stock_prices WHERE ticker = '{ticker}' ORDER BY date DESC LIMIT {limit}"
        else:
            query = f"SELECT * FROM fact_stock_prices ORDER BY date DESC LIMIT {limit}"
        return db.execute_query(query)
    except Exception as e:
        st.warning(f"Could not load stock data: {e}")
        return pd.DataFrame()


def load_macro_data(db):
    """Load macroeconomic data."""
    try:
        return db.execute_query("SELECT * FROM feat_macro_ratios ORDER BY date DESC LIMIT 100")
    except:
        return pd.DataFrame()


def load_predictions(db, ticker: str = None, limit: int = 30):
    """Load prediction history."""
    try:
        if ticker:
            query = f"SELECT * FROM fact_predictions WHERE ticker = '{ticker}' ORDER BY prediction_date DESC LIMIT {limit}"
        else:
            query = f"SELECT * FROM fact_predictions ORDER BY prediction_date DESC LIMIT {limit}"
        return db.execute_query(query)
    except:
        return pd.DataFrame()


def load_fmp_data(db, ticker: str = None):
    """Load FMP data."""
    try:
        if ticker:
            query = f"SELECT * FROM feat_fmp_metrics WHERE ticker = '{ticker}' ORDER BY date DESC"
        else:
            query = "SELECT * FROM feat_fmp_metrics ORDER BY date DESC LIMIT 100"
        return db.execute_query(query)
    except:
        return pd.DataFrame()


# =============================================================================
# Dashboard Components
# =============================================================================

def render_header():
    """Render dashboard header."""
    available_dbs = get_available_databases()
    
    db_badges = ""
    for db, is_available in available_dbs.items():
        status = "✓" if is_available else "✗"
        db_badges += f'<span class="db-badge db-{db}">{db.upper()} {status}</span>'
    
    st.markdown(f"""
    <div class="hero">
        <div style="margin-bottom: 0.8rem;">{db_badges}</div>
        <h1 class="hero-title">Financial Data Warehouse</h1>
        <p class="hero-subtitle">Unified access to PostgreSQL, MySQL, and MongoDB</p>
    </div>
    """, unsafe_allow_html=True)


def render_database_selector():
    """Database selector in sidebar."""
    st.sidebar.title("🔌 Database")
    
    db_type = st.sidebar.selectbox(
        "Select Database",
        ["postgresql", "mysql", "mongodb"],
        format_func=lambda x: x.upper()
    )
    
    return db_type


def render_ticker_selector(db):
    """Ticker selector with available stocks."""
    st.sidebar.title("📈 Stocks")
    
    try:
        # Try to get available tickers
        if db:
            try:
                tickers = db.execute_query("SELECT DISTINCT ticker FROM dim_company")['ticker'].tolist()
            except:
                tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
        else:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
    except:
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT']
    
    ticker = st.sidebar.selectbox("Select Ticker", tickers)
    return ticker


def render_stock_chart(df: pd.DataFrame, ticker: str):
    """Render stock price chart."""
    if df.empty:
        st.warning("No stock data available")
        return
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=ticker
    ))
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        template="plotly_dark",
        height=400,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_prediction_card(prediction: dict):
    """Render prediction result card."""
    pred = prediction.get('prediction', 'N/A')
    conf = prediction.get('confidence', 0)
    
    # Color based on prediction
    if pred == 'buy':
        color = '#22c55e'
        icon = '📈'
    elif pred == 'sell':
        color = '#ef4444'
        icon = '📉'
    else:
        color = '#eab308'
        icon = '➡️'
    
    st.markdown(f"""
    <div class="chart-card" style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 1.5rem; font-weight: 600; color: {color};">{pred.upper()}</div>
        <div style="font-size: 0.9rem; color: #8b949e;">Confidence: {conf:.1%}</div>
    </div>
    """, unsafe_allow_html=True)


def render_metrics(df: pd.DataFrame):
    """Render key metrics."""
    if df.empty:
        return
    
    cols = st.columns(4)
    
    # Latest price
    latest = df.iloc[0] if len(df) > 0 else None
    if latest is not None and 'close' in df.columns:
        cols[0].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${latest.get('close', 0):.2f}</div>
            <div class="metric-label">Latest Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Volume
    if 'volume' in df.columns:
        vol = latest.get('volume', 0) if latest else 0
        cols[1].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{vol/1e6:.1f}M</div>
            <div class="metric-label">Volume</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 30-day change
    if len(df) >= 30 and 'close' in df.columns:
        change = (df.iloc[0]['close'] - df.iloc[29]['close']) / df.iloc[29]['close'] * 100
        color = '#22c55e' if change > 0 else '#ef4444'
        cols[2].markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{change:+.1f}%</div>
            <div class="metric-label">30-Day Change</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data points
    cols[3].markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(df)}</div>
        <div class="metric-label">Data Points</div>
    </div>
    """, unsafe_allow_html=True)


def render_macro_indicators(df: pd.DataFrame):
    """Render macroeconomic indicators."""
    if df.empty:
        st.info("No macro data available")
        return
    
    # Get latest values
    latest = df.iloc[0]
    
    cols = st.columns(4)
    
    if 'interest_rate' in df.columns:
        cols[0].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latest.get('interest_rate', 0):.2f}%</div>
            <div class="metric-label">Interest Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    if 'inflation' in df.columns:
        cols[1].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latest.get('inflation', 0):.2f}%</div>
            <div class="metric-label">Inflation</div>
        </div>
        """, unsafe_allow_html=True)
    
    if 'gdp_growth' in df.columns:
        cols[2].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latest.get('gdp_growth', 0):.2f}%</div>
            <div class="metric-label">GDP Growth</div>
        </div>
        """, unsafe_allow_html=True)
    
    if 'unemployment' in df.columns:
        cols[3].markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{latest.get('unemployment', 0):.2f}%</div>
            <div class="metric-label">Unemployment</div>
        </div>
        """, unsafe_allow_html=True)


def render_prediction_history(df: pd.DataFrame):
    """Render prediction history chart."""
    if df.empty:
        st.info("No prediction history available")
        return
    
    fig = px.line(
        df,
        x='prediction_date',
        y='confidence',
        color='predicted_label',
        title="Prediction Confidence Over Time",
        template="plotly_dark"
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_fmp_data(df: pd.DataFrame):
    """Render FMP metrics."""
    if df.empty:
        st.info("No FMP data available")
        return
    
    # Display key metrics
    latest = df.iloc[0]
    
    metrics = []
    for col in ['pe_ratio', 'eps', 'market_cap', 'dividend_yield', 'beta']:
        if col in df.columns:
            metrics.append({
                'Metric': col.replace('_', ' ').title(),
                'Value': latest.get(col, 'N/A')
            })
    
    if metrics:
        st.table(pd.DataFrame(metrics))


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main dashboard application."""
    
    # Render header
    render_header()
    
    # Sidebar controls
    db_type = render_database_selector()
    db = get_database(db_type)
    
    if not db:
        st.error("No database connection available. Please check your configuration.")
        st.stop()
    
    ticker = render_ticker_selector(db)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Data", "🔮 Predictions", "📈 Macro", "🗄️ Data Warehouse"])
    
    with tab1:
        st.title("📊 Stock Data")
        
        # Load and display stock data
        stock_df = load_stock_data(db, ticker)
        
        if not stock_df.empty:
            render_metrics(stock_df)
            render_stock_chart(stock_df, ticker)
        else:
            st.warning(f"No stock data for {ticker}")
    
    with tab2:
        st.title("🔮 ML Predictions")
        
        # Try to load model and make prediction
        try:
            # For now, show mock prediction (replace with actual model loading)
            st.info("Loading ML model...")
            
            # Prediction history
            pred_df = load_predictions(db, ticker)
            if not pred_df.empty:
                render_prediction_history(pred_df)
            else:
                st.info("No prediction history available. Train a model first.")
        except Exception as e:
            st.warning(f"Could not load predictions: {e}")
    
    with tab3:
        st.title("📈 Macroeconomic Indicators")
        
        macro_df = load_macro_data(db)
        render_macro_indicators(macro_df)
        
        if not macro_df.empty:
            # Plot macro trends
            fig = px.line(
                macro_df,
                x='date',
                y=['interest_rate', 'inflation'],
                title="Interest Rate vs Inflation",
                template="plotly_dark"
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.title("🗄️ Data Warehouse")
        
        # Show database info
        st.subheader("Database Status")
        
        available = get_available_databases()
        for db_name, is_available in available.items():
            status = "✅ Connected" if is_available else "❌ Not Available"
            st.write(f"**{db_name.upper()}**: {status}")
        
        # Show tables/collections
        st.subheader("Available Data")
        
        if db_type == 'mongodb':
            collections = db.get_collections() if db else []
            st.write("**Collections:**")
            for coll in collections:
                st.write(f"  - {coll}")
        else:
            tables = db.get_tables() if db else []
            st.write("**Tables:**")
            for table in tables:
                st.write(f"  - {table}")
        
        # FMP Data section
        st.subheader("FMP Data")
        fmp_df = load_fmp_data(db, ticker)
        render_fmp_data(fmp_df)
    
    # Close database connection
    if db:
        db.close()


if __name__ == "__main__":
    main()