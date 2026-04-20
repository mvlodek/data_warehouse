import streamlit as st

# ── MUST be the very first Streamlit call ──────────────────────────────────────
st.set_page_config(
    page_title="Financial Data Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from analysis import (
    interest_rate_impact,
    inflation_vs_volatility,
    earnings_vs_returns,
    macro_prediction_signal,
)
from ml.prediction_analysis import (
    get_latest_prediction,
    get_prediction_history,
    plot_prediction_confidence_timeline,
    plot_prediction_probabilities,
)

# ── Custom CSS – dark financial terminal aesthetic ─────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d2137 100%);
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #f0f6fc;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin: 0;
        font-weight: 300;
    }
    .badge {
        display: inline-block;
        background: rgba(56,189,248,0.15);
        color: #38bdf8;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 10px;
        border-radius: 20px;
        border: 1px solid rgba(56,189,248,0.3);
        margin-bottom: 0.8rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* Ticker selector label */
    label[data-testid="stSelectboxLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        color: #38bdf8 !important;
    }

    /* Section headings */
    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #38bdf8;
        margin: 0 0 0.5rem 0;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f0f6fc;
        margin: 0 0 1rem 0;
    }

    /* Card wrapper for each chart */
    .chart-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.4rem 1.6rem 1rem;
    }

    /* Correlation metric pills */
    .metric-row {
        display: flex;
        gap: 0.8rem;
        margin-top: 0.8rem;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: #0d2137;
        border: 1px solid rgba(56,189,248,0.25);
        border-radius: 6px;
        padding: 0.45rem 0.9rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #e6edf3;
    }
    .metric-pill span {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Warning box */
    .warn-box {
        background: rgba(230,162,60,0.1);
        border: 1px solid rgba(230,162,60,0.3);
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.85rem;
        color: #e6a23c;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}

    /* Divider */
    hr {border: none; border-top: 1px solid #21262d; margin: 2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero header 
st.markdown(
    """
    <div class="hero">
        <div class="badge">Financial Analytics Platform</div>
        <p class="hero-title">Market & Macro Intelligence</p>
        <p class="hero-subtitle">
            Exploring relationships between macroeconomic indicators and stock market performance
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Ticker selector ────────────────────────────────────────────────────────────
ticker = st.selectbox(
    "Active Ticker",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    help="All four analyses update when you change the ticker.",
)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 1: Interest Rate Impact  |  Inflation vs Volatility ───────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p class="section-header">Research Q1</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Interest Rate Impact on Stock Price</p>', unsafe_allow_html=True)
    with st.spinner("Loading interest rate data…"):
        fig_ir = interest_rate_impact(ticker)
    st.pyplot(fig_ir, use_container_width=True)

with col2:
    st.markdown('<p class="section-header">Research Q2</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Inflation vs Stock Volatility</p>', unsafe_allow_html=True)
    with st.spinner("Loading inflation data…"):
        fig_vol = inflation_vs_volatility(ticker)
    st.pyplot(fig_vol, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Row 2: Earnings vs Returns  |  Macro Prediction Signal ────────────────────
col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown('<p class="section-header">Research Q3</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Earnings Growth vs Stock Returns</p>', unsafe_allow_html=True)
    with st.spinner("Loading earnings data…"):
        fig_earn = earnings_vs_returns(ticker=ticker)
    if fig_earn:
        st.pyplot(fig_earn, use_container_width=True)
    else:
        st.markdown(
            '<div class="warn-box">⚠ Insufficient overlapping data for earnings vs returns.</div>',
            unsafe_allow_html=True,
        )

with col4:
    st.markdown('<p class="section-header">Research Q5</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Macro Indicators as Return Predictors</p>', unsafe_allow_html=True)
    with st.spinner("Computing macro signals…"):
        fig_macro, corr_ir, corr_inf = macro_prediction_signal(ticker=ticker)
    st.pyplot(fig_macro, use_container_width=True)

    # Correlation metrics with color-coded signs
    ir_sign  = "+" if corr_ir  >= 0 else ""
    inf_sign = "+" if corr_inf >= 0 else ""
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-pill">
                Interest Rate Corr &nbsp;<span>{ir_sign}{corr_ir:.4f}</span>
            </div>
            <div class="metric-pill">
                Inflation Corr &nbsp;<span>{inf_sign}{corr_inf:.4f}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;color:#484f58;text-align:center;">'
    "Financial Data Warehouse &amp; Market Analytics Platform · Matt Vlodek · March 2026"
    "</p>",
    unsafe_allow_html=True,
)

# ── ML PREDICTIONS SECTION ────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero">
        <div class="badge">ML Predictions</div>
        <p class="hero-title">Stock Purchase Recommendations</p>
        <p class="hero-subtitle">
            Machine learning model predictions for buy/hold/sell decisions (30-day horizon)
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Get latest prediction
prediction = get_latest_prediction(ticker)

if prediction:
    col_pred1, col_pred2 = st.columns(2, gap="large")
    
    with col_pred1:
        st.markdown('<p class="section-header">Latest Prediction</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Recommendation for 30-Day Horizon</p>', unsafe_allow_html=True)
        
        # Color-coded recommendation
        pred_label = prediction['predicted_label']
        confidence = prediction['confidence']
        
        color_map = {'BUY': '#2ecc71', 'HOLD': '#f39c12', 'SELL': '#e74c3c'}
        color = color_map.get(pred_label, '#95a5a6')
        
        st.markdown(
            f"""
            <div style="background: {color}; padding: 2rem; border-radius: 10px; text-align: center; margin: 1rem 0;">
                <p style="font-size: 0.9rem; color: #fff; margin: 0;">RECOMMENDATION</p>
                <p style="font-size: 2.5rem; font-weight: bold; color: #fff; margin: 0.5rem 0;">{pred_label}</p>
                <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0;">Confidence: {confidence:.1%}</p>
                <p style="font-size: 0.8rem; color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0;">
                    Prediction Date: {prediction['prediction_date']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col_pred2:
        st.markdown('<p class="section-header">Prediction Probabilities</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Model Confidence by Class</p>', unsafe_allow_html=True)
        
        with st.spinner("Rendering probabilities..."):
            fig_probs = plot_prediction_probabilities(ticker)
        st.pyplot(fig_probs, use_container_width=True)
    
    # Confidence timeline
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Prediction History</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Confidence & Recommendation Trends (60 Days)</p>', unsafe_allow_html=True)
    
    with st.spinner("Loading confidence timeline..."):
        fig_timeline = plot_prediction_confidence_timeline(ticker, days=60)
    st.pyplot(fig_timeline, use_container_width=True)
    
    # Recent prediction history table
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Recent Predictions</p>', unsafe_allow_html=True)
    
    history = get_prediction_history(ticker, limit=10)
    if not history.empty:
        history_display = history.copy()
        history_display['predicted_probs_buy'] = history_display['predicted_probs_buy'].apply(lambda x: f"{x:.1%}")
        history_display['predicted_probs_hold'] = history_display['predicted_probs_hold'].apply(lambda x: f"{x:.1%}")
        history_display['predicted_probs_sell'] = history_display['predicted_probs_sell'].apply(lambda x: f"{x:.1%}")
        history_display['confidence'] = history_display['confidence'].apply(lambda x: f"{x:.1%}")
        
        st.dataframe(history_display, use_container_width=True, hide_index=True)
else:
    st.warning(f"⚠ No predictions available for {ticker}. Run the ML pipeline to generate predictions.")