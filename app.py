import streamlit as st
from analysis import interest_rate_impact, interest_rate_vs_stock_price, inflation_vs_volatility, earnings_vs_returns

st.markdown("### Exploring Relationships Between Macroeconomic Indicators and Stock Performance")

st.set_page_config(page_title="Financial Data Analysis", layout="wide")
st.write("---")

# Stock Trend
if st.button("Interest Rates vs Stock Price"):
    fig = interest_rate_vs_stock_price(ticker="AAPL")
    st.pyplot(fig)


# Interest Rate
if st.button("Interest Rate Impact"):
    fig, corr = interest_rate_impact(ticker="AAPL")
    st.pyplot(fig)
    st.write(f"Correlation: {corr:.2f}")

# Inflation
if st.button("Inflation vs Volatility"):
    fig = inflation_vs_volatility(ticker="AAPL")
    st.pyplot(fig)