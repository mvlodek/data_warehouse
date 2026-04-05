import requests
import pandas as pd

def fetch_financials(ticker, api_key):
    print(f"Fetching financial data for {ticker} from FMP...")
    # Get income statement data from FMP
    url = f"https://financialmodelingprep.com/stable/income-statement?symbol={ticker}&apikey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to fetch data for {ticker}. Status code: {response.status_code}")
        return None
    
    data = response.json()

    if not data:
        print(f"No financial data found for {ticker}.")
        return None

    # Convert to DataFrame
    financials_df = pd.DataFrame(data)

    # Keep only relevant columns and add ticker
    financials_df = financials_df[["date","revenue","netIncome","eps"]].copy()
    financials_df.rename(columns={"symbol": "ticker"}, inplace=True)
    financials_df["ticker"] = ticker
    print(f"Financial data fetched for {ticker}.")
    
    return financials_df