import yfinance as yf

def fetch_stock_data(ticker, start_date, end_date):

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print(f"Data fetched for {ticker}")
    return stock_data