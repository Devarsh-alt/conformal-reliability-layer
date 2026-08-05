from config import TICKERS, START_DATE, END_DATE, CACHE_DIR
import os
import pandas as pd
import yfinance as yf

os.makedirs(CACHE_DIR, exist_ok=True)

def get_ticker_data(ticker):
    file_path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col='Date', parse_dates=True)
    try:
        data = yf.download(ticker, start=START_DATE, end=END_DATE, interval="1d")
        if data.empty:
            print(f"WARNING: no data for {ticker}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.index.name = "Date"
        data.to_csv(file_path)
        return data
    except Exception as e:
        print(f"FAILED: {ticker} — {e}")
        return None

def get_all_data():
    all_data = {}
    for ticker in TICKERS:
        df = get_ticker_data(ticker)
        if df is not None:
            all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    all_data = get_all_data()
    print(f"Fetched {len(all_data)} / {len(TICKERS)} tickers successfully")
    print(all_data)

