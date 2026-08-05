import os
import numpy as np
import pandas as pd
import yfinance as yf

from data import get_all_data
from config import TICKERS, START_DATE, END_DATE, CACHE_DIR, BASE_DIR

PROCESSED_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_vix():

    file_path = os.path.join(CACHE_DIR, "VIX.csv")
    if os.path.exists(file_path):
        vix = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    else:
        vix = yf.download("^VIX", start=START_DATE, end=END_DATE, interval="1d")
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        vix.index.name = "Date"
        vix.to_csv(file_path)
    return vix['Close'].rename("VIX")


def compute_features(data, vix):
    features = {}
    for ticker, df in data.items():
        df = df.copy()

        df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

        df['return_lag1'] = df['Log_Return'].shift(1)
        df['return_lag5'] = df['Log_Return'].shift(5)
        df['return_lag10'] = df['Log_Return'].shift(10)

        df['vol_10d'] = df['Log_Return'].rolling(window=10).std()
        df['vol_20d'] = df['Log_Return'].rolling(window=20).std()

        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Join VIX onto this ticker's dataframe by date
        df = df.join(vix, how='left')

        feature_cols = ['Log_Return', 'return_lag1', 'return_lag5', 'return_lag10',
                         'vol_10d', 'vol_20d', 'volume_ratio', 'VIX']
        df = df[feature_cols].dropna()

        features[ticker] = df
    return features


def get_all_features():
    all_cached = True
    features = {}
    for ticker in TICKERS:
        file_path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            features[ticker] = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        else:
            all_cached = False
            break

    if all_cached:
        return features

    raw_data = get_all_data()
    vix = get_vix()
    features = compute_features(raw_data, vix)

    for ticker, df in features.items():
        file_path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        df.to_csv(file_path)

    return features


if __name__ == "__main__":
    features = get_all_features()
    print(f"Computed features for {len(features)} / {len(TICKERS)} tickers")
    for ticker, df in features.items():
        print(f"{ticker}: {len(df)} rows, columns: {list(df.columns)}")