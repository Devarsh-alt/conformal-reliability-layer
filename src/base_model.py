import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import get_all_features
from config import TICKERS, BASE_DIR

PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# --- Walk-forward config (defaults — tune later with your supervisor) ---
INITIAL_TRAIN_DAYS = 252 * 5   # ~5 years of trading days
TEST_CHUNK_DAYS = 126          # ~6 months per fold
FEATURE_COLS = ['Log_Return', 'return_lag1', 'return_lag5', 'return_lag10',
                 'vol_10d', 'vol_20d', 'volume_ratio', 'VIX']


def build_target(df):
    """Target = next day's log return. Drops the last row (no 'tomorrow' available)."""
    df = df.copy()
    df['target'] = df['Log_Return'].shift(-1)
    df = df.dropna(subset=['target'])
    return df


def walk_forward_predict(df, ticker):
    """
    Expanding-window walk-forward training + prediction.
    Returns a dataframe of date, actual, predicted, fold_id.
    """
    results = []
    n = len(df)
    fold_id = 0
    train_end = INITIAL_TRAIN_DAYS

    if train_end >= n:
        print(f"WARNING: {ticker} has insufficient history ({n} rows) for initial train window ({INITIAL_TRAIN_DAYS}). Skipping.")
        return None

    while train_end < n:
        test_end = min(train_end + TEST_CHUNK_DAYS, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train, y_train = train_df[FEATURE_COLS], train_df['target']
        X_test, y_test = test_df[FEATURE_COLS], test_df['target']

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        fold_results = pd.DataFrame({
            'date': test_df.index,
            'actual': y_test.values,
            'predicted': preds,
            'fold_id': fold_id
        })
        results.append(fold_results)

        fold_id += 1
        train_end = test_end

    return pd.concat(results, ignore_index=True)


def run_base_model():
    features = get_all_features()
    all_results = {}

    for ticker in TICKERS:
        if ticker not in features:
            continue
        df = build_target(features[ticker])
        result = walk_forward_predict(df, ticker)
        if result is not None:
            all_results[ticker] = result
            file_path = os.path.join(PREDICTIONS_DIR, f"{ticker}.csv")
            result.to_csv(file_path, index=False)

    return all_results


if __name__ == "__main__":
    results = run_base_model()
    print(f"Base model predictions generated for {len(results)} / {len(TICKERS)} tickers")
    for ticker, df in results.items():
        n_folds = df['fold_id'].nunique()
        print(f"{ticker}: {len(df)} predictions across {n_folds} folds")