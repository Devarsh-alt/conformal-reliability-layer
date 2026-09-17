import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from conformal import run_aci, nonconformity_scores
from features import get_all_features
from base_model import build_target, FEATURE_COLS
from config import BASE_DIR, TICKERS
from ks_drift import rolling_ks_drift
from rolling_miscoverage import rolling_miscoverage
from bootstrap_disagreement import run_bootstrap_disagreement_walk_forward

SIGNALS_DIR = os.path.join(BASE_DIR, "..", "data", "signals")
os.makedirs(SIGNALS_DIR, exist_ok=True)


def build_aligned_signals(ticker="AAPL", warmup=30, roll_window=20, n_bootstrap=10):
    predictions_dir = os.path.join(BASE_DIR, "..", "data", "predictions")
    pred_df = pd.read_csv(os.path.join(predictions_dir, f"{ticker}.csv"))
    pred_df = pred_df.sort_values('date').reset_index(drop=True)
    pred_df['date'] = pd.to_datetime(pred_df['date'])

    # --- rolling miscoverage (needs ACI's covered_t) ---
    aci_result = run_aci(pred_df['actual'], pred_df['predicted'], alpha_target=0.10, gamma=0.01, warmup=warmup)
    covered_t = aci_result['covered_t']
    roll_miscov = rolling_miscoverage(covered_t, window=roll_window)
    miscov_dates = pred_df['date'].iloc[warmup:].reset_index(drop=True)
    miscov_df = pd.DataFrame({'date': miscov_dates, 'miscoverage': roll_miscov})

    # --- KS drift ---
    scores = nonconformity_scores(pred_df['actual'], pred_df['predicted'])
    ks_result = rolling_ks_drift(scores, window=roll_window, reference_window=roll_window)
    ks_df = pd.DataFrame({'date': pred_df['date'], 'ks_stat': ks_result['ks_stat']})

    # --- bootstrap disagreement (needs raw features, not predictions CSV) ---
    features = get_all_features()
    feat_df = build_target(features[ticker])
    disagreement_df = run_bootstrap_disagreement_walk_forward(
        feat_df, FEATURE_COLS, n_bootstrap=n_bootstrap
    )
    disagreement_df['date'] = pd.to_datetime(disagreement_df['date'])
    disagreement_df = disagreement_df[['date', 'disagreement']]

    # --- merge all three on date ---
    merged = ks_df.merge(miscov_df, on='date', how='outer')
    merged = merged.merge(disagreement_df, on='date', how='outer')
    merged = merged.sort_values('date').reset_index(drop=True)
    merged['ticker'] = ticker

    return merged


def build_all_tickers(tickers=None, warmup=30, roll_window=20, n_bootstrap=10, skip_tickers=("COIN",)):
    """
    Runs build_aligned_signals for every ticker (skipping any in
    skip_tickers), concatenates into one long dataframe, and saves it
    to data/signals/all_tickers_signals.csv so this doesn't need to be
    recomputed every time.
    """
    if tickers is None:
        tickers = TICKERS

    all_rows = []
    for ticker in tickers:
        if ticker in skip_tickers:
            print(f"Skipping {ticker} (in skip_tickers)")
            continue
        print(f"Building signals for {ticker}...")
        try:
            merged = build_aligned_signals(ticker, warmup=warmup, roll_window=roll_window, n_bootstrap=n_bootstrap)
            all_rows.append(merged)
            print(f"  -> {len(merged)} rows")
        except Exception as e:
            print(f"  FAILED: {ticker} — {e}")

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(SIGNALS_DIR, "all_tickers_signals.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined signals to {out_path}")
    print(f"Total rows: {len(combined)} across {combined['ticker'].nunique()} tickers")

    return combined


if __name__ == "__main__":
    combined = build_all_tickers()

    print("\n--- Per-ticker summary ---")
    summary = combined.groupby('ticker').agg(
        n_rows=('date', 'count'),
        mean_ks=('ks_stat', 'mean'),
        mean_miscov=('miscoverage', 'mean'),
        mean_disagreement=('disagreement', 'mean'),
    )
    pd.set_option('display.width', 120)
    print(summary.to_string())


if __name__ == "__main__":
    merged = build_aligned_signals(ticker="AAPL")

    # window around the COVID miscoverage peak
    start = pd.Timestamp("2020-02-01")
    end = pd.Timestamp("2020-03-20")
    window_view = merged[(merged['date'] >= start) & (merged['date'] <= end)]

    pd.set_option('display.width', 120)
    pd.set_option('display.max_rows', None)
    print(window_view.to_string(index=False))