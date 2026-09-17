import numpy as np
import pandas as pd


def forward_miscoverage(covered_t, horizons=(5, 10, 20)):
    """
    Computes forward-looking realized miscoverage rate(s) for each
    horizon in `horizons`.

    covered_t: array of 1s (covered) and 0s (missed), in chronological
               order -- e.g. the 'covered_t' output from conformal.run_aci()
    horizons: tuple of forward window lengths (in days) to compute

    For each day t, forward_miscoverage_Nd[t] = mean miscoverage rate
    over the NEXT N days (t+1 ... t+N), NOT including day t itself.
    The last N rows of the series will be NaN for horizon N, since
    there isn't a full future window to look into yet.

    Returns: dict mapping horizon -> numpy array (same length as
    covered_t), e.g. {5: array(...), 10: array(...), 20: array(...)}
    """
    covered_t = np.asarray(covered_t, dtype=float)
    miscovered_t = 1 - covered_t
    n = len(miscovered_t)

    series = pd.Series(miscovered_t)

    results = {}
    for N in horizons:
        # shift(-N) style: for each t, we want mean of t+1..t+N.
        # reverse the series, take a trailing rolling mean, reverse back,
        # then shift so day t sees days AFTER it, not including itself.
        reversed_series = series[::-1].reset_index(drop=True)
        reversed_roll = reversed_series.rolling(window=N, min_periods=N).mean()
        forward_roll = reversed_roll[::-1].reset_index(drop=True)
        # shift by -1 so day t's value excludes day t itself, only t+1..t+N
        forward_roll = forward_roll.shift(-1)
        results[N] = forward_roll.to_numpy()

    return results


def build_labels_for_ticker(covered_t, dates, horizons=(5, 10, 20)):
    """
    Convenience wrapper: builds a dataframe with 'date' plus one column
    per horizon, e.g. 'label_fwd_miscov_5d', 'label_fwd_miscov_10d', etc.

    covered_t: array of 1s/0s, chronological, aligned to `dates`
    dates: array-like of dates, same length as covered_t
    """
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    fwd = forward_miscoverage(covered_t, horizons=horizons)

    df = pd.DataFrame({'date': dates})
    for N, values in fwd.items():
        df[f'label_fwd_miscov_{N}d'] = values

    return df


def build_labels_all_tickers(tickers=None, warmup=30, horizons=(5, 10, 20), skip_tickers=("COIN",)):
    """
    Runs build_labels_for_ticker for every ticker (skipping any in
    skip_tickers), concatenates into one long dataframe, and saves it
    to data/labels/all_tickers_labels.csv.
    """
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conformal import run_aci
    from config import BASE_DIR, TICKERS

    if tickers is None:
        tickers = TICKERS

    LABELS_DIR = os.path.join(BASE_DIR, "..", "data", "labels")
    os.makedirs(LABELS_DIR, exist_ok=True)
    PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")

    all_rows = []
    for ticker in tickers:
        if ticker in skip_tickers:
            print(f"Skipping {ticker} (in skip_tickers)")
            continue

        file_path = os.path.join(PREDICTIONS_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"  FAILED: {ticker} — no predictions file")
            continue

        print(f"Building labels for {ticker}...")
        df = pd.read_csv(file_path)
        df = df.sort_values('date').reset_index(drop=True)

        aci_result = run_aci(df['actual'], df['predicted'], alpha_target=0.10, gamma=0.01, warmup=warmup)
        covered_t = aci_result['covered_t']
        dates = df['date'].iloc[warmup:].reset_index(drop=True)

        labels_df = build_labels_for_ticker(covered_t, dates, horizons=horizons)
        labels_df['ticker'] = ticker
        all_rows.append(labels_df)
        print(f"  -> {len(labels_df)} rows")

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(LABELS_DIR, "all_tickers_labels.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved combined labels to {out_path}")
    print(f"Total rows: {len(combined)} across {combined['ticker'].nunique()} tickers")

    return combined


if __name__ == "__main__":
    combined = build_labels_all_tickers()

    print("\n--- Per-ticker label summary ---")
    summary = combined.groupby('ticker').agg(
        n_rows=('date', 'count'),
        mean_5d=('label_fwd_miscov_5d', 'mean'),
        mean_10d=('label_fwd_miscov_10d', 'mean'),
        mean_20d=('label_fwd_miscov_20d', 'mean'),
    )
    pd.set_option('display.width', 120)
    print(summary.to_string())