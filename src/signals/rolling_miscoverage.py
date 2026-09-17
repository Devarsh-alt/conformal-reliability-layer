import numpy as np
import pandas as pd


def rolling_miscoverage(covered_t, window=20):
    """
    Computes the rolling miscoverage rate over a specified window.

    covered_t: array of 1s (covered) and 0s (missed), in chronological order,
               e.g. the 'covered_t' output from conformal.run_aci()
    window: trailing window size in days

    Returns: numpy array, same length as covered_t, with NaN for the
             first (window - 1) entries where a full window isn't yet available.
    """
    covered_t = np.asarray(covered_t, dtype=float)
    miscovered_t = 1 - covered_t

    series = pd.Series(miscovered_t)
    rolling_rate = series.rolling(window=window, min_periods=window).mean()

    return rolling_rate.to_numpy()


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conformal import run_aci
    from config import BASE_DIR

    PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")
    df = pd.read_csv(os.path.join(PREDICTIONS_DIR, "AAPL.csv"))
    df = df.sort_values('date').reset_index(drop=True)

    aci_result = run_aci(df['actual'], df['predicted'], alpha_target=0.10, gamma=0.01, warmup=30)
    covered_t = aci_result['covered_t']

    roll_miscov = rolling_miscoverage(covered_t, window=20)

    print(f"Total points: {len(covered_t)}")
    print(f"Valid rolling values (non-NaN): {(~np.isnan(roll_miscov)).sum()}")
    print(f"Mean rolling miscoverage (ignoring NaN): {np.nanmean(roll_miscov):.4f}")
    print(f"Max rolling miscoverage: {np.nanmax(roll_miscov):.4f}")
    print(f"Min rolling miscoverage: {np.nanmin(roll_miscov):.4f}")
        # find where the max rolling miscoverage occurred
    dates_aligned = df['date'].iloc[30:].reset_index(drop=True)  # 30 = warmup used above
    max_idx = np.nanargmax(roll_miscov)
    window = 20  # matches the window used in rolling_miscoverage() above

    print(f"\nMax miscoverage (0.55) occurred in the 20-day window ending: {dates_aligned.iloc[max_idx]}")
    print(f"That window spans roughly: {dates_aligned.iloc[max_idx - window + 1]} to {dates_aligned.iloc[max_idx]}")