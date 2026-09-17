import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def rolling_ks_drift(scores, window=20, reference_window=20):
    """
    Rolling two-sample KS drift detector.

    At each time t (once enough history exists), compares:
      - reference sample: scores in [t - window - reference_window, t - window)
      - recent sample:     scores in [t - window, t)
    using a two-sample KS test. A large KS statistic (or small p-value)
    indicates the recent error distribution has shifted relative to the
    immediately preceding period -- i.e. distribution drift.

    scores: array-like of nonconformity scores, chronological order
    window: size of the "recent" comparison window
    reference_window: size of the "reference" comparison window

    Returns: dict with 'ks_stat' and 'ks_pvalue' arrays, same length as
             scores, NaN where not enough history exists yet.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    min_start = window + reference_window

    ks_stat = np.full(n, np.nan)
    ks_pvalue = np.full(n, np.nan)

    for t in range(min_start, n):
        reference = scores[t - window - reference_window: t - window]
        recent = scores[t - window: t]
        stat, pvalue = ks_2samp(reference, recent)
        ks_stat[t] = stat
        ks_pvalue[t] = pvalue

    return {'ks_stat': ks_stat, 'ks_pvalue': ks_pvalue}


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from conformal import nonconformity_scores
    from config import BASE_DIR

    PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")
    df = pd.read_csv(os.path.join(PREDICTIONS_DIR, "AAPL.csv"))
    df = df.sort_values('date').reset_index(drop=True)

    scores = nonconformity_scores(df['actual'], df['predicted'])
    result = rolling_ks_drift(scores, window=20, reference_window=20)

    ks_stat = result['ks_stat']
    valid = ~np.isnan(ks_stat)

    print(f"Total points: {len(scores)}")
    print(f"Valid KS values (non-NaN): {valid.sum()}")
    print(f"Mean KS stat: {np.nanmean(ks_stat):.4f}")
    print(f"Max KS stat: {np.nanmax(ks_stat):.4f}")

    max_idx = np.nanargmax(ks_stat)
    print(f"\nMax KS drift occurred at date: {df['date'].iloc[max_idx]}")
    print(f"p-value at that point: {result['ks_pvalue'][max_idx]:.6f}")
        # check how often the test fires as "significant" at common thresholds
    pvals = result['ks_pvalue']
    valid_pvals = pvals[~np.isnan(pvals)]

    for threshold in [0.05, 0.01, 0.001]:
        frac_significant = (valid_pvals < threshold).mean()
        print(f"Fraction of points with p < {threshold}: {frac_significant:.3f}")
        # compare volatility of reference vs recent windows at the two dates of interest
    def inspect_window(date_str, label):
        idx = df.index[df['date'] == date_str][0]
        ref = scores[idx - 40: idx - 20]
        rec = scores[idx - 20: idx]
        print(f"\n{label} ({date_str}):")
        print(f"  reference window std: {ref.std():.5f}, mean: {ref.mean():.5f}")
        print(f"  recent window std:    {rec.std():.5f}, mean: {rec.mean():.5f}")

    inspect_window("2016-09-06", "2016 event (max KS)")
    # find a COVID-window date to inspect from your rolling_miscoverage max
    inspect_window("2020-03-13", "COVID crash window")