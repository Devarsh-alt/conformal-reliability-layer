import numpy as np
import pandas as pd


def _causal_threshold(series, percentile, min_periods):
    """
    Computes an expanding, causal percentile threshold for each point in
    the series: threshold(t) is based only on values strictly before t
    (shift(1) before expanding), so there's no look-ahead leakage.

    Returns a pd.Series of thresholds, same length/index as input,
    with NaN wherever there isn't yet enough history (< min_periods).
    """
    series = pd.Series(np.asarray(series, dtype=float))
    q = percentile / 100.0
    return series.shift(1).expanding(min_periods=min_periods).quantile(q)


def find_event_onsets(series, dates, percentile=90, merge_gap_days=10, min_periods=252):
    """
    Finds 'onset' dates: the first day a series crosses above its OWN
    causal (expanding, look-ahead-free) percentile threshold, after
    having been below it. Nearby onsets (within merge_gap_days) are
    merged into a single event, keeping the earliest onset date.

    min_periods: minimum history (in rows) required before a threshold
    is computed at all. Defaults to ~1 trading year (252). Days before
    this warm-up period can never register as events.

    Returns: list of onset dates (pd.Timestamp)
    """
    series = pd.Series(np.asarray(series, dtype=float))
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)

    threshold = _causal_threshold(series, percentile, min_periods)
    above = (series > threshold).fillna(False)
    rising_edge = above & ~above.shift(1, fill_value=False)

    onset_dates = dates[rising_edge].tolist()

    merged = []
    for d in onset_dates:
        if merged and (d - merged[-1]).days <= merge_gap_days:
            continue  # part of the same event as the previous onset
        merged.append(d)

    return merged


def compute_lead_times(event_onsets, signal_series, signal_dates,
                        signal_percentile=90, max_lookback_days=30, min_periods=252):
    """
    For each event onset date, finds the earliest date within
    max_lookback_days beforehand where signal_series first crossed its
    OWN causal (expanding) threshold. Lead time is in days
    (positive = signal led the event).

    Returns: list of dicts, one per event:
             {event_date, signal_crossing_date, lead_days}
             lead_days is None if the signal never crossed within the
             lookback window (or didn't have enough history yet).
    """
    signal_series = pd.Series(np.asarray(signal_series, dtype=float))
    signal_dates = pd.to_datetime(pd.Series(signal_dates)).reset_index(drop=True)

    threshold = _causal_threshold(signal_series, signal_percentile, min_periods)
    above = (signal_series > threshold).fillna(False)

    results = []
    for event_date in event_onsets:
        window_start = event_date - pd.Timedelta(days=max_lookback_days)
        mask = (signal_dates >= window_start) & (signal_dates <= event_date) & above
        candidate_dates = signal_dates[mask]

        if len(candidate_dates) == 0:
            results.append({'event_date': event_date, 'signal_crossing_date': None, 'lead_days': None})
        else:
            first_crossing = candidate_dates.min()
            lead_days = (event_date - first_crossing).days
            results.append({'event_date': event_date, 'signal_crossing_date': first_crossing, 'lead_days': lead_days})

    return results


def summarize(results, label):
    """Reports hit rate and mean lead time TOGETHER, since one without
    the other is misleading (a signal that leads by a lot on a third
    of events isn't directly comparable to one that leads by a little
    on nearly all of them)."""
    total = len(results)
    valid = [r['lead_days'] for r in results if r['lead_days'] is not None]
    hit_rate = len(valid) / total if total else float('nan')
    mean_lead = np.mean(valid) if valid else float('nan')
    print(f"{label}: hit rate {len(valid)}/{total} ({hit_rate:.0%}), "
          f"mean lead {mean_lead:.1f} days (over detected events only)")
    return hit_rate, mean_lead


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from compare_signals import build_aligned_signals

    merged = build_aligned_signals(ticker="AAPL")

    miscov_onsets = find_event_onsets(merged['miscoverage'], merged['date'],
                                       percentile=90, merge_gap_days=10, min_periods=252)
    print(f"Found {len(miscov_onsets)} miscoverage events for AAPL (causal thresholds, 1yr warm-up)")
    for d in miscov_onsets:
        print(f"  {d.date()}")

    print("\n--- KS-stat lead times ---")
    ks_leads = compute_lead_times(miscov_onsets, merged['ks_stat'], merged['date'],
                                    signal_percentile=90, max_lookback_days=30, min_periods=252)
    for r in ks_leads:
        print(r)

    print("\n--- Disagreement lead times ---")
    dis_leads = compute_lead_times(miscov_onsets, merged['disagreement'], merged['date'],
                                     signal_percentile=90, max_lookback_days=30, min_periods=252)
    for r in dis_leads:
        print(r)

    print()
    summarize(ks_leads, "KS-stat")
    summarize(dis_leads, "Disagreement")