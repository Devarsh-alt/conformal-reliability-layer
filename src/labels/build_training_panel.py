import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR


def build_training_panel():
    """
    Merges data/signals/all_tickers_signals.csv (features: ks_stat,
    miscoverage, disagreement) with data/labels/all_tickers_labels.csv
    (targets: label_fwd_miscov_5d/10d/20d) on ['ticker', 'date'].

    Saves the result to data/training_panel.csv.
    """
    signals_path = os.path.join(BASE_DIR, "..", "data", "signals", "all_tickers_signals.csv")
    labels_path = os.path.join(BASE_DIR, "..", "data", "labels", "all_tickers_labels.csv")

    signals_df = pd.read_csv(signals_path, parse_dates=['date'])
    labels_df = pd.read_csv(labels_path, parse_dates=['date'])

    merged = signals_df.merge(labels_df, on=['ticker', 'date'], how='inner')
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)

    out_path = os.path.join(BASE_DIR, "..", "data", "training_panel.csv")
    merged.to_csv(out_path, index=False)

    print(f"Signals rows: {len(signals_df)}")
    print(f"Labels rows: {len(labels_df)}")
    print(f"Merged (inner join) rows: {len(merged)}")
    print(f"Saved training panel to {out_path}")

    print("\n--- Missing-value check (should be near-zero except tail rows per ticker) ---")
    print(merged.isna().sum())

    return merged


if __name__ == "__main__":
    panel = build_training_panel()
    print("\n--- Sample rows ---")
    pd.set_option('display.width', 140)
    print(panel.head(10).to_string(index=False))