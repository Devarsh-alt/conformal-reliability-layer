import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR

TRAINING_PANEL_PATH = os.path.join(BASE_DIR, "..", "data", "training_panel.csv")
TEST_START_DATE = "2024-01-01"  # last ~2.5 years held out as test; everything before is "train"
HORIZONS = (5, 10, 20)


def evaluate_baselines():
    df = pd.read_csv(TRAINING_PANEL_PATH, parse_dates=['date'])

    results = []

    for N in HORIZONS:
        label_col = f'label_fwd_miscov_{N}d'
        valid = df.dropna(subset=['miscoverage', label_col]).copy()

        train = valid[valid['date'] < TEST_START_DATE]
        test = valid[valid['date'] >= TEST_START_DATE]

        # Baseline 1: persistence -- predict trailing miscoverage as-is
        persistence_pred = test['miscoverage']
        persistence_mae = (persistence_pred - test[label_col]).abs().mean()
        persistence_rmse = np.sqrt(((persistence_pred - test[label_col]) ** 2).mean())

        # Baseline 2: naive mean -- predict the TRAIN set's mean label (no leakage from test)
        train_mean_label = train[label_col].mean()
        mean_mae = (train_mean_label - test[label_col]).abs().mean()
        mean_rmse = np.sqrt(((train_mean_label - test[label_col]) ** 2).mean())

        results.append({
            'horizon': N,
            'n_train': len(train),
            'n_test': len(test),
            'persistence_mae': persistence_mae,
            'persistence_rmse': persistence_rmse,
            'mean_baseline_value': train_mean_label,
            'mean_baseline_mae': mean_mae,
            'mean_baseline_rmse': mean_rmse,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    results = evaluate_baselines()
    pd.set_option('display.width', 140)
    print(results.to_string(index=False))

    print(f"\nTest period: dates >= {TEST_START_DATE}")
    print("\nA meta-model needs to beat BOTH baselines (lower MAE/RMSE) to be worth using.")
    print("If persistence_mae is already very low, that means trailing miscoverage")
    print("alone is already a strong predictor of forward miscoverage -- worth knowing")
    print("BEFORE training anything, since it sets expectations for how much headroom")
    print("KS-stat and disagreement actually have to improve on.")