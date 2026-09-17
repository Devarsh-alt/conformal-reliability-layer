import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import BASE_DIR

TRAINING_PANEL_PATH = os.path.join(BASE_DIR, "..", "data", "training_panel.csv")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "meta_model_predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

TEST_START_DATE = "2024-01-01"  # matches the baseline split exactly
HORIZONS = (5, 10, 20)
FEATURE_COLS = ['ks_stat', 'miscoverage', 'disagreement']


def train_and_evaluate(N):
    df = pd.read_csv(TRAINING_PANEL_PATH, parse_dates=['date'])
    label_col = f'label_fwd_miscov_{N}d'

    valid = df.dropna(subset=FEATURE_COLS + [label_col]).copy()

    train = valid[valid['date'] < TEST_START_DATE]
    test = valid[valid['date'] >= TEST_START_DATE].copy()

    X_train, y_train = train[FEATURE_COLS], train[label_col]
    X_test, y_test = test[FEATURE_COLS], test[label_col]

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = np.abs(preds - y_test).mean()
    rmse = np.sqrt(((preds - y_test) ** 2).mean())

    test['predicted_miscoverage'] = preds
    test['reliability_score'] = 1 - preds
    test['actual_label'] = y_test

    out_path = os.path.join(PREDICTIONS_DIR, f"meta_model_{N}d.csv")
    test[['date', 'ticker', 'ks_stat', 'miscoverage', 'disagreement',
          'actual_label', 'predicted_miscoverage', 'reliability_score']].to_csv(out_path, index=False)

    importances = dict(zip(FEATURE_COLS, model.feature_importances_))

    return {
        'horizon': N,
        'n_train': len(train),
        'n_test': len(test),
        'mae': mae,
        'rmse': rmse,
        'importance_ks_stat': importances['ks_stat'],
        'importance_miscoverage': importances['miscoverage'],
        'importance_disagreement': importances['disagreement'],
    }


if __name__ == "__main__":
    results = []
    for N in HORIZONS:
        print(f"Training meta-model for {N}d horizon...")
        result = train_and_evaluate(N)
        results.append(result)
        print(f"  -> MAE: {result['mae']:.4f} | RMSE: {result['rmse']:.4f}")

    results_df = pd.DataFrame(results)
    pd.set_option('display.width', 140)
    print("\n--- Meta-model results ---")
    print(results_df.to_string(index=False))

    print("\n--- Reference: baseline MAE from earlier run ---")
    print("horizon  persistence_mae  mean_baseline_mae")
    print("5        0.1267           0.1275")
    print("10       0.1015           0.0840")
    print("20       0.0861           0.0641")

    print(f"\nPer-horizon predictions saved to {PREDICTIONS_DIR}/meta_model_{{N}}d.csv")
    print("(these files will be needed for calibration diagrams and lead-time analysis next)")