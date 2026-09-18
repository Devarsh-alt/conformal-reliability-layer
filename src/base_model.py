import os
import numpy as np
import pandas as pd
import lightgbm as lgb

from features import get_all_features
from config import TICKERS, BASE_DIR


PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ~5 years of trading-day observations before the first prediction.
INITIAL_TRAIN_DAYS = 252 * 5

# ~6 months of trading days per walk-forward test fold.
TEST_CHUNK_DAYS = 126


FEATURE_COLS = [
    "Log_Return",
    "return_lag1",
    "return_lag5",
    "return_lag10",
    "vol_10d",
    "vol_20d",
    "volume_ratio",
    "VIX",
]


def build_target(df):
    """
    Construct a one-step-ahead forecasting dataset.

    Information available on prediction date t:
        X_t

    Target realized on outcome date t+1:
        y_{t+1}

    The returned row therefore has two distinct timestamps:
        prediction_date = t
        outcome_date    = t+1
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # Next trading day's return.
    df["target"] = df["Log_Return"].shift(-1)

    # Explicitly store when the target becomes observable.
    df["outcome_date"] = pd.Series(
        df.index,
        index=df.index
    ).shift(-1)

    # The final row has no future outcome.
    df = df.dropna(subset=["target", "outcome_date"])

    # The index is the date on which the prediction is made.
    df["prediction_date"] = df.index

    return df


def walk_forward_predict(df, ticker):
    """
    Expanding-window walk-forward training and prediction.

    For each test observation:
        prediction_date = date on which the forecast is produced
        outcome_date    = date on which the realized target is known

    No future observations are used to train a fold.
    """
    results = []

    n = len(df)
    fold_id = 0
    train_end = INITIAL_TRAIN_DAYS

    if train_end >= n:
        print(
            f"WARNING: {ticker} has insufficient history "
            f"({n} rows) for initial train window "
            f"({INITIAL_TRAIN_DAYS}). Skipping."
        )
        return None

    while train_end < n:
        test_end = min(train_end + TEST_CHUNK_DAYS, n)

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["target"]

        X_test = test_df[FEATURE_COLS]
        y_test = test_df["target"]

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        fold_results = pd.DataFrame(
            {
                # Primary timestamp definitions.
                "prediction_date": test_df["prediction_date"].values,
                "outcome_date": test_df["outcome_date"].values,

                # The old column is retained for compatibility with
                # existing signal-generation scripts.
                "date": test_df["prediction_date"].values,

                # Realized next-day target and prediction.
                "actual": y_test.values,
                "predicted": preds,

                # Useful for diagnostics and walk-forward analysis.
                "fold_id": fold_id,
                "ticker": ticker,

                # Explicitly document this as a one-step forecast.
                "forecast_horizon": 1,
            }
        )

        results.append(fold_results)

        fold_id += 1
        train_end = test_end

    if not results:
        return None

    result = pd.concat(results, ignore_index=True)

    # Enforce chronological order.
    result["prediction_date"] = pd.to_datetime(result["prediction_date"])
    result["outcome_date"] = pd.to_datetime(result["outcome_date"])
    result["date"] = pd.to_datetime(result["date"])

    result = result.sort_values(
        ["prediction_date", "outcome_date"]
    ).reset_index(drop=True)

    return result


def run_base_model():
    """
    Generate walk-forward predictions for all configured tickers.
    """
    features = get_all_features()
    all_results = {}

    for ticker in TICKERS:
        if ticker not in features:
            print(f"WARNING: no feature data for {ticker}, skipping.")
            continue

        try:
            df = build_target(features[ticker])

            result = walk_forward_predict(df, ticker)

            if result is None:
                continue

            all_results[ticker] = result

            file_path = os.path.join(
                PREDICTIONS_DIR,
                f"{ticker}.csv"
            )

            result.to_csv(file_path, index=False)

            print(
                f"{ticker}: "
                f"{len(result)} predictions | "
                f"{result['prediction_date'].min().date()} -> "
                f"{result['prediction_date'].max().date()}"
            )

        except Exception as e:
            print(f"FAILED: {ticker} — {e}")

    return all_results


if __name__ == "__main__":
    results = run_base_model()

    print(
        f"\nBase model predictions generated for "
        f"{len(results)} / {len(TICKERS)} tickers"
    )

    for ticker, df in results.items():
        n_folds = df["fold_id"].nunique()

        print(
            f"{ticker}: "
            f"{len(df)} predictions across "
            f"{n_folds} folds"
        )

        print(
            f"  prediction_date: "
            f"{df['prediction_date'].min().date()} -> "
            f"{df['prediction_date'].max().date()}"
        )

        print(
            f"  outcome_date: "
            f"{df['outcome_date'].min().date()} -> "
            f"{df['outcome_date'].max().date()}"
        )