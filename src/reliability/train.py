import os
import sys
import itertools
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
import lightgbm as lgb


# -------------------------------------------------------------------
# IMPORT PATH
# -------------------------------------------------------------------

SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(
    0,
    SRC_DIR,
)

from config import BASE_DIR


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "reliability",
)

DATASET_PATH = os.path.join(
    DATA_DIR,
    "reliability_dataset.csv",
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "experiments",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "primary_test_predictions.csv",
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "primary_ablation_results.csv",
)


# -------------------------------------------------------------------
# TIME SPLIT
# -------------------------------------------------------------------

TRAIN_END = pd.Timestamp(
    "2022-01-01"
)

VALIDATION_END = pd.Timestamp(
    "2024-01-01"
)

# Test therefore starts at 2024-01-01.
#
# Train:
#     prediction_date < 2022-01-01
#
# Validation:
#     2022-01-01 <= date < 2024-01-01
#
# Test:
#     date >= 2024-01-01
#
# The test set is never used for model fitting.


# -------------------------------------------------------------------
# FEATURES
# -------------------------------------------------------------------

M = "miscoverage_pct"
K = "ks_stat_pct"
D = "disagreement_pct"

FEATURE_SETS = {
    "baseline": [],
    "M": [M],
    "K": [K],
    "D": [D],
    "M+K": [M, K],
    "M+D": [M, D],
    "K+D": [K, D],
    "M+K+D": [M, K, D],
}


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def load_dataset():
    """
    Load and validate the pointwise reliability dataset.
    """

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_PATH}\n\n"
            "Run build_dataset.py first."
        )

    df = pd.read_csv(
        DATASET_PATH,
        parse_dates=[
            "prediction_date",
            "outcome_date",
        ],
    )

    required = {
        "prediction_date",
        "outcome_date",
        "ticker",
        M,
        K,
        D,
        "covered",
        "miscovered",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Dataset is missing columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values(
            [
                "prediction_date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    return df


# -------------------------------------------------------------------
# TIME SPLIT
# -------------------------------------------------------------------

def split_dataset(df):
    """
    Create chronological train / validation / test sets.

    No random shuffling is performed.
    """

    train = df[
        df["prediction_date"]
        < TRAIN_END
    ].copy()

    validation = df[
        (
            df["prediction_date"]
            >= TRAIN_END
        )
        & (
            df["prediction_date"]
            < VALIDATION_END
        )
    ].copy()

    test = df[
        df["prediction_date"]
        >= VALIDATION_END
    ].copy()

    if len(train) == 0:
        raise ValueError(
            "Training set is empty."
        )

    if len(validation) == 0:
        raise ValueError(
            "Validation set is empty."
        )

    if len(test) == 0:
        raise ValueError(
            "Test set is empty."
        )

    return train, validation, test


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
):
    """
    Calculate probability-quality metrics.

    Brier and log loss are primary metrics because the model output
    is intended to represent probability of miscoverage.

    ROC-AUC and PR-AUC provide ranking diagnostics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    # Keep probabilities strictly inside (0, 1) for log loss.
    probabilities_clipped = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    metrics = {
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities_clipped,
        ),
        "mean_predicted_miscov": probabilities.mean(),
        "actual_miscov": y_true.mean(),
    }

    # These require both classes to exist.
    if len(np.unique(y_true)) == 2:

        metrics["roc_auc"] = roc_auc_score(
            y_true,
            probabilities,
        )

        metrics["pr_auc"] = average_precision_score(
            y_true,
            probabilities,
        )

    else:

        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# -------------------------------------------------------------------
# LOGISTIC REGRESSION
# -------------------------------------------------------------------

def fit_logistic(
    X_train,
    y_train,
    X_test,
):
    """
    Simple interpretable probability model.

    No class weighting is used because we care about calibrated
    probability estimates, not classification accuracy at a fixed
    threshold.
    """

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return model, probabilities


# -------------------------------------------------------------------
# LIGHTGBM
# -------------------------------------------------------------------

def fit_lightgbm(
    X_train,
    y_train,
    X_test,
):
    """
    Nonlinear probability model.

    Parameters are deliberately conservative; we are testing whether
    the signals contain information, not conducting extensive
    hyperparameter optimization yet.
    """

    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return model, probabilities


# -------------------------------------------------------------------
# BASELINE
# -------------------------------------------------------------------

def constant_baseline(
    y_train,
    n_test,
):
    """
    Constant probability baseline.

    Every test observation receives the historical training-set
    miscoverage rate.
    """

    probability = float(
        np.mean(y_train)
    )

    return np.full(
        n_test,
        probability,
        dtype=float,
    )


# -------------------------------------------------------------------
# RUN ONE EXPERIMENT
# -------------------------------------------------------------------

def run_experiment(
    feature_name,
    feature_cols,
    train,
    validation,
    test,
):
    """
    Run both Logistic Regression and LightGBM for one feature subset.
    """

    y_train = train[
        "miscovered"
    ].to_numpy(dtype=int)

    y_validation = validation[
        "miscovered"
    ].to_numpy(dtype=int)

    y_test = test[
        "miscovered"
    ].to_numpy(dtype=int)

    results = []
    prediction_rows = []

    # ---------------------------------------------------------------
    # CONSTANT BASELINE
    # ---------------------------------------------------------------

    if feature_name == "baseline":

        train_rate = y_train.mean()

        probabilities = constant_baseline(
            y_train,
            len(test),
        )

        metrics = calculate_metrics(
            y_test,
            probabilities,
        )

        results.append(
            {
                "feature_set": feature_name,
                "model": "constant_baseline",
                "n_features": 0,
                "features": "",
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_miscov": train_rate,
                "validation_miscov": y_validation.mean(),
                **metrics,
            }
        )

        for i, (_, row) in enumerate(
            test.iterrows()
        ):

            prediction_rows.append(
                {
                    "prediction_date": row[
                        "prediction_date"
                    ],
                    "ticker": row[
                        "ticker"
                    ],
                    "feature_set": feature_name,
                    "model": "constant_baseline",
                    "predicted_miscov": probabilities[i],
                    "actual_miscov": row[
                        "miscovered"
                    ],
                }
            )

        return results, prediction_rows

    # ---------------------------------------------------------------
    # FEATURE MATRICES
    # ---------------------------------------------------------------

    X_train = train[
        feature_cols
    ].to_numpy(dtype=float)

    X_validation = validation[
        feature_cols
    ].to_numpy(dtype=float)

    X_test = test[
        feature_cols
    ].to_numpy(dtype=float)

    # Explicitly verify absence of NaNs/infinities.
    if not np.isfinite(
        X_train
    ).all():
        raise ValueError(
            f"{feature_name}: non-finite training feature values."
        )

    if not np.isfinite(
        X_validation
    ).all():
        raise ValueError(
            f"{feature_name}: non-finite validation feature values."
        )

    if not np.isfinite(
        X_test
    ).all():
        raise ValueError(
            f"{feature_name}: non-finite test feature values."
        )

    # ---------------------------------------------------------------
    # LOGISTIC REGRESSION
    # ---------------------------------------------------------------

    logistic_model, logistic_probs = fit_logistic(
        X_train,
        y_train,
        X_test,
    )

    logistic_metrics = calculate_metrics(
        y_test,
        logistic_probs,
    )

    results.append(
        {
            "feature_set": feature_name,
            "model": "logistic",
            "n_features": len(feature_cols),
            "features": ",".join(feature_cols),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_miscov": y_train.mean(),
            "validation_miscov": y_validation.mean(),
            **logistic_metrics,
        }
    )

    for i, (_, row) in enumerate(
        test.iterrows()
    ):

        prediction_rows.append(
            {
                "prediction_date": row[
                    "prediction_date"
                ],
                "ticker": row[
                    "ticker"
                ],
                "feature_set": feature_name,
                "model": "logistic",
                "predicted_miscov": logistic_probs[i],
                "actual_miscov": row[
                    "miscovered"
                ],
            }
        )

    # ---------------------------------------------------------------
    # LIGHTGBM
    # ---------------------------------------------------------------

    lgb_model, lgb_probs = fit_lightgbm(
        X_train,
        y_train,
        X_test,
    )

    lgb_metrics = calculate_metrics(
        y_test,
        lgb_probs,
    )

    results.append(
        {
            "feature_set": feature_name,
            "model": "lightgbm",
            "n_features": len(feature_cols),
            "features": ",".join(feature_cols),
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_miscov": y_train.mean(),
            "validation_miscov": y_validation.mean(),
            **lgb_metrics,
        }
    )

    for i, (_, row) in enumerate(
        test.iterrows()
    ):

        prediction_rows.append(
            {
                "prediction_date": row[
                    "prediction_date"
                ],
                "ticker": row[
                    "ticker"
                ],
                "feature_set": feature_name,
                "model": "lightgbm",
                "predicted_miscov": lgb_probs[i],
                "actual_miscov": row[
                    "miscovered"
                ],
            }
        )

    return results, prediction_rows


# -------------------------------------------------------------------
# MAIN RESEARCH EXPERIMENT
# -------------------------------------------------------------------

def main():

    print(
        "Loading reliability dataset..."
    )

    df = load_dataset()

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Tickers: "
        f"{df['ticker'].nunique()}"
    )

    # ---------------------------------------------------------------
    # TIME SPLIT
    # ---------------------------------------------------------------

    train, validation, test = split_dataset(
        df
    )

    print(
        "\n--- Temporal split ---"
    )

    print(
        f"Train:      "
        f"{train['prediction_date'].min().date()} "
        f"-> "
        f"{train['prediction_date'].max().date()} "
        f"({len(train)} rows)"
    )

    print(
        f"Validation: "
        f"{validation['prediction_date'].min().date()} "
        f"-> "
        f"{validation['prediction_date'].max().date()} "
        f"({len(validation)} rows)"
    )

    print(
        f"Test:       "
        f"{test['prediction_date'].min().date()} "
        f"-> "
        f"{test['prediction_date'].max().date()} "
        f"({len(test)} rows)"
    )

    print(
        "\n--- Miscoverage rates ---"
    )

    print(
        f"Train:      "
        f"{train['miscovered'].mean():.4f}"
    )

    print(
        f"Validation: "
        f"{validation['miscovered'].mean():.4f}"
    )

    print(
        f"Test:       "
        f"{test['miscovered'].mean():.4f}"
    )

    # ---------------------------------------------------------------
    # EXPERIMENTS
    # ---------------------------------------------------------------

    all_results = []
    all_predictions = []

    for feature_name, feature_cols in (
        FEATURE_SETS.items()
    ):

        print(
            f"\nRunning feature set: "
            f"{feature_name}"
        )

        results, predictions = (
            run_experiment(
                feature_name,
                feature_cols,
                train,
                validation,
                test,
            )
        )

        all_results.extend(
            results
        )

        all_predictions.extend(
            predictions
        )

        for result in results:

            print(
                f"  {result['model']:<18} "
                f"Brier={result['brier']:.6f} | "
                f"LogLoss={result['log_loss']:.6f} | "
                f"ROC-AUC={result['roc_auc']:.4f} | "
                f"PR-AUC={result['pr_auc']:.4f}"
            )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    # ---------------------------------------------------------------
    # BASELINE COMPARISON
    # ---------------------------------------------------------------

    baseline_rows = results_df[
        results_df["model"]
        == "constant_baseline"
    ]

    if len(baseline_rows) != 1:
        raise AssertionError(
            "Expected exactly one constant baseline."
        )

    baseline_brier = float(
        baseline_rows.iloc[0][
            "brier"
        ]
    )

    baseline_logloss = float(
        baseline_rows.iloc[0][
            "log_loss"
        ]
    )

    results_df[
        "brier_improvement_vs_constant"
    ] = (
        baseline_brier
        - results_df["brier"]
    )

    results_df[
        "logloss_improvement_vs_constant"
    ] = (
        baseline_logloss
        - results_df["log_loss"]
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # FINAL TABLE
    # ---------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "PRIMARY ABLATION RESULTS"
    )

    print(
        "========================================"
    )

    display_cols = [
        "feature_set",
        "model",
        "brier",
        "brier_improvement_vs_constant",
        "log_loss",
        "roc_auc",
        "pr_auc",
        "actual_miscov",
        "mean_predicted_miscov",
    ]

    print(
        results_df[
            display_cols
        ]
        .sort_values(
            [
                "feature_set",
                "model",
            ]
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved results to:"
        f"\n{RESULTS_PATH}"
    )

    print(
        f"\nSaved test predictions to:"
        f"\n{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()