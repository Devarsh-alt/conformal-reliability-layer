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
    "final_candidate_dataset.csv",
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "nested_ablation",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# -------------------------------------------------------------------
# CORE SIGNALS
# -------------------------------------------------------------------

SIGNALS = {
    "D": "disagreement_pct",
    "O": "novelty_pct",
    "M": "ewma_miscoverage_pct",
    "W": "interval_width_pct",
}


# -------------------------------------------------------------------
# ALL FEATURE COMBINATIONS
# -------------------------------------------------------------------

def generate_feature_sets():

    feature_sets = {
        "baseline": []
    }

    names = list(
        SIGNALS.keys()
    )

    for r in range(
        1,
        len(names) + 1,
    ):

        for combination in itertools.combinations(
            names,
            r,
        ):

            name = "+".join(
                combination
            )

            columns = [
                SIGNALS[name]
                for name in combination
            ]

            feature_sets[name] = columns

    return feature_sets


FEATURE_SETS = generate_feature_sets()


# -------------------------------------------------------------------
# NESTED TEMPORAL FOLDS
# -------------------------------------------------------------------

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


def make_fold(
    df,
    test_year,
):
    """
    Construct one nested temporal fold.

    For test year Y:

        TRAIN:
            before Jan 1 of Y-2

        VALIDATION:
            Y-2 through Y-1

        TEST:
            calendar year Y

    Labels are additionally purged by outcome_date.

    This prevents a training/validation observation from using a
    target whose outcome was not yet observable at the corresponding
    split boundary.
    """

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year + 1}-01-01"
    )

    validation_start = pd.Timestamp(
        f"{test_year - 2}-01-01"
    )

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------
    #
    # Both the prediction and outcome must be before validation_start.
    #
    train = df[
        (
            df["prediction_date"]
            < validation_start
        )
        &
        (
            df["outcome_date"]
            < validation_start
        )
    ].copy()

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------
    #
    # Prediction occurs within the validation period and its outcome
    # must also be known before the test period begins.
    #
    validation = df[
        (
            df["prediction_date"]
            >= validation_start
        )
        &
        (
            df["prediction_date"]
            < test_start
        )
        &
        (
            df["outcome_date"]
            < test_start
        )
    ].copy()

    # ---------------------------------------------------------------
    # Test
    # ---------------------------------------------------------------

    test = df[
        (
            df["prediction_date"]
            >= test_start
        )
        &
        (
            df["prediction_date"]
            < test_end
        )
    ].copy()

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):
        raise FileNotFoundError(
            f"Final candidate dataset not found:\n"
            f"{DATASET_PATH}\n\n"
            "Run build_final_dataset.py first."
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
        "miscovered",
        *SIGNALS.values(),
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Dataset missing columns: "
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
# LOGISTIC MODEL
# -------------------------------------------------------------------

def fit_logistic(
    train,
    test,
    feature_cols,
):
    """
    Fit a probability model for pointwise miscoverage.

    The output is:

        P(miscoverage | reliability signals)
    """

    if len(feature_cols) == 0:
        raise ValueError(
            "fit_logistic requires at least one feature."
        )

    X_train = train[
        feature_cols
    ].to_numpy(
        dtype=float
    )

    y_train = train[
        "miscovered"
    ].to_numpy(
        dtype=int
    )

    X_test = test[
        feature_cols
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "Training features contain "
            "non-finite values."
        )

    if not np.isfinite(
        X_test
    ).all():

        raise ValueError(
            "Test features contain "
            "non-finite values."
        )

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
# BASELINE
# -------------------------------------------------------------------

def constant_probability(
    train,
    n_test,
):
    """
    Constant historical miscoverage baseline.
    """

    probability = float(
        train["miscovered"].mean()
    )

    return np.full(
        n_test,
        probability,
        dtype=float,
    )


# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

def metrics(
    y_true,
    probabilities,
):
    """
    Probability and ranking metrics.
    """

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities_clipped = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    result = {
        "brier": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities_clipped,
        ),
        "actual_miscov": y_true.mean(),
        "mean_predicted_miscov": probabilities.mean(),
    }

    if len(
        np.unique(y_true)
    ) == 2:

        result["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result["pr_auc"] = (
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        result["roc_auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


# -------------------------------------------------------------------
# VALIDATION EVALUATION
# -------------------------------------------------------------------

def evaluate_validation(
    train,
    validation,
    feature_name,
    feature_cols,
):
    """
    Evaluate one feature set on validation data.

    This is used ONLY for architecture selection.
    """

    y_validation = validation[
        "miscovered"
    ].to_numpy(
        dtype=int
    )

    # ---------------------------------------------------------------
    # Constant baseline
    # ---------------------------------------------------------------

    if feature_name == "baseline":

        probabilities = (
            constant_probability(
                train,
                len(validation),
            )
        )

        return metrics(
            y_validation,
            probabilities,
        )

    # ---------------------------------------------------------------
    # Logistic model
    # ---------------------------------------------------------------

    _, probabilities = fit_logistic(
        train,
        validation,
        feature_cols,
    )

    return metrics(
        y_validation,
        probabilities,
    )


# -------------------------------------------------------------------
# TEST EVALUATION
# -------------------------------------------------------------------

def evaluate_test(
    train,
    validation,
    test,
    feature_name,
    feature_cols,
):
    """
    Train using ONLY information that would have been available
    before the test period.

    For this first nested experiment, train on the combined
    train + validation sample after architecture selection.

    This is a standard nested temporal procedure:

        inner validation -> choose architecture
        outer test       -> evaluate selected architecture

    The test period is never used for feature selection.
    """

    training_data = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    y_test = test[
        "miscovered"
    ].to_numpy(
        dtype=int
    )

    if feature_name == "baseline":

        probabilities = (
            constant_probability(
                training_data,
                len(test),
            )
        )

    else:

        _, probabilities = fit_logistic(
            training_data,
            test,
            feature_cols,
        )

    return metrics(
        y_test,
        probabilities
    ), probabilities


# -------------------------------------------------------------------
# RUN ONE FOLD
# -------------------------------------------------------------------

def run_fold(
    df,
    test_year,
):
    """
    Run architecture selection and independent evaluation
    for one outer test year.
    """

    (
        train,
        validation,
        test,
    ) = make_fold(
        df,
        test_year,
    )

    if len(train) == 0:
        raise ValueError(
            f"{test_year}: empty training set."
        )

    if len(validation) == 0:
        raise ValueError(
            f"{test_year}: empty validation set."
        )

    if len(test) == 0:
        raise ValueError(
            f"{test_year}: empty test set."
        )

    print(
        "\n========================================"
    )

    print(
        f"OUTER TEST YEAR: {test_year}"
    )

    print(
        "========================================"
    )

    print(
        f"Train:      "
        f"{train['prediction_date'].min().date()} "
        f"-> "
        f"{train['prediction_date'].max().date()} "
        f""
        f"({len(train)} rows)"
    )

    print(
        f"Validation: "
        f"{validation['prediction_date'].min().date()} "
        f"-> "
        f"{validation['prediction_date'].max().date()} "
        f""
        f"({len(validation)} rows)"
    )

    print(
        f"Test:       "
        f"{test['prediction_date'].min().date()} "
        f"-> "
        f"{test['prediction_date'].max().date()} "
        f""
        f"({len(test)} rows)"
    )

    print(
        "\nMiscoverage:"
    )

    print(
        f"  Train:      "
        f"{train['miscovered'].mean():.6f}"
    )

    print(
        f"  Validation: "
        f"{validation['miscovered'].mean():.6f}"
    )

    print(
        f"  Test:       "
        f"{test['miscovered'].mean():.6f}"
    )

    # ---------------------------------------------------------------
    # Architecture selection
    # ---------------------------------------------------------------

    validation_results = []

    print(
        "\n--- Validation ablation ---"
    )

    for feature_name, feature_cols in (
        FEATURE_SETS.items()
    ):

        validation_metrics = (
            evaluate_validation(
                train,
                validation,
                feature_name,
                feature_cols,
            )
        )

        validation_results.append(
            {
                "test_year": test_year,
                "feature_set": feature_name,
                "features": ",".join(
                    feature_cols
                ),
                "validation_brier":
                    validation_metrics["brier"],
                "validation_log_loss":
                    validation_metrics["log_loss"],
                "validation_auc":
                    validation_metrics["roc_auc"],
                "validation_pr_auc":
                    validation_metrics["pr_auc"],
            }
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    validation_df = (
        validation_df
        .sort_values(
            "validation_brier"
        )
        .reset_index(drop=True)
    )

    print(
        validation_df.to_string(
            index=False
        )
    )

    # Best architecture is selected ONLY from validation.
    selected_feature_set = (
        validation_df.iloc[0][
            "feature_set"
        ]
    )

    selected_features = FEATURE_SETS[
        selected_feature_set
    ]

    print(
        f"\nSelected architecture: "
        f"{selected_feature_set}"
    )

    # ---------------------------------------------------------------
    # Outer test evaluation
    # ---------------------------------------------------------------

    test_metrics, probabilities = (
        evaluate_test(
            train,
            validation,
            test,
            selected_feature_set,
            selected_features,
        )
    )

    test_result = {
        "test_year": test_year,
        "selected_feature_set":
            selected_feature_set,
        "selected_features":
            ",".join(selected_features),
        "train_rows":
            len(train),
        "validation_rows":
            len(validation),
        "test_rows":
            len(test),
        "test_brier":
            test_metrics["brier"],
        "test_log_loss":
            test_metrics["log_loss"],
        "test_roc_auc":
            test_metrics["roc_auc"],
        "test_pr_auc":
            test_metrics["pr_auc"],
        "test_actual_miscov":
            test_metrics["actual_miscov"],
        "test_mean_predicted_miscov":
            test_metrics[
                "mean_predicted_miscov"
            ],
    }

    # ---------------------------------------------------------------
    # Save predictions for this fold
    # ---------------------------------------------------------------

    prediction_df = test[
        [
            "prediction_date",
            "outcome_date",
            "ticker",
            "miscovered",
        ]
    ].copy()

    prediction_df[
        "predicted_miscov"
    ] = probabilities

    prediction_df[
        "test_year"
    ] = test_year

    prediction_df[
        "selected_feature_set"
    ] = selected_feature_set

    return (
        validation_df,
        pd.DataFrame(
            [test_result]
        ),
        prediction_df,
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():

    print(
        "Loading final candidate dataset..."
    )

    df = load_dataset()

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Tickers: "
        f"{df['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['prediction_date'].min().date()} "
        f"-> "
        f"{df['prediction_date'].max().date()}"
    )

    all_validation = []
    all_test = []
    all_predictions = []

    for test_year in TEST_YEARS:

        try:

            (
                validation_results,
                test_result,
                prediction_df,
            ) = run_fold(
                df,
                test_year,
            )

            all_validation.append(
                validation_results
            )

            all_test.append(
                test_result
            )

            all_predictions.append(
                prediction_df
            )

        except Exception as e:

            print(
                f"\nFAILED test year "
                f"{test_year}: {e}"
            )

    if not all_test:

        raise RuntimeError(
            "No outer test folds completed."
        )

    validation_all = pd.concat(
        all_validation,
        ignore_index=True,
    )

    test_all = pd.concat(
        all_test,
        ignore_index=True,
    )

    predictions_all = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    validation_path = os.path.join(
        OUTPUT_DIR,
        "nested_validation_results.csv",
    )

    test_path = os.path.join(
        OUTPUT_DIR,
        "nested_test_results.csv",
    )

    predictions_path = os.path.join(
        OUTPUT_DIR,
        "nested_test_predictions.csv",
    )

    validation_all.to_csv(
        validation_path,
        index=False,
    )

    test_all.to_csv(
        test_path,
        index=False,
    )

    predictions_all.to_csv(
        predictions_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "NESTED WALK-FORWARD RESULTS"
    )

    print(
        "========================================"
    )

    print(
        test_all.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Selected architecture frequency
    # ---------------------------------------------------------------

    print(
        "\n--- Selected architecture by year ---"
    )

    print(
        test_all[
            [
                "test_year",
                "selected_feature_set",
                "test_brier",
                "test_roc_auc",
                "test_pr_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n--- Selection frequency ---"
    )

    print(
        test_all[
            "selected_feature_set"
        ]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Aggregate selected-model performance
    # ---------------------------------------------------------------

    print(
        "\n--- Aggregate outer-test performance ---"
    )

    print(
        f"Mean Brier: "
        f"{test_all['test_brier'].mean():.6f}"
    )

    print(
        f"Median Brier: "
        f"{test_all['test_brier'].median():.6f}"
    )

    print(
        f"Mean ROC-AUC: "
        f"{test_all['test_roc_auc'].mean():.6f}"
    )

    print(
        f"Median ROC-AUC: "
        f"{test_all['test_roc_auc'].median():.6f}"
    )

    print(
        f"Mean PR-AUC: "
        f"{test_all['test_pr_auc'].mean():.6f}"
    )

    # ---------------------------------------------------------------
    # Save paths
    # ---------------------------------------------------------------

    print(
        "\nSaved:"
    )

    print(
        validation_path
    )

    print(
        test_path
    )

    print(
        predictions_path
    )


if __name__ == "__main__":
    main()