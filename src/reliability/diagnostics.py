import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
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
    "reliability_dataset.csv",
)

PREDICTIONS_PATH = os.path.join(
    DATA_DIR,
    "experiments",
    "primary_test_predictions.csv",
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "diagnostics",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

TEST_START = pd.Timestamp(
    "2024-01-01"
)

TEST_END = pd.Timestamp(
    "2026-06-30"
)

D_FEATURE = "disagreement_pct"

TARGET = "miscovered"


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------

def load_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    dataset = pd.read_csv(
        DATASET_PATH,
        parse_dates=[
            "prediction_date",
            "outcome_date",
        ],
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        parse_dates=[
            "prediction_date",
        ],
    )

    return dataset, predictions


# -------------------------------------------------------------------
# MERGE TEST PREDICTIONS WITH ORIGINAL DATA
# -------------------------------------------------------------------

def build_test_panel(
    dataset,
    predictions,
):

    test_dataset = dataset[
        (
            dataset["prediction_date"]
            >= TEST_START
        )
        &
        (
            dataset["prediction_date"]
            <= TEST_END
        )
    ].copy()

    # Keep only models of interest.
    prediction_models = predictions[
        predictions["model"].isin(
            [
                "logistic",
                "lightgbm",
            ]
        )
    ].copy()

    panel = test_dataset.merge(
        prediction_models,
        on=[
            "prediction_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_many",
    )

    return panel


# -------------------------------------------------------------------
# OVERALL METRICS
# -------------------------------------------------------------------

def overall_metrics(panel):

    rows = []

    for (
        feature_set,
        model
    ), group in panel.groupby(
        [
            "feature_set",
            "model",
        ]
    ):

        y = group[
            "actual_miscov"
        ].astype(int).to_numpy()

        p = group[
            "predicted_miscov"
        ].astype(float).to_numpy()

        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "n": len(group),
                "actual_miscov": y.mean(),
                "mean_predicted_miscov": p.mean(),
                "brier": brier_score_loss(
                    y,
                    p,
                ),
                "roc_auc": (
                    roc_auc_score(y, p)
                    if len(np.unique(y)) == 2
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# PER-TICKER AUC
# -------------------------------------------------------------------

def per_ticker_auc(panel):

    rows = []

    # Focus on the two main specifications:
    #
    # D only
    # all three signals
    #
    for feature_set in [
        "D",
        "M+K+D",
    ]:

        for model in [
            "logistic",
            "lightgbm",
        ]:

            subset = panel[
                (
                    panel["feature_set"]
                    == feature_set
                )
                &
                (
                    panel["model"]
                    == model
                )
            ]

            for ticker, group in subset.groupby(
                "ticker"
            ):

                y = group[
                    "actual_miscov"
                ].astype(int)

                p = group[
                    "predicted_miscov"
                ].astype(float)

                if y.nunique() < 2:
                    auc = np.nan
                else:
                    auc = roc_auc_score(
                        y,
                        p,
                    )

                brier = brier_score_loss(
                    y,
                    p,
                )

                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model,
                        "ticker": ticker,
                        "n": len(group),
                        "actual_miscov": y.mean(),
                        "brier": brier,
                        "roc_auc": auc,
                    }
                )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# YEARLY AUC
# -------------------------------------------------------------------

def yearly_auc(panel):

    panel = panel.copy()

    panel["year"] = (
        panel[
            "prediction_date"
        ]
        .dt.year
    )

    rows = []

    for (
        feature_set,
        model,
        year
    ), group in panel.groupby(
        [
            "feature_set",
            "model",
            "year",
        ]
    ):

        y = group[
            "actual_miscov"
        ].astype(int)

        p = group[
            "predicted_miscov"
        ].astype(float)

        if y.nunique() < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y,
                p,
            )

        rows.append(
            {
                "feature_set": feature_set,
                "model": model,
                "year": year,
                "n": len(group),
                "actual_miscov": y.mean(),
                "mean_predicted_miscov": p.mean(),
                "brier": brier_score_loss(
                    y,
                    p,
                ),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# SIGNAL CORRELATION
# -------------------------------------------------------------------

def signal_correlations(
    dataset
):

    test = dataset[
        dataset[
            "prediction_date"
        ] >= TEST_START
    ].copy()

    cols = [
        "miscoverage_pct",
        "ks_stat_pct",
        "disagreement_pct",
    ]

    return test[
        cols
    ].corr(
        method="spearman"
    )


# -------------------------------------------------------------------
# DISAGREEMENT CONDITIONAL RISK
# -------------------------------------------------------------------

def disagreement_bins(
    dataset
):

    test = dataset[
        dataset[
            "prediction_date"
        ] >= TEST_START
    ].copy()

    # These are fixed thresholds, not quantiles recalculated from
    # the test set. Therefore the interpretation remains tied to the
    # causal historical percentile feature.
    bins = [
        -np.inf,
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        np.inf,
    ]

    labels = [
        "0-10%",
        "10-20%",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ]

    test[
        "disagreement_bin"
    ] = pd.cut(
        test[
            D_FEATURE
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        test
        .groupby(
            "disagreement_bin",
            observed=False,
        )
        .agg(
            n=(
                TARGET,
                "size",
            ),
            miscoverage_rate=(
                TARGET,
                "mean",
            ),
            mean_abs_error=(
                "abs_error",
                "mean",
            ),
            mean_interval_width=(
                "interval_width",
                "mean",
            ),
            mean_disagreement=(
                D_FEATURE,
                "mean",
            ),
        )
        .reset_index()
    )

    return result


# -------------------------------------------------------------------
# PER-TICKER DISAGREEMENT EFFECT
# -------------------------------------------------------------------

def ticker_disagreement_effect(
    dataset
):

    test = dataset[
        dataset[
            "prediction_date"
        ] >= TEST_START
    ].copy()

    rows = []

    for ticker, group in test.groupby(
        "ticker"
    ):

        low = group[
            group[D_FEATURE] <= 0.20
        ]

        high = group[
            group[D_FEATURE] >= 0.80
        ]

        if len(low) == 0 or len(high) == 0:
            continue

        low_rate = low[
            TARGET
        ].mean()

        high_rate = high[
            TARGET
        ].mean()

        rows.append(
            {
                "ticker": ticker,
                "n_low": len(low),
                "n_high": len(high),
                "low_disagreement_miscov": low_rate,
                "high_disagreement_miscov": high_rate,
                "difference_high_minus_low": (
                    high_rate - low_rate
                ),
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# SUMMARY
# -------------------------------------------------------------------

def main():

    print(
        "Loading data..."
    )

    dataset, predictions = (
        load_data()
    )

    panel = build_test_panel(
        dataset,
        predictions,
    )

    print(
        f"Test prediction rows: "
        f"{len(panel)}"
    )

    print(
        f"Tickers: "
        f"{panel['ticker'].nunique()}"
    )

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------

    overall = overall_metrics(
        panel
    )

    overall_path = os.path.join(
        OUTPUT_DIR,
        "overall_metrics.csv",
    )

    overall.to_csv(
        overall_path,
        index=False,
    )

    print(
        "\n--- Overall metrics ---"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Per ticker
    # ---------------------------------------------------------------

    ticker_auc = per_ticker_auc(
        panel
    )

    ticker_auc_path = os.path.join(
        OUTPUT_DIR,
        "per_ticker_metrics.csv",
    )

    ticker_auc.to_csv(
        ticker_auc_path,
        index=False,
    )

    print(
        "\n--- Per-ticker AUC ---"
    )

    print(
        ticker_auc.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Year
    # ---------------------------------------------------------------

    yearly = yearly_auc(
        panel
    )

    yearly_path = os.path.join(
        OUTPUT_DIR,
        "yearly_metrics.csv",
    )

    yearly.to_csv(
        yearly_path,
        index=False,
    )

    print(
        "\n--- Yearly metrics ---"
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Signal correlation
    # ---------------------------------------------------------------

    correlations = signal_correlations(
        dataset
    )

    correlations_path = os.path.join(
        OUTPUT_DIR,
        "signal_correlations.csv",
    )

    correlations.to_csv(
        correlations_path
    )

    print(
        "\n--- Spearman signal correlations ---"
    )

    print(
        correlations.to_string()
    )

    # ---------------------------------------------------------------
    # Disagreement conditional risk
    # ---------------------------------------------------------------

    bins = disagreement_bins(
        dataset
    )

    bins_path = os.path.join(
        OUTPUT_DIR,
        "disagreement_bins.csv",
    )

    bins.to_csv(
        bins_path,
        index=False,
    )

    print(
        "\n--- Miscoverage by disagreement percentile ---"
    )

    print(
        bins.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Per-ticker disagreement effect
    # ---------------------------------------------------------------

    ticker_effect = (
        ticker_disagreement_effect(
            dataset
        )
    )

    ticker_effect_path = os.path.join(
        OUTPUT_DIR,
        "ticker_disagreement_effect.csv",
    )

    ticker_effect.to_csv(
        ticker_effect_path,
        index=False,
    )

    print(
        "\n--- Per-ticker disagreement effect ---"
    )

    print(
        ticker_effect.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Compact research summary
    # ---------------------------------------------------------------

    d_logistic = panel[
        (
            panel["feature_set"]
            == "D"
        )
        &
        (
            panel["model"]
            == "logistic"
        )
    ]

    all_logistic = panel[
        (
            panel["feature_set"]
            == "M+K+D"
        )
        &
        (
            panel["model"]
            == "logistic"
        )
    ]

    print(
        "\n========================================"
    )

    print(
        "RESEARCH DIAGNOSTIC SUMMARY"
    )

    print(
        "========================================"
    )

    # D.
    y_d = d_logistic[
        "actual_miscov"
    ].astype(int)

    p_d = d_logistic[
        "predicted_miscov"
    ].astype(float)

    print(
        f"D-only logistic AUC: "
        f"{roc_auc_score(y_d, p_d):.4f}"
    )

    # All three.
    y_all = all_logistic[
        "actual_miscov"
    ].astype(int)

    p_all = all_logistic[
        "predicted_miscov"
    ].astype(float)

    print(
        f"M+K+D logistic AUC: "
        f"{roc_auc_score(y_all, p_all):.4f}"
    )

    # Per-ticker D AUC.
    d_ticker = ticker_auc[
        (
            ticker_auc["feature_set"]
            == "D"
        )
        &
        (
            ticker_auc["model"]
            == "logistic"
        )
    ].dropna(
        subset=["roc_auc"]
    )

    if len(d_ticker):

        print(
            f"D-only ticker AUC mean: "
            f"{d_ticker['roc_auc'].mean():.4f}"
        )

        print(
            f"D-only ticker AUC median: "
            f"{d_ticker['roc_auc'].median():.4f}"
        )

        print(
            "Tickers with AUC > 0.50: "
            f"{(d_ticker['roc_auc'] > 0.50).sum()} "
            f"/ {len(d_ticker)}"
        )

    # Yearly D AUC.
    d_year = yearly[
        (
            yearly["feature_set"]
            == "D"
        )
        &
        (
            yearly["model"]
            == "logistic"
        )
    ]

    print(
        "\nD-only yearly AUC:"
    )

    print(
        d_year[
            [
                "year",
                "n",
                "actual_miscov",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved diagnostics to:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()