import os
import sys
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# -------------------------------------------------------------------
# IMPORT PATH
# -------------------------------------------------------------------

SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, SRC_DIR)

from config import BASE_DIR, TICKERS
from features import get_all_features
from base_model import FEATURE_COLS

from signals.feature_novelty import (
    causal_feature_novelty,
    causal_signal_percentile,
)


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
    "novelty_diagnostics",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

WINDOW = 60
MIN_PERIODS = 60

TEST_START = pd.Timestamp(
    "2024-01-01"
)

TEST_END = pd.Timestamp(
    "2026-06-30"
)

TARGET_COL = "miscovered"
NOVELTY_COL = "feature_novelty_pct"

SKIP_TICKERS = {
    "COIN",
}


# -------------------------------------------------------------------
# LOAD RELIABILITY DATASET
# -------------------------------------------------------------------

def load_reliability_dataset():
    """
    Load the already-built reliability dataset.

    This contains:
        - prediction dates
        - realized outcomes
        - miscoverage labels
        - absolute prediction error
    """

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Reliability dataset not found:\n"
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
        TARGET_COL,
        "abs_error",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Reliability dataset is missing: "
            f"{sorted(missing)}"
        )

    return (
        df
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------
# BUILD NOVELTY FOR ONE TICKER
# -------------------------------------------------------------------

def build_ticker_novelty(
    ticker,
):
    """
    Calculate causal feature novelty for one ticker.

    Important:
        This calculation uses only the feature history.

    It does NOT use:
        actual outcomes
        prediction errors
        conformal coverage
        miscoverage
    """

    features = get_all_features()

    if ticker not in features:
        raise ValueError(
            f"No feature data found for {ticker}."
        )

    feature_df = features[
        ticker
    ].copy()

    novelty_df = causal_feature_novelty(
        feature_df,
        feature_cols=FEATURE_COLS,
        window=WINDOW,
        min_periods=MIN_PERIODS,
    )

    novelty_df[
        NOVELTY_COL
    ] = causal_signal_percentile(
        novelty_df[
            "feature_novelty"
        ],
        window=WINDOW,
        min_periods=MIN_PERIODS,
    )

    novelty_df["ticker"] = ticker

    novelty_df = novelty_df[
        [
            "prediction_date",
            "ticker",
            "feature_novelty",
            NOVELTY_COL,
        ]
    ].copy()

    novelty_df = novelty_df.dropna(
        subset=[
            NOVELTY_COL
        ]
    )

    return novelty_df


# -------------------------------------------------------------------
# BUILD ALL TICKERS
# -------------------------------------------------------------------

def build_all_novelty():
    """
    Build novelty features for the research universe.
    """

    results = []

    for ticker in TICKERS:

        if ticker in SKIP_TICKERS:
            print(
                f"Skipping {ticker} "
                f"(insufficient historical coverage)"
            )
            continue

        print(
            f"Computing novelty for {ticker}..."
        )

        ticker_novelty = (
            build_ticker_novelty(
                ticker
            )
        )

        results.append(
            ticker_novelty
        )

        print(
            f"  -> {len(ticker_novelty)} "
            f"valid rows"
        )

    if not results:
        raise RuntimeError(
            "No novelty data was generated."
        )

    novelty = pd.concat(
        results,
        ignore_index=True,
    )

    novelty = (
        novelty
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    path = os.path.join(
        OUTPUT_DIR,
        "feature_novelty_panel.csv",
    )

    novelty.to_csv(
        path,
        index=False,
    )

    print(
        f"\nSaved novelty panel to:\n"
        f"{path}"
    )

    return novelty


# -------------------------------------------------------------------
# MERGE NOVELTY WITH RELIABILITY OUTCOMES
# -------------------------------------------------------------------

def build_test_panel(
    reliability,
    novelty,
):
    """
    Merge causal novelty with realized test-period outcomes.
    """

    test = reliability[
        (
            reliability["prediction_date"]
            >= TEST_START
        )
        &
        (
            reliability["prediction_date"]
            <= TEST_END
        )
    ].copy()

    merged = test.merge(
        novelty,
        on=[
            "prediction_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    return merged


# -------------------------------------------------------------------
# OVERALL RANKING METRICS
# -------------------------------------------------------------------

def overall_metrics(
    df,
):
    """
    Evaluate novelty as a ranking signal.

    ROC-AUC:
        Can novelty distinguish miscovered from covered predictions?

    Spearman:
        Is higher novelty associated with larger prediction error?
    """

    y = (
        df[TARGET_COL]
        .astype(int)
        .to_numpy()
    )

    novelty = (
        df[NOVELTY_COL]
        .astype(float)
        .to_numpy()
    )

    abs_error = (
        df["abs_error"]
        .astype(float)
        .to_numpy()
    )

    auc = roc_auc_score(
        y,
        novelty,
    )

    spearman_miscov, misc_pvalue = (
        spearmanr(
            novelty,
            y,
        )
    )

    spearman_error, error_pvalue = (
        spearmanr(
            novelty,
            abs_error,
        )
    )

    return {
        "n": len(df),
        "actual_miscov": y.mean(),
        "mean_novelty_pct": novelty.mean(),
        "roc_auc": auc,
        "spearman_novelty_miscov": spearman_miscov,
        "spearman_miscov_pvalue": misc_pvalue,
        "spearman_novelty_abs_error": spearman_error,
        "spearman_error_pvalue": error_pvalue,
    }


# -------------------------------------------------------------------
# PER-TICKER METRICS
# -------------------------------------------------------------------

def per_ticker_metrics(
    df,
):
    rows = []

    for ticker, group in df.groupby(
        "ticker"
    ):

        y = (
            group[TARGET_COL]
            .astype(int)
            .to_numpy()
        )

        novelty = (
            group[NOVELTY_COL]
            .astype(float)
            .to_numpy()
        )

        if len(np.unique(y)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y,
                novelty,
            )

        rho, pvalue = spearmanr(
            novelty,
            group["abs_error"].astype(float),
        )

        rows.append(
            {
                "ticker": ticker,
                "n": len(group),
                "actual_miscov": y.mean(),
                "roc_auc": auc,
                "spearman_novelty_abs_error": rho,
                "pvalue_abs_error": pvalue,
            }
        )

    return pd.DataFrame(
        rows
    )


# -------------------------------------------------------------------
# YEARLY METRICS
# -------------------------------------------------------------------

def yearly_metrics(
    df,
):
    data = df.copy()

    data["year"] = (
        data[
            "prediction_date"
        ].dt.year
    )

    rows = []

    for year, group in data.groupby(
        "year"
    ):

        y = (
            group[TARGET_COL]
            .astype(int)
            .to_numpy()
        )

        novelty = (
            group[NOVELTY_COL]
            .astype(float)
            .to_numpy()
        )

        if len(np.unique(y)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(
                y,
                novelty,
            )

        rows.append(
            {
                "year": year,
                "n": len(group),
                "actual_miscov": y.mean(),
                "mean_novelty_pct": novelty.mean(),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(
        rows
    )


# -------------------------------------------------------------------
# NOVELTY DECILES
# -------------------------------------------------------------------

def novelty_deciles(
    df,
):
    """
    Examine realized miscoverage and absolute error
    as causal novelty percentile increases.
    """

    data = df.copy()

    bins = [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
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

    data[
        "novelty_bin"
    ] = pd.cut(
        data[NOVELTY_COL],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    result = (
        data
        .groupby(
            "novelty_bin",
            observed=False,
        )
        .agg(
            n=(
                TARGET_COL,
                "size",
            ),
            miscoverage_rate=(
                TARGET_COL,
                "mean",
            ),
            mean_abs_error=(
                "abs_error",
                "mean",
            ),
            mean_novelty_pct=(
                NOVELTY_COL,
                "mean",
            ),
        )
        .reset_index()
    )

    return result


# -------------------------------------------------------------------
# HIGH VS LOW NOVELTY
# -------------------------------------------------------------------

def high_low_effect(
    df,
):
    """
    Compare the lowest 20% and highest 20% of causal
    novelty percentile.
    """

    rows = []

    for ticker, group in df.groupby(
        "ticker"
    ):

        low = group[
            group[NOVELTY_COL] <= 0.20
        ]

        high = group[
            group[NOVELTY_COL] >= 0.80
        ]

        if (
            len(low) == 0
            or len(high) == 0
        ):
            continue

        low_miscov = low[
            TARGET_COL
        ].mean()

        high_miscov = high[
            TARGET_COL
        ].mean()

        low_error = low[
            "abs_error"
        ].mean()

        high_error = high[
            "abs_error"
        ].mean()

        rows.append(
            {
                "ticker": ticker,
                "n_low": len(low),
                "n_high": len(high),
                "low_miscov": low_miscov,
                "high_miscov": high_miscov,
                "difference_high_minus_low_miscov": (
                    high_miscov
                    - low_miscov
                ),
                "low_abs_error": low_error,
                "high_abs_error": high_error,
                "difference_high_minus_low_error": (
                    high_error
                    - low_error
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():

    print(
        "Loading reliability dataset..."
    )

    reliability = (
        load_reliability_dataset()
    )

    print(
        f"Reliability rows: "
        f"{len(reliability)}"
    )

    print(
        f"Reliability tickers: "
        f"{reliability['ticker'].nunique()}"
    )

    # ---------------------------------------------------------------
    # Build novelty
    # ---------------------------------------------------------------

    novelty = (
        build_all_novelty()
    )

    # ---------------------------------------------------------------
    # Merge with test outcomes
    # ---------------------------------------------------------------

    print(
        "\nMerging novelty with "
        "test-period outcomes..."
    )

    test = build_test_panel(
        reliability,
        novelty,
    )

    if len(test) == 0:
        raise RuntimeError(
            "No rows survived novelty/reliability merge."
        )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        f"Test tickers: "
        f"{test['ticker'].nunique()}"
    )

    print(
        f"Test date range: "
        f"{test['prediction_date'].min().date()} "
        f"-> "
        f"{test['prediction_date'].max().date()}"
    )

    # ---------------------------------------------------------------
    # Overall
    # ---------------------------------------------------------------

    overall = overall_metrics(
        test
    )

    print(
        "\n--- Overall novelty metrics ---"
    )

    for key, value in overall.items():

        if isinstance(
            value,
            (float, np.floating),
        ):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    # ---------------------------------------------------------------
    # Per ticker
    # ---------------------------------------------------------------

    ticker_results = (
        per_ticker_metrics(
            test
        )
    )

    print(
        "\n--- Per-ticker novelty metrics ---"
    )

    print(
        ticker_results.to_string(
            index=False
        )
    )

    valid_auc = ticker_results[
        "roc_auc"
    ].dropna()

    if len(valid_auc):

        print(
            f"\nMean ticker AUC: "
            f"{valid_auc.mean():.4f}"
        )

        print(
            f"Median ticker AUC: "
            f"{valid_auc.median():.4f}"
        )

        print(
            f"Count AUC > 0.50: "
            f"{(valid_auc > 0.50).sum()} "
            f"/ {len(valid_auc)}"
        )

    # ---------------------------------------------------------------
    # Yearly
    # ---------------------------------------------------------------

    yearly = yearly_metrics(
        test
    )

    print(
        "\n--- Yearly novelty metrics ---"
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Deciles
    # ---------------------------------------------------------------

    deciles = novelty_deciles(
        test
    )

    print(
        "\n--- Miscoverage by novelty percentile ---"
    )

    print(
        deciles.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # High vs low
    # ---------------------------------------------------------------

    effects = high_low_effect(
        test
    )

    print(
        "\n--- High vs low novelty ---"
    )

    print(
        effects.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    test.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "test_novelty_panel.csv",
        ),
        index=False,
    )

    ticker_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "per_ticker_metrics.csv",
        ),
        index=False,
    )

    yearly.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "yearly_metrics.csv",
        ),
        index=False,
    )

    deciles.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "novelty_deciles.csv",
        ),
        index=False,
    )

    effects.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "high_low_effect.csv",
        ),
        index=False,
    )

    pd.DataFrame(
        [overall]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "overall_metrics.csv",
        ),
        index=False,
    )

    print(
        f"\nSaved diagnostics to:\n"
        f"{OUTPUT_DIR}"
    )


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()