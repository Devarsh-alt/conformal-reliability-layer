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

INPUT_PATH = os.path.join(
    DATA_DIR,
    "reliability_dataset_candidates.csv",
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "candidate_diagnostics",
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

TARGET = "miscovered"

SIGNALS = {
    "M": "ewma_miscoverage_pct",
    "W": "interval_width_pct",
}


# -------------------------------------------------------------------
# LOAD
# -------------------------------------------------------------------

def load_data():

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Candidate dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run candidate_signals.py first."
        )

    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "prediction_date",
            "outcome_date",
        ],
    )

    required = {
        "prediction_date",
        "outcome_date",
        "ticker",
        TARGET,
        "abs_error",
        "interval_width",
        "ewma_miscoverage",
        "ewma_miscoverage_pct",
        "interval_width_pct",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
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
# TEST PANEL
# -------------------------------------------------------------------

def get_test_panel(df):

    test = df[
        df["prediction_date"]
        >= TEST_START
    ].copy()

    test = test.dropna(
        subset=[
            TARGET,
            *SIGNALS.values(),
        ]
    )

    if len(test) == 0:
        raise ValueError(
            "No usable test observations."
        )

    return test


# -------------------------------------------------------------------
# OVERALL METRICS
# -------------------------------------------------------------------

def overall_metrics(
    test,
):

    rows = []

    y = (
        test[TARGET]
        .astype(int)
        .to_numpy()
    )

    for name, col in SIGNALS.items():

        x = (
            test[col]
            .astype(float)
            .to_numpy()
        )

        auc = roc_auc_score(
            y,
            x,
        )

        rho_miscov, p_miscov = (
            spearmanr(
                x,
                y,
            )
        )

        rho_error, p_error = (
            spearmanr(
                x,
                test["abs_error"].astype(float),
            )
        )

        rows.append(
            {
                "signal": name,
                "column": col,
                "n": len(test),
                "actual_miscov": y.mean(),
                "mean_signal": x.mean(),
                "roc_auc": auc,
                "spearman_miscov": rho_miscov,
                "spearman_miscov_pvalue": p_miscov,
                "spearman_abs_error": rho_error,
                "spearman_error_pvalue": p_error,
            }
        )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# PER-TICKER
# -------------------------------------------------------------------

def per_ticker_metrics(
    test,
):

    rows = []

    for signal_name, signal_col in SIGNALS.items():

        for ticker, group in test.groupby(
            "ticker"
        ):

            y = (
                group[TARGET]
                .astype(int)
                .to_numpy()
            )

            x = (
                group[signal_col]
                .astype(float)
                .to_numpy()
            )

            if len(np.unique(y)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(
                    y,
                    x,
                )

            rho, pvalue = spearmanr(
                x,
                group[
                    "abs_error"
                ].astype(float),
            )

            rows.append(
                {
                    "signal": signal_name,
                    "ticker": ticker,
                    "n": len(group),
                    "actual_miscov": y.mean(),
                    "roc_auc": auc,
                    "spearman_abs_error": rho,
                    "pvalue_abs_error": pvalue,
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# YEARLY
# -------------------------------------------------------------------

def yearly_metrics(
    test,
):

    data = test.copy()

    data["year"] = (
        data[
            "prediction_date"
        ].dt.year
    )

    rows = []

    for signal_name, signal_col in SIGNALS.items():

        for year, group in data.groupby(
            "year"
        ):

            y = (
                group[TARGET]
                .astype(int)
                .to_numpy()
            )

            x = (
                group[signal_col]
                .astype(float)
                .to_numpy()
            )

            if len(np.unique(y)) < 2:
                auc = np.nan
            else:
                auc = roc_auc_score(
                    y,
                    x,
                )

            rows.append(
                {
                    "signal": signal_name,
                    "year": year,
                    "n": len(group),
                    "actual_miscov": y.mean(),
                    "mean_signal": x.mean(),
                    "roc_auc": auc,
                }
            )

    return pd.DataFrame(rows)


# -------------------------------------------------------------------
# DECILE ANALYSIS
# -------------------------------------------------------------------

def signal_deciles(
    test,
):

    rows = []

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

    for signal_name, signal_col in SIGNALS.items():

        data = test.copy()

        data[
            "signal_bin"
        ] = pd.cut(
            data[signal_col],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

        grouped = (
            data
            .groupby(
                "signal_bin",
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
                mean_signal=(
                    signal_col,
                    "mean",
                ),
            )
            .reset_index()
        )

        grouped["signal"] = signal_name

        rows.append(
            grouped
        )

    return pd.concat(
        rows,
        ignore_index=True,
    )


# -------------------------------------------------------------------
# LOW VS HIGH
# -------------------------------------------------------------------

def low_high_effect(
    test,
):

    rows = []

    for signal_name, signal_col in SIGNALS.items():

        for ticker, group in test.groupby(
            "ticker"
        ):

            low = group[
                group[signal_col] <= 0.20
            ]

            high = group[
                group[signal_col] >= 0.80
            ]

            if (
                len(low) == 0
                or len(high) == 0
            ):
                continue

            low_miscov = low[
                TARGET
            ].mean()

            high_miscov = high[
                TARGET
            ].mean()

            low_error = low[
                "abs_error"
            ].mean()

            high_error = high[
                "abs_error"
            ].mean()

            rows.append(
                {
                    "signal": signal_name,
                    "ticker": ticker,
                    "n_low": len(low),
                    "n_high": len(high),
                    "low_miscov": low_miscov,
                    "high_miscov": high_miscov,
                    "difference_miscov": (
                        high_miscov
                        - low_miscov
                    ),
                    "low_abs_error": low_error,
                    "high_abs_error": high_error,
                    "difference_abs_error": (
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
        "Loading candidate dataset..."
    )

    df = load_data()

    test = get_test_panel(
        df
    )

    print(
        f"\nTest rows: {len(test)}"
    )

    print(
        f"Tickers: "
        f"{test['ticker'].nunique()}"
    )

    print(
        f"Test period: "
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
        "\n========================================"
    )

    print(
        "OVERALL CANDIDATE SIGNALS"
    )

    print(
        "========================================"
    )

    print(
        overall.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Per ticker
    # ---------------------------------------------------------------

    ticker = per_ticker_metrics(
        test
    )

    print(
        "\n--- Per-ticker results ---"
    )

    print(
        ticker.to_string(
            index=False
        )
    )

    for signal in SIGNALS:

        subset = ticker[
            ticker["signal"]
            == signal
        ]

        valid = subset[
            "roc_auc"
        ].dropna()

        print(
            f"\n{signal}:"
        )

        print(
            f"  mean ticker AUC: "
            f"{valid.mean():.4f}"
        )

        print(
            f"  median ticker AUC: "
            f"{valid.median():.4f}"
        )

        print(
            f"  AUC > 0.50: "
            f"{(valid > 0.50).sum()} "
            f"/ {len(valid)}"
        )

    # ---------------------------------------------------------------
    # Yearly
    # ---------------------------------------------------------------

    yearly = yearly_metrics(
        test
    )

    print(
        "\n--- Yearly results ---"
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Deciles
    # ---------------------------------------------------------------

    deciles = signal_deciles(
        test
    )

    print(
        "\n--- Miscoverage by signal percentile ---"
    )

    print(
        deciles.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # High vs low
    # ---------------------------------------------------------------

    effects = low_high_effect(
        test
    )

    print(
        "\n--- High vs low signal ---"
    )

    print(
        effects.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    overall.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "overall_metrics.csv",
        ),
        index=False,
    )

    ticker.to_csv(
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
            "signal_deciles.csv",
        ),
        index=False,
    )

    effects.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "low_high_effect.csv",
        ),
        index=False,
    )

    print(
        f"\nSaved diagnostics to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()