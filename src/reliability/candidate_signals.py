import os
import sys
import numpy as np
import pandas as pd


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
    "reliability_dataset.csv",
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "reliability_dataset_candidates.csv",
)


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

EWMA_SPAN = 20
EWMA_MIN_PERIODS = 20

PERCENTILE_WINDOW = 60
PERCENTILE_MIN_PERIODS = 60


# -------------------------------------------------------------------
# CAUSAL ROLLING PERCENTILE
# -------------------------------------------------------------------

def causal_rolling_percentile(
    series,
    window=PERCENTILE_WINDOW,
    min_periods=PERCENTILE_MIN_PERIODS,
):
    """
    Strictly causal rolling empirical percentile.

    The percentile at t is calculated relative to observations
    strictly before t.

    Current observation is NEVER included in its own reference
    distribution.
    """

    values = pd.Series(
        series,
        dtype=float,
    ).to_numpy()

    n = len(values)

    result = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for i in range(n):

        start = max(
            0,
            i - window,
        )

        historical = values[
            start:i
        ]

        historical = historical[
            np.isfinite(historical)
        ]

        if len(historical) < min_periods:
            continue

        current = values[i]

        if not np.isfinite(current):
            continue

        less = np.sum(
            historical < current
        )

        equal = np.sum(
            historical == current
        )

        result[i] = (
            less
            + 0.5 * equal
        ) / len(historical)

    return pd.Series(
        result,
        index=series.index,
    )


# -------------------------------------------------------------------
# CAUSAL EWMA MIScoverage
# -------------------------------------------------------------------

def causal_ewma_miscoverage(
    miscovered,
    span=EWMA_SPAN,
    min_periods=EWMA_MIN_PERIODS,
):
    """
    Calculate exponentially weighted recent miscoverage.

    Critical timing rule:

        miscoverage at t is unknown at prediction time t.

    Therefore the EWMA used for prediction t only uses:

        miscovered[0 ... t-1]

    The current observation is shifted out before the EWMA is
    calculated.

    This produces the actual historical failure state available
    to a risk manager before the current prediction outcome arrives.
    """

    history = pd.Series(
        miscovered,
        dtype=float,
    ).shift(1)

    ewma = (
        history
        .ewm(
            span=span,
            adjust=False,
            min_periods=min_periods,
        )
        .mean()
    )

    return ewma


# -------------------------------------------------------------------
# NORMALIZE EWMA
# -------------------------------------------------------------------

def add_ewma_signals(
    df,
):
    """
    Add:
        ewma_miscoverage
        ewma_miscoverage_pct
    """

    df = df.copy()

    df[
        "ewma_miscoverage"
    ] = causal_ewma_miscoverage(
        df["miscovered"]
    )

    df[
        "ewma_miscoverage_pct"
    ] = causal_rolling_percentile(
        df["ewma_miscoverage"],
        window=PERCENTILE_WINDOW,
        min_periods=PERCENTILE_MIN_PERIODS,
    )

    return df


# -------------------------------------------------------------------
# NORMALIZE CONFORMAL INTERVAL WIDTH
# -------------------------------------------------------------------

def add_interval_width_signal(
    df,
):
    """
    Add a causal percentile representation of conformal interval width.

    interval_width is already available on prediction date t.

    The percentile asks:

        How unusually wide is today's conformal interval relative
        to this ticker's own recent history?
    """

    df = df.copy()

    if "interval_width" not in df.columns:
        raise ValueError(
            "Dataset does not contain interval_width."
        )

    df[
        "interval_width_pct"
    ] = causal_rolling_percentile(
        df["interval_width"],
        window=PERCENTILE_WINDOW,
        min_periods=PERCENTILE_MIN_PERIODS,
    )

    return df


# -------------------------------------------------------------------
# BUILD CANDIDATE SIGNAL DATASET
# -------------------------------------------------------------------

def build_candidate_dataset():
    """
    Add the two remaining candidate reliability signals.

    Existing columns are preserved.

    Added columns:

        ewma_miscoverage
        ewma_miscoverage_pct
        interval_width_pct
    """

    if not os.path.exists(
        INPUT_PATH
    ):
        raise FileNotFoundError(
            f"Reliability dataset not found:\n"
            f"{INPUT_PATH}\n\n"
            "Run build_dataset.py first."
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
        "miscovered",
        "interval_width",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Dataset missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        df
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Build independently within each ticker.
    # ---------------------------------------------------------------

    parts = []

    for ticker, ticker_df in df.groupby(
        "ticker",
        sort=False,
    ):

        print(
            f"Building candidate signals for {ticker}..."
        )

        ticker_df = ticker_df.copy()

        ticker_df = add_ewma_signals(
            ticker_df
        )

        ticker_df = add_interval_width_signal(
            ticker_df
        )

        parts.append(
            ticker_df
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Causal / range validation
    # ---------------------------------------------------------------

    candidate_cols = [
        "ewma_miscoverage_pct",
        "interval_width_pct",
    ]

    for col in candidate_cols:

        valid = result[
            col
        ].dropna()

        if not valid.between(
            0,
            1,
        ).all():

            raise AssertionError(
                f"{col} contains values outside [0,1]."
            )

    # Prediction must occur before outcome.
    if not (
        result["prediction_date"]
        < result["outcome_date"]
    ).all():

        raise AssertionError(
            "Invalid prediction/outcome chronology."
        )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "CANDIDATE SIGNAL SUMMARY"
    )

    print(
        "========================================"
    )

    print(
        f"Rows: {len(result)}"
    )

    print(
        f"Tickers: "
        f"{result['ticker'].nunique()}"
    )

    print(
        "\n--- Candidate NaNs ---"
    )

    print(
        result[
            [
                "ewma_miscoverage",
                "ewma_miscoverage_pct",
                "interval_width_pct",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print(
        "\n--- Candidate statistics ---"
    )

    print(
        result[
            [
                "ewma_miscoverage",
                "ewma_miscoverage_pct",
                "interval_width_pct",
            ]
        ]
        .describe()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved candidate-signal dataset to:"
        f"\n{OUTPUT_PATH}"
    )

    # ---------------------------------------------------------------
    # AAPL example
    # ---------------------------------------------------------------

    print(
        "\n--- AAPL example ---"
    )

    aapl = result[
        result["ticker"] == "AAPL"
    ].head(15)

    print(
        aapl[
            [
                "prediction_date",
                "miscovered",
                "ewma_miscoverage",
                "ewma_miscoverage_pct",
                "interval_width",
                "interval_width_pct",
            ]
        ].to_string(
            index=False
        )
    )

    return result


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    build_candidate_dataset()