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

SIGNALS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "signals",
)

RAW_SIGNALS_PATH = os.path.join(
    SIGNALS_DIR,
    "all_tickers_signals.csv",
)

NORMALIZED_SIGNALS_PATH = os.path.join(
    SIGNALS_DIR,
    "all_tickers_signals_normalized.csv",
)


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

NORMALIZATION_WINDOW = 60
MIN_PERIODS = 60

SIGNAL_COLS = [
    "miscoverage",
    "ks_stat",
    "disagreement",
]


# -------------------------------------------------------------------
# CAUSAL ROLLING PERCENTILE
# -------------------------------------------------------------------

def causal_rolling_percentile(
    series,
    window=NORMALIZATION_WINDOW,
    min_periods=MIN_PERIODS,
):
    """
    Compute a strictly causal empirical percentile.

    For observation t:

        percentile_t =
            rank(X_t relative to X_{t-window}, ..., X_{t-1})

    The current observation X_t is NEVER included in the
    reference window.

    Ties receive half weight.
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

        # -----------------------------------------------------------
        # Historical window ends at t-1.
        # -----------------------------------------------------------

        start = max(
            0,
            i - window,
        )

        past = values[start:i]

        # Need enough historical observations.
        if len(past) < min_periods:
            continue

        current = values[i]

        if not np.isfinite(current):
            continue

        past = past[
            np.isfinite(past)
        ]

        if len(past) < min_periods:
            continue

        less = np.sum(
            past < current
        )

        equal = np.sum(
            past == current
        )

        # Mid-rank percentile:
        #
        # values strictly below current
        # + half of tied observations
        #
        percentile = (
            less
            + 0.5 * equal
        ) / len(past)

        result[i] = percentile

    return pd.Series(
        result,
        index=series.index,
    )


# -------------------------------------------------------------------
# NORMALIZE ONE TICKER
# -------------------------------------------------------------------

def normalize_ticker(
    ticker_df,
    signal_cols=SIGNAL_COLS,
    window=NORMALIZATION_WINDOW,
    min_periods=MIN_PERIODS,
):
    """
    Add causal percentile-rank versions of the reliability signals.

    New columns:

        miscoverage_pct
        ks_stat_pct
        disagreement_pct

    Interpretation:

        0.95 = unusually high relative to the ticker's own
               recent history

        0.50 = approximately typical

        0.05 = unusually low
    """

    df = (
        ticker_df
        .copy()
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )

    for col in signal_cols:

        if col not in df.columns:
            raise ValueError(
                f"Missing signal column: {col}"
            )

        pct_col = f"{col}_pct"

        df[pct_col] = (
            causal_rolling_percentile(
                df[col],
                window=window,
                min_periods=min_periods,
            )
        )

    return df


# -------------------------------------------------------------------
# BUILD NORMALIZED PANEL
# -------------------------------------------------------------------

def build_normalized_panel(
    input_path=RAW_SIGNALS_PATH,
    output_path=NORMALIZED_SIGNALS_PATH,
    window=NORMALIZATION_WINDOW,
    min_periods=MIN_PERIODS,
):
    """
    Convert the raw aligned signal panel into a causally normalized
    panel.

    Normalization is performed independently for each ticker.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Raw signal panel not found:\n{input_path}\n\n"
            "Run compare_signals.py first."
        )

    df = pd.read_csv(
        input_path,
        parse_dates=[
            "prediction_date",
            "outcome_date",
            "date",
        ],
    )

    required_cols = {
        "prediction_date",
        "outcome_date",
        "date",
        "ticker",
        *SIGNAL_COLS,
    }

    missing = required_cols.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Raw signal panel is missing columns: "
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
    # Normalize independently for each ticker.
    # ---------------------------------------------------------------

    normalized_parts = []

    for ticker, ticker_df in df.groupby(
        "ticker",
        sort=False,
    ):

        print(
            f"Normalizing {ticker}..."
        )

        normalized = normalize_ticker(
            ticker_df,
            signal_cols=SIGNAL_COLS,
            window=window,
            min_periods=min_periods,
        )

        normalized_parts.append(
            normalized
        )

    normalized_df = pd.concat(
        normalized_parts,
        ignore_index=True,
    )

    normalized_df = (
        normalized_df
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    normalized_cols = [
        f"{col}_pct"
        for col in SIGNAL_COLS
    ]

    print(
        "\n--- Validation ---"
    )

    print(
        f"Rows: {len(normalized_df)}"
    )

    print(
        f"Tickers: "
        f"{normalized_df['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{normalized_df['prediction_date'].min().date()} "
        f"-> "
        f"{normalized_df['prediction_date'].max().date()}"
    )

    # ---------------------------------------------------------------
    # Chronology
    # ---------------------------------------------------------------

    for ticker, ticker_df in normalized_df.groupby(
        "ticker"
    ):

        if not ticker_df[
            "prediction_date"
        ].is_monotonic_increasing:

            raise AssertionError(
                f"{ticker}: prediction dates are not chronological."
            )

        if not (
            ticker_df["outcome_date"]
            > ticker_df["prediction_date"]
        ).all():

            raise AssertionError(
                f"{ticker}: outcome dates are not after "
                f"prediction dates."
            )

    # ---------------------------------------------------------------
    # Range validation
    # ---------------------------------------------------------------

    for col in normalized_cols:

        if col not in normalized_df.columns:

            raise AssertionError(
                f"Missing normalized column: {col}"
            )

        valid = normalized_df[
            col
        ].dropna()

        if not (
            (valid >= 0)
            & (valid <= 1)
        ).all():

            raise AssertionError(
                f"{col} contains values outside [0, 1]."
            )

    # ---------------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------------

    print(
        "\n--- Normalized NaNs ---"
    )

    print(
        normalized_df[
            normalized_cols
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    normalized_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved normalized signal panel to:"
        f"\n{output_path}"
    )

    # ---------------------------------------------------------------
    # Compare raw ticker scales
    # ---------------------------------------------------------------

    print(
        "\n--- Raw ticker-level means ---"
    )

    raw_summary = (
        normalized_df
        .groupby("ticker")[
            SIGNAL_COLS
        ]
        .mean()
    )

    print(
        raw_summary.to_string()
    )

    print(
        "\n--- Normalized ticker-level means ---"
    )

    normalized_summary = (
        normalized_df
        .groupby("ticker")[
            normalized_cols
        ]
        .mean()
    )

    print(
        normalized_summary.to_string()
    )

    # ---------------------------------------------------------------
    # Global normalized distribution
    # ---------------------------------------------------------------

    print(
        "\n--- Global normalized statistics ---"
    )

    print(
        normalized_df[
            normalized_cols
        ]
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
        .to_string()
    )

    # ---------------------------------------------------------------
    # Example
    # ---------------------------------------------------------------

    print(
        "\n--- Example: AAPL ---"
    )

    aapl = normalized_df[
        normalized_df["ticker"] == "AAPL"
    ].head(10)

    print(
        aapl[
            [
                "prediction_date",
                "miscoverage",
                "miscoverage_pct",
                "ks_stat",
                "ks_stat_pct",
                "disagreement",
                "disagreement_pct",
            ]
        ].to_string(
            index=False
        )
    )

    return normalized_df


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    build_normalized_panel(
        window=60,
        min_periods=60,
    )