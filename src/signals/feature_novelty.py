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
from features import get_all_features
from base_model import FEATURE_COLS


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

DEFAULT_WINDOW = 60
DEFAULT_MIN_PERIODS = 60

SIGNAL_NAME = "feature_novelty"


# -------------------------------------------------------------------
# CAUSAL ROBUST FEATURE DISTANCE
# -------------------------------------------------------------------

def causal_feature_novelty(
    df,
    feature_cols=FEATURE_COLS,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
):
    """
    Compute a strictly causal multivariate feature-novelty score.

    For each date t:

        1. Take only the previous `window` observations.
        2. Estimate the historical median of every feature.
        3. Estimate the historical IQR of every feature.
        4. Robustly standardize today's feature vector.
        5. Aggregate the standardized components using their RMS.

    Thus:

        novelty_t =
            sqrt(mean(z_1^2 + ... + z_p^2))

    where all normalization statistics come strictly from dates < t.

    This is deliberately a robust standardized distance rather than a
    classical Mahalanobis distance. It avoids repeatedly inverting a
    noisy covariance matrix on a small rolling window.

    Interpretation:

        low novelty  -> current market/input state resembles recent history
        high novelty -> current market/input state is unusual
    """

    data = df.copy()

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        data.index = pd.to_datetime(data.index)

    data = (
        data
        .sort_index()
        .copy()
    )

    missing = [
        col
        for col in feature_cols
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = data[
        feature_cols
    ].astype(float)

    n = len(X)

    novelty = np.full(
        n,
        np.nan,
        dtype=float,
    )

    # ---------------------------------------------------------------
    # Current value is NEVER part of the reference window.
    # ---------------------------------------------------------------

    for i in range(n):

        start = max(
            0,
            i - window,
        )

        historical = X.iloc[
            start:i
        ]

        if len(historical) < min_periods:
            continue

        current = X.iloc[
            i
        ]

        if not np.isfinite(
            current.to_numpy()
        ).all():
            continue

        # -----------------------------------------------------------
        # Historical robust center and scale.
        # -----------------------------------------------------------

        median = historical.median()

        q25 = historical.quantile(
            0.25
        )

        q75 = historical.quantile(
            0.75
        )

        iqr = q75 - q25

        # -----------------------------------------------------------
        # Replace zero-IQR dimensions with historical standard
        # deviation where possible.
        # -----------------------------------------------------------

        std = historical.std(
            ddof=0
        )

        scale = iqr.copy()

        zero_scale = (
            scale <= 1e-12
        )

        scale.loc[
            zero_scale
        ] = std.loc[
            zero_scale
        ]

        # If a feature is completely constant in history, use 1 as
        # scale. Its resulting standardized contribution will then be
        # its absolute deviation from the historical median rather
        # than causing division by zero.
        scale = scale.replace(
            0,
            1.0,
        )

        # -----------------------------------------------------------
        # Robust standardized vector.
        # -----------------------------------------------------------

        z = (
            current - median
        ) / scale

        # Guard against numerical issues.
        z = z.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

        if z.isna().any():
            continue

        # -----------------------------------------------------------
        # Multivariate novelty.
        #
        # RMS prevents the number of features from automatically
        # increasing the scale.
        # -----------------------------------------------------------

        novelty[i] = float(
            np.sqrt(
                np.mean(
                    np.square(
                        z.to_numpy()
                    )
                )
            )
        )

    result = pd.DataFrame(
        {
            "prediction_date": data.index,
            SIGNAL_NAME: novelty,
        }
    )

    return result


# -------------------------------------------------------------------
# CAUSAL PERCENTILE OF NOVELTY
# -------------------------------------------------------------------

def causal_signal_percentile(
    series,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
):
    """
    Convert a novelty score into a causal historical percentile.

    At date t, the percentile is calculated relative to previous
    novelty observations only.

    Example:

        feature_novelty_pct = 0.95

    means today's novelty is unusually high relative to its recent
    historical novelty distribution.
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

        percentile = (
            less
            + 0.5 * equal
        ) / len(historical)

        result[i] = percentile

    return pd.Series(
        result,
        index=series.index,
    )


# -------------------------------------------------------------------
# BUILD SIGNAL FOR ONE TICKER
# -------------------------------------------------------------------

def build_feature_novelty(
    ticker,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
):
    """
    Build feature novelty for one ticker.

    Returns:

        prediction_date
        feature_novelty
        feature_novelty_pct

    No target, actual outcome, residual, or conformal information is
    used anywhere in this calculation.
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
        window=window,
        min_periods=min_periods,
    )

    novelty_df[
        "feature_novelty_pct"
    ] = causal_signal_percentile(
        novelty_df[
            "feature_novelty"
        ],
        window=window,
        min_periods=min_periods,
    )

    novelty_df["ticker"] = ticker

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    valid = novelty_df[
        "feature_novelty"
    ].dropna()

    if len(valid) == 0:
        raise ValueError(
            f"{ticker}: no valid novelty observations."
        )

    pct_valid = novelty_df[
        "feature_novelty_pct"
    ].dropna()

    if not (
        pct_valid.between(
            0,
            1,
        )
    ).all():
        raise AssertionError(
            f"{ticker}: novelty percentile outside [0, 1]."
        )

    if not novelty_df[
        "prediction_date"
    ].is_monotonic_increasing:
        raise AssertionError(
            f"{ticker}: dates are not chronological."
        )

    return novelty_df[
        [
            "prediction_date",
            "ticker",
            "feature_novelty",
            "feature_novelty_pct",
        ]
    ].reset_index(
        drop=True
    )


# -------------------------------------------------------------------
# BUILD ALL TICKERS
# -------------------------------------------------------------------

def build_all_feature_novelty(
    tickers,
    window=DEFAULT_WINDOW,
    min_periods=DEFAULT_MIN_PERIODS,
):
    """
    Build feature novelty for the supplied research universe.
    """

    results = []

    for ticker in tickers:

        print(
            f"Building feature novelty for {ticker}..."
        )

        result = build_feature_novelty(
            ticker,
            window=window,
            min_periods=min_periods,
        )

        results.append(
            result
        )

        print(
            f"  -> {len(result)} rows"
        )

    combined = pd.concat(
        results,
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    return combined


# -------------------------------------------------------------------
# MAIN TEST
# -------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "Testing feature novelty on AAPL..."
    )

    aapl = build_feature_novelty(
        "AAPL",
        window=60,
        min_periods=60,
    )

    print(
        "\n--- AAPL novelty summary ---"
    )

    print(
        aapl[
            [
                "feature_novelty",
                "feature_novelty_pct",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\n--- First 15 rows ---"
    )

    print(
        aapl.head(15)
        .to_string(
            index=False
        )
    )

    print(
        "\n--- NaNs ---"
    )

    print(
        aapl.isna()
        .sum()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Critical causal sanity check.
    # ---------------------------------------------------------------

    first_valid = aapl[
        "feature_novelty"
    ].first_valid_index()

    print(
        "\nFirst valid novelty index:",
        first_valid,
    )

    if first_valid is not None:

        print(
            "First valid novelty date:",
            aapl.loc[
                first_valid,
                "prediction_date",
            ],
        )

    print(
        "\nFeature novelty test completed."
    )