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

from config import BASE_DIR, TICKERS
from conformal import run_aci


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

SIGNALS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "signals",
)

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "predictions",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "reliability",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


NORMALIZED_SIGNALS_PATH = os.path.join(
    SIGNALS_DIR,
    "all_tickers_signals_normalized.csv",
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "reliability_dataset.csv",
)


# -------------------------------------------------------------------
# SETTINGS
# -------------------------------------------------------------------

WARMUP = 30

HORIZONS = (
    5,
    10,
    20,
)

SKIP_TICKERS = (
    "COIN",
)


# -------------------------------------------------------------------
# FORWARD MIScoverage
# -------------------------------------------------------------------

def forward_miscoverage(
    miscovered,
    horizons=HORIZONS,
):
    """
    For each prediction at time t, calculate the realized
    miscoverage rate over the NEXT N predictions.

    Important:

        forward_miscov_Nd[t]
            =
        mean(miscovered[t+1 : t+N])

    The current prediction is NOT included.

    These are secondary research targets. The primary target is
    the pointwise miscoverage label for the current prediction.
    """

    series = pd.Series(
        np.asarray(
            miscovered,
            dtype=float,
        )
    )

    results = {}

    for horizon in horizons:

        values = np.full(
            len(series),
            np.nan,
            dtype=float,
        )

        for i in range(len(series)):

            start = i + 1
            end = i + 1 + horizon

            if end > len(series):
                continue

            window = series.iloc[
                start:end
            ]

            if window.isna().any():
                continue

            values[i] = window.mean()

        results[horizon] = values

    return results


# -------------------------------------------------------------------
# BUILD ACI LABELS FOR ONE TICKER
# -------------------------------------------------------------------

def build_aci_labels(
    predictions,
    warmup=WARMUP,
):
    """
    Recompute ACI on the walk-forward prediction stream and construct
    labels for each prediction.

    Primary target:

        miscovered
            1 = prediction interval failed to cover outcome
            0 = prediction interval covered outcome

    Secondary quantities:

        abs_error
        interval_width
        normalized_error
        future 5/10/20-day miscoverage
    """

    predictions = (
        predictions
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )

    required = {
        "prediction_date",
        "outcome_date",
        "actual",
        "predicted",
        "ticker",
    }

    missing = required.difference(
        predictions.columns
    )

    if missing:
        raise ValueError(
            f"Prediction data missing columns: "
            f"{sorted(missing)}"
        )

    aci = run_aci(
        predictions["actual"],
        predictions["predicted"],
        prediction_dates=predictions[
            "prediction_date"
        ],
        outcome_dates=predictions[
            "outcome_date"
        ],
        alpha_target=0.10,
        gamma=0.01,
        warmup=warmup,
    )

    # ACI results begin at the warmup index.
    usable = predictions.iloc[
        warmup:
    ].copy().reset_index(drop=True)

    usable["aci_alpha"] = aci[
        "alpha_t"
    ]

    usable["interval_margin"] = aci[
        "margin_t"
    ]

    usable["lower"] = aci[
        "lower_t"
    ]

    usable["upper"] = aci[
        "upper_t"
    ]

    usable["covered"] = (
        aci["covered_t"]
        .astype(int)
    )

    usable["miscovered"] = (
        1 - usable["covered"]
    )

    # ---------------------------------------------------------------
    # Prediction error
    # ---------------------------------------------------------------

    usable["abs_error"] = (
        usable["actual"]
        - usable["predicted"]
    ).abs()

    usable["signed_error"] = (
        usable["actual"]
        - usable["predicted"]
    )

    usable["interval_width"] = (
        usable["upper"]
        - usable["lower"]
    )

    # How large is the realized error relative to the current
    # conformal uncertainty margin?
    usable["normalized_error"] = (
        usable["abs_error"]
        / (
            usable["interval_margin"]
            + 1e-12
        )
    )

    # ---------------------------------------------------------------
    # Forward-horizon secondary labels
    # ---------------------------------------------------------------

    future_labels = forward_miscoverage(
        usable["miscovered"].to_numpy(),
        horizons=HORIZONS,
    )

    for horizon, values in future_labels.items():

        usable[
            f"future_miscov_{horizon}d"
        ] = values

    # ---------------------------------------------------------------
    # Explicit information timing
    # ---------------------------------------------------------------

    usable["label_known_date"] = (
        usable["outcome_date"]
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    if not (
        usable["prediction_date"]
        < usable["outcome_date"]
    ).all():

        raise AssertionError(
            "prediction_date must occur before "
            "outcome_date."
        )

    if not usable[
        "covered"
    ].isin([0, 1]).all():

        raise AssertionError(
            "covered must contain only 0/1."
        )

    if not usable[
        "miscovered"
    ].isin([0, 1]).all():

        raise AssertionError(
            "miscovered must contain only 0/1."
        )

    return usable


# -------------------------------------------------------------------
# LOAD NORMALIZED SIGNALS
# -------------------------------------------------------------------

def load_normalized_signals():
    """
    Load the causally normalized signal panel.
    """

    if not os.path.exists(
        NORMALIZED_SIGNALS_PATH
    ):
        raise FileNotFoundError(
            f"Normalized signal file not found:\n"
            f"{NORMALIZED_SIGNALS_PATH}\n\n"
            "Run normalization.py first."
        )

    df = pd.read_csv(
        NORMALIZED_SIGNALS_PATH,
        parse_dates=[
            "prediction_date",
            "outcome_date",
            "date",
        ],
    )

    required = {
        "prediction_date",
        "outcome_date",
        "ticker",
        "miscoverage",
        "ks_stat",
        "disagreement",
        "miscoverage_pct",
        "ks_stat_pct",
        "disagreement_pct",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Normalized signal panel is missing: "
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
# LOAD PREDICTIONS
# -------------------------------------------------------------------

def load_predictions(
    ticker,
):
    """
    Load one ticker's walk-forward predictions.
    """

    path = os.path.join(
        PREDICTIONS_DIR,
        f"{ticker}.csv",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No predictions file for {ticker}: "
            f"{path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "prediction_date",
            "outcome_date",
            "date",
        ],
    )

    return (
        df
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )


# -------------------------------------------------------------------
# BUILD COMPLETE DATASET
# -------------------------------------------------------------------

def build_reliability_dataset(
    tickers=None,
    skip_tickers=SKIP_TICKERS,
):
    """
    Build the complete research dataset.

    One row = one prediction made at prediction_date=t.

    Features:
        raw reliability signals
        causal ticker-normalized signals

    Primary labels:
        covered
        miscovered

    Secondary labels:
        future_miscov_5d
        future_miscov_10d
        future_miscov_20d
    """

    if tickers is None:
        tickers = TICKERS

    normalized_signals = (
        load_normalized_signals()
    )

    all_rows = []

    for ticker in tickers:

        if ticker in skip_tickers:

            print(
                f"Skipping {ticker} "
                f"(insufficient history)"
            )

            continue

        print(
            f"\nBuilding reliability dataset "
            f"for {ticker}..."
        )

        predictions = load_predictions(
            ticker
        )

        labels = build_aci_labels(
            predictions,
            warmup=WARMUP,
        )

        ticker_signals = (
            normalized_signals[
                normalized_signals["ticker"]
                == ticker
            ]
            .copy()
        )

        # -----------------------------------------------------------
        # Only retain rows for which all three causally normalized
        # signals are available.
        # -----------------------------------------------------------

        ticker_signals = ticker_signals.dropna(
            subset=[
                "miscoverage_pct",
                "ks_stat_pct",
                "disagreement_pct",
            ]
        )

        # -----------------------------------------------------------
        # Merge using prediction_date.
        #
        # This is the critical alignment:
        #
        # features(t)
        #        +
        # label(t), whose outcome is observed at t+1
        #
        # The label is NEVER used as a feature.
        # -----------------------------------------------------------

        merged = ticker_signals.merge(
            labels[
                [
                    "prediction_date",
                    "outcome_date",
                    "actual",
                    "predicted",
                    "lower",
                    "upper",
                    "aci_alpha",
                    "interval_margin",
                    "interval_width",
                    "covered",
                    "miscovered",
                    "abs_error",
                    "signed_error",
                    "normalized_error",
                    "future_miscov_5d",
                    "future_miscov_10d",
                    "future_miscov_20d",
                    "label_known_date",
                ]
            ],
            on=[
                "prediction_date",
                "outcome_date",
            ],
            how="inner",
            validate="one_to_one",
        )

        if len(merged) == 0:

            raise ValueError(
                f"{ticker}: no rows survived "
                "signal/label merge."
            )

        # -----------------------------------------------------------
        # Strong chronology validation.
        # -----------------------------------------------------------

        if not (
            merged["prediction_date"]
            < merged["outcome_date"]
        ).all():

            raise AssertionError(
                f"{ticker}: invalid prediction/outcome chronology."
            )

        # -----------------------------------------------------------
        # Ensure normalized features are valid.
        # -----------------------------------------------------------

        feature_cols = [
            "miscoverage_pct",
            "ks_stat_pct",
            "disagreement_pct",
        ]

        for col in feature_cols:

            if not (
                merged[col]
                .between(0, 1)
            ).all():

                raise AssertionError(
                    f"{ticker}: {col} outside [0,1]."
                )

        all_rows.append(
            merged
        )

        print(
            f"  -> {len(merged)} usable rows"
        )

    if not all_rows:

        raise RuntimeError(
            "No valid ticker datasets were produced."
        )

    dataset = pd.concat(
        all_rows,
        ignore_index=True,
    )

    dataset = (
        dataset
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------------
    # FINAL VALIDATION
    # ----------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL RELIABILITY DATASET"
    )

    print(
        "========================================"
    )

    print(
        f"Rows: {len(dataset)}"
    )

    print(
        f"Tickers: "
        f"{dataset['ticker'].nunique()}"
    )

    print(
        f"Prediction date range: "
        f"{dataset['prediction_date'].min().date()} "
        f"-> "
        f"{dataset['prediction_date'].max().date()}"
    )

    print(
        f"Outcome date range: "
        f"{dataset['outcome_date'].min().date()} "
        f"-> "
        f"{dataset['outcome_date'].max().date()}"
    )

    # ---------------------------------------------------------------
    # Required columns
    # ---------------------------------------------------------------

    required_final = [
        "prediction_date",
        "outcome_date",
        "ticker",
        "miscoverage_pct",
        "ks_stat_pct",
        "disagreement_pct",
        "covered",
        "miscovered",
    ]

    missing_final = [
        col
        for col in required_final
        if col not in dataset.columns
    ]

    if missing_final:

        raise AssertionError(
            f"Final dataset missing: "
            f"{missing_final}"
        )

    # ---------------------------------------------------------------
    # Missing values in primary features/labels
    # ---------------------------------------------------------------

    print(
        "\n--- Primary feature/label NaNs ---"
    )

    primary_cols = [
        "miscoverage_pct",
        "ks_stat_pct",
        "disagreement_pct",
        "covered",
        "miscovered",
    ]

    print(
        dataset[
            primary_cols
        ]
        .isna()
        .sum()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Primary target balance
    # ---------------------------------------------------------------

    print(
        "\n--- Primary target balance ---"
    )

    print(
        dataset["covered"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        .to_string()
    )

    print(
        "\nEmpirical miscoverage rate: "
        f"{dataset['miscovered'].mean():.4f}"
    )

    print(
        "Empirical coverage rate: "
        f"{dataset['covered'].mean():.4f}"
    )

    # ---------------------------------------------------------------
    # Secondary target means
    # ---------------------------------------------------------------

    print(
        "\n--- Secondary forward miscoverage ---"
    )

    print(
        dataset[
            [
                "future_miscov_5d",
                "future_miscov_10d",
                "future_miscov_20d",
            ]
        ]
        .mean()
        .to_string()
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nSaved reliability dataset to:"
        f"\n{OUTPUT_PATH}"
    )

    return dataset


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    dataset = (
        build_reliability_dataset()
    )

    print(
        "\n--- Example rows ---"
    )

    print(
        dataset[
            [
                "prediction_date",
                "outcome_date",
                "ticker",
                "miscoverage_pct",
                "ks_stat_pct",
                "disagreement_pct",
                "covered",
                "miscovered",
                "abs_error",
                "interval_width",
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )