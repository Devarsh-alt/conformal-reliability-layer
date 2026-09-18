import os
import sys
import numpy as np
import pandas as pd

# Allow imports from src/
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
    ),
)

from conformal import run_aci, nonconformity_scores
from features import get_all_features
from base_model import build_target, FEATURE_COLS
from config import BASE_DIR, TICKERS

from ks_drift import rolling_ks_drift
from rolling_miscoverage import rolling_miscoverage
from bootstrap_disagreement import (
    run_bootstrap_disagreement_walk_forward,
)


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

SIGNALS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "signals",
)

os.makedirs(
    SIGNALS_DIR,
    exist_ok=True,
)

PREDICTIONS_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "predictions",
)


# -------------------------------------------------------------------
# LOAD PREDICTIONS
# -------------------------------------------------------------------

def _load_predictions(ticker):
    """
    Load the walk-forward prediction stream for one ticker.

    Each row represents a prediction made on prediction_date=t
    whose realized outcome becomes available on outcome_date.
    """

    path = os.path.join(
        PREDICTIONS_DIR,
        f"{ticker}.csv",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Prediction file not found: {path}"
        )

    df = pd.read_csv(
        path,
        parse_dates=[
            "prediction_date",
            "outcome_date",
        ],
    )

    required = {
        "prediction_date",
        "outcome_date",
        "actual",
        "predicted",
        "ticker",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{ticker}: missing required columns: "
            f"{sorted(missing)}"
        )

    df = (
        df.sort_values("prediction_date")
        .reset_index(drop=True)
    )

    # Chronological checks.
    if not df["prediction_date"].is_monotonic_increasing:
        raise ValueError(
            f"{ticker}: prediction dates are not chronological."
        )

    if not (
        df["outcome_date"] > df["prediction_date"]
    ).all():
        raise ValueError(
            f"{ticker}: every outcome_date must occur "
            f"after prediction_date."
        )

    return df


# -------------------------------------------------------------------
# MIScoverage SIGNAL
# -------------------------------------------------------------------

def _build_miscoverage_signal(
    pred_df,
    warmup=30,
    roll_window=20,
):
    """
    Build a causal rolling-miscoverage signal.

    Important timing:

        Prediction made at t
              |
              v
        Outcome observed at t+1
              |
              v
        Coverage becomes known
              |
              v
        This information can be used for the next prediction.

    Therefore the coverage result for a prediction is associated
    with its outcome_date rather than its original prediction_date.
    """

    aci_result = run_aci(
        pred_df["actual"],
        pred_df["predicted"],
        prediction_dates=pred_df["prediction_date"],
        outcome_dates=pred_df["outcome_date"],
        alpha_target=0.10,
        gamma=0.01,
        warmup=warmup,
    )

    covered_t = aci_result["covered_t"]

    rolling_values = rolling_miscoverage(
        covered_t,
        window=roll_window,
    )

    # ACI evaluation begins at the original observation index
    # `warmup`. The coverage at that observation becomes known
    # on its outcome_date.
    outcome_dates = (
        pred_df["outcome_date"]
        .iloc[warmup:]
        .reset_index(drop=True)
    )

    miscov_df = pd.DataFrame(
        {
            "prediction_date": outcome_dates,
            "miscoverage": rolling_values,
        }
    )

    return miscov_df


# -------------------------------------------------------------------
# KS DRIFT SIGNAL
# -------------------------------------------------------------------

def _build_ks_signal(
    pred_df,
    roll_window=20,
    reference_window=20,
):
    """
    Build the causal KS drift signal.

    The KS statistic is calculated from historical prediction errors.
    It is therefore aligned to the current prediction_date.
    """

    scores = nonconformity_scores(
        pred_df["actual"],
        pred_df["predicted"],
    )

    ks_result = rolling_ks_drift(
        scores,
        window=roll_window,
        reference_window=reference_window,
    )

    ks_df = pd.DataFrame(
        {
            "prediction_date": pred_df[
                "prediction_date"
            ],
            "ks_stat": ks_result["ks_stat"],
            "ks_pvalue": ks_result["ks_pvalue"],
        }
    )

    return ks_df


# -------------------------------------------------------------------
# DISAGREEMENT SIGNAL
# -------------------------------------------------------------------

def _build_disagreement_signal(
    ticker,
    n_bootstrap=10,
):
    """
    Build the prediction-instability / disagreement signal.

    The disagreement signal is generated from predictions available
    at the current prediction date.
    """

    features = get_all_features()

    if ticker not in features:
        raise ValueError(
            f"No feature data available for {ticker}."
        )

    feat_df = build_target(
        features[ticker]
    )

    disagreement_df = (
        run_bootstrap_disagreement_walk_forward(
            feat_df,
            FEATURE_COLS,
            n_bootstrap=n_bootstrap,
        )
    )

    disagreement_df["prediction_date"] = (
        pd.to_datetime(
            disagreement_df["date"]
        )
    )

    disagreement_df = disagreement_df[
        [
            "prediction_date",
            "disagreement",
            "ensemble_mean",
            "fold_id",
        ]
    ].copy()

    return disagreement_df


# -------------------------------------------------------------------
# BUILD ALIGNED SIGNAL PANEL FOR ONE TICKER
# -------------------------------------------------------------------

def build_aligned_signals(
    ticker="AAPL",
    warmup=30,
    roll_window=20,
    reference_window=20,
    n_bootstrap=10,
):
    """
    Build the three reliability signals on a common timeline.

    Signals:

        1. miscoverage
           Recent conformal failures whose outcomes are already known.

        2. KS drift
           Distributional change in historical prediction errors.

        3. disagreement
           Prediction instability at the current prediction date.

    Every returned row corresponds to a prediction date and contains
    only information available at that date.
    """

    print(
        f"\nBuilding aligned signals for {ticker}..."
    )

    pred_df = _load_predictions(ticker)

    # ---------------------------------------------------------------
    # 1. CAUSAL ROLLING MIScoverage
    # ---------------------------------------------------------------

    print(
        "  -> computing causal rolling miscoverage"
    )

    miscov_df = _build_miscoverage_signal(
        pred_df,
        warmup=warmup,
        roll_window=roll_window,
    )

    # ---------------------------------------------------------------
    # 2. CAUSAL KS DRIFT
    # ---------------------------------------------------------------

    print(
        "  -> computing causal KS drift"
    )

    ks_df = _build_ks_signal(
        pred_df,
        roll_window=roll_window,
        reference_window=reference_window,
    )

    # ---------------------------------------------------------------
    # 3. PREDICTION DISAGREEMENT
    # ---------------------------------------------------------------

    print(
        f"  -> computing bootstrap disagreement "
        f"(B={n_bootstrap})"
    )

    disagreement_df = _build_disagreement_signal(
        ticker,
        n_bootstrap=n_bootstrap,
    )

    # ---------------------------------------------------------------
    # MERGE ON COMMON PREDICTION TIMELINE
    # ---------------------------------------------------------------

    merged = disagreement_df.merge(
        ks_df,
        on="prediction_date",
        how="inner",
    )

    merged = merged.merge(
        miscov_df,
        on="prediction_date",
        how="inner",
    )

    # Add the actual outcome date belonging to this prediction.
    timeline = pred_df[
        [
            "prediction_date",
            "outcome_date",
        ]
    ].copy()

    merged = merged.merge(
        timeline,
        on="prediction_date",
        how="left",
        validate="one_to_one",
    )

    merged["ticker"] = ticker

    # Backward-compatible date alias.
    merged["date"] = merged["prediction_date"]

    merged = (
        merged
        .sort_values("prediction_date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # REMOVE WARM-UP ROWS
    # ---------------------------------------------------------------

    signal_cols = [
        "miscoverage",
        "ks_stat",
        "disagreement",
    ]

    # Initial rolling windows legitimately contain NaN because
    # insufficient historical information exists.
    #
    # These rows do not represent a complete information set, so
    # they are removed rather than filled or interpolated.
    before_drop = len(merged)

    merged = (
        merged
        .dropna(subset=signal_cols)
        .reset_index(drop=True)
    )

    dropped = before_drop - len(merged)

    print(
        f"  -> dropped {dropped} warm-up rows"
    )

    if len(merged) == 0:
        raise ValueError(
            f"{ticker}: no rows remain after signal warm-up."
        )

    # ---------------------------------------------------------------
    # CAUSALITY VALIDATION
    # ---------------------------------------------------------------

    # Every signal date must correspond to a real prediction date.
    prediction_dates = set(
        pred_df["prediction_date"]
    )

    if not merged["prediction_date"].isin(
        prediction_dates
    ).all():
        raise AssertionError(
            f"{ticker}: signal dates do not match "
            f"prediction dates."
        )

    # A prediction must occur before its own realized outcome.
    if not (
        merged["prediction_date"]
        < merged["outcome_date"]
    ).all():
        raise AssertionError(
            f"{ticker}: prediction_date is not before "
            f"outcome_date."
        )

    # No missing signal values may remain.
    if merged[signal_cols].isna().any().any():
        raise AssertionError(
            f"{ticker}: NaNs remain in final signal panel."
        )

    # Ensure each prediction_date appears exactly once.
    if merged["prediction_date"].duplicated().any():
        raise AssertionError(
            f"{ticker}: duplicate prediction dates detected."
        )

    # ---------------------------------------------------------------
    # RETURN FINAL COLUMNS
    # ---------------------------------------------------------------

    return merged[
        [
            "prediction_date",
            "outcome_date",
            "date",
            "ticker",
            "ks_stat",
            "ks_pvalue",
            "miscoverage",
            "disagreement",
            "ensemble_mean",
            "fold_id",
        ]
    ]


# -------------------------------------------------------------------
# BUILD SIGNAL PANEL FOR ALL RESEARCH TICKERS
# -------------------------------------------------------------------

def build_all_tickers(
    tickers=None,
    warmup=30,
    roll_window=20,
    reference_window=20,
    n_bootstrap=10,
    skip_tickers=("COIN",),
):
    """
    Build the causal reliability-signal panel for the research universe.

    COIN is excluded by default because it has insufficient historical
    predictions in the current dataset.
    """

    if tickers is None:
        tickers = TICKERS

    all_rows = []

    for ticker in tickers:

        if ticker in skip_tickers:
            print(
                f"\nSkipping {ticker} "
                f"(insufficient historical coverage)"
            )
            continue

        try:

            merged = build_aligned_signals(
                ticker=ticker,
                warmup=warmup,
                roll_window=roll_window,
                reference_window=reference_window,
                n_bootstrap=n_bootstrap,
            )

            all_rows.append(
                merged
            )

            print(
                f"  -> {len(merged)} aligned rows"
            )

        except Exception as e:

            print(
                f"  FAILED: {ticker} — {e}"
            )

    if not all_rows:
        raise RuntimeError(
            "No ticker produced a valid signal panel."
        )

    combined = pd.concat(
        all_rows,
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

    # Final global validation.
    required_cols = [
        "prediction_date",
        "outcome_date",
        "ticker",
        "ks_stat",
        "miscoverage",
        "disagreement",
    ]

    if combined[required_cols].isna().any().any():
        raise AssertionError(
            "Final combined signal panel contains NaN values."
        )

    # Save.
    out_path = os.path.join(
        SIGNALS_DIR,
        "all_tickers_signals.csv",
    )

    combined.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\nSaved causal signal panel to:"
        f"\n{out_path}"
    )

    print(
        f"\nTotal rows: {len(combined)}"
    )

    print(
        f"Tickers: {combined['ticker'].nunique()}"
    )

    return combined


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    combined = build_all_tickers(
        warmup=30,
        roll_window=20,
        reference_window=20,
        n_bootstrap=10,
        skip_tickers=("COIN",),
    )

    pd.set_option(
        "display.width",
        160,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    # ---------------------------------------------------------------
    # PER-TICKER SUMMARY
    # ---------------------------------------------------------------

    print(
        "\n--- Per-ticker summary ---"
    )

    summary = (
        combined
        .groupby("ticker")
        .agg(
            n_rows=(
                "prediction_date",
                "count",
            ),
            first_date=(
                "prediction_date",
                "min",
            ),
            last_date=(
                "prediction_date",
                "max",
            ),
            mean_ks=(
                "ks_stat",
                "mean",
            ),
            mean_miscov=(
                "miscoverage",
                "mean",
            ),
            mean_disagreement=(
                "disagreement",
                "mean",
            ),
        )
    )

    print(
        summary.to_string()
    )

    # ---------------------------------------------------------------
    # AAPL EXAMPLE
    # ---------------------------------------------------------------

    print(
        "\n--- Example: AAPL ---"
    )

    aapl = combined[
        combined["ticker"] == "AAPL"
    ].head(10)

    print(
        aapl.to_string(
            index=False
        )
    )