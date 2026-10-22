"""
23_define_causal_deterioration_targets.py

Create causal binary deterioration targets from the QRF reliability panel.

Main target:
    future_pinball_ratio_h

For each horizon h in {5, 10, 20}:

    deterioration_h =
        1[
            future_pinball_ratio_h(t)
            >
            causal historical 75th percentile
        ]

    severe_deterioration_h =
        1[
            future_pinball_ratio_h(t)
            >
            causal historical 90th percentile
        ]

The threshold at time t is based ONLY on previously completed
future-loss windows.

For a horizon h, the target at time s is only fully known after
s + h - 1. Therefore, at forecast origin t, the latest target
that may be used to establish the threshold is t-h.

This prevents target leakage.

A fixed 1.5x threshold is also retained as a simple sensitivity
benchmark, but it is NOT the primary target.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [5, 10, 20]

PRIMARY_QUANTILE = 0.75
SEVERE_QUANTILE = 0.90

FIXED_RATIO = 1.50

# Minimum number of previously completed target observations
# before creating a percentile-based label.
MIN_HISTORY = 100

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)


# ============================================================
# HELPERS
# ============================================================

def causal_expanding_quantile(
    series: pd.Series,
    horizon: int,
    quantile: float,
    min_history: int = MIN_HISTORY,
) -> pd.Series:
    """
    Compute a causal expanding quantile.

    For target value T_t with horizon h, the target T_s is fully
    observable by forecast origin t only when:

        s + h - 1 <= t - 1

    which is equivalent to:

        s <= t - h

    Therefore we use:

        T.shift(h)

    before expanding.

    At row t, the expanding quantile therefore uses only target
    observations up through t-h.
    """

    completed_history = series.shift(horizon)

    return completed_history.expanding(
        min_periods=min_history
    ).quantile(quantile)


def add_targets_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal deterioration targets for one ticker.
    """

    g = (
        g
        .sort_values("forecast_date")
        .reset_index(drop=True)
        .copy()
    )

    for h in HORIZONS:

        target_col = (
            f"future_pinball_ratio_{h}d"
        )

        if target_col not in g.columns:
            raise ValueError(
                f"Missing target column: {target_col}"
            )

        target = pd.to_numeric(
            g[target_col],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Causal 75th percentile threshold
        # ----------------------------------------------------

        threshold_75 = causal_expanding_quantile(
            target,
            horizon=h,
            quantile=PRIMARY_QUANTILE,
        )

        # ----------------------------------------------------
        # Causal 90th percentile threshold
        # ----------------------------------------------------

        threshold_90 = causal_expanding_quantile(
            target,
            horizon=h,
            quantile=SEVERE_QUANTILE,
        )

        g[
            f"historical_deterioration_threshold_75_{h}d"
        ] = threshold_75

        g[
            f"historical_deterioration_threshold_90_{h}d"
        ] = threshold_90

        # ----------------------------------------------------
        # Primary label
        # ----------------------------------------------------

        g[
            f"deterioration_{h}d"
        ] = np.where(
            target.notna() & threshold_75.notna(),
            (
                target > threshold_75
            ).astype(int),
            np.nan,
        )

        # ----------------------------------------------------
        # Severe label
        # ----------------------------------------------------

        g[
            f"severe_deterioration_{h}d"
        ] = np.where(
            target.notna() & threshold_90.notna(),
            (
                target > threshold_90
            ).astype(int),
            np.nan,
        )

        # ----------------------------------------------------
        # Fixed threshold sensitivity
        # ----------------------------------------------------

        g[
            f"deterioration_fixed_1p5_{h}d"
        ] = np.where(
            target.notna(),
            (
                target >= FIXED_RATIO
            ).astype(int),
            np.nan,
        )

    return g


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("DEFINE CAUSAL DETERIORATION TARGETS")
    print("=" * 90)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "ticker",
        "forecast_date",
    ]

    for h in HORIZONS:
        required.append(
            f"future_pinball_ratio_{h}d"
        )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce"
    )

    if df["forecast_date"].isna().any():
        raise ValueError(
            "Invalid forecast dates detected."
        )

    print(
        f"\nLoaded: {INPUT_FILE}"
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Tickers: {df['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # BUILD TARGETS TICKER BY TICKER
    # ========================================================

    panels = []

    for ticker, g in df.groupby(
        "ticker",
        sort=True
    ):

        print(
            "\n" + "-" * 90
        )

        print(
            f"PROCESSING {ticker}"
        )

        print(
            "-" * 90
        )

        g = add_targets_for_ticker(
            g
        )

        panels.append(g)

    out = (
        pd.concat(
            panels,
            ignore_index=True
        )
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SANITY CHECK: NO FUTURE TARGET LEAKAGE
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "CAUSALITY SANITY CHECK"
    )

    print(
        "=" * 90
    )

    # For each ticker/horizon verify that the threshold at t
    # does not use target observations later than t-h.

    causal_failures = 0

    for ticker, g in out.groupby(
        "ticker",
        sort=True
    ):

        g = (
            g
            .sort_values("forecast_date")
            .reset_index(drop=True)
        )

        for h in HORIZONS:

            target_col = (
                f"future_pinball_ratio_{h}d"
            )

            threshold_col = (
                f"historical_deterioration_threshold_75_{h}d"
            )

            target = pd.to_numeric(
                g[target_col],
                errors="coerce"
            )

            threshold = pd.to_numeric(
                g[threshold_col],
                errors="coerce"
            )

            # Basic expected availability:
            # threshold should be unavailable at least during
            # the first MIN_HISTORY + h observations.

            first_valid = threshold.first_valid_index()

            if first_valid is not None:

                if first_valid < (
                    MIN_HISTORY + h - 1
                ):

                    print(
                        f"WARNING {ticker} h={h}: "
                        f"threshold became available unexpectedly early "
                        f"at row {first_valid}"
                    )

                    causal_failures += 1

    if causal_failures == 0:
        print(
            "PASS: causal threshold construction looks correct."
        )
    else:
        print(
            f"WARNING: {causal_failures} causality checks flagged."
        )

    # ========================================================
    # OVERALL TARGET SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "OVERALL TARGET SUMMARY"
    )

    print(
        "=" * 90
    )

    for h in HORIZONS:

        print(
            f"\n{h}-DAY HORIZON"
        )

        # Primary
        col = f"deterioration_{h}d"

        x = out[col].dropna()

        print(
            f"Primary 75th-percentile target:"
        )

        print(
            f"  Valid:       {len(x):,}"
        )

        print(
            f"  Positive:    {int(x.sum()):,}"
        )

        print(
            f"  Positive %:  {x.mean():.2%}"
        )

        # Severe
        col = (
            f"severe_deterioration_{h}d"
        )

        x = out[col].dropna()

        print(
            f"Severe 90th-percentile target:"
        )

        print(
            f"  Valid:       {len(x):,}"
        )

        print(
            f"  Positive:    {int(x.sum()):,}"
        )

        print(
            f"  Positive %:  {x.mean():.2%}"
        )

        # Fixed
        col = (
            f"deterioration_fixed_1p5_{h}d"
        )

        x = out[col].dropna()

        print(
            f"Fixed 1.5x sensitivity target:"
        )

        print(
            f"  Valid:       {len(x):,}"
        )

        print(
            f"  Positive:    {int(x.sum()):,}"
        )

        print(
            f"  Positive %:  {x.mean():.2%}"
        )

    # ========================================================
    # PER-TICKER POSITIVE RATES
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "PRIMARY TARGET POSITIVE RATE BY TICKER"
    )

    print(
        "=" * 90
    )

    rows = []

    for ticker, g in out.groupby(
        "ticker",
        sort=True
    ):

        row = {
            "ticker": ticker,
            "n_rows": len(g),
        }

        for h in HORIZONS:

            col = (
                f"deterioration_{h}d"
            )

            x = g[col].dropna()

            row[
                f"h{h}_n"
            ] = len(x)

            row[
                f"h{h}_positive"
            ] = int(x.sum())

            row[
                f"h{h}_positive_rate"
            ] = (
                x.mean()
                if len(x) > 0
                else np.nan
            )

        rows.append(row)

    ticker_summary = pd.DataFrame(
        rows
    )

    print(
        ticker_summary.round(4)
        .to_string(index=False)
    )

    # ========================================================
    # THRESHOLD SUMMARY
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "CAUSAL THRESHOLD SUMMARY"
    )

    print(
        "=" * 90
    )

    threshold_rows = []

    for ticker, g in out.groupby(
        "ticker",
        sort=True
    ):

        for h in HORIZONS:

            p75_col = (
                f"historical_deterioration_threshold_75_{h}d"
            )

            p90_col = (
                f"historical_deterioration_threshold_90_{h}d"
            )

            p75 = g[p75_col].dropna()
            p90 = g[p90_col].dropna()

            threshold_rows.append({
                "ticker": ticker,
                "horizon": h,
                "threshold_75_median": (
                    p75.median()
                    if len(p75)
                    else np.nan
                ),
                "threshold_75_mean": (
                    p75.mean()
                    if len(p75)
                    else np.nan
                ),
                "threshold_90_median": (
                    p90.median()
                    if len(p90)
                    else np.nan
                ),
                "threshold_90_mean": (
                    p90.mean()
                    if len(p90)
                    else np.nan
                ),
            })

    threshold_summary = pd.DataFrame(
        threshold_rows
    )

    print(
        threshold_summary.round(4)
        .to_string(index=False)
    )

    # ========================================================
    # CORRELATION OF PRIMARY BINARY TARGETS
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "PRIMARY TARGET CORRELATION"
    )

    print(
        "=" * 90
    )

    target_cols = [
        f"deterioration_{h}d"
        for h in HORIZONS
    ]

    corr = (
        out[target_cols]
        .corr()
    )

    corr.index = [
        "5d",
        "10d",
        "20d"
    ]

    corr.columns = [
        "5d",
        "10d",
        "20d"
    ]

    print(
        corr.round(4)
        .to_string()
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    out.to_csv(
        OUTPUT_FILE,
        index=False
    )

    ticker_summary_file = (
        ROOT
        / "data"
        / "monitoring"
        / "deterioration_target_positive_rates.csv"
    )

    threshold_summary_file = (
        ROOT
        / "data"
        / "monitoring"
        / "deterioration_target_causal_thresholds.csv"
    )

    ticker_summary.to_csv(
        ticker_summary_file,
        index=False
    )

    threshold_summary.to_csv(
        threshold_summary_file,
        index=False
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 90
    )

    print(
        "SAVED"
    )

    print(
        "=" * 90
    )

    print(
        OUTPUT_FILE
    )

    print(
        ticker_summary_file
    )

    print(
        threshold_summary_file
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "CAUSAL DETERIORATION TARGETS COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            "\nFATAL ERROR:",
            exc,
            file=sys.stderr
        )

        raise