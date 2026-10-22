"""
21_build_qrf_reliability_panel.py

Build a causal reliability-monitoring panel from the already-generated
25-ticker QRF OOS forecasts.

IMPORTANT
---------
At forecast origin t, only information available through t is allowed
in monitoring features.

QRF convention:
    forecast_date = t
    realized_date = t+1
    q05_forecast  = forecasted 5% return quantile for t+1

Therefore:
    - current breach/loss is NOT available at forecast time t
    - all breach/loss monitoring features use shift(1)
    - realized volatility uses returns through t
    - KS drift compares recent returns ending at t with an older reference sample
    - future pinball/breach targets deliberately use t+1 onward

PS is intentionally NOT included here because the fitted QRF trees are
not stored in the OOS CSV. It will be added in a separate step.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# CONFIG
# ============================================================

ALPHA = 0.05
EWMA_LAMBDA = 0.94

VOL_WINDOWS = (20, 60)
BREACH_WINDOWS = (5, 20, 60)

KS_RECENT_WINDOW = 60
KS_REFERENCE_WINDOW = 250

MIN_PAST_PINBALL = 60

TICKERS_EXPECTED = [
    "AAPL", "AMZN", "BA", "BAC", "CAT", "COIN", "CVX", "DIS",
    "GME", "GOOGL", "GS", "JNJ", "JPM", "KO", "MCD", "MSFT",
    "NFLX", "NVDA", "PFE", "PG", "TSLA", "UNH", "V", "WMT", "XOM"
]

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_oos_predictions.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel.csv"
)


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def pinball_loss(actual, q, alpha=ALPHA):
    """
    Lower-tail quantile pinball loss.

    I = 1(actual < q)

    rho_alpha(actual, q)
        = (actual - q) * (alpha - I)
    """
    hit = (actual < q).astype(int)
    return (actual - q) * (alpha - hit)


def consecutive_breach_streak(breach_series):
    """
    For each time t, return the number of consecutive breaches
    immediately BEFORE t.

    Example:
        breach sequence: 0 1 1 1 0
        streak feature: 0 0 1 2 3

    We calculate this before shifting so the final feature only uses
    information known by the forecast origin.
    """
    values = breach_series.astype(int).to_numpy()
    streak = np.zeros(len(values), dtype=float)

    current = 0

    for i, x in enumerate(values):
        if x == 1:
            current += 1
        else:
            current = 0

        streak[i] = current

    return pd.Series(streak, index=breach_series.index)


def compute_ks_series(
    returns,
    recent_window=KS_RECENT_WINDOW,
    reference_window=KS_REFERENCE_WINDOW
):
    """
    Causal rolling KS drift.

    At date t:

        Recent sample:
            returns[t-59 : t]

        Reference sample:
            returns[t-309 : t-60]

    Both samples are entirely known at the end of t.

    If insufficient history exists, return NaN.
    """

    arr = np.asarray(returns, dtype=float)
    out = np.full(len(arr), np.nan)

    minimum_needed = recent_window + reference_window

    for i in range(len(arr)):
        if i + 1 < minimum_needed:
            continue

        recent_start = i + 1 - recent_window
        recent_end = i + 1

        ref_end = recent_start
        ref_start = ref_end - reference_window

        recent = arr[recent_start:recent_end]
        reference = arr[ref_start:ref_end]

        if (
            len(recent) != recent_window
            or len(reference) != reference_window
            or not np.isfinite(recent).all()
            or not np.isfinite(reference).all()
        ):
            continue

        result = ks_2samp(
            recent,
            reference,
            alternative="two-sided",
            mode="auto"
        )

        out[i] = result.statistic

    return pd.Series(out)


def add_causal_features(g):
    """
    Add all monitoring features that are available at forecast origin t.
    """

    g = g.sort_values("forecast_date").reset_index(drop=True).copy()

    # --------------------------------------------------------
    # Basic forecast quantities
    # --------------------------------------------------------

    g["VaR_5pct"] = -g["q05_forecast"]

    g["q05_change_1d"] = (
        g["q05_forecast"]
        .diff()
    )

    g["VaR_change_1d"] = (
        g["VaR_5pct"]
        .diff()
    )

    # --------------------------------------------------------
    # Recompute breach + pinball for sanity / consistency
    # --------------------------------------------------------

    g["breach_recomputed"] = (
        g["actual_return"] < g["q05_forecast"]
    ).astype(int)

    g["pinball_recomputed"] = pinball_loss(
        g["actual_return"],
        g["q05_forecast"],
        ALPHA
    )

    # Validate against stored columns if present
    if "breach" in g.columns:
        stored_breach = pd.to_numeric(
            g["breach"],
            errors="coerce"
        ).fillna(-999).astype(int)

        if not np.array_equal(
            stored_breach.to_numpy(),
            g["breach_recomputed"].to_numpy()
        ):
            raise ValueError(
                "Stored breach column does not match "
                "(actual_return < q05_forecast)."
            )

    if "pinball_loss" in g.columns:
        stored_loss = pd.to_numeric(
            g["pinball_loss"],
            errors="coerce"
        )

        max_diff = np.nanmax(
            np.abs(
                stored_loss.to_numpy()
                - g["pinball_recomputed"].to_numpy()
            )
        )

        if max_diff > 1e-10:
            raise ValueError(
                f"Stored pinball_loss does not match recomputed loss. "
                f"Maximum difference = {max_diff:.12e}"
            )

    # Use recomputed values as authoritative
    g["breach"] = g["breach_recomputed"]
    g["pinball_loss"] = g["pinball_recomputed"]

    # --------------------------------------------------------
    # CRITICAL CAUSAL SHIFT
    #
    # At forecast origin t, today's outcome is unknown.
    # Therefore all breach/loss-history features use t-1 and earlier.
    # --------------------------------------------------------

    prior_breach = g["breach"].shift(1)
    prior_pinball = g["pinball_loss"].shift(1)

    # --------------------------------------------------------
    # Breach persistence
    # --------------------------------------------------------

    g["breach_lag1"] = prior_breach

    # Consecutive breach streak known before current forecast
    streak_all = consecutive_breach_streak(g["breach"])
    g["breach_streak_prior"] = streak_all.shift(1)

    # Rolling breach rates
    for w in BREACH_WINDOWS:
        g[f"breach_rate_{w}d"] = (
            prior_breach
            .rolling(
                window=w,
                min_periods=w
            )
            .mean()
        )

    # --------------------------------------------------------
    # EWMA breach rate
    # --------------------------------------------------------

    # RiskMetrics-style lambda:
    # EWMA_t = lambda * EWMA_(t-1) + (1-lambda) * breach_(t-1)

    g["ewma_breach_rate"] = (
        prior_breach
        .ewm(
            alpha=1.0 - EWMA_LAMBDA,
            adjust=False,
            min_periods=20
        )
        .mean()
    )

    # --------------------------------------------------------
    # Historical pinball-loss information
    # --------------------------------------------------------

    g["pinball_lag1"] = prior_pinball

    g["pinball_mean_5d_prior"] = (
        prior_pinball
        .rolling(5, min_periods=5)
        .mean()
    )

    g["pinball_mean_20d_prior"] = (
        prior_pinball
        .rolling(20, min_periods=20)
        .mean()
    )

    g["pinball_mean_60d_prior"] = (
        prior_pinball
        .rolling(60, min_periods=60)
        .mean()
    )

    # Difference between short and medium recent loss levels
    g["pinball_acceleration"] = (
        g["pinball_mean_5d_prior"]
        - g["pinball_mean_20d_prior"]
    )

    # --------------------------------------------------------
    # Causal realized volatility
    #
    # Actual return on forecast origin t is known at end of t,
    # so it is legitimate to use it here.
    # --------------------------------------------------------

    for w in VOL_WINDOWS:
        g[f"realized_vol_{w}d"] = (
            g["origin_return"]
            .rolling(w, min_periods=w)
            .std()
        )

    g["vol_ratio_20_60"] = (
        g["realized_vol_20d"]
        / g["realized_vol_60d"]
    )

    # --------------------------------------------------------
    # KS distribution drift
    # --------------------------------------------------------

    g["ks_drift_60_vs_250"] = compute_ks_series(
        g["origin_return"].to_numpy(),
        recent_window=KS_RECENT_WINDOW,
        reference_window=KS_REFERENCE_WINDOW
    ).to_numpy()

    # --------------------------------------------------------
    # Current forecast relative to recent history
    # --------------------------------------------------------

    prior_var = g["VaR_5pct"].shift(1)

    g["VaR_ratio_to_prior"] = (
        g["VaR_5pct"]
        / prior_var
    )

    prior_var_mean_20 = (
        prior_var
        .rolling(20, min_periods=20)
        .mean()
    )

    g["VaR_ratio_to_20d_mean"] = (
        g["VaR_5pct"]
        / prior_var_mean_20
    )

    # --------------------------------------------------------
    # Future targets
    #
    # Current row t represents a forecast made at t.
    # Its loss is realized at t+1.
    #
    # Therefore the next h forecast losses are:
    #   rows t, t+1, ..., t+h-1
    # --------------------------------------------------------

    for h in [5, 10, 20]:

        g[f"future_pinball_{h}d"] = (
            g["pinball_loss"]
            .rolling(
                window=h,
                min_periods=h
            )
            .mean()
            .shift(-(h - 1))
        )

        g[f"future_breach_rate_{h}d"] = (
            g["breach"]
            .rolling(
                window=h,
                min_periods=h
            )
            .mean()
            .shift(-(h - 1))
        )

        # Binary high-loss indicator is NOT defined yet.
        # We keep the continuous target so that the threshold
        # can be established later using training-only data.

    # --------------------------------------------------------
    # Causal normalization baseline
    #
    # Expanding mean of previously observed OOS losses.
    # This is available at t and therefore does not leak future data.
    # --------------------------------------------------------

    g["past_mean_pinball"] = (
        prior_pinball
        .expanding(
            min_periods=MIN_PAST_PINBALL
        )
        .mean()
    )

    for h in [5, 10, 20]:

        g[f"future_pinball_ratio_{h}d"] = (
            g[f"future_pinball_{h}d"]
            / g["past_mean_pinball"]
        )

    # --------------------------------------------------------
    # Secondary target: average future breach severity
    #
    # severity = max(q - return, 0)
    # because q is a negative return quantile.
    # --------------------------------------------------------

    g["breach_severity"] = np.maximum(
        g["q05_forecast"] - g["actual_return"],
        0.0
    )

    for h in [5, 10, 20]:

        g[f"future_severity_{h}d"] = (
            g["breach_severity"]
            .rolling(
                window=h,
                min_periods=h
            )
            .mean()
            .shift(-(h - 1))
        )

    return g


# ============================================================
# LOAD QRF OOS DATA
# ============================================================

def main():

    print("=" * 88)
    print("BUILD QRF RELIABILITY MONITORING PANEL")
    print("=" * 88)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    qrf = pd.read_csv(INPUT_FILE)

    print(f"\nLoaded: {INPUT_FILE}")
    print(f"Shape:  {qrf.shape}")
    print(f"Columns: {qrf.columns.tolist()}")

    required = [
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "q05_forecast",
    ]

    require_columns(qrf, required)

    # --------------------------------------------------------
    # Parse types
    # --------------------------------------------------------

    qrf["forecast_date"] = pd.to_datetime(
        qrf["forecast_date"],
        errors="coerce"
    )

    qrf["realized_date"] = pd.to_datetime(
        qrf["realized_date"],
        errors="coerce"
    )

    qrf["actual_return"] = pd.to_numeric(
        qrf["actual_return"],
        errors="coerce"
    )

    qrf["q05_forecast"] = pd.to_numeric(
        qrf["q05_forecast"],
        errors="coerce"
    )

    if qrf[required].isna().any().any():
        bad_counts = qrf[required].isna().sum()
        raise ValueError(
            f"Missing values detected:\n{bad_counts}"
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    if qrf["q05_forecast"].median() >= 0:
        raise ValueError(
            "Median q05_forecast is >= 0. "
            "This does not look like a signed lower-tail return quantile."
        )

    qrf["breach_check"] = (
        qrf["actual_return"] < qrf["q05_forecast"]
    ).astype(int)

    overall_breach = qrf["breach_check"].mean()

    print(
        f"\nOverall OOS breach rate: "
        f"{overall_breach:.6f}"
    )

    if not (0.005 <= overall_breach <= 0.25):
        raise ValueError(
            "Implausible overall breach rate. "
            "Check q05 column/sign convention."
        )

    print(
        f"Tickers found: "
        f"{qrf['ticker'].nunique()}"
    )

    # --------------------------------------------------------
    # Load processed returns
    # --------------------------------------------------------

    panels = []
    skipped = []

    for ticker in sorted(qrf["ticker"].unique()):

        print("\n" + "-" * 88)
        print(f"PROCESSING {ticker}")
        print("-" * 88)

        processed_file = (
            ROOT
            / "data"
            / "processed"
            / f"{ticker}.csv"
        )

        if not processed_file.exists():
            skipped.append(
                (ticker, f"processed file missing: {processed_file}")
            )
            print(
                f"SKIPPED: processed file missing"
            )
            continue

        processed = pd.read_csv(processed_file)

        require_columns(
            processed,
            ["Date", "Log_Return"]
        )

        processed["Date"] = pd.to_datetime(
            processed["Date"],
            errors="coerce"
        )

        processed["Log_Return"] = pd.to_numeric(
            processed["Log_Return"],
            errors="coerce"
        )

        processed = (
            processed[
                ["Date", "Log_Return"]
            ]
            .dropna()
            .sort_values("Date")
            .drop_duplicates("Date")
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Select ticker's QRF OOS rows
        # ----------------------------------------------------

        g = (
            qrf[
                qrf["ticker"] == ticker
            ]
            .copy()
            .sort_values("forecast_date")
            .reset_index(drop=True)
        )

        print(f"QRF OOS rows: {len(g)}")
        print(
            f"QRF dates: "
            f"{g['forecast_date'].min().date()} "
            f"to "
            f"{g['forecast_date'].max().date()}"
        )

        # ----------------------------------------------------
        # Map forecast date to processed return history
        # ----------------------------------------------------

        date_to_idx = {
            d: i
            for i, d in enumerate(processed["Date"])
        }

        g["processed_origin_idx"] = (
            g["forecast_date"]
            .map(date_to_idx)
        )

        if g["processed_origin_idx"].isna().any():
            n_missing = int(
                g["processed_origin_idx"].isna().sum()
            )
            skipped.append(
                (
                    ticker,
                    f"{n_missing} forecast dates missing in processed data"
                )
            )
            print(
                f"SKIPPED: {n_missing} forecast dates not found"
            )
            continue

        g["processed_origin_idx"] = (
            g["processed_origin_idx"]
            .astype(int)
        )

        # ----------------------------------------------------
        # Verify forecast-date -> realized-date alignment
        # ----------------------------------------------------

        processed_dates = processed["Date"].to_numpy()

        expected_realized = []

        for idx in g["processed_origin_idx"]:
            if idx + 1 >= len(processed_dates):
                expected_realized.append(pd.NaT)
            else:
                expected_realized.append(
                    pd.Timestamp(processed_dates[idx + 1])
                )

        g["expected_realized_date"] = expected_realized

        alignment_ok = (
            g["realized_date"].values
            == g["expected_realized_date"].values
        )

        if not alignment_ok.all():

            bad_n = int((~alignment_ok).sum())

            sample = g.loc[
                ~alignment_ok,
                [
                    "forecast_date",
                    "realized_date",
                    "expected_realized_date"
                ]
            ].head(5)

            print(
                f"Alignment errors: {bad_n}"
            )
            print(sample.to_string(index=False))

            skipped.append(
                (
                    ticker,
                    f"{bad_n} forecast/realized-date alignment failures"
                )
            )
            continue

        # ----------------------------------------------------
        # Get return known at forecast origin t
        # ----------------------------------------------------

        origin_return_map = (
            processed
            .set_index("Date")["Log_Return"]
        )

        g["origin_return"] = (
            g["forecast_date"]
            .map(origin_return_map)
        )

        if g["origin_return"].isna().any():
            skipped.append(
                (
                    ticker,
                    "missing origin-day return"
                )
            )
            continue

        # ----------------------------------------------------
        # Check next-day actual return against processed file
        # ----------------------------------------------------

        realized_return_map = (
            processed
            .set_index("Date")["Log_Return"]
        )

        expected_actual = (
            g["realized_date"]
            .map(realized_return_map)
        )

        max_return_diff = np.max(
            np.abs(
                g["actual_return"].to_numpy()
                - expected_actual.to_numpy()
            )
        )

        print(
            f"Actual-return alignment max diff: "
            f"{max_return_diff:.12e}"
        )

        if max_return_diff > 1e-10:
            raise ValueError(
                f"{ticker}: QRF actual_return does not match "
                f"processed Log_Return on realized_date."
            )

        # ----------------------------------------------------
        # Build causal monitoring features
        # ----------------------------------------------------

        g = add_causal_features(g)

        # ----------------------------------------------------
        # Add ticker metadata
        # ----------------------------------------------------

        g["ticker"] = ticker

        panels.append(g)

        print(
            f"Panel rows generated: {len(g)}"
        )

    # ========================================================
    # COMBINE
    # ========================================================

    if not panels:
        raise RuntimeError(
            "No ticker panels were generated."
        )

    panel = (
        pd.concat(
            panels,
            axis=0,
            ignore_index=True
        )
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # FINAL COLUMN SELECTION
    # ========================================================

    output_columns = [
        # identifiers
        "ticker",
        "forecast_date",
        "realized_date",

        # forecast / outcome
        "actual_return",
        "q05_forecast",
        "VaR_5pct",
        "breach",
        "pinball_loss",

        # current forecast movement
        "q05_change_1d",
        "VaR_change_1d",
        "VaR_ratio_to_prior",
        "VaR_ratio_to_20d_mean",

        # breach persistence
        "breach_lag1",
        "breach_streak_prior",
        "breach_rate_5d",
        "breach_rate_20d",
        "breach_rate_60d",
        "ewma_breach_rate",

        # historical loss state
        "pinball_lag1",
        "pinball_mean_5d_prior",
        "pinball_mean_20d_prior",
        "pinball_mean_60d_prior",
        "pinball_acceleration",

        # volatility
        "realized_vol_20d",
        "realized_vol_60d",
        "vol_ratio_20_60",

        # drift
        "ks_drift_60_vs_250",

        # causal normalization
        "past_mean_pinball",

        # future targets
        "future_pinball_5d",
        "future_pinball_10d",
        "future_pinball_20d",
        "future_pinball_ratio_5d",
        "future_pinball_ratio_10d",
        "future_pinball_ratio_20d",

        # secondary future targets
        "future_breach_rate_5d",
        "future_breach_rate_10d",
        "future_breach_rate_20d",
        "future_severity_5d",
        "future_severity_10d",
        "future_severity_20d",
    ]

    output_columns = [
        c for c in output_columns
        if c in panel.columns
    ]

    panel = panel[output_columns]

    # ========================================================
    # DATA QUALITY SUMMARY
    # ========================================================

    print("\n" + "=" * 88)
    print("FINAL PANEL SUMMARY")
    print("=" * 88)

    print(
        f"Rows:             {len(panel):,}"
    )

    print(
        f"Tickers:          {panel['ticker'].nunique()}"
    )

    print(
        f"Forecast dates:   "
        f"{panel['forecast_date'].min().date()} "
        f"to "
        f"{panel['forecast_date'].max().date()}"
    )

    print(
        f"Overall breach:   "
        f"{panel['breach'].mean():.6f}"
    )

    print("\nRows by ticker:")
    print(
        panel.groupby("ticker")
        .size()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Feature availability
    # --------------------------------------------------------

    feature_columns = [
        "breach_lag1",
        "breach_streak_prior",
        "breach_rate_5d",
        "breach_rate_20d",
        "breach_rate_60d",
        "ewma_breach_rate",
        "pinball_mean_20d_prior",
        "realized_vol_20d",
        "realized_vol_60d",
        "vol_ratio_20_60",
        "ks_drift_60_vs_250",
        "VaR_ratio_to_20d_mean",
    ]

    print("\nFeature availability:")
    for c in feature_columns:
        if c in panel.columns:
            valid = panel[c].notna().sum()
            pct = valid / len(panel)
            print(
                f"{c:<30} "
                f"{valid:>6}/{len(panel):<6} "
                f"({pct:6.2%})"
            )

    # --------------------------------------------------------
    # Target availability
    # --------------------------------------------------------

    print("\nTarget availability:")
    for h in [5, 10, 20]:
        c = f"future_pinball_ratio_{h}d"
        valid = panel[c].notna().sum()
        pct = valid / len(panel)
        print(
            f"{c:<30} "
            f"{valid:>6}/{len(panel):<6} "
            f"({pct:6.2%})"
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    panel.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 88)
    print("SAVED")
    print("=" * 88)

    print(OUTPUT_FILE)

    if skipped:
        print("\nSkipped tickers:")
        for ticker, reason in skipped:
            print(
                f"  {ticker}: {reason}"
            )

    print("\n" + "=" * 88)
    print("QRF RELIABILITY PANEL COMPLETE")
    print("=" * 88)


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