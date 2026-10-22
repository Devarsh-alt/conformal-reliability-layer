# ============================================================
# 26_build_new_reliability_signals.py
#
# Purpose:
#   Construct three PRE-SPECIFIED reliability signals that
#   contain information different from the QRF's own recent
#   breach/loss history.
#
# Signals:
#
#   1. Cross-sectional breach stress
#      -----------------------------
#      5-day average fraction of available QRF assets that
#      breached their VaR.
#
#      IMPORTANT:
#      The signal is lagged by one trading day so that today's
#      monitor never uses today's realized breach.
#
#
#   2. QRF training-regime extremeness / extrapolation
#      -----------------------------------------------
#      Compare today's vol_20d and VIX against the previous
#      1000 observations used by the frozen QRF window.
#
#      Outputs:
#        - vol20 training percentile
#        - VIX training percentile
#        - training-regime extremeness
#        - out-of-range flags
#
#
#   3. QRF-EWMA model disagreement
#      ----------------------------
#      Compare QRF 5% quantile with EWMA 5% quantile.
#
#      Positive values mean QRF is LESS negative / less
#      conservative than EWMA.
#
#      Formula:
#
#          (QRF_q05 - EWMA_q05) / |EWMA_q05|
#
#
# IMPORTANT:
#   This script DOES NOT evaluate predictive performance.
#   It only constructs and validates the frozen candidate
#   signals.
#
# Inputs:
#
#   data/monitoring/QRF_25ticker_reliability_panel_with_targets.csv
#   data/monitoring/QRF_25ticker_oos_predictions.csv
#   data/processed/<TICKER>.csv
#   data/var/<TICKER>_var_5pct.csv
#
# Outputs:
#
#   data/monitoring/QRF_25ticker_new_signals.csv
#   data/monitoring/new_signal_availability.csv
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_WINDOW = 1000

CROSS_SECTIONAL_WINDOW = 5

MIN_TRAINING_OBS = 900

TICKERS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CAT",
    "COIN",
    "CVX",
    "DIS",
    "GME",
    "GOOGL",
    "GS",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MSFT",
    "NFLX",
    "NVDA",
    "PFE",
    "PG",
    "TSLA",
    "UNH",
    "V",
    "WMT",
    "XOM",
]


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

VAR_DIR = (
    PROJECT_ROOT
    / "data"
    / "var"
)

PANEL_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

QRF_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_oos_predictions.csv"
)

OUTPUT_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_new_signals.csv"
)

AVAILABILITY_FILE = (
    MONITORING_DIR
    / "new_signal_availability.csv"
)


# ============================================================
# HELPERS
# ============================================================

def find_column(
    df,
    candidates,
    required=True,
):
    """
    Find a column using exact case-insensitive matching.
    """

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lower_map:

            return lower_map[key]

    if required:

        raise ValueError(
            "\nCould not find any of:"
            f"\n{candidates}"
            f"\nAvailable columns:"
            f"\n{df.columns.tolist()}"
        )

    return None


def percentile_rank(
    training_values,
    current_value,
):
    """
    Empirical percentile rank.

    Returns the fraction of training observations less
    than or equal to the current observation.
    """

    values = np.asarray(
        training_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if (
        len(values) == 0
        or not np.isfinite(current_value)
    ):
        return np.nan

    return float(
        np.mean(
            values <= current_value
        )
    )


def load_processed_ticker(
    ticker,
):
    """
    Load the processed ticker data used by QRF.
    """

    path = (
        PROCESSED_DIR
        / f"{ticker}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nProcessed file not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    date_col = find_column(
        df,
        ["Date", "date"],
    )

    required = [
        "vol_20d",
        "VIX",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\n{ticker}: missing columns "
            f"{missing}"
        )

    out = df[
        [
            date_col,
            "vol_20d",
            "VIX",
        ]
    ].copy()

    out.columns = [
        "date",
        "vol_20d",
        "VIX",
    ]

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    out["vol_20d"] = pd.to_numeric(
        out["vol_20d"],
        errors="coerce",
    )

    out["VIX"] = pd.to_numeric(
        out["VIX"],
        errors="coerce",
    )

    out = (
        out
        .dropna(
            subset=["date"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out


def load_ewma_for_ticker(
    ticker,
):
    """
    Load EWMA VaR from the existing traditional VaR file.

    The historical project convention stores the VaR as the
    signed lower-tail quantile. If a file happens to contain
    positive VaR magnitudes instead, convert them to negative
    quantiles automatically.
    """

    path = (
        VAR_DIR
        / f"{ticker}_var_5pct.csv"
    )

    if not path.exists():

        print(
            f"WARNING: EWMA file missing for {ticker}:"
            f"\n  {path}"
        )

        return pd.DataFrame(
            columns=[
                "realized_date",
                "ewma_q05",
            ]
        )

    df = pd.read_csv(
        path
    )

    date_col = find_column(
        df,
        ["Date", "date"],
    )

    ewma_col = find_column(
        df,
        [
            "EWMA_VaR",
            "EWMA_q05",
            "EWMA",
        ],
    )

    out = df[
        [
            date_col,
            ewma_col,
        ]
    ].copy()

    out.columns = [
        "realized_date",
        "ewma_q05",
    ]

    out["realized_date"] = pd.to_datetime(
        out["realized_date"],
        errors="coerce",
    )

    out["ewma_q05"] = pd.to_numeric(
        out["ewma_q05"],
        errors="coerce",
    )

    out = (
        out
        .dropna()
        .sort_values("realized_date")
        .drop_duplicates(
            subset=["realized_date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Detect sign convention
    # --------------------------------------------------------

    median_value = (
        out["ewma_q05"]
        .median()
    )

    if (
        np.isfinite(median_value)
        and median_value > 0
    ):

        print(
            f"  {ticker}: EWMA file appears "
            "to use positive VaR magnitudes."
        )

        out["ewma_q05"] = (
            -out["ewma_q05"].abs()
        )

    else:

        print(
            f"  {ticker}: EWMA treated as "
            "signed lower-tail quantile."
        )

    return out


# ============================================================
# START
# ============================================================

print("=" * 100)
print("BUILD NEW PRE-SPECIFIED QRF RELIABILITY SIGNALS")
print("=" * 100)

print(
    f"\nProject root:"
    f"\n{PROJECT_ROOT}"
)

print(
    f"\nQRF training window:"
    f" {TRAINING_WINDOW}"
)

print(
    f"\nCross-sectional window:"
    f" {CROSS_SECTIONAL_WINDOW} trading days"
)


# ============================================================
# LOAD EXISTING QRF PANEL
# ============================================================

print("\n" + "=" * 100)
print("LOADING QRF RELIABILITY PANEL")
print("=" * 100)

if not PANEL_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{PANEL_FILE}"
    )

panel = pd.read_csv(
    PANEL_FILE
)

print(
    f"\nPanel rows: {len(panel):,}"
)

print(
    f"Panel columns: {len(panel.columns)}"
)


# ============================================================
# LOAD RAW QRF PREDICTIONS
# ============================================================

print("\n" + "=" * 100)
print("LOADING QRF OOS PREDICTIONS")
print("=" * 100)

if not QRF_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{QRF_FILE}"
    )

qrf = pd.read_csv(
    QRF_FILE
)

print(
    f"\nQRF rows: {len(qrf):,}"
)

print(
    f"QRF columns:"
    f"\n{qrf.columns.tolist()}"
)


# ============================================================
# NORMALIZE QRF COLUMNS
# ============================================================

qrf_ticker_col = find_column(
    qrf,
    ["ticker"],
)

qrf_forecast_date_col = find_column(
    qrf,
    ["forecast_date"],
)

qrf_realized_date_col = find_column(
    qrf,
    ["realized_date"],
)

qrf_q05_col = find_column(
    qrf,
    [
        "q05_forecast",
        "q05",
    ],
)

qrf_breach_col = find_column(
    qrf,
    ["breach"],
)

qrf_return_col = find_column(
    qrf,
    ["actual_return"],
)


qrf = qrf[
    [
        qrf_ticker_col,
        qrf_forecast_date_col,
        qrf_realized_date_col,
        qrf_q05_col,
        qrf_breach_col,
        qrf_return_col,
    ]
].copy()


qrf.columns = [
    "ticker",
    "forecast_date",
    "realized_date",
    "q05_forecast",
    "breach",
    "actual_return_qrf",
]


qrf["forecast_date"] = pd.to_datetime(
    qrf["forecast_date"],
    errors="coerce",
)

qrf["realized_date"] = pd.to_datetime(
    qrf["realized_date"],
    errors="coerce",
)

qrf["q05_forecast"] = pd.to_numeric(
    qrf["q05_forecast"],
    errors="coerce",
)

qrf["breach"] = pd.to_numeric(
    qrf["breach"],
    errors="coerce",
)

qrf["actual_return_qrf"] = pd.to_numeric(
    qrf["actual_return_qrf"],
    errors="coerce",
)


qrf = (
    qrf
    .dropna(
        subset=[
            "ticker",
            "forecast_date",
            "realized_date",
            "q05_forecast",
            "breach",
        ]
    )
    .sort_values(
        [
            "ticker",
            "realized_date",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# PANEL DATE NORMALIZATION
# ============================================================

panel["realized_date"] = pd.to_datetime(
    panel["realized_date"],
    errors="coerce",
)

panel["forecast_date"] = pd.to_datetime(
    panel["forecast_date"],
    errors="coerce",
)


# ============================================================
# VERIFY QRF / PANEL ALIGNMENT
# ============================================================

print("\n" + "=" * 100)
print("VERIFYING QRF / PANEL ALIGNMENT")
print("=" * 100)

qrf_keys = qrf[
    [
        "ticker",
        "forecast_date",
        "realized_date",
        "q05_forecast",
        "breach",
    ]
].copy()

panel_keys = panel[
    [
        "ticker",
        "forecast_date",
        "realized_date",
        "q05_forecast",
        "breach",
    ]
].copy()

qrf_keys = qrf_keys.sort_values(
    [
        "ticker",
        "realized_date",
    ]
).reset_index(drop=True)

panel_keys = panel_keys.sort_values(
    [
        "ticker",
        "realized_date",
    ]
).reset_index(drop=True)


print(
    f"\nQRF rows:   {len(qrf_keys):,}"
)

print(
    f"Panel rows: {len(panel_keys):,}"
)


# ============================================================
# CROSS-SECTIONAL BREACH STRESS
# ============================================================

print("\n" + "=" * 100)
print("SIGNAL 1 — CROSS-SECTIONAL BREACH STRESS")
print("=" * 100)

print(
    "\nDefinition:"
    "\n5-day average share of available QRF assets"
    "\nthat breached their 5% VaR."
)

print(
    "\nThe completed signal is shifted by one trading day."
    "\nTherefore a breach occurring on date t cannot"
    "\nbe used to predict deterioration on date t."
)


daily_breach = (
    qrf
    .groupby(
        "realized_date",
        as_index=False,
    )
    .agg(
        cross_sectional_breach_share=(
            "breach",
            "mean",
        ),
        cross_sectional_n=(
            "ticker",
            "nunique",
        ),
    )
    .sort_values(
        "realized_date"
    )
    .reset_index(drop=True)
)


daily_breach[
    "cs_breach_stress_5d"
] = (
    daily_breach[
        "cross_sectional_breach_share"
    ]
    .rolling(
        CROSS_SECTIONAL_WINDOW,
        min_periods=CROSS_SECTIONAL_WINDOW,
    )
    .mean()
    .shift(1)
)


print(
    "\nCross-sectional daily rows:"
    f" {len(daily_breach):,}"
)

print(
    "\nCross-sectional stress availability:"
    f" {daily_breach['cs_breach_stress_5d'].notna().sum():,}"
)

print(
    "\nCross-sectional stress summary:"
)

print(
    daily_breach[
        [
            "cross_sectional_breach_share",
            "cross_sectional_n",
            "cs_breach_stress_5d",
        ]
    ].describe().to_string()
)


# ============================================================
# MERGE CROSS-SECTIONAL SIGNAL INTO QRF PANEL
# ============================================================

panel = panel.merge(
    daily_breach[
        [
            "realized_date",
            "cross_sectional_breach_share",
            "cross_sectional_n",
            "cs_breach_stress_5d",
        ]
    ],
    on="realized_date",
    how="left",
)


# ============================================================
# QRF-EWMA DISAGREEMENT
# ============================================================

print("\n" + "=" * 100)
print("SIGNAL 2 — QRF / EWMA MODEL DISAGREEMENT")
print("=" * 100)

print(
    "\nPrimary definition:"
    "\n(QRF q05 - EWMA q05) / |EWMA q05|"
)

print(
    "\nPositive values mean the QRF is"
    "\nless negative / less conservative than EWMA."
)


ewma_rows = []

missing_ewma = []

for ticker in TICKERS:

    print(
        f"\nLoading EWMA: {ticker}"
    )

    ewma = load_ewma_for_ticker(
        ticker
    )

    if len(ewma) == 0:

        missing_ewma.append(
            ticker
        )

        continue

    ewma["ticker"] = ticker

    ewma_rows.append(
        ewma[
            [
                "ticker",
                "realized_date",
                "ewma_q05",
            ]
        ]
    )


if len(ewma_rows) > 0:

    ewma_all = pd.concat(
        ewma_rows,
        ignore_index=True,
    )

else:

    ewma_all = pd.DataFrame(
        columns=[
            "ticker",
            "realized_date",
            "ewma_q05",
        ]
    )


print(
    f"\nEWMA tickers loaded:"
    f" {ewma_all['ticker'].nunique()}"
)

if missing_ewma:

    print(
        "\nMissing EWMA tickers:"
        f" {missing_ewma}"
    )


# ============================================================
# MERGE EWMA WITH QRF
# ============================================================

qrf_with_ewma = qrf.merge(
    ewma_all,
    on=[
        "ticker",
        "realized_date",
    ],
    how="left",
)


# ============================================================
# CHECK ACTUAL-RETURN ALIGNMENT
# ============================================================

# The EWMA files do not necessarily expose their actual return
# under a standardized name, so we validate forecast coverage
# rather than assuming alignment.

ewma_available = (
    qrf_with_ewma["ewma_q05"]
    .notna()
)

print(
    "\nQRF rows with EWMA:"
    f" {ewma_available.sum():,}"
    f"/{len(qrf_with_ewma):,}"
)


# ============================================================
# DISAGREEMENT
# ============================================================

denominator = (
    qrf_with_ewma[
        "ewma_q05"
    ].abs()
)

valid_disagreement = (
    denominator > 1e-12
)

qrf_with_ewma[
    "qrf_ewma_disagreement"
] = np.nan

qrf_with_ewma.loc[
    valid_disagreement,
    "qrf_ewma_disagreement",
] = (
    (
        qrf_with_ewma.loc[
            valid_disagreement,
            "q05_forecast",
        ]
        -
        qrf_with_ewma.loc[
            valid_disagreement,
            "ewma_q05",
        ]
    )
    /
    denominator[
        valid_disagreement
    ]
)


# Positive means QRF is less conservative
qrf_with_ewma[
    "qrf_less_conservative_than_ewma"
] = (
    qrf_with_ewma[
        "qrf_ewma_disagreement"
    ]
    > 0
).astype(float)


# ============================================================
# MERGE DISAGREEMENT INTO PANEL
# ============================================================

panel = panel.merge(
    qrf_with_ewma[
        [
            "ticker",
            "realized_date",
            "ewma_q05",
            "qrf_ewma_disagreement",
            "qrf_less_conservative_than_ewma",
        ]
    ],
    on=[
        "ticker",
        "realized_date",
    ],
    how="left",
)


print(
    "\nQRF-EWMA disagreement summary:"
)

print(
    panel[
        "qrf_ewma_disagreement"
    ].describe().to_string()
)


# ============================================================
# TRAINING-RANGE REGIME SIGNAL
# ============================================================

print("\n" + "=" * 100)
print(
    "SIGNAL 3 — QRF TRAINING-REGIME EXTREMENESS"
)
print("=" * 100)

print(
    "\nDefinition:"
    "\nCompare current vol_20d and VIX with the"
    "\nprevious 1000 observations available to"
    "\nthe QRF fit."
)

print(
    "\nFor forecast date t, the training window ends"
    "\nat the last available observation strictly before t."
)


regime_results = []

for ticker in TICKERS:

    print(
        f"\nProcessing training regime: {ticker}"
    )

    processed = load_processed_ticker(
        ticker
    )

    ticker_rows = panel[
        panel["ticker"] == ticker
    ].copy()

    ticker_rows = ticker_rows.sort_values(
        "forecast_date"
    ).reset_index(drop=False)

    if len(ticker_rows) == 0:

        continue

    processed_dates = (
        processed["date"]
        .to_numpy()
    )

    for _, row in ticker_rows.iterrows():

        forecast_date = row[
            "forecast_date"
        ]

        # ----------------------------------------------------
        # Last processed observation STRICTLY BEFORE
        # forecast_date.
        #
        # This reproduces the temporal information set:
        # forecast_date features are NOT allowed into the
        # training window.
        # ----------------------------------------------------

        position = np.searchsorted(
            processed_dates,
            np.datetime64(
                forecast_date
            ),
            side="left",
        )

        end_idx = (
            position - 1
        )

        if end_idx < 0:

            regime_results.append(
                {
                    "ticker": ticker,
                    "realized_date":
                        row["realized_date"],
                    "train_regime_vol20_pct":
                        np.nan,
                    "train_regime_vix_pct":
                        np.nan,
                    "train_regime_extremeness":
                        np.nan,
                    "train_out_of_range_vol20":
                        np.nan,
                    "train_out_of_range_vix":
                        np.nan,
                    "train_out_of_range_any":
                        np.nan,
                    "training_obs_available":
                        0,
                }
            )

            continue

        start_idx = max(
            0,
            end_idx
            - TRAINING_WINDOW
            + 1,
        )

        training = processed.iloc[
            start_idx:
            end_idx + 1
        ].copy()

        training_obs = len(
            training
        )

        if training_obs < MIN_TRAINING_OBS:

            regime_results.append(
                {
                    "ticker": ticker,
                    "realized_date":
                        row["realized_date"],
                    "train_regime_vol20_pct":
                        np.nan,
                    "train_regime_vix_pct":
                        np.nan,
                    "train_regime_extremeness":
                        np.nan,
                    "train_out_of_range_vol20":
                        np.nan,
                    "train_out_of_range_vix":
                        np.nan,
                    "train_out_of_range_any":
                        np.nan,
                    "training_obs_available":
                        training_obs,
                }
            )

            continue

        # ----------------------------------------------------
        # Current values
        #
        # They are obtained from the row corresponding to
        # forecast_date, not realized_date.
        # ----------------------------------------------------

        current_match = processed[
            processed["date"]
            == forecast_date
        ]

        if len(current_match) == 0:

            # Should normally never happen for this project.
            # Fall back to latest observation at/before date.
            current_match = (
                processed[
                    processed["date"]
                    <= forecast_date
                ]
                .tail(1)
            )

        if len(current_match) == 0:

            regime_results.append(
                {
                    "ticker": ticker,
                    "realized_date":
                        row["realized_date"],
                    "train_regime_vol20_pct":
                        np.nan,
                    "train_regime_vix_pct":
                        np.nan,
                    "train_regime_extremeness":
                        np.nan,
                    "train_out_of_range_vol20":
                        np.nan,
                    "train_out_of_range_vix":
                        np.nan,
                    "train_out_of_range_any":
                        np.nan,
                    "training_obs_available":
                        training_obs,
                }
            )

            continue

        current = current_match.iloc[
            0
        ]

        current_vol20 = (
            current["vol_20d"]
        )

        current_vix = (
            current["VIX"]
        )

        # ----------------------------------------------------
        # Empirical percentile ranks
        # ----------------------------------------------------

        vol_training = (
            training["vol_20d"]
            .dropna()
            .to_numpy()
        )

        vix_training = (
            training["VIX"]
            .dropna()
            .to_numpy()
        )

        vol_pct = percentile_rank(
            vol_training,
            current_vol20,
        )

        vix_pct = percentile_rank(
            vix_training,
            current_vix,
        )

        # ----------------------------------------------------
        # Extremeness:
        #
        #   0.50 = around middle of training distribution
        #   close to 1 = near one of the training extremes
        #
        # Example:
        #   p = 0.99 -> extremeness = 0.99
        #   p = 0.01 -> extremeness = 0.99
        #   p = 0.50 -> extremeness = 0.50
        # ----------------------------------------------------

        extremeness_values = []

        if np.isfinite(vol_pct):

            extremeness_values.append(
                max(
                    vol_pct,
                    1.0 - vol_pct,
                )
            )

        if np.isfinite(vix_pct):

            extremeness_values.append(
                max(
                    vix_pct,
                    1.0 - vix_pct,
                )
            )

        if len(extremeness_values) > 0:

            regime_extremeness = max(
                extremeness_values
            )

        else:

            regime_extremeness = np.nan

        # ----------------------------------------------------
        # Out-of-range indicators
        # ----------------------------------------------------

        vol_out = np.nan
        vix_out = np.nan

        if (
            np.isfinite(current_vol20)
            and len(vol_training) > 0
        ):

            vol_out = float(
                (
                    current_vol20
                    < np.min(vol_training)
                )
                or
                (
                    current_vol20
                    > np.max(vol_training)
                )
            )

        if (
            np.isfinite(current_vix)
            and len(vix_training) > 0
        ):

            vix_out = float(
                (
                    current_vix
                    < np.min(vix_training)
                )
                or
                (
                    current_vix
                    > np.max(vix_training)
                )
            )

        if (
            np.isfinite(vol_out)
            or np.isfinite(vix_out)
        ):

            out_any = float(
                (
                    vol_out == 1
                )
                or
                (
                    vix_out == 1
                )
            )

        else:

            out_any = np.nan

        regime_results.append(
            {
                "ticker": ticker,

                "realized_date":
                    row["realized_date"],

                "train_regime_vol20_pct":
                    vol_pct,

                "train_regime_vix_pct":
                    vix_pct,

                "train_regime_extremeness":
                    regime_extremeness,

                "train_out_of_range_vol20":
                    vol_out,

                "train_out_of_range_vix":
                    vix_out,

                "train_out_of_range_any":
                    out_any,

                "training_obs_available":
                    training_obs,
            }
        )


regime_df = pd.DataFrame(
    regime_results
)


# ============================================================
# MERGE REGIME SIGNALS
# ============================================================

panel = panel.merge(
    regime_df,
    on=[
        "ticker",
        "realized_date",
    ],
    how="left",
)


# ============================================================
# SANITY CHECK: FORECAST DATE vs REALIZED DATE
# ============================================================

print("\n" + "=" * 100)
print("TEMPORAL SANITY CHECK")
print("=" * 100)

date_gap = (
    panel["realized_date"]
    -
    panel["forecast_date"]
).dt.days


print(
    "\nForecast-to-realized calendar-day gaps:"
)

print(
    date_gap.describe().to_string()
)


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

signal_columns = [
    "cs_breach_stress_5d",
    "train_regime_extremeness",
    "train_out_of_range_any",
    "qrf_ewma_disagreement",
]


# ============================================================
# CHECK SIGNAL AVAILABILITY
# ============================================================

availability_rows = []

for signal in signal_columns:

    available = int(
        panel[signal]
        .notna()
        .sum()
    )

    total = len(panel)

    pct = (
        100.0
        * available
        / total
    )

    availability_rows.append(
        {
            "signal": signal,
            "available": available,
            "total": total,
            "availability_pct": pct,
        }
    )


availability_df = pd.DataFrame(
    availability_rows
)


# ============================================================
# PRINT AVAILABILITY
# ============================================================

print("\n" + "=" * 100)
print("SIGNAL AVAILABILITY")
print("=" * 100)

print(
    availability_df.to_string(
        index=False
    )
)


# ============================================================
# PRINT SIGNAL SUMMARIES
# ============================================================

print("\n" + "=" * 100)
print("SIGNAL DISTRIBUTIONS")
print("=" * 100)

for signal in signal_columns:

    print(
        f"\n{signal}"
    )

    print(
        panel[
            signal
        ].describe().to_string()
    )


# ============================================================
# CROSS-SECTIONAL CHECKS
# ============================================================

print("\n" + "=" * 100)
print("CROSS-SECTIONAL STRESS CHECK")
print("=" * 100)

cs_summary = (
    panel[
        [
            "realized_date",
            "ticker",
            "cs_breach_stress_5d",
        ]
    ]
    .drop_duplicates()
    .groupby("realized_date")
    .agg(
        stress_mean=(
            "cs_breach_stress_5d",
            "mean",
        ),
        ticker_count=(
            "ticker",
            "nunique",
        ),
    )
)

print(
    cs_summary.describe().to_string()
)


# ============================================================
# EWMA SANITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("QRF-EWMA SANITY CHECK")
print("=" * 100)

ewma_check = panel[
    [
        "q05_forecast",
        "ewma_q05",
        "qrf_ewma_disagreement",
    ]
].dropna()


print(
    f"\nRows with both QRF and EWMA:"
    f" {len(ewma_check):,}"
)

if len(ewma_check) > 0:

    print(
        "\nQRF q05:"
    )

    print(
        ewma_check[
            "q05_forecast"
        ].describe().to_string()
    )

    print(
        "\nEWMA q05:"
    )

    print(
        ewma_check[
            "ewma_q05"
        ].describe().to_string()
    )

    print(
        "\nDisagreement:"
    )

    print(
        ewma_check[
            "qrf_ewma_disagreement"
        ].describe().to_string()
    )


# ============================================================
# TRAINING-RANGE SANITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("TRAINING-RANGE SANITY CHECK")
print("=" * 100)

range_check = panel[
    [
        "train_regime_vol20_pct",
        "train_regime_vix_pct",
        "train_regime_extremeness",
        "train_out_of_range_vol20",
        "train_out_of_range_vix",
        "train_out_of_range_any",
        "training_obs_available",
    ]
]

print(
    range_check.describe().to_string()
)


# ============================================================
# SAVE
# ============================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

panel.to_csv(
    OUTPUT_FILE,
    index=False,
)

availability_df.to_csv(
    AVAILABILITY_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL NEW SIGNAL SUMMARY")
print("=" * 100)

print(
    "\nRows:"
    f" {len(panel):,}"
)

print(
    "\nTickers:"
    f" {panel['ticker'].nunique()}"
)

print(
    "\nRealized-date range:"
    f" {panel['realized_date'].min().date()}"
    f" to "
    f"{panel['realized_date'].max().date()}"
)

print(
    "\nSignal availability:"
)

print(
    availability_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE PATHS
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_FILE
)

print(
    AVAILABILITY_FILE
)

print("\n" + "=" * 100)
print(
    "NEW RELIABILITY SIGNAL CONSTRUCTION COMPLETE"
)
print("=" * 100)