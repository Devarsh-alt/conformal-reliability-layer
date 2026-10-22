from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# FINAL 24-TICKER COMMON VAR MONITORING PANEL
# =============================================================================
#
# Paper:
# "Simple Signals or Complex Monitors? An Empirical Study of Early Warning
#  for VaR Forecast Deterioration"
#
# FINAL UNIVERSE:
# 24 tickers
# COIN excluded because the GARCH OOS panel has only 27 observations.
#
# MODELS:
# 1. Historical Simulation (HS)
# 2. EWMA
# 3. GARCH(1,1)-Student-t
# 4. Quantile Random Forest (QRF)
#
# MAIN TARGET:
# Future h-day average pinball loss.
#
# For forecast origin t:
#
# D(t,h) = mean(L(t+1), ..., L(t+h))
#
# where L(.) is the model's 5% quantile pinball loss.
#
# PRIMARY DETERIORATION:
#
# D(t,h) > causal historical 75th percentile
#
# STRICT CAUSALITY:
#
# threshold(t,h) =
# Q_0.75(D(s,h) : s <= t-h-1)
#
# Implemented by:
#
# future_loss.shift(h+1).expanding().quantile(0.75)
#
# This prevents future-window overlap from contaminating the threshold.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MONITORING_DIR = (
    ROOT / "data" / "monitoring"
)

QRF_PATH = (
    MONITORING_DIR
    / "QRF_25ticker_oos_predictions.csv"
)

EWMA_PATH = (
    MONITORING_DIR
    / "EWMA_25ticker_oos_predictions.csv"
)

GARCH_PATH = (
    MONITORING_DIR
    / "GARCH_25ticker_oos_predictions.csv"
)

HS_PATH = (
    MONITORING_DIR
    / "pooled_hs_reliability_panel.csv"
)

OUTPUT_PANEL = (
    MONITORING_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_MODEL_SUMMARY = (
    MONITORING_DIR
    / "final_24ticker_model_summary.csv"
)

OUTPUT_TARGET_SUMMARY = (
    MONITORING_DIR
    / "final_24ticker_target_summary.csv"
)

OUTPUT_ALIGNMENT_AUDIT = (
    MONITORING_DIR
    / "final_24ticker_alignment_audit.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA = 0.05

TARGET_QUANTILE = 0.75
SEVERE_QUANTILE = 0.90

HORIZONS = [5, 10, 20]

MIN_THRESHOLD_OBS = 75

EXPECTED_TICKERS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CAT",
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

DROP_TICKERS = {"COIN"}


# =============================================================================
# BASIC HELPERS
# =============================================================================

def pinball_loss(
    actual: float,
    forecast: float,
    alpha: float = ALPHA,
) -> float:
    """
    Lower-tail quantile pinball loss.
    """

    actual = float(actual)
    forecast = float(forecast)

    if actual < forecast:
        return (
            (alpha - 1.0)
            * (actual - forecast)
        )

    return (
        alpha
        * (actual - forecast)
    )


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    """
    Exact case-insensitive column-name matching.
    """

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def clean_keys(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize ticker/date fields.
    """

    df = df.copy()

    if "ticker" in df.columns:

        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    if "forecast_date" in df.columns:

        df["forecast_date"] = pd.to_datetime(
            df["forecast_date"],
            errors="coerce",
        )

    if "realized_date" in df.columns:

        df["realized_date"] = pd.to_datetime(
            df["realized_date"],
            errors="coerce",
        )

    return df


def assert_unique(
    df: pd.DataFrame,
    keys: list[str],
    label: str,
) -> None:
    """
    Verify uniqueness of a key.
    """

    duplicates = int(
        df.duplicated(keys).sum()
    )

    if duplicates > 0:

        raise ValueError(
            f"{label}: {duplicates} duplicate rows "
            f"for keys {keys}."
        )


# =============================================================================
# FUTURE TARGET HELPERS
# =============================================================================

def make_future_mean(
    values: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    Row t gets the average of:

        values[t+1], ..., values[t+horizon]

    Current row is deliberately excluded.
    """

    x = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .to_numpy(dtype=float)
    )

    n = len(x)

    result = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if n <= horizon:
        return pd.Series(
            result,
            index=values.index,
        )

    for i in range(
        0,
        n - horizon,
    ):

        window = x[
            i + 1:
            i + horizon + 1
        ]

        if np.all(
            np.isfinite(window)
        ):

            result[i] = float(
                np.mean(window)
            )

    return pd.Series(
        result,
        index=values.index,
    )


def causal_expanding_quantile(
    future_loss: pd.Series,
    horizon: int,
    quantile: float,
    min_obs: int = MIN_THRESHOLD_OBS,
) -> pd.Series:
    """
    Strictly causal threshold.

    The future loss D(t,h) contains observations from t+1 ... t+h.

    Therefore, when setting the threshold at t, the latest historical
    target origin allowed is t-h-1.

    Hence:
        future_loss.shift(h+1)

    before taking the expanding quantile.
    """

    historical_targets = (
        future_loss
        .shift(horizon + 1)
    )

    return (
        historical_targets
        .expanding(
            min_periods=min_obs
        )
        .quantile(
            quantile
        )
    )


# =============================================================================
# CAUSAL FEATURE CONSTRUCTION
# =============================================================================

def build_causal_features(
    group: pd.DataFrame,
    q05_col: str,
    loss_col: str,
    breach_col: str,
) -> pd.DataFrame:
    """
    Construct only information available before the current forecast outcome.

    Current breach/current loss are NOT predictors.

    Only lagged and prior-window information is used.
    """

    g = (
        group
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
        .copy()
    )

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    g[q05_col] = pd.to_numeric(
        g[q05_col],
        errors="coerce",
    )

    g[loss_col] = pd.to_numeric(
        g[loss_col],
        errors="coerce",
    )

    g[breach_col] = pd.to_numeric(
        g[breach_col],
        errors="coerce",
    )

    g["actual_return"] = pd.to_numeric(
        g["actual_return"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Current forecast magnitude
    #
    # q05 is signed and negative.
    # -------------------------------------------------------------------------

    g["var_abs_current"] = (
        -g[q05_col]
    )

    # -------------------------------------------------------------------------
    # VaR dynamics
    # -------------------------------------------------------------------------

    g["var_change_1d"] = (
        g["var_abs_current"]
        -
        g["var_abs_current"].shift(1)
    )

    g["var_ratio_to_prior"] = (
        g["var_abs_current"]
        /
        g["var_abs_current"].shift(1)
    )

    prior_var_mean_20 = (
        g["var_abs_current"]
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    g["var_ratio_to_20d_mean"] = (
        g["var_abs_current"]
        /
        prior_var_mean_20
    )

    # -------------------------------------------------------------------------
    # Loss history
    # -------------------------------------------------------------------------

    g["pinball_lag1"] = (
        g[loss_col]
        .shift(1)
    )

    g["pinball_mean_5d_prior"] = (
        g[loss_col]
        .shift(1)
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
    )

    g["pinball_mean_20d_prior"] = (
        g[loss_col]
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    g["pinball_mean_60d_prior"] = (
        g[loss_col]
        .shift(1)
        .rolling(
            60,
            min_periods=60,
        )
        .mean()
    )

    g["pinball_acceleration"] = (
        g["pinball_mean_5d_prior"]
        -
        g["pinball_mean_20d_prior"]
    )

    # -------------------------------------------------------------------------
    # Breach lag
    # -------------------------------------------------------------------------

    g["breach_lag1"] = (
        g[breach_col]
        .shift(1)
    )

    breach_values = (
        g[breach_col]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    n = len(
        breach_values
    )

    streak_prior = np.zeros(
        n,
        dtype=float,
    )

    days_since_breach = np.full(
        n,
        np.nan,
        dtype=float,
    )

    previous_breach_index = None
    current_streak = 0

    for i in range(n):

        # State available before row i.
        streak_prior[i] = (
            current_streak
        )

        if previous_breach_index is not None:

            days_since_breach[i] = (
                i
                -
                previous_breach_index
                -
                1
            )

        # Update only after constructing current-row features.
        if breach_values[i] == 1:

            current_streak += 1
            previous_breach_index = i

        else:

            current_streak = 0

    g["breach_streak_prior"] = (
        streak_prior
    )

    g["days_since_last_breach"] = (
        days_since_breach
    )

    # -------------------------------------------------------------------------
    # Historical breach rates
    # -------------------------------------------------------------------------

    g["breach_rate_5d_prior"] = (
        g[breach_col]
        .shift(1)
        .rolling(
            5,
            min_periods=5,
        )
        .mean()
    )

    g["breach_rate_20d_prior"] = (
        g[breach_col]
        .shift(1)
        .rolling(
            20,
            min_periods=20,
        )
        .mean()
    )

    g["breach_rate_60d_prior"] = (
        g[breach_col]
        .shift(1)
        .rolling(
            60,
            min_periods=60,
        )
        .mean()
    )

    # -------------------------------------------------------------------------
    # Realized volatility from RETURNS ALREADY OBSERVED.
    # -------------------------------------------------------------------------

    prior_returns = (
        g["actual_return"]
        .shift(1)
    )

    g["realized_vol_20d"] = (
        prior_returns
        .rolling(
            20,
            min_periods=20,
        )
        .std()
    )

    g["realized_vol_60d"] = (
        prior_returns
        .rolling(
            60,
            min_periods=60,
        )
        .std()
    )

    g["vol_ratio_20_60"] = (
        g["realized_vol_20d"]
        /
        g["realized_vol_60d"]
    )

    # -------------------------------------------------------------------------
    # Historical average loss
    # -------------------------------------------------------------------------

    g["past_mean_pinball"] = (
        g[loss_col]
        .shift(1)
        .expanding(
            min_periods=1
        )
        .mean()
    )

    return g


# =============================================================================
# MODEL-SPECIFIC TARGETS
# =============================================================================

def build_model_targets(
    group: pd.DataFrame,
    prefix: str,
    loss_col: str,
) -> pd.DataFrame:
    """
    Construct future loss and strictly causal deterioration labels.
    """

    g = (
        group
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
        .copy()
    )

    loss = pd.to_numeric(
        g[loss_col],
        errors="coerce",
    )

    for horizon in HORIZONS:

        # ---------------------------------------------------------------------
        # Future average loss
        # ---------------------------------------------------------------------

        future_loss = make_future_mean(
            loss,
            horizon,
        )

        future_col = (
            f"{prefix}_future_pinball_{horizon}d"
        )

        g[future_col] = (
            future_loss
        )

        # ---------------------------------------------------------------------
        # Causal 75th percentile
        # ---------------------------------------------------------------------

        threshold_75 = (
            causal_expanding_quantile(
                future_loss,
                horizon,
                TARGET_QUANTILE,
            )
        )

        threshold_75_col = (
            f"{prefix}_causal_threshold_75_{horizon}d"
        )

        g[threshold_75_col] = (
            threshold_75
        )

        # ---------------------------------------------------------------------
        # Causal 90th percentile
        # ---------------------------------------------------------------------

        threshold_90 = (
            causal_expanding_quantile(
                future_loss,
                horizon,
                SEVERE_QUANTILE,
            )
        )

        threshold_90_col = (
            f"{prefix}_causal_threshold_90_{horizon}d"
        )

        g[threshold_90_col] = (
            threshold_90
        )

        # ---------------------------------------------------------------------
        # Primary deterioration target
        # ---------------------------------------------------------------------

        deterioration_col = (
            f"{prefix}_deterioration_{horizon}d"
        )

        g[deterioration_col] = np.nan

        valid_75 = (
            future_loss.notna()
            &
            threshold_75.notna()
        )

        g.loc[
            valid_75,
            deterioration_col,
        ] = (
            future_loss.loc[
                valid_75
            ]
            >
            threshold_75.loc[
                valid_75
            ]
        ).astype(float)

        # ---------------------------------------------------------------------
        # Severe deterioration target
        # ---------------------------------------------------------------------

        severe_col = (
            f"{prefix}_severe_deterioration_{horizon}d"
        )

        g[severe_col] = np.nan

        valid_90 = (
            future_loss.notna()
            &
            threshold_90.notna()
        )

        g.loc[
            valid_90,
            severe_col,
        ] = (
            future_loss.loc[
                valid_90
            ]
            >
            threshold_90.loc[
                valid_90
            ]
        ).astype(float)

        # ---------------------------------------------------------------------
        # Fixed 1.5x historical mean sensitivity
        # ---------------------------------------------------------------------

        historical_mean = (
            loss
            .shift(horizon + 1)
            .expanding(
                min_periods=MIN_THRESHOLD_OBS
            )
            .mean()
        )

        fixed_col = (
            f"{prefix}_deterioration_fixed_1p5_{horizon}d"
        )

        g[fixed_col] = np.nan

        valid_fixed = (
            future_loss.notna()
            &
            historical_mean.notna()
        )

        g.loc[
            valid_fixed,
            fixed_col,
        ] = (
            future_loss.loc[
                valid_fixed
            ]
            >
            (
                1.5
                *
                historical_mean.loc[
                    valid_fixed
                ]
            )
        ).astype(float)

    return g


# =============================================================================
# MODEL SUMMARY
# =============================================================================

def summarize_model(
    df: pd.DataFrame,
    prefix: str,
) -> dict:

    q05_col = (
        f"{prefix}_q05"
    )

    breach_col = (
        f"{prefix}_breach"
    )

    loss_col = (
        f"{prefix}_pinball_loss"
    )

    q05 = pd.to_numeric(
        df[q05_col],
        errors="coerce",
    )

    breach = pd.to_numeric(
        df[breach_col],
        errors="coerce",
    )

    loss = pd.to_numeric(
        df[loss_col],
        errors="coerce",
    )

    return {
        "model":
            prefix.upper(),

        "rows":
            int(len(df)),

        "tickers":
            int(
                df["ticker"].nunique()
            ),

        "breaches":
            int(
                breach.sum()
            ),

        "breach_rate":
            float(
                breach.mean()
            ),

        "mean_pinball_loss":
            float(
                loss.mean()
            ),

        "median_pinball_loss":
            float(
                loss.median()
            ),

        "mean_q05":
            float(
                q05.mean()
            ),
    }


# =============================================================================
# START
# =============================================================================

print(
    "=" * 100,
    flush=True,
)

print(
    "FINAL 24-TICKER COMMON VAR MONITORING PANEL",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    f"Project root: {ROOT}",
    flush=True,
)

print(
    f"Target quantile: {TARGET_QUANTILE}",
    flush=True,
)

print(
    f"Severe quantile: {SEVERE_QUANTILE}",
    flush=True,
)

print(
    f"Horizons: {HORIZONS}",
    flush=True,
)

print(
    f"Minimum causal threshold observations: "
    f"{MIN_THRESHOLD_OBS}",
    flush=True,
)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

required_files = {
    "QRF": QRF_PATH,
    "EWMA": EWMA_PATH,
    "GARCH": GARCH_PATH,
    "HS": HS_PATH,
}

for label, path in required_files.items():

    if not path.exists():

        raise FileNotFoundError(
            f"{label} file not found:\n{path}"
        )


# =============================================================================
# LOAD FILES
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "LOADING MODEL FILES",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

qrf = pd.read_csv(
    QRF_PATH
)

ewma = pd.read_csv(
    EWMA_PATH
)

garch = pd.read_csv(
    GARCH_PATH
)

hs = pd.read_csv(
    HS_PATH
)

print(
    f"QRF rows:   {len(qrf):,}",
    flush=True,
)

print(
    f"EWMA rows:  {len(ewma):,}",
    flush=True,
)

print(
    f"GARCH rows: {len(garch):,}",
    flush=True,
)

print(
    f"HS rows:    {len(hs):,}",
    flush=True,
)


# =============================================================================
# CLEAN DATA
# =============================================================================

qrf = clean_keys(qrf)
ewma = clean_keys(ewma)
garch = clean_keys(garch)
hs = clean_keys(hs)


# =============================================================================
# PREPARE QRF
# =============================================================================

print(
    "\nPreparing QRF...",
    flush=True,
)

qrf_required = [
    "ticker",
    "forecast_date",
    "realized_date",
    "actual_return",
    "q05_forecast",
]

for column in qrf_required:

    if column not in qrf.columns:

        raise ValueError(
            f"QRF missing required column: {column}"
        )

qrf = qrf[
    qrf_required
].copy()

qrf = qrf.rename(
    columns={
        "q05_forecast":
            "qrf_q05",
    }
)

qrf["actual_return"] = pd.to_numeric(
    qrf["actual_return"],
    errors="coerce",
)

qrf["qrf_q05"] = pd.to_numeric(
    qrf["qrf_q05"],
    errors="coerce",
)

qrf = qrf.dropna(
    subset=[
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "qrf_q05",
    ]
)

qrf = (
    qrf
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

assert_unique(
    qrf,
    [
        "ticker",
        "forecast_date",
    ],
    "QRF",
)

qrf["qrf_pinball_loss"] = [
    pinball_loss(
        actual,
        forecast,
        ALPHA,
    )
    for actual, forecast in zip(
        qrf["actual_return"],
        qrf["qrf_q05"],
    )
]

qrf["qrf_breach"] = (
    qrf["actual_return"]
    <
    qrf["qrf_q05"]
).astype(int)


# =============================================================================
# PREPARE EWMA
# =============================================================================

print(
    "Preparing EWMA...",
    flush=True,
)

ewma_required = [
    "ticker",
    "forecast_date",
    "realized_date",
    "actual_return",
    "ewma_q05",
]

for column in ewma_required:

    if column not in ewma.columns:

        raise ValueError(
            f"EWMA missing required column: {column}"
        )

ewma = ewma[
    ewma_required
].copy()

ewma["actual_return"] = pd.to_numeric(
    ewma["actual_return"],
    errors="coerce",
)

ewma["ewma_q05"] = pd.to_numeric(
    ewma["ewma_q05"],
    errors="coerce",
)

ewma = ewma.dropna(
    subset=[
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "ewma_q05",
    ]
)

ewma = (
    ewma
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

assert_unique(
    ewma,
    [
        "ticker",
        "forecast_date",
    ],
    "EWMA",
)

ewma["ewma_pinball_loss"] = [
    pinball_loss(
        actual,
        forecast,
        ALPHA,
    )
    for actual, forecast in zip(
        ewma["actual_return"],
        ewma["ewma_q05"],
    )
]

ewma["ewma_breach"] = (
    ewma["actual_return"]
    <
    ewma["ewma_q05"]
).astype(int)


# =============================================================================
# PREPARE GARCH
# =============================================================================

print(
    "Preparing GARCH...",
    flush=True,
)

garch_required = [
    "ticker",
    "forecast_date",
    "realized_date",
    "actual_return",
    "garch_q05",
]

for column in garch_required:

    if column not in garch.columns:

        raise ValueError(
            f"GARCH missing required column: {column}"
        )

garch = garch[
    garch_required
].copy()

garch["actual_return"] = pd.to_numeric(
    garch["actual_return"],
    errors="coerce",
)

garch["garch_q05"] = pd.to_numeric(
    garch["garch_q05"],
    errors="coerce",
)

garch = garch.dropna(
    subset=[
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "garch_q05",
    ]
)

garch = (
    garch
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

assert_unique(
    garch,
    [
        "ticker",
        "forecast_date",
    ],
    "GARCH",
)

garch["garch_pinball_loss"] = [
    pinball_loss(
        actual,
        forecast,
        ALPHA,
    )
    for actual, forecast in zip(
        garch["actual_return"],
        garch["garch_q05"],
    )
]

garch["garch_breach"] = (
    garch["actual_return"]
    <
    garch["garch_q05"]
).astype(int)


# =============================================================================
# PREPARE HS
# =============================================================================

print(
    "Preparing HS...",
    flush=True,
)

hs_date_col = find_column(
    hs,
    [
        "realized_date",
        "date",
        "Date",
        "outcome_date",
    ],
    "HS realized-date column",
)

hs_return_col = find_column(
    hs,
    [
        "actual_return",
        "return",
        "Return",
        "Log_Return",
        "log_return",
    ],
    "HS return column",
)

hs_q05_col = find_column(
    hs,
    [
        "HS_VaR",
        "hs_q05",
        "HS_q05",
        "q05_forecast",
    ],
    "HS lower-tail quantile column",
)

hs = hs[
    [
        "ticker",
        hs_date_col,
        hs_return_col,
        hs_q05_col,
    ]
].copy()

hs = hs.rename(
    columns={
        hs_date_col:
            "realized_date",

        hs_return_col:
            "actual_return",

        hs_q05_col:
            "hs_q05",
    }
)

hs["realized_date"] = pd.to_datetime(
    hs["realized_date"],
    errors="coerce",
)

hs["actual_return"] = pd.to_numeric(
    hs["actual_return"],
    errors="coerce",
)

hs["hs_q05"] = pd.to_numeric(
    hs["hs_q05"],
    errors="coerce",
)

hs = hs.dropna(
    subset=[
        "ticker",
        "realized_date",
        "actual_return",
        "hs_q05",
    ]
)

hs = (
    hs
    .sort_values(
        [
            "ticker",
            "realized_date",
        ]
    )
    .reset_index(drop=True)
)

assert_unique(
    hs,
    [
        "ticker",
        "realized_date",
    ],
    "HS",
)

hs["hs_pinball_loss"] = [
    pinball_loss(
        actual,
        forecast,
        ALPHA,
    )
    for actual, forecast in zip(
        hs["actual_return"],
        hs["hs_q05"],
    )
]

hs["hs_breach"] = (
    hs["actual_return"]
    <
    hs["hs_q05"]
).astype(int)


# =============================================================================
# DROP COIN
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "FINAL TICKER UNIVERSE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

qrf = qrf[
    ~qrf["ticker"].isin(
        DROP_TICKERS
    )
].copy()

ewma = ewma[
    ~ewma["ticker"].isin(
        DROP_TICKERS
    )
].copy()

garch = garch[
    ~garch["ticker"].isin(
        DROP_TICKERS
    )
].copy()

hs = hs[
    ~hs["ticker"].isin(
        DROP_TICKERS
    )
].copy()

final_qrf_tickers = sorted(
    qrf["ticker"].unique()
)

print(
    "QRF tickers:",
    final_qrf_tickers,
    flush=True,
)

print(
    "\nGARCH tickers:",
    sorted(
        garch["ticker"].unique()
    ),
    flush=True,
)

print(
    "\nHS tickers:",
    sorted(
        hs["ticker"].unique()
    ),
    flush=True,
)

print(
    "\nEWMA tickers:",
    sorted(
        ewma["ticker"].unique()
    ),
    flush=True,
)

if final_qrf_tickers != sorted(
    EXPECTED_TICKERS
):

    raise ValueError(
        "QRF final universe does not exactly match "
        "the expected 24 tickers."
    )


# =============================================================================
# BUILD MASTER COMMON PANEL
# =============================================================================

panel = qrf.copy()


# =============================================================================
# MERGE EWMA
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "ALIGNING QRF + EWMA",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

ewma_merge = ewma[
    [
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "ewma_q05",
        "ewma_pinball_loss",
        "ewma_breach",
    ]
].copy()

ewma_merge = ewma_merge.rename(
    columns={
        "realized_date":
            "ewma_realized_date",

        "actual_return":
            "ewma_actual_return",
    }
)

panel = panel.merge(
    ewma_merge,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    f"Rows after QRF + EWMA: "
    f"{len(panel):,}",
    flush=True,
)


# =============================================================================
# MERGE GARCH
# =============================================================================

print(
    "\nAligning GARCH...",
    flush=True,
)

garch_merge = garch[
    [
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",
        "garch_q05",
        "garch_pinball_loss",
        "garch_breach",
    ]
].copy()

garch_merge = garch_merge.rename(
    columns={
        "realized_date":
            "garch_realized_date",

        "actual_return":
            "garch_actual_return",
    }
)

panel = panel.merge(
    garch_merge,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    f"Rows after QRF + EWMA + GARCH: "
    f"{len(panel):,}",
    flush=True,
)


# =============================================================================
# MERGE HS BY REALIZED DATE
# =============================================================================

print(
    "\nAligning HS using realized date...",
    flush=True,
)

hs_merge = hs[
    [
        "ticker",
        "realized_date",
        "actual_return",
        "hs_q05",
        "hs_pinball_loss",
        "hs_breach",
    ]
].copy()

hs_merge = hs_merge.rename(
    columns={
        "actual_return":
            "hs_actual_return",
    }
)

panel = panel.merge(
    hs_merge,
    on=[
        "ticker",
        "realized_date",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    f"Rows after all four models: "
    f"{len(panel):,}",
    flush=True,
)


# =============================================================================
# CORE ALIGNMENT AUDIT
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "CORE ALIGNMENT AUDIT",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

qrf_ewma_diff = (
    panel["actual_return"]
    -
    panel["ewma_actual_return"]
).abs()

qrf_garch_diff = (
    panel["actual_return"]
    -
    panel["garch_actual_return"]
).abs()

qrf_hs_diff = (
    panel["actual_return"]
    -
    panel["hs_actual_return"]
).abs()

print(
    f"Max QRF-EWMA actual-return diff: "
    f"{qrf_ewma_diff.max():.12e}",
    flush=True,
)

print(
    f"Max QRF-GARCH actual-return diff: "
    f"{qrf_garch_diff.max():.12e}",
    flush=True,
)

print(
    f"Max QRF-HS actual-return diff: "
    f"{qrf_hs_diff.max():.12e}",
    flush=True,
)

if qrf_ewma_diff.max() > 1e-10:

    raise RuntimeError(
        "QRF/EWMA actual returns are not aligned."
    )

if qrf_garch_diff.max() > 1e-10:

    raise RuntimeError(
        "QRF/GARCH actual returns are not aligned."
    )

if qrf_hs_diff.max() > 1e-10:

    raise RuntimeError(
        "QRF/HS actual returns are not aligned."
    )


ewma_realized_mismatch = (
    panel["realized_date"]
    !=
    panel["ewma_realized_date"]
)

garch_realized_mismatch = (
    panel["realized_date"]
    !=
    panel["garch_realized_date"]
)

print(
    f"EWMA realized-date mismatches: "
    f"{int(ewma_realized_mismatch.sum())}",
    flush=True,
)

print(
    f"GARCH realized-date mismatches: "
    f"{int(garch_realized_mismatch.sum())}",
    flush=True,
)

if ewma_realized_mismatch.any():

    raise RuntimeError(
        "EWMA realized-date alignment failed."
    )

if garch_realized_mismatch.any():

    raise RuntimeError(
        "GARCH realized-date alignment failed."
    )


# =============================================================================
# DUPLICATE AUDIT
# =============================================================================

forecast_duplicates = int(
    panel.duplicated(
        [
            "ticker",
            "forecast_date",
        ]
    ).sum()
)

realized_duplicates = int(
    panel.duplicated(
        [
            "ticker",
            "realized_date",
        ]
    ).sum()
)

print(
    f"Duplicate ticker-forecast_date rows: "
    f"{forecast_duplicates}",
    flush=True,
)

print(
    f"Duplicate ticker-realized_date rows: "
    f"{realized_duplicates}",
    flush=True,
)

if forecast_duplicates > 0:

    raise RuntimeError(
        "Duplicate forecast-date rows found."
    )

if realized_duplicates > 0:

    raise RuntimeError(
        "Duplicate realized-date rows found."
    )


# =============================================================================
# FINAL TICKER FILTER
# =============================================================================

panel = panel[
    panel["ticker"].isin(
        EXPECTED_TICKERS
    )
].copy()

panel = (
    panel
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)


# =============================================================================
# CAUSAL FEATURES
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "BUILDING CAUSAL MONITORING FEATURES",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

model_specs = {

    "qrf": {
        "q05":
            "qrf_q05",
        "loss":
            "qrf_pinball_loss",
        "breach":
            "qrf_breach",
    },

    "ewma": {
        "q05":
            "ewma_q05",
        "loss":
            "ewma_pinball_loss",
        "breach":
            "ewma_breach",
    },

    "garch": {
        "q05":
            "garch_q05",
        "loss":
            "garch_pinball_loss",
        "breach":
            "garch_breach",
    },

    "hs": {
        "q05":
            "hs_q05",
        "loss":
            "hs_pinball_loss",
        "breach":
            "hs_breach",
    },
}

feature_frames = []

for ticker, ticker_group in panel.groupby(
    "ticker",
    sort=False,
):

    print(
        f"  Features: {ticker}",
        flush=True,
    )

    model_feature_frames = []

    for prefix, spec in model_specs.items():

        temp_input = ticker_group[
            [
                "ticker",
                "forecast_date",
                "realized_date",
                "actual_return",
                spec["q05"],
                spec["loss"],
                spec["breach"],
            ]
        ].copy()

        temp = build_causal_features(
            temp_input,
            q05_col=spec["q05"],
            loss_col=spec["loss"],
            breach_col=spec["breach"],
        )

        keep_columns = [
            "ticker",
            "forecast_date",

            "var_abs_current",
            "var_change_1d",
            "var_ratio_to_prior",
            "var_ratio_to_20d_mean",

            "pinball_lag1",
            "pinball_mean_5d_prior",
            "pinball_mean_20d_prior",
            "pinball_mean_60d_prior",
            "pinball_acceleration",

            "breach_lag1",
            "breach_streak_prior",
            "days_since_last_breach",

            "breach_rate_5d_prior",
            "breach_rate_20d_prior",
            "breach_rate_60d_prior",

            "realized_vol_20d",
            "realized_vol_60d",
            "vol_ratio_20_60",

            "past_mean_pinball",
        ]

        temp = temp[
            keep_columns
        ].copy()

        rename_map = {}

        for column in temp.columns:

            if column in {
                "ticker",
                "forecast_date",
            }:
                continue

            rename_map[column] = (
                f"{prefix}_{column}"
            )

        temp = temp.rename(
            columns=rename_map
        )

        model_feature_frames.append(
            temp
        )

    merged_features = (
        model_feature_frames[0]
    )

    for next_features in (
        model_feature_frames[1:]
    ):

        # FIX:
        # how must be "inner", while one_to_one belongs in validate.
        merged_features = merged_features.merge(
            next_features,
            on=[
                "ticker",
                "forecast_date",
            ],
            how="inner",
            validate="one_to_one",
        )

    feature_frames.append(
        merged_features
    )


causal_features = pd.concat(
    feature_frames,
    ignore_index=True,
)

panel = panel.merge(
    causal_features,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="left",
    validate="one_to_one",
)


# =============================================================================
# TARGETS
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "BUILDING STRICTLY CAUSAL DETERIORATION TARGETS",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

target_summary_rows = []

for prefix, spec in model_specs.items():

    print(
        f"\nTarget model: {prefix.upper()}",
        flush=True,
    )

    model_target_frames = []

    for ticker, ticker_group in panel.groupby(
        "ticker",
        sort=False,
    ):

        temp = ticker_group[
            [
                "ticker",
                "forecast_date",
                spec["loss"],
            ]
        ].copy()

        temp = build_model_targets(
            temp,
            prefix=prefix,
            loss_col=spec["loss"],
        )

        keep_columns = [
            "ticker",
            "forecast_date",
        ]

        for horizon in HORIZONS:

            keep_columns.extend(
                [
                    (
                        f"{prefix}"
                        f"_future_pinball_{horizon}d"
                    ),

                    (
                        f"{prefix}"
                        f"_causal_threshold_75_{horizon}d"
                    ),

                    (
                        f"{prefix}"
                        f"_causal_threshold_90_{horizon}d"
                    ),

                    (
                        f"{prefix}"
                        f"_deterioration_{horizon}d"
                    ),

                    (
                        f"{prefix}"
                        f"_severe_deterioration_{horizon}d"
                    ),

                    (
                        f"{prefix}"
                        f"_deterioration_fixed_1p5_{horizon}d"
                    ),
                ]
            )

        model_target_frames.append(
            temp[
                keep_columns
            ]
        )

    model_targets = pd.concat(
        model_target_frames,
        ignore_index=True,
    )

    panel = panel.merge(
        model_targets,
        on=[
            "ticker",
            "forecast_date",
        ],
        how="left",
        validate="one_to_one",
    )


# =============================================================================
# TARGET SANITY CHECKS
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "TARGET SANITY CHECKS",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

for prefix in model_specs.keys():

    print(
        f"\n{prefix.upper()}",
        flush=True,
    )

    loss_col = (
        f"{prefix}_pinball_loss"
    )

    print(
        f"  Mean instantaneous pinball loss: "
        f"{panel[loss_col].mean():.8f}",
        flush=True,
    )

    for horizon in HORIZONS:

        future_col = (
            f"{prefix}"
            f"_future_pinball_{horizon}d"
        )

        threshold75_col = (
            f"{prefix}"
            f"_causal_threshold_75_{horizon}d"
        )

        threshold90_col = (
            f"{prefix}"
            f"_causal_threshold_90_{horizon}d"
        )

        target_col = (
            f"{prefix}"
            f"_deterioration_{horizon}d"
        )

        severe_col = (
            f"{prefix}"
            f"_severe_deterioration_{horizon}d"
        )

        fixed_col = (
            f"{prefix}"
            f"_deterioration_fixed_1p5_{horizon}d"
        )

        valid_target = (
            panel[future_col].notna()
            &
            panel[threshold75_col].notna()
        )

        valid_severe = (
            panel[future_col].notna()
            &
            panel[threshold90_col].notna()
        )

        valid_fixed = (
            panel[future_col].notna()
            &
            panel[
                fixed_col
            ].notna()
        )

        positive_rate = (
            panel.loc[
                valid_target,
                target_col,
            ].mean()
            if valid_target.any()
            else np.nan
        )

        severe_rate = (
            panel.loc[
                valid_severe,
                severe_col,
            ].mean()
            if valid_severe.any()
            else np.nan
        )

        fixed_rate = (
            panel.loc[
                valid_fixed,
                fixed_col,
            ].mean()
            if valid_fixed.any()
            else np.nan
        )

        print(
            f"  {horizon}d:"
            f" valid={int(valid_target.sum()):,}"
            f" positive={positive_rate:.4%}"
            f" severe={severe_rate:.4%}"
            f" fixed1.5={fixed_rate:.4%}",
            flush=True,
        )

        target_summary_rows.append(
            {
                "model":
                    prefix.upper(),

                "horizon":
                    horizon,

                "valid_target_rows":
                    int(
                        valid_target.sum()
                    ),

                "positive_rate":
                    float(
                        positive_rate
                    ),

                "valid_severe_rows":
                    int(
                        valid_severe.sum()
                    ),

                "severe_positive_rate":
                    float(
                        severe_rate
                    ),

                "valid_fixed_rows":
                    int(
                        valid_fixed.sum()
                    ),

                "fixed_1p5_positive_rate":
                    float(
                        fixed_rate
                    ),
            }
        )


# =============================================================================
# CROSS-MODEL DISAGREEMENT FEATURES
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "ADDING CROSS-MODEL DISAGREEMENT VARIABLES",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

# -------------------------------------------------------------------------
# QRF vs EWMA
# -------------------------------------------------------------------------

panel["qrf_ewma_disagreement"] = (
    (
        panel["qrf_q05"]
        -
        panel["ewma_q05"]
    )
    /
    panel["ewma_q05"].abs()
)


# -------------------------------------------------------------------------
# QRF vs GARCH
# -------------------------------------------------------------------------

panel["qrf_garch_disagreement"] = (
    (
        panel["qrf_q05"]
        -
        panel["garch_q05"]
    )
    /
    panel["garch_q05"].abs()
)


# -------------------------------------------------------------------------
# QRF vs HS
# -------------------------------------------------------------------------

panel["qrf_hs_disagreement"] = (
    (
        panel["qrf_q05"]
        -
        panel["hs_q05"]
    )
    /
    panel["hs_q05"].abs()
)


# -------------------------------------------------------------------------
# Mean absolute disagreement from QRF
# -------------------------------------------------------------------------

panel[
    "mean_absolute_model_disagreement"
] = (
    (
        (
            panel["qrf_q05"]
            -
            panel["ewma_q05"]
        ).abs()

        +

        (
            panel["qrf_q05"]
            -
            panel["garch_q05"]
        ).abs()

        +

        (
            panel["qrf_q05"]
            -
            panel["hs_q05"]
        ).abs()
    )
    /
    3.0
)


# =============================================================================
# FINAL SORT
# =============================================================================

panel = (
    panel
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)


# =============================================================================
# MODEL SUMMARY
# =============================================================================

model_summary_rows = []

for prefix in model_specs.keys():

    model_summary_rows.append(
        summarize_model(
            panel,
            prefix,
        )
    )

model_summary = pd.DataFrame(
    model_summary_rows
)


# =============================================================================
# COMMON SAMPLE SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "FINAL COMMON SAMPLE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    f"Total rows: "
    f"{len(panel):,}",
    flush=True,
)

print(
    f"Tickers: "
    f"{panel['ticker'].nunique()}",
    flush=True,
)

print(
    f"Forecast date range: "
    f"{panel['forecast_date'].min().date()} "
    f"to "
    f"{panel['forecast_date'].max().date()}",
    flush=True,
)

print(
    f"Realized date range: "
    f"{panel['realized_date'].min().date()} "
    f"to "
    f"{panel['realized_date'].max().date()}",
    flush=True,
)


# =============================================================================
# ROWS PER TICKER
# =============================================================================

rows_per_ticker = (
    panel
    .groupby("ticker")
    .size()
    .sort_index()
)

print(
    "\nRows per ticker:",
    flush=True,
)

print(
    rows_per_ticker.to_string(),
    flush=True,
)


# =============================================================================
# FINAL TICKER CHECK
# =============================================================================

observed_tickers = sorted(
    panel["ticker"].unique()
)

if observed_tickers != sorted(
    EXPECTED_TICKERS
):

    raise RuntimeError(
        "Final ticker universe mismatch.\n"
        f"Expected: {sorted(EXPECTED_TICKERS)}\n"
        f"Observed: {observed_tickers}"
    )


# =============================================================================
# TARGET SUMMARY
# =============================================================================

target_summary = pd.DataFrame(
    target_summary_rows
)


# =============================================================================
# ALIGNMENT AUDIT OUTPUT
# =============================================================================

alignment_audit_rows = [

    {
        "check":
            "QRF_EWMA_max_actual_return_difference",

        "value":
            float(
                qrf_ewma_diff.max()
            ),

        "status":
            "PASS"
            if qrf_ewma_diff.max()
            <= 1e-10
            else "FAIL",
    },

    {
        "check":
            "QRF_GARCH_max_actual_return_difference",

        "value":
            float(
                qrf_garch_diff.max()
            ),

        "status":
            "PASS"
            if qrf_garch_diff.max()
            <= 1e-10
            else "FAIL",
    },

    {
        "check":
            "QRF_HS_max_actual_return_difference",

        "value":
            float(
                qrf_hs_diff.max()
            ),

        "status":
            "PASS"
            if qrf_hs_diff.max()
            <= 1e-10
            else "FAIL",
    },

    {
        "check":
            "EWMA_realized_date_mismatches",

        "value":
            int(
                ewma_realized_mismatch.sum()
            ),

        "status":
            "PASS"
            if ewma_realized_mismatch.sum()
            == 0
            else "FAIL",
    },

    {
        "check":
            "GARCH_realized_date_mismatches",

        "value":
            int(
                garch_realized_mismatch.sum()
            ),

        "status":
            "PASS"
            if garch_realized_mismatch.sum()
            == 0
            else "FAIL",
    },

    {
        "check":
            "duplicate_ticker_forecast_date",

        "value":
            forecast_duplicates,

        "status":
            "PASS"
            if forecast_duplicates == 0
            else "FAIL",
    },

    {
        "check":
            "duplicate_ticker_realized_date",

        "value":
            realized_duplicates,

        "status":
            "PASS"
            if realized_duplicates == 0
            else "FAIL",
    },

    {
        "check":
            "final_ticker_count",

        "value":
            int(
                panel["ticker"].nunique()
            ),

        "status":
            "PASS"
            if panel["ticker"].nunique()
            == 24
            else "FAIL",
    },

    {
        "check":
            "final_row_count",

        "value":
            int(
                len(panel)
            ),

        "status":
            "PASS"
            if len(panel) > 0
            else "FAIL",
    },
]

alignment_audit = pd.DataFrame(
    alignment_audit_rows
)


# =============================================================================
# SAVE
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "SAVING FINAL FILES",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

panel.to_csv(
    OUTPUT_PANEL,
    index=False,
)

model_summary.to_csv(
    OUTPUT_MODEL_SUMMARY,
    index=False,
)

target_summary.to_csv(
    OUTPUT_TARGET_SUMMARY,
    index=False,
)

alignment_audit.to_csv(
    OUTPUT_ALIGNMENT_AUDIT,
    index=False,
)

print(
    f"\nMain panel:\n{OUTPUT_PANEL}",
    flush=True,
)

print(
    f"\nModel summary:\n{OUTPUT_MODEL_SUMMARY}",
    flush=True,
)

print(
    f"\nTarget summary:\n{OUTPUT_TARGET_SUMMARY}",
    flush=True,
)

print(
    f"\nAlignment audit:\n{OUTPUT_ALIGNMENT_AUDIT}",
    flush=True,
)


# =============================================================================
# FINAL CORE-COLUMN CHECK
# =============================================================================

core_columns = [
    "qrf_q05",
    "ewma_q05",
    "garch_q05",
    "hs_q05",

    "qrf_pinball_loss",
    "ewma_pinball_loss",
    "garch_pinball_loss",
    "hs_pinball_loss",
]

for column in core_columns:

    if panel[column].isna().any():

        raise RuntimeError(
            f"Unexpected missing values in "
            f"core column: {column}"
        )


# =============================================================================
# FINAL ROW-COUNT CHECK
# =============================================================================

expected_rows = (
    24 * 853
)

if len(panel) != expected_rows:

    print(
        "\nWARNING:",
        flush=True,
    )

    print(
        f"Expected common overlap under current "
        f"data endpoints: {expected_rows:,}",
        flush=True,
    )

    print(
        f"Observed: {len(panel):,}",
        flush=True,
    )

    print(
        "This is not automatically a failure; "
        "inspect the row counts above.",
        flush=True,
    )


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "FINAL 24-TICKER COMMON PANEL COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    "\nUniverse: 24 tickers",
    flush=True,
)

print(
    "COIN: excluded",
    flush=True,
)

print(
    f"Common observations: {len(panel):,}",
    flush=True,
)

print(
    "Core alignment: PASS",
    flush=True,
)

print(
    "Strictly causal targets: BUILT",
    flush=True,
)

print(
    "Next stage: final simple-vs-complex evaluation",
    flush=True,
)