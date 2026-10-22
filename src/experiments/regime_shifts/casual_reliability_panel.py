from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


# =============================================================================
# RS04 — REVISED CAUSAL RELIABILITY PANEL
# =============================================================================
#
# PURPOSE
# -------
# Build a strictly causal monitoring dataset answering:
#
# "During a market regime shift, can a monitoring layer determine how much
# confidence a risk manager should place in an existing VaR model?"
#
#
# DEPENDENCY CHAIN
# ----------------
#
# RS01:
#   Market data
#       ↓
#   HS / EWMA / GARCH-t / QRF VaR
#
# RS03:
#   VIX + realized volatility
#       ↓
#   causal market regime
#
# RS04:
#   information available at t
#       ↓
#   causal monitoring features
#       ↓
#   future deterioration labels
#
# RS05:
#   reliability / trust model
#
#
# IMPORTANT
# ---------
# The main RS04 panel MUST NOT contain raw next-day realized outcomes as
# predictor columns.
#
# Raw next-day breach/loss outcomes are saved separately.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"D:\reliability_layer"
)

INPUT_PANEL = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS03_RS01_regime_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PANEL = (
    OUTPUT_DIR
    / "RS04_causal_reliability_panel.csv"
)

OUTPUT_OUTCOMES = (
    OUTPUT_DIR
    / "RS04_quarantined_outcomes.csv"
)

OUTPUT_TARGET_SUMMARY = (
    OUTPUT_DIR
    / "RS04_target_summary.csv"
)

OUTPUT_LABEL_CORRELATION = (
    OUTPUT_DIR
    / "RS04_label_correlation.csv"
)

OUTPUT_TRUNCATION_AUDIT = (
    OUTPUT_DIR
    / "RS04_truncation_audit.csv"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR
    / "RS04_column_manifest.json"
)

OUTPUT_LOG = (
    OUTPUT_DIR
    / "RS04_causal_reliability_panel.log"
)

PREVIOUS_PANEL_BACKUP = (
    OUTPUT_DIR
    / "RS04_causal_reliability_panel_PRE_REVISION.csv"
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            OUTPUT_LOG,
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

TICKERS = [
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

MODELS = {
    "HS": {
        "var": "hs_var",
        "breach": "hs_breach",
        "loss": "hs_pinball_loss",
    },

    "EWMA": {
        "var": "ewma_var",
        "breach": "ewma_breach",
        "loss": "ewma_pinball_loss",
    },

    "GARCH-t": {
        "var": "garch_var",
        "breach": "garch_breach",
        "loss": "garch_pinball_loss",
    },

    "QRF": {
        "var": "qrf_var",
        "breach": "qrf_breach",
        "loss": "qrf_pinball_loss",
    },
}

HORIZONS = [
    5,
    10,
    20,
]

MONITOR_WINDOWS = [
    5,
    10,
    20,
]

VAR_ALPHA = 0.05

PRIMARY_QUANTILE = 0.75

MIN_HISTORY = 252

COVERAGE_TEST_LEVEL = 0.05

TRUNCATION_TEST_DATES = [
    pd.Timestamp("2018-02-05"),
    pd.Timestamp("2020-03-16"),
    pd.Timestamp("2022-06-01"),
]


# =============================================================================
# HELPERS
# =============================================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    denominator = denominator.replace(
        [np.inf, -np.inf, 0],
        np.nan,
    )

    result = numerator / denominator

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


def safe_pct_change(
    series: pd.Series,
    periods: int,
) -> pd.Series:

    previous = series.shift(periods)

    return (
        safe_divide(
            series,
            previous,
        )
        - 1.0
    )


def causal_zscore(
    series: pd.Series,
    window: int = 252,
) -> pd.Series:

    past = series.shift(1)

    rolling_mean = (
        past
        .rolling(
            window,
            min_periods=window,
        )
        .mean()
    )

    rolling_std = (
        past
        .rolling(
            window,
            min_periods=window,
        )
        .std()
    )

    return safe_divide(
        series - rolling_mean,
        rolling_std,
    )


def assert_contiguous_dates(
    df: pd.DataFrame,
    ticker: str,
) -> None:

    if len(df) <= 1:
        return

    realized_dates = (
        df["realized_date"]
        .iloc[:-1]
        .to_numpy()
    )

    next_forecast_dates = (
        df["forecast_date"]
        .iloc[1:]
        .to_numpy()
    )

    if not np.array_equal(
        realized_dates,
        next_forecast_dates,
    ):

        bad_positions = np.where(
            realized_dates
            != next_forecast_dates
        )[0]

        raise RuntimeError(
            f"{ticker}: date contiguity failure at "
            f"positions {bad_positions[:10].tolist()}"
        )


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_input_panel(
    panel: pd.DataFrame,
) -> None:

    required = [
        "ticker",
        "forecast_date",
        "realized_date",
        "actual_return",

        "vix",
        "median_rv20",
        "vix_change_10d",
        "rv20_change_10d",
        "stress_dimensions",
        "regime",
    ]

    for model_name, cols in MODELS.items():

        required.extend(
            [
                cols["var"],
                cols["breach"],
                cols["loss"],
            ]
        )

    missing = [
        column
        for column in required
        if column not in panel.columns
    ]

    if missing:

        raise RuntimeError(
            "RS03 + RS01 panel is missing required columns:\n"
            + "\n".join(missing)
        )


# =============================================================================
# CROSS-MODEL FEATURES
# =============================================================================

def build_cross_model_features(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build model-disagreement features using ONLY current VaR forecasts.

    No realized returns.
    No breaches.
    No realized loss.
    """

    key_columns = [
        "ticker",
        "forecast_date",
    ]

    var_columns = [
        MODELS[m]["var"]
        for m in MODELS
    ]

    base = (
        panel[
            key_columns
            + var_columns
        ]
        .copy()
    )

    duplicate_keys = (
        base
        .duplicated(
            subset=key_columns
        )
        .sum()
    )

    if duplicate_keys:

        raise RuntimeError(
            "Duplicate ticker/date keys before "
            "cross-model feature construction."
        )

    # -------------------------------------------------------------------------
    # Absolute VaRs
    # -------------------------------------------------------------------------

    abs_var_columns = []

    for model_name in MODELS:

        abs_col = (
            f"{model_name}_abs_var_tmp"
        )

        base[abs_col] = (
            base[
                MODELS[model_name]["var"]
            ]
            .abs()
        )

        abs_var_columns.append(
            abs_col
        )


    # -------------------------------------------------------------------------
    # Cross-model median
    # -------------------------------------------------------------------------

    cross_median = (
        base[abs_var_columns]
        .median(
            axis=1
        )
    )


    cross_std = (
        base[abs_var_columns]
        .std(
            axis=1
        )
    )


    cross_max = (
        base[abs_var_columns]
        .max(
            axis=1
        )
    )


    cross_min = (
        base[abs_var_columns]
        .min(
            axis=1
        )
    )


    disagreement_cv = safe_divide(
        cross_std,
        cross_median,
    )


    disagreement_range_pct = safe_divide(
        cross_max - cross_min,
        cross_median,
    )


    # -------------------------------------------------------------------------
    # Cross-sectional model rank
    # -------------------------------------------------------------------------

    ranks = (
        base[abs_var_columns]
        .rank(
            axis=1,
            pct=True,
            method="average",
        )
    )


    # -------------------------------------------------------------------------
    # Create one row per ticker/date/model
    # -------------------------------------------------------------------------

    parts = []


    for model_name in MODELS:

        abs_col = (
            f"{model_name}_abs_var_tmp"
        )

        out = base[
            key_columns
        ].copy()

        out["model_name"] = (
            model_name
        )

        out[
            "cross_model_disagreement_cv"
        ] = disagreement_cv

        out[
            "cross_model_disagreement_range_pct"
        ] = disagreement_range_pct

        out[
            "model_abs_var_to_cross_median"
        ] = safe_divide(
            base[abs_col],
            cross_median,
        )

        out[
            "model_abs_var_rank_pct"
        ] = ranks[
            abs_col
        ]

        parts.append(
            out
        )


    result = pd.concat(
        parts,
        ignore_index=True,
    )

    return result


# =============================================================================
# COMMON MARKET FEATURES
# =============================================================================

def add_common_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        df
        .sort_values("forecast_date")
        .reset_index(drop=True)
        .copy()
    )


    # Existing RS03 causal regime variables remain untouched.
    #
    # These are:
    #   vix
    #   median_rv20
    #   vix_change_10d
    #   rv20_change_10d
    #   stress_dimensions
    #   regime


    df["vix_pct_change_10d"] = (
        safe_pct_change(
            df["vix"],
            periods=10,
        )
    )


    df["rv20_pct_change_10d"] = (
        safe_pct_change(
            df["median_rv20"],
            periods=10,
        )
    )


    # Current value compared against the PRIOR 252 observations only.

    df["vix_zscore_252d"] = (
        causal_zscore(
            df["vix"],
            window=252,
        )
    )


    df["rv20_zscore_252d"] = (
        causal_zscore(
            df["median_rv20"],
            window=252,
        )
    )


    return df


# =============================================================================
# CAUSAL HISTORICAL MODEL FEATURES
# =============================================================================

def add_causal_historical_features(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    cols = MODELS[
        model_name
    ]

    var_col = cols["var"]
    breach_col = cols["breach"]
    loss_col = cols["loss"]


    # =========================================================================
    # CURRENT VaR STATE
    # =========================================================================

    df["current_abs_var"] = (
        df[var_col]
        .abs()
    )


    # =========================================================================
    # VaR DYNAMICS
    # =========================================================================

    df["current_var_change_pct_1d"] = (
        safe_pct_change(
            df["current_abs_var"],
            periods=1,
        )
    )


    df["current_var_change_pct_5d"] = (
        safe_pct_change(
            df["current_abs_var"],
            periods=5,
        )
    )


    df["current_var_change_pct_20d"] = (
        safe_pct_change(
            df["current_abs_var"],
            periods=20,
        )
    )


    df["current_abs_var_zscore_252d"] = (
        causal_zscore(
            df["current_abs_var"],
            window=252,
        )
    )


    # =========================================================================
    # PAST BREACHES
    #
    # shift(1) ensures row t does not use breach[t].
    #
    # breach[t-1] is known at t because it corresponds to realized return t.
    # =========================================================================

    observed_breach = (
        df[breach_col]
        .shift(1)
    )


    for window in MONITOR_WINDOWS:

        df[
            f"past_{window}d_breach_rate"
        ] = (
            observed_breach
            .rolling(
                window,
                min_periods=window,
            )
            .mean()
        )


    # =========================================================================
    # NORMALIZED PINBALL LOSS
    #
    # raw loss / |VaR|
    #
    # This reduces the mechanical relationship between loss magnitude and
    # volatility scale.
    # =========================================================================

    normalized_loss = safe_divide(
        df[loss_col],
        df[var_col].abs(),
    )


    observed_normalized_loss = (
        normalized_loss
        .shift(1)
    )


    # =========================================================================
    # PAST NORMALIZED LOSS FEATURES
    # =========================================================================

    for window in MONITOR_WINDOWS:

        df[
            f"past_{window}d_mean_norm_loss"
        ] = (
            observed_normalized_loss
            .rolling(
                window,
                min_periods=window,
            )
            .mean()
        )


        df[
            f"past_{window}d_median_norm_loss"
        ] = (
            observed_normalized_loss
            .rolling(
                window,
                min_periods=window,
            )
            .median()
        )


        df[
            f"past_{window}d_max_norm_loss"
        ] = (
            observed_normalized_loss
            .rolling(
                window,
                min_periods=window,
            )
            .max()
        )


    # =========================================================================
    # SHORT-vs-LONG BREACH DIFFERENCE
    # =========================================================================

    df[
        "past_breach_diff_5d_20d"
    ] = (
        df["past_5d_breach_rate"]
        - df["past_20d_breach_rate"]
    )


    # =========================================================================
    # SHORT-vs-LONG NORMALIZED LOSS RATIO
    # =========================================================================

    df[
        "past_normloss_ratio_5d_20d"
    ] = safe_divide(
        df["past_5d_mean_norm_loss"],
        df["past_20d_mean_norm_loss"],
    )


    return df


# =============================================================================
# FUTURE TARGETS
# =============================================================================

def add_future_targets(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
) -> pd.DataFrame:

    cols = MODELS[
        model_name
    ]

    var_col = cols["var"]
    breach_col = cols["breach"]
    loss_col = cols["loss"]

    h = horizon


    # =========================================================================
    # SCALE-NORMALIZED FUTURE LOSS
    # =========================================================================

    normalized_loss = safe_divide(
        df[loss_col],
        df[var_col].abs(),
    )


    # =========================================================================
    # FUTURE H-DAY MEAN NORMALIZED LOSS
    #
    # At t:
    #
    #   future loss = loss[t] ... loss[t+h-1]
    #
    # These correspond to realized returns:
    #
    #   t+1 ... t+h
    # =========================================================================

    future_mean_norm_loss = (
        normalized_loss
        .rolling(
            h,
            min_periods=h,
        )
        .mean()
        .shift(
            -(h - 1)
        )
    )


    # =========================================================================
    # FUTURE H-DAY BREACH COUNT / RATE
    # =========================================================================

    future_breach_count = (
        df[breach_col]
        .rolling(
            h,
            min_periods=h,
        )
        .sum()
        .shift(
            -(h - 1)
        )
    )


    future_breach_rate = (
        future_breach_count
        / h
    )


    # =========================================================================
    # LABEL DATES
    # =========================================================================

    df[
        f"label_start_date_{h}d"
    ] = (
        df["realized_date"]
    )


    df[
        f"label_end_date_{h}d"
    ] = (
        df["realized_date"]
        .shift(
            -(h - 1)
        )
    )


    # =========================================================================
    # FUTURE TARGET VALUES
    # =========================================================================

    df[
        f"future_{h}d_mean_norm_loss"
    ] = (
        future_mean_norm_loss
    )


    df[
        f"future_{h}d_breach_rate"
    ] = (
        future_breach_rate
    )


    # =========================================================================
    # HISTORICALLY KNOWN H-DAY LOSS
    #
    # At t:
    #
    #   loss[t-1], ..., loss[t-h]
    #
    # are known.
    # =========================================================================

    historical_known_loss = (
        normalized_loss
        .shift(1)
        .rolling(
            h,
            min_periods=h,
        )
        .mean()
    )


    df[
        f"historical_{h}d_mean_norm_loss"
    ] = (
        historical_known_loss
    )


    # =========================================================================
    # CAUSAL EXPANDING THRESHOLD
    # =========================================================================

    causal_threshold = (
        historical_known_loss
        .expanding(
            min_periods=MIN_HISTORY,
        )
        .quantile(
            PRIMARY_QUANTILE
        )
    )


    df[
        f"causal_{h}d_loss_threshold"
    ] = (
        causal_threshold
    )


    # =========================================================================
    # PRIMARY DETERIORATION TARGET
    # =========================================================================

    valid_relative = (
        future_mean_norm_loss.notna()
        &
        causal_threshold.notna()
    )


    df[
        f"deterioration_{h}d"
    ] = np.where(
        valid_relative,
        (
            future_mean_norm_loss
            > causal_threshold
        ).astype(float),
        np.nan,
    )


    # =========================================================================
    # CONTINUOUS LOSS EXCESS
    # =========================================================================

    df[
        f"loss_excess_{h}d"
    ] = (
        future_mean_norm_loss
        - causal_threshold
    )


    # =========================================================================
    # ABSOLUTE COVERAGE FAILURE
    #
    # Under nominal alpha=0.05:
    #
    # H=5  -> 2 breaches
    # H=10 -> 3 breaches
    # H=20 -> 4 breaches
    #
    # using the one-sided 5% binomial critical threshold.
    # =========================================================================

    critical_breach_count = (
        int(
            binom.ppf(
                1.0 - COVERAGE_TEST_LEVEL,
                h,
                VAR_ALPHA,
            )
        )
        + 1
    )


    df[
        f"coverage_failure_{h}d"
    ] = np.where(
        future_breach_count.notna(),
        (
            future_breach_count
            >= critical_breach_count
        ).astype(float),
        np.nan,
    )


    df[
        f"coverage_failure_k_{h}d"
    ] = (
        critical_breach_count
    )


    return df


# =============================================================================
# PROCESS ONE MODEL
# =============================================================================

def process_model(
    panel: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    cols = MODELS[
        model_name
    ]

    parts = []


    for ticker, group in panel.groupby(
        "ticker",
        sort=True,
    ):

        g = (
            group
            .sort_values("forecast_date")
            .reset_index(drop=True)
            .copy()
        )


        # ---------------------------------------------------------------------
        # DATE CONTIGUITY
        # ---------------------------------------------------------------------

        assert_contiguous_dates(
            g,
            ticker,
        )


        # ---------------------------------------------------------------------
        # COMMON MARKET FEATURES
        # ---------------------------------------------------------------------

        g = add_common_market_features(
            g
        )


        # ---------------------------------------------------------------------
        # MODEL-SPECIFIC CAUSAL FEATURES
        # ---------------------------------------------------------------------

        g = add_causal_historical_features(
            g,
            model_name,
        )


        # ---------------------------------------------------------------------
        # FUTURE TARGETS
        # ---------------------------------------------------------------------

        for horizon in HORIZONS:

            g = add_future_targets(
                g,
                model_name,
                horizon,
            )


        # ---------------------------------------------------------------------
        # MODEL IDENTIFIER
        # ---------------------------------------------------------------------

        g["model_name"] = (
            model_name
        )


        # ---------------------------------------------------------------------
        # REMOVE RAW MODEL VaR
        #
        # The raw signed VaR itself is not retained.
        # current_abs_var already captures its scale.
        # ---------------------------------------------------------------------

        g = g.drop(
            columns=[
                cols["var"],
            ]
        )


        # ---------------------------------------------------------------------
        # REMOVE RAW NEXT-DAY OUTCOMES
        #
        # These remain in the original input and are saved separately as
        # quarantined outcomes in main().
        # ---------------------------------------------------------------------

        g = g.drop(
            columns=[
                cols["breach"],
                cols["loss"],
            ]
        )


        parts.append(
            g
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
                "forecast_date",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    return result


# =============================================================================
# CROSS-MODEL ATTACHMENT
# =============================================================================

def attach_cross_model_features(
    model_panel: pd.DataFrame,
    cross_features: pd.DataFrame,
) -> pd.DataFrame:

    keys = [
        "ticker",
        "forecast_date",
        "model_name",
    ]


    duplicate_cross_keys = (
        cross_features
        .duplicated(
            subset=keys
        )
        .sum()
    )


    if duplicate_cross_keys:

        raise RuntimeError(
            "Duplicate keys in cross-model features."
        )


    result = model_panel.merge(
        cross_features,
        on=keys,
        how="left",
        validate="one_to_one",
    )


    return result


# =============================================================================
# DECLARED FEATURES
# =============================================================================

MODEL_FEATURES = [

    "current_abs_var",

    "current_var_change_pct_1d",
    "current_var_change_pct_5d",
    "current_var_change_pct_20d",

    "current_abs_var_zscore_252d",

    "past_5d_breach_rate",
    "past_10d_breach_rate",
    "past_20d_breach_rate",

    "past_5d_mean_norm_loss",
    "past_10d_mean_norm_loss",
    "past_20d_mean_norm_loss",

    "past_5d_median_norm_loss",
    "past_10d_median_norm_loss",
    "past_20d_median_norm_loss",

    "past_5d_max_norm_loss",
    "past_10d_max_norm_loss",
    "past_20d_max_norm_loss",

    "past_breach_diff_5d_20d",

    "past_normloss_ratio_5d_20d",
]


HISTORICAL_FEATURES = [
    f"historical_{h}d_mean_norm_loss"
    for h in HORIZONS
]


CAUSAL_THRESHOLD_FEATURES = [
    f"causal_{h}d_loss_threshold"
    for h in HORIZONS
]


COMMON_NUMERIC_FEATURES = [

    "vix",
    "median_rv20",

    "vix_change_10d",
    "rv20_change_10d",

    "vix_pct_change_10d",
    "rv20_pct_change_10d",

    "vix_zscore_252d",
    "rv20_zscore_252d",

    "stress_dimensions",
]


CROSS_MODEL_FEATURES = [

    "cross_model_disagreement_cv",

    "cross_model_disagreement_range_pct",

    "model_abs_var_to_cross_median",

    "model_abs_var_rank_pct",
]


CATEGORICAL_FEATURES = [
    "regime",
    "model_name",
]


FEATURE_COLUMNS = (
    MODEL_FEATURES
    + HISTORICAL_FEATURES
    + CAUSAL_THRESHOLD_FEATURES
    + COMMON_NUMERIC_FEATURES
    + CROSS_MODEL_FEATURES
    + CATEGORICAL_FEATURES
)


# =============================================================================
# TARGET / LABEL / OUTCOME DEFINITIONS
# =============================================================================

TARGET_PREFIXES = (
    "deterioration_",
    "coverage_failure_",
    "future_",
    "loss_excess_",
)


LABEL_METADATA_PREFIXES = (
    "label_start_date_",
    "label_end_date_",
)


TARGET_METADATA_PREFIXES = (
    "coverage_failure_k_",
)


QUARANTINED_OUTCOME_COLUMNS = [
    "outcome_next_day_breach",
    "outcome_next_day_pinball_loss",
]


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def validate_feature_columns(
    panel: pd.DataFrame,
) -> None:

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in panel.columns
    ]


    if missing:

        raise RuntimeError(
            "Declared features missing from final panel:\n"
            + "\n".join(missing)
        )


# =============================================================================
# TARGET SUMMARY
# =============================================================================

def build_target_summary(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    for model_name in MODELS:

        model_data = (
            panel[
                panel["model_name"]
                == model_name
            ]
        )


        for horizon in HORIZONS:

            deterioration_col = (
                f"deterioration_{horizon}d"
            )

            coverage_col = (
                f"coverage_failure_{horizon}d"
            )

            future_loss_col = (
                f"future_{horizon}d_mean_norm_loss"
            )

            future_breach_col = (
                f"future_{horizon}d_breach_rate"
            )


            valid_det = (
                model_data[
                    deterioration_col
                ]
                .notna()
            )


            valid_cov = (
                model_data[
                    coverage_col
                ]
                .notna()
            )


            valid_loss = (
                model_data[
                    future_loss_col
                ]
                .notna()
            )


            valid_breach = (
                model_data[
                    future_breach_col
                ]
                .notna()
            )


            rows.append(
                {
                    "model":
                        model_name,

                    "horizon_days":
                        horizon,

                    "deterioration_valid":
                        int(valid_det.sum()),

                    "deterioration_positive":
                        int(
                            model_data.loc[
                                valid_det,
                                deterioration_col,
                            ].sum()
                        ),

                    "deterioration_positive_rate":
                        float(
                            model_data.loc[
                                valid_det,
                                deterioration_col,
                            ].mean()
                        )
                        if valid_det.any()
                        else np.nan,

                    "coverage_failure_valid":
                        int(valid_cov.sum()),

                    "coverage_failure_positive":
                        int(
                            model_data.loc[
                                valid_cov,
                                coverage_col,
                            ].sum()
                        ),

                    "coverage_failure_positive_rate":
                        float(
                            model_data.loc[
                                valid_cov,
                                coverage_col,
                            ].mean()
                        )
                        if valid_cov.any()
                        else np.nan,

                    "future_norm_loss_valid":
                        int(valid_loss.sum()),

                    "mean_future_norm_loss":
                        float(
                            model_data.loc[
                                valid_loss,
                                future_loss_col,
                            ].mean()
                        )
                        if valid_loss.any()
                        else np.nan,

                    "future_breach_rate_valid":
                        int(valid_breach.sum()),

                    "mean_future_breach_rate":
                        float(
                            model_data.loc[
                                valid_breach,
                                future_breach_col,
                            ].mean()
                        )
                        if valid_breach.any()
                        else np.nan,
                }
            )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# CROSS-MODEL LABEL CORRELATION
# =============================================================================

def build_label_correlation_diagnostic(
    panel: pd.DataFrame,
) -> pd.DataFrame:

    rows = []


    key_columns = [
        "ticker",
        "forecast_date",
    ]


    for horizon in HORIZONS:

        for target_type in [
            "deterioration",
            "coverage_failure",
        ]:

            target_col = (
                f"{target_type}_{horizon}d"
            )


            wide = (
                panel[
                    key_columns
                    + [
                        "model_name",
                        target_col,
                    ]
                ]
                .pivot(
                    index=key_columns,
                    columns="model_name",
                    values=target_col,
                )
            )


            model_names = list(
                MODELS.keys()
            )


            for i, model_a in enumerate(
                model_names
            ):

                for model_b in model_names[
                    i + 1:
                ]:

                    pair = (
                        wide[
                            [
                                model_a,
                                model_b,
                            ]
                        ]
                        .dropna()
                    )


                    if len(pair) >= 2:

                        correlation = (
                            pair[
                                model_a
                            ]
                            .corr(
                                pair[
                                    model_b
                                ]
                            )
                        )

                    else:

                        correlation = np.nan


                    rows.append(
                        {
                            "horizon_days":
                                horizon,

                            "target_type":
                                target_type,

                            "model_a":
                                model_a,

                            "model_b":
                                model_b,

                            "n_common":
                                len(pair),

                            "correlation":
                                correlation,
                        }
                    )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# TRUNCATION-INVARIANCE TEST
# =============================================================================

def build_full_processed_model(
    panel: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:

    result = process_model(
        panel,
        model_name,
    )


    cross_features = (
        build_cross_model_features(
            panel
        )
    )


    result = (
        attach_cross_model_features(
            result,
            cross_features,
        )
    )


    return result


def truncation_test(
    original_panel: pd.DataFrame,
    model_name: str,
    test_date: pd.Timestamp,
) -> dict:

    cols = MODELS[
        model_name
    ]


    # =========================================================================
    # FULL DATASET
    # =========================================================================

    full_model = (
        build_full_processed_model(
            original_panel,
            model_name,
        )
    )


    # =========================================================================
    # TRUNCATED DATASET
    #
    # Information available by T only.
    # The row-T realized outcome is deliberately removed because it is not
    # known at forecast date T.
    # =========================================================================

    truncated = (
        original_panel[
            original_panel["forecast_date"]
            <= test_date
        ]
        .copy()
    )


    row_mask = (
        truncated["forecast_date"]
        == test_date
    )


    if not row_mask.any():

        raise RuntimeError(
            f"{model_name}: truncation date "
            f"{test_date.date()} not found."
        )


    truncated.loc[
        row_mask,
        cols["breach"],
    ] = np.nan


    truncated.loc[
        row_mask,
        cols["loss"],
    ] = np.nan


    truncated_model = (
        build_full_processed_model(
            truncated,
            model_name,
        )
    )


    # =========================================================================
    # COMPARE CAUSAL FEATURES
    # =========================================================================

    key = [
        "ticker",
        "forecast_date",
    ]


    full_features = (
        full_model[
            full_model["forecast_date"]
            <= test_date
        ]
        .set_index(key)
        .sort_index()
    )


    truncated_features = (
        truncated_model
        .set_index(key)
        .sort_index()
    )


    available_feature_columns = [
        column
        for column in FEATURE_COLUMNS
        if column in full_features.columns
        and column in truncated_features.columns
    ]


    missing_test_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in available_feature_columns
    ]


    if missing_test_features:

        raise RuntimeError(
            f"{model_name}: truncation test missing features:\n"
            + "\n".join(
                missing_test_features
            )
        )


    full_compare = (
        full_features[
            available_feature_columns
        ]
    )


    truncated_compare = (
        truncated_features[
            available_feature_columns
        ]
    )


    common_index = (
        full_compare.index
        .intersection(
            truncated_compare.index
        )
    )


    full_compare = (
        full_compare
        .loc[common_index]
    )


    truncated_compare = (
        truncated_compare
        .loc[common_index]
    )


    # =========================================================================
    # NUMERIC FEATURES
    # =========================================================================

    numeric_columns = [
        column
        for column in available_feature_columns
        if pd.api.types.is_numeric_dtype(
            full_compare[column]
        )
    ]


    categorical_columns = [
        column
        for column in available_feature_columns
        if column not in numeric_columns
    ]


    numeric_pass = True


    for column in numeric_columns:

        a = (
            full_compare[column]
            .to_numpy(
                dtype=float
            )
        )


        b = (
            truncated_compare[column]
            .to_numpy(
                dtype=float
            )
        )


        if not np.allclose(
            a,
            b,
            equal_nan=True,
            rtol=1e-12,
            atol=0.0,
        ):

            numeric_pass = False

            break


    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================

    categorical_pass = True


    for column in categorical_columns:

        a = (
            full_compare[column]
            .astype(str)
            .to_numpy()
        )


        b = (
            truncated_compare[column]
            .astype(str)
            .to_numpy()
        )


        if not np.array_equal(
            a,
            b,
        ):

            categorical_pass = False

            break


    passed = (
        numeric_pass
        and categorical_pass
    )


    return {
        "model": model_name,
        "test_date": test_date,
        "n_rows_compared": len(common_index),
        "n_features_compared": len(available_feature_columns),
        "numeric_pass": numeric_pass,
        "categorical_pass": categorical_pass,
        "passed": passed,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    log.info(
        "=" * 100
    )

    log.info(
        "RS04 — REVISED CAUSAL RELIABILITY PANEL"
    )

    log.info(
        "=" * 100
    )


    log.info("")

    log.info(
        "Primary target: future normalized VaR loss deterioration"
    )

    log.info(
        "Secondary target: absolute coverage failure"
    )

    log.info(
        f"MIN_HISTORY = {MIN_HISTORY}"
    )

    log.info(
        f"HORIZONS = {HORIZONS}"
    )

    log.info(
        f"MONITOR_WINDOWS = {MONITOR_WINDOWS}"
    )


    # =========================================================================
    # LOAD INPUT
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "LOADING RS03 + RS01 PANEL"
    )

    log.info(
        "=" * 100
    )


    if not INPUT_PANEL.exists():

        raise FileNotFoundError(
            f"Input panel not found:\n"
            f"{INPUT_PANEL}"
        )


    panel = pd.read_csv(
        INPUT_PANEL
    )


    panel["forecast_date"] = pd.to_datetime(
        panel["forecast_date"],
        errors="coerce",
    )


    panel["realized_date"] = pd.to_datetime(
        panel["realized_date"],
        errors="coerce",
    )


    validate_input_panel(
        panel
    )


    log.info(
        f"Rows: "
        f"{len(panel):,}"
    )

    log.info(
        f"Tickers: "
        f"{panel['ticker'].nunique()}"
    )

    log.info(
        f"Forecast date range: "
        f"{panel['forecast_date'].min().date()} "
        f"to "
        f"{panel['forecast_date'].max().date()}"
    )

    log.info(
        f"Realized date range: "
        f"{panel['realized_date'].min().date()} "
        f"to "
        f"{panel['realized_date'].max().date()}"
    )


    # =========================================================================
    # INPUT KEY CHECK
    # =========================================================================

    duplicate_keys = (
        panel
        .duplicated(
            subset=[
                "ticker",
                "forecast_date",
            ]
        )
        .sum()
    )


    if duplicate_keys:

        raise RuntimeError(
            f"Input panel contains "
            f"{duplicate_keys} duplicate ticker/date rows."
        )


    # =========================================================================
    # CONTIGUITY CHECK
    # =========================================================================

    for ticker, group in panel.groupby(
        "ticker",
        sort=True,
    ):

        g = (
            group
            .sort_values("forecast_date")
            .reset_index(drop=True)
        )


        assert_contiguous_dates(
            g,
            ticker,
        )


    log.info(
        "Input date contiguity: OK"
    )


    # =========================================================================
    # BACKUP OLD RS04
    # =========================================================================

    if OUTPUT_PANEL.exists():

        if not PREVIOUS_PANEL_BACKUP.exists():

            shutil.copy2(
                OUTPUT_PANEL,
                PREVIOUS_PANEL_BACKUP,
            )

            log.info(
                f"Previous RS04 panel backed up to:\n"
                f"{PREVIOUS_PANEL_BACKUP}"
            )


    # =========================================================================
    # BUILD CROSS-MODEL FEATURES
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "BUILDING CROSS-MODEL CAUSAL FEATURES"
    )

    log.info(
        "=" * 100
    )


    cross_features = (
        build_cross_model_features(
            panel
        )
    )


    expected_cross_rows = (
        len(panel)
        * len(MODELS)
    )


    log.info(
        f"Cross-model rows: "
        f"{len(cross_features):,}"
    )


    if len(cross_features) != expected_cross_rows:

        raise RuntimeError(
            "Unexpected cross-model feature row count."
        )


    # =========================================================================
    # PROCESS EACH MODEL
    # =========================================================================

    processed_models = []


    for model_name in MODELS:

        log.info("")

        log.info(
            "=" * 100
        )

        log.info(
            f"PROCESSING MODEL — {model_name}"
        )

        log.info(
            "=" * 100
        )


        model_panel = process_model(
            panel,
            model_name,
        )


        model_panel = (
            attach_cross_model_features(
                model_panel,
                cross_features,
            )
        )


        log.info(
            f"{model_name}: "
            f"{len(model_panel):,} rows"
        )


        processed_models.append(
            model_panel
        )


    # =========================================================================
    # COMBINE
    # =========================================================================

    reliability_panel = pd.concat(
        processed_models,
        ignore_index=True,
    )


    reliability_panel = (
        reliability_panel
        .sort_values(
            [
                "ticker",
                "forecast_date",
                "model_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # =========================================================================
    # PANEL SIZE
    # =========================================================================

    expected_rows = (
        len(panel)
        * len(MODELS)
    )


    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "FINAL RS04 PANEL"
    )

    log.info(
        "=" * 100
    )


    log.info(
        f"Rows: "
        f"{len(reliability_panel):,}"
    )


    log.info(
        f"Expected rows: "
        f"{expected_rows:,}"
    )


    if len(reliability_panel) != expected_rows:

        raise RuntimeError(
            "Unexpected RS04 row count."
        )


    log.info(
        f"Tickers: "
        f"{reliability_panel['ticker'].nunique()}"
    )


    log.info(
        f"Models: "
        f"{reliability_panel['model_name'].nunique()}"
    )


    # =========================================================================
    # MODEL KEY CHECK
    # =========================================================================

    duplicate_model_keys = (
        reliability_panel
        .duplicated(
            subset=[
                "ticker",
                "forecast_date",
                "model_name",
            ]
        )
        .sum()
    )


    log.info(
        f"Duplicate ticker/date/model rows: "
        f"{duplicate_model_keys}"
    )


    if duplicate_model_keys:

        raise RuntimeError(
            "Duplicate model-date keys."
        )


    # =========================================================================
    # IDENTIFY TARGET COLUMNS
    # =========================================================================

    target_columns = [
        column
        for column in reliability_panel.columns
        if column.startswith(
            TARGET_PREFIXES
        )
    ]


    label_metadata_columns = [
        column
        for column in reliability_panel.columns
        if column.startswith(
            LABEL_METADATA_PREFIXES
        )
    ]


    target_metadata_columns = [
        column
        for column in reliability_panel.columns
        if column.startswith(
            TARGET_METADATA_PREFIXES
        )
    ]


    log.info(
        f"Declared numeric/categorical features: "
        f"{len(FEATURE_COLUMNS)}"
    )


    log.info(
        f"Target columns: "
        f"{len(target_columns)}"
    )


    log.info(
        f"Label metadata columns: "
        f"{len(label_metadata_columns)}"
    )


    log.info(
        f"Target metadata columns: "
        f"{len(target_metadata_columns)}"
    )


    # =========================================================================
    # FEATURE CHECK
    # =========================================================================

    validate_feature_columns(
        reliability_panel
    )


    # =========================================================================
    # CRITICAL LEAKAGE CHECK
    #
    # Raw next-day outcomes must NOT be in the main RS04 panel.
    # =========================================================================

    forbidden_columns = [
        column
        for column in QUARANTINED_OUTCOME_COLUMNS
        if column in reliability_panel.columns
    ]


    if forbidden_columns:

        raise RuntimeError(
            "Raw outcome columns were not properly quarantined:\n"
            + "\n".join(
                forbidden_columns
            )
        )


    log.info(
        "Raw next-day outcomes absent from main RS04 panel: OK"
    )


    # =========================================================================
    # ADDITIONAL LEAKAGE CHECK
    #
    # actual_return must not be a feature column in RS04.
    # =========================================================================

    if "actual_return" in FEATURE_COLUMNS:

        raise RuntimeError(
            "actual_return is incorrectly listed as a feature."
        )


    if "realized_date" in FEATURE_COLUMNS:

        raise RuntimeError(
            "realized_date is incorrectly listed as a predictor feature."
        )


    for column in [
        "future_5d_mean_norm_loss",
        "future_10d_mean_norm_loss",
        "future_20d_mean_norm_loss",
    ]:

        if column in FEATURE_COLUMNS:

            raise RuntimeError(
                f"Future target {column} is incorrectly listed "
                "as a predictor feature."
            )


    # =========================================================================
    # QUARANTINE RAW OUTCOMES
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "QUARANTINING RAW NEXT-DAY OUTCOMES"
    )

    log.info(
        "=" * 100
    )


    outcome_parts = []


    for model_name in MODELS:

        cols = MODELS[
            model_name
        ]


        out = (
            panel[
                [
                    "ticker",
                    "forecast_date",
                    "realized_date",
                    cols["breach"],
                    cols["loss"],
                ]
            ]
            .copy()
        )


        out["model_name"] = (
            model_name
        )


        out = out.rename(
            columns={
                cols["breach"]:
                    "outcome_next_day_breach",

                cols["loss"]:
                    "outcome_next_day_pinball_loss",
            }
        )


        outcome_parts.append(
            out
        )


    quarantined_outcomes = pd.concat(
        outcome_parts,
        ignore_index=True,
    )


    quarantined_outcomes = (
        quarantined_outcomes
        .sort_values(
            [
                "ticker",
                "forecast_date",
                "model_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    expected_outcome_rows = (
        len(panel)
        * len(MODELS)
    )


    if len(quarantined_outcomes) != expected_outcome_rows:

        raise RuntimeError(
            "Unexpected quarantined outcome row count."
        )


    log.info(
        f"Quarantined outcome rows: "
        f"{len(quarantined_outcomes):,}"
    )


    # =========================================================================
    # TARGET SUMMARY
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "TARGET SUMMARY"
    )

    log.info(
        "=" * 100
    )


    target_summary = (
        build_target_summary(
            reliability_panel
        )
    )


    log.info(
        target_summary.to_string(
            index=False
        )
    )


    # =========================================================================
    # LABEL CORRELATION
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "CROSS-MODEL LABEL CORRELATION DIAGNOSTIC"
    )

    log.info(
        "=" * 100
    )


    label_correlation = (
        build_label_correlation_diagnostic(
            reliability_panel
        )
    )


    log.info(
        label_correlation.to_string(
            index=False
        )
    )


    # =========================================================================
    # TRUNCATION TEST
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "TRUNCATION-INVARIANCE CAUSALITY TEST"
    )

    log.info(
        "=" * 100
    )


    truncation_results = []


    available_dates = set(
        panel[
            "forecast_date"
        ]
        .dropna()
    )


    for test_date in TRUNCATION_TEST_DATES:

        if test_date not in available_dates:

            raise RuntimeError(
                f"Truncation date "
                f"{test_date.date()} not found in input."
            )


        for model_name in MODELS:

            log.info(
                f"Testing {model_name} "
                f"at {test_date.date()}..."
            )


            result = (
                truncation_test(
                    panel,
                    model_name,
                    test_date,
                )
            )


            truncation_results.append(
                result
            )


            if not result["passed"]:

                raise RuntimeError(
                    f"TRUNCATION TEST FAILED: "
                    f"{model_name} "
                    f"{test_date.date()}"
                )


            log.info(
                f"{model_name} "
                f"{test_date.date()} "
                f"features identical under truncation: OK"
            )


    truncation_audit = pd.DataFrame(
        truncation_results
    )


    # =========================================================================
    # FEATURE MISSINGNESS
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "FEATURE MISSINGNESS CHECK"
    )

    log.info(
        "=" * 100
    )


    feature_missingness = (
        reliability_panel[
            FEATURE_COLUMNS
        ]
        .isna()
        .mean()
        .sort_values(
            ascending=False
        )
    )


    log.info(
        feature_missingness.to_string()
    )


    # =========================================================================
    # BUILD FINAL MAIN PANEL
    # =========================================================================

    main_panel_columns = [

        "ticker",
        "forecast_date",
        "realized_date",
        "model_name",

    ]


    main_panel_columns += (
        FEATURE_COLUMNS
    )


    main_panel_columns += (
        target_columns
    )


    main_panel_columns += (
        label_metadata_columns
    )


    main_panel_columns += (
        target_metadata_columns
    )


    # Preserve first occurrence only.
    main_panel_columns = list(
        dict.fromkeys(
            main_panel_columns
        )
    )


    main_panel = (
        reliability_panel[
            main_panel_columns
        ]
        .copy()
    )


    # =========================================================================
    # FINAL MAIN-PANEL LEAKAGE CHECK
    # =========================================================================

    final_forbidden = [
        "actual_return",
        "outcome_next_day_breach",
        "outcome_next_day_pinball_loss",
    ]


    accidentally_present = [
        column
        for column in final_forbidden
        if column in main_panel.columns
    ]


    if accidentally_present:

        raise RuntimeError(
            "Leakage columns found in final main panel:\n"
            + "\n".join(
                accidentally_present
            )
        )


    log.info(
        "Final main-panel leakage check: OK"
    )


    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    main_panel.to_csv(
        OUTPUT_PANEL,
        index=False,
    )


    quarantined_outcomes.to_csv(
        OUTPUT_OUTCOMES,
        index=False,
    )


    target_summary.to_csv(
        OUTPUT_TARGET_SUMMARY,
        index=False,
    )


    label_correlation.to_csv(
        OUTPUT_LABEL_CORRELATION,
        index=False,
    )


    truncation_audit.to_csv(
        OUTPUT_TRUNCATION_AUDIT,
        index=False,
    )


    # =========================================================================
    # SAVE EXPLICIT FEATURE / TARGET MANIFEST
    # =========================================================================

    manifest = {

        "project":
            "RS04 revised causal reliability panel",

        "purpose":
            (
                "Construct strictly causal monitoring features "
                "and future VaR deterioration targets."
            ),

        "input_panel":
            str(INPUT_PANEL),

        "output_panel":
            str(OUTPUT_PANEL),

        "quarantined_outcomes":
            str(OUTPUT_OUTCOMES),

        "models":
            list(MODELS.keys()),

        "tickers":
            TICKERS,

        "configuration": {

            "var_alpha":
                VAR_ALPHA,

            "primary_quantile":
                PRIMARY_QUANTILE,

            "min_history":
                MIN_HISTORY,

            "monitor_windows":
                MONITOR_WINDOWS,

            "horizons":
                HORIZONS,

            "coverage_test_level":
                COVERAGE_TEST_LEVEL,
        },

        "feature_columns":
            FEATURE_COLUMNS,

        "model_features":
            MODEL_FEATURES,

        "historical_features":
            HISTORICAL_FEATURES,

        "causal_threshold_features":
            CAUSAL_THRESHOLD_FEATURES,

        "common_market_features":
            COMMON_NUMERIC_FEATURES,

        "cross_model_features":
            CROSS_MODEL_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "metadata_columns": [
            "ticker",
            "forecast_date",
            "realized_date",
            "model_name",
        ],

        "target_columns":
            target_columns,

        "label_metadata_columns":
            label_metadata_columns,

        "target_metadata_columns":
            target_metadata_columns,

        "quarantined_outcome_columns":
            QUARANTINED_OUTCOME_COLUMNS,

        "target_definitions": {

            "normalized_loss":
                (
                    "pinball_loss / abs(VaR)"
                ),

            "deterioration_H":
                (
                    "Future H-day mean normalized loss "
                    "> causal expanding 75th percentile "
                    "of historically known H-day normalized-loss windows."
                ),

            "coverage_failure_H":
                (
                    "At least k breaches in the next H observations, "
                    "where k is the one-sided 5% binomial critical count "
                    "under nominal alpha=0.05."
                ),

            "label_start_date_H":
                (
                    "First realized date in the H-day future outcome window."
                ),

            "label_end_date_H":
                (
                    "Last realized date in the H-day future outcome window."
                ),
        },

        "causality_policy": {

            "past_breach_features":
                "Use breach.shift(1).",

            "past_loss_features":
                "Use normalized_loss.shift(1).",

            "historical_loss":
                "Use normalized_loss.shift(1).rolling(H).",

            "causal_threshold":
                "Expanding quantile of historically known H-day loss windows.",

            "cross_model_features":
                "Use current VaR forecasts only.",

            "raw_realized_outcomes":
                "Quarantined outside the main feature panel.",
        },

        "truncation_test_dates": [
            str(date.date())
            for date in TRUNCATION_TEST_DATES
        ],

        "leakage_exclusions": [
            "actual_return",
            "outcome_next_day_breach",
            "outcome_next_day_pinball_loss",
            "raw signed VaR columns",
        ],
    }


    with open(
        OUTPUT_MANIFEST,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            default=str,
        )


    # =========================================================================
    # RELOAD SAVED PANEL
    # =========================================================================

    saved_panel = pd.read_csv(
        OUTPUT_PANEL
    )


    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "FINAL SANITY CHECKS"
    )

    log.info(
        "=" * 100
    )


    # =========================================================================
    # SAVED ROW COUNT
    # =========================================================================

    log.info(
        f"Saved main panel rows: "
        f"{len(saved_panel):,}"
    )


    if len(saved_panel) != expected_rows:

        raise RuntimeError(
            "Saved panel row count mismatch."
        )


    # =========================================================================
    # SAVED DUPLICATE CHECK
    # =========================================================================

    saved_duplicates = (
        saved_panel
        .duplicated(
            subset=[
                "ticker",
                "forecast_date",
                "model_name",
            ]
        )
        .sum()
    )


    log.info(
        f"Saved duplicate model keys: "
        f"{saved_duplicates}"
    )


    if saved_duplicates:

        raise RuntimeError(
            "Saved RS04 panel contains duplicate keys."
        )


    # =========================================================================
    # SAVED FEATURE CHECK
    # =========================================================================

    missing_saved_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in saved_panel.columns
    ]


    if missing_saved_features:

        raise RuntimeError(
            "Saved RS04 panel lost declared features:\n"
            + "\n".join(
                missing_saved_features
            )
        )


    # =========================================================================
    # SAVED LEAKAGE CHECK
    # =========================================================================

    saved_leaks = [
        column
        for column in final_forbidden
        if column in saved_panel.columns
    ]


    if saved_leaks:

        raise RuntimeError(
            "Saved RS04 panel contains leakage columns:\n"
            + "\n".join(
                saved_leaks
            )
        )


    log.info(
        "Saved feature/leakage validation: OK"
    )


    # =========================================================================
    # FINAL MODEL COVERAGE
    # =========================================================================

    log.info("")

    log.info(
        "MODEL ROW COVERAGE:"
    )


    for model_name in MODELS:

        model_rows = (
            saved_panel[
                saved_panel["model_name"]
                == model_name
            ]
            .shape[0]
        )


        log.info(
            f"{model_name:8s}: "
            f"{model_rows:,}"
        )


        if model_rows != len(panel):

            raise RuntimeError(
                f"{model_name} does not have "
                f"{len(panel):,} rows."
            )


    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "OUTPUTS SAVED"
    )

    log.info(
        "=" * 100
    )


    log.info(
        f"Main causal panel:\n"
        f"{OUTPUT_PANEL}"
    )


    log.info(
        f"Quarantined outcomes:\n"
        f"{OUTPUT_OUTCOMES}"
    )


    log.info(
        f"Target summary:\n"
        f"{OUTPUT_TARGET_SUMMARY}"
    )


    log.info(
        f"Label correlation:\n"
        f"{OUTPUT_LABEL_CORRELATION}"
    )


    log.info(
        f"Truncation audit:\n"
        f"{OUTPUT_TRUNCATION_AUDIT}"
    )


    log.info(
        f"Column manifest:\n"
        f"{OUTPUT_MANIFEST}"
    )


    log.info(
        f"Log:\n"
        f"{OUTPUT_LOG}"
    )


    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        "RS04 REVISED CAUSAL RELIABILITY PANEL COMPLETE"
    )

    log.info(
        "=" * 100
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()