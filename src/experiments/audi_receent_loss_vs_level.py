from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# RECENT LOSS VS LONG-RUN LOSS LEVEL AUDIT
# =============================================================================
#
# Purpose:
#
# Determine whether the strong "past_mean_pinball" result is driven by:
#
#   1. genuinely recent forecast-loss information,
#   2. a slow long-run loss level,
#   3. or a property of the relative deterioration target.
#
#
# Signals:
#
#   pinball_lag1
#   pinball_mean_5d_prior
#   pinball_mean_20d_prior
#   pinball_mean_60d_prior
#   past_mean_pinball
#   mean20 / expanding_mean
#   mean60 / expanding_mean
#   breach_rate_20d_prior
#   vol_ratio_20_60
#
#
# Targets:
#
# A) Existing primary target:
#
#    future_loss > causal 75th percentile
#
# B) Normalized deterioration:
#
#    future_loss / prior-60d-loss-level
#       >
#    causal 75th percentile of that ratio
#
#
# Direction is selected using VALIDATION ONLY.
#
# Evaluation:
#
#   - full test sample pooled AUC
#   - non-overlapping test sample pooled AUC
#   - continuous Spearman correlation with future loss
#
# We do NOT use XGBoost here.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data" / "monitoring"
)

PANEL_PATH = (
    DATA_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_PATH = (
    DATA_DIR
    / "audit_recent_loss_vs_level.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20

TARGET_QUANTILE = 0.75

MIN_TARGET_OBS = 75

HORIZONS = [5, 10, 20]

MODELS = [
    "qrf",
    "ewma",
    "garch",
    "hs",
]

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


# =============================================================================
# HELPERS
# =============================================================================

def safe_auc(y, score):

    y = np.asarray(y)
    score = np.asarray(
        score,
        dtype=float,
    )

    mask = (
        pd.notna(y)
        &
        np.isfinite(score)
    )

    y = y[mask]
    score = score[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def add_splits(df):

    df = (
        df
        .sort_values(
            [
                "ticker",
                "forecast_date",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    labels = []

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):

        n = len(group)

        train_end = int(
            np.floor(
                TRAIN_FRAC * n
            )
        )

        val_end = int(
            np.floor(
                (
                    TRAIN_FRAC
                    +
                    VALIDATION_FRAC
                ) * n
            )
        )

        x = np.array(
            ["test"] * n,
            dtype=object,
        )

        x[:train_end] = "train"

        x[
            train_end:val_end
        ] = "validation"

        labels.extend(
            x.tolist()
        )

    df["split"] = labels

    return df


def choose_direction(
    df,
    target_col,
    signal_col,
):

    valid = df[
        df[target_col].notna()
        &
        df[signal_col].notna()
    ].copy()

    if len(valid) == 0:
        return np.nan, np.nan

    positive = safe_auc(
        valid[target_col],
        valid[signal_col],
    )

    negative = safe_auc(
        valid[target_col],
        -valid[signal_col],
    )

    if not np.isfinite(positive):
        return -1.0, negative

    if not np.isfinite(negative):
        return 1.0, positive

    if positive >= negative:

        return 1.0, positive

    return -1.0, negative


def nonoverlap_mask(
    df,
    horizon,
):

    mask = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    for (
        ticker,
        split,
    ), group in df.groupby(
        [
            "ticker",
            "split",
        ],
        sort=False,
    ):

        indices = group.index.to_numpy()

        selected = indices[
            ::horizon
        ]

        mask.loc[
            selected
        ] = True

    return mask


def make_future_mean(
    series,
    horizon,
):

    x = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    out = np.full(
        len(x),
        np.nan,
        dtype=float,
    )

    for i in range(
        len(x) - horizon
    ):

        window = x[
            i + 1:
            i + horizon + 1
        ]

        if np.all(
            np.isfinite(window)
        ):

            out[i] = (
                np.mean(window)
            )

    return pd.Series(
        out,
        index=series.index,
    )


def causal_quantile(
    target_series,
    horizon,
    q=TARGET_QUANTILE,
):

    historical_targets = (
        target_series
        .shift(horizon + 1)
    )

    return (
        historical_targets
        .expanding(
            min_periods=MIN_TARGET_OBS
        )
        .quantile(q)
    )


def add_normalized_targets(
    df,
    model,
    horizon,
):

    loss_col = (
        f"{model}_pinball_loss"
    )

    baseline_col = (
        f"{model}_pinball_mean_60d_prior"
    )

    target_ratio_col = (
        f"{model}_future_loss_ratio_{horizon}d"
    )

    threshold_col = (
        f"{model}_causal_ratio_threshold_{horizon}d"
    )

    target_col = (
        f"{model}_normalized_deterioration_{horizon}d"
    )

    future_loss = make_future_mean(
        df[loss_col],
        horizon,
    )

    baseline = pd.to_numeric(
        df[baseline_col],
        errors="coerce",
    )

    ratio = (
        future_loss
        /
        baseline
    )

    threshold = causal_quantile(
        ratio,
        horizon,
        TARGET_QUANTILE,
    )

    target = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    valid = (
        ratio.notna()
        &
        threshold.notna()
    )

    target.loc[
        valid
    ] = (
        ratio.loc[valid]
        >
        threshold.loc[valid]
    ).astype(float)

    df[target_ratio_col] = ratio

    df[threshold_col] = threshold

    df[target_col] = target

    return df


# =============================================================================
# LOAD
# =============================================================================

print(
    "=" * 100
)

print(
    "RECENT LOSS VS LONG-RUN LOSS LEVEL AUDIT"
)

print(
    "=" * 100
)

if not PANEL_PATH.exists():

    raise FileNotFoundError(
        PANEL_PATH
    )

df = pd.read_csv(
    PANEL_PATH
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"],
    errors="coerce",
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df[
    df["ticker"].isin(
        TICKERS
    )
].copy()

df = add_splits(
    df
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Tickers: {df['ticker'].nunique()}"
)


# =============================================================================
# BUILD TARGET B
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "BUILDING NORMALIZED FUTURE-LOSS TARGETS"
)

print(
    "=" * 100
)

for model in MODELS:

    print(
        f"  {model.upper()}"
    )

    for horizon in HORIZONS:

        df = add_normalized_targets(
            df,
            model,
            horizon,
        )


# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

summary_rows = []


for model in MODELS:

    signal_prefix = model

    signals = {

        "lag1":
            f"{signal_prefix}_pinball_lag1",

        "mean5":
            f"{signal_prefix}_pinball_mean_5d_prior",

        "mean20":
            f"{signal_prefix}_pinball_mean_20d_prior",

        "mean60":
            f"{signal_prefix}_pinball_mean_60d_prior",

        "expanding_mean":
            f"{signal_prefix}_past_mean_pinball",

        "mean20_to_expanding":
            None,

        "mean60_to_expanding":
            None,

        "breach_rate20":
            f"{signal_prefix}_breach_rate_20d_prior",

        "vol_ratio":
            f"{signal_prefix}_vol_ratio_20_60",
    }

    # -------------------------------------------------------------------------
    # Relative recent-loss signals.
    # -------------------------------------------------------------------------

    df[
        f"{model}_mean20_to_expanding"
    ] = (
        df[
            f"{model}_pinball_mean_20d_prior"
        ]
        /
        df[
            f"{model}_past_mean_pinball"
        ]
    )

    df[
        f"{model}_mean60_to_expanding"
    ] = (
        df[
            f"{model}_pinball_mean_60d_prior"
        ]
        /
        df[
            f"{model}_past_mean_pinball"
        ]
    )

    signals[
        "mean20_to_expanding"
    ] = (
        f"{model}_mean20_to_expanding"
    )

    signals[
        "mean60_to_expanding"
    ] = (
        f"{model}_mean60_to_expanding"
    )

    print(
        "\n" + "-" * 100
    )

    print(
        f"MODEL: {model.upper()}"
    )

    print(
        "-" * 100
    )

    for horizon in HORIZONS:

        primary_target = (
            f"{model}_deterioration_{horizon}d"
        )

        normalized_target = (
            f"{model}_normalized_deterioration_{horizon}d"
        )

        future_loss = (
            f"{model}_future_pinball_{horizon}d"
        )

        # =====================================================================
        # VALIDATION / TEST
        # =====================================================================

        validation = df[
            df["split"] == "validation"
        ].copy()

        test = df[
            df["split"] == "test"
        ].copy()

        validation_nonoverlap = (
            validation[
                nonoverlap_mask(
                    validation,
                    horizon,
                )
            ]
            .copy()
        )

        test_nonoverlap = (
            test[
                nonoverlap_mask(
                    test,
                    horizon,
                )
            ]
            .copy()
        )

        print(
            f"\n{horizon}D"
        )

        # =====================================================================
        # SIGNAL LOOP
        # =====================================================================

        for signal_name, signal_col in signals.items():

            # ---------------------------------------------------------------
            # Direction using VALIDATION ONLY.
            # Use the ORIGINAL primary target for selection.
            # ---------------------------------------------------------------

            direction, validation_auc = (
                choose_direction(
                    validation_nonoverlap,
                    primary_target,
                    signal_col,
                )
            )

            if not np.isfinite(
                direction
            ):
                continue

            # ---------------------------------------------------------------
            # FULL TEST SAMPLE
            # ---------------------------------------------------------------

            full_test = test[
                test[primary_target].notna()
                &
                test[signal_col].notna()
            ].copy()

            full_score = (
                direction
                *
                full_test[
                    signal_col
                ]
            )

            full_primary_auc = safe_auc(
                full_test[
                    primary_target
                ],
                full_score,
            )

            full_normalized_auc = safe_auc(
                full_test[
                    normalized_target
                ],
                full_score,
            )

            # ---------------------------------------------------------------
            # NON-OVERLAP TEST
            # ---------------------------------------------------------------

            no_test = test_nonoverlap[
                test_nonoverlap[
                    primary_target
                ].notna()
                &
                test_nonoverlap[
                    signal_col
                ].notna()
            ].copy()

            no_score = (
                direction
                *
                no_test[
                    signal_col
                ]
            )

            no_primary_auc = safe_auc(
                no_test[
                    primary_target
                ],
                no_score,
            )

            no_normalized_auc = safe_auc(
                no_test[
                    normalized_target
                ],
                no_score,
            )

            # ---------------------------------------------------------------
            # CONTINUOUS FUTURE LOSS
            # ---------------------------------------------------------------

            continuous_valid = (
                full_test[
                    future_loss
                ].notna()
                &
                no_score.notna()
            )

            if (
                continuous_valid.sum()
                > 10
            ):

                future_values = (
                    full_test[
                        future_loss
                    ]
                )

                score_values = (
                    full_score
                )

                continuous_spearman = (
                    future_values.corr(
                        score_values,
                        method="spearman",
                    )
                )

            else:

                continuous_spearman = np.nan

            # ---------------------------------------------------------------
            # Target prevalence
            # ---------------------------------------------------------------

            primary_rate = (
                no_test[
                    primary_target
                ].mean()
                if len(no_test)
                else np.nan
            )

            normalized_rate = (
                no_test[
                    normalized_target
                ].mean()
                if len(no_test)
                else np.nan
            )

            summary_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "signal":
                        signal_name,

                    "signal_column":
                        signal_col,

                    "direction":
                        direction,

                    "validation_auc_primary":
                        validation_auc,

                    "full_test_rows":
                        len(full_test),

                    "nonoverlap_test_rows":
                        len(no_test),

                    "full_test_auc_primary":
                        full_primary_auc,

                    "nonoverlap_test_auc_primary":
                        no_primary_auc,

                    "full_test_auc_normalized":
                        full_normalized_auc,

                    "nonoverlap_test_auc_normalized":
                        no_normalized_auc,

                    "future_loss_spearman":
                        continuous_spearman,

                    "primary_target_rate":
                        primary_rate,

                    "normalized_target_rate":
                        normalized_rate,
                }
            )

            print(
                f"  {signal_name:24s}"
                f" | val={validation_auc:.4f}"
                f" | full={full_primary_auc:.4f}"
                f" | nonoverlap={no_primary_auc:.4f}"
                f" | normalized={no_normalized_auc:.4f}"
            )


# =============================================================================
# SAVE
# =============================================================================

results = pd.DataFrame(
    summary_rows
)

results.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =============================================================================
# PRINT KEY TABLE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "KEY RESULTS: NON-OVERLAP TEST"
)

print(
    "=" * 100
)

key = results[
    [
        "model",
        "horizon",
        "signal",
        "validation_auc_primary",
        "nonoverlap_test_auc_primary",
        "nonoverlap_test_auc_normalized",
        "future_loss_spearman",
    ]
].copy()

print(
    key.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# BEST RECENT SIGNALS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "BEST RECENT / LEVEL SIGNALS BY MODEL AND HORIZON"
)

print(
    "=" * 100
)

for model in MODELS:

    for horizon in HORIZONS:

        block = results[
            (
                results["model"]
                ==
                model.upper()
            )
            &
            (
                results["horizon"]
                ==
                horizon
            )
        ].copy()

        block = (
            block
            .sort_values(
                "nonoverlap_test_auc_normalized",
                ascending=False,
            )
        )

        if len(block) == 0:
            continue

        best = block.iloc[0]

        print(
            f"{model.upper():5s} {horizon:2d}d | "
            f"best normalized signal="
            f"{best['signal']}"
            f" | AUC="
            f"{best['nonoverlap_test_auc_normalized']:.4f}"
        )


# =============================================================================
# SAVE PATH
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 100
)

print(
    f"\nSaved:"
)

print(
    OUTPUT_PATH
)