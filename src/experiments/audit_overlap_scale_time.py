from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# AUDIT: OVERLAP, TICKER SCALE, AND TIME TREND
# =============================================================================
#
# Research question being audited:
#
# Why is past_mean_pinball producing surprisingly strong 10d/20d AUCs?
#
# We test:
#
# 1. OVERLAP
#    Evaluate only non-overlapping future-loss windows.
#
#       h = 5   -> every 5th observation
#       h = 10  -> every 10th observation
#       h = 20  -> every 20th observation
#
# 2. TICKER SCALE
#    Compare raw past_mean_pinball with a causal within-ticker standardized
#    version.
#
# 3. TIME / REGIME TREND
#    Compare:
#
#       a) pure time trend
#       b) raw past_mean_pinball
#       c) past_mean_pinball with a training-only quadratic time trend removed
#
# ALL ORIENTATIONS ARE CHOSEN USING VALIDATION ONLY.
#
# MAIN EVALUATION IS ON THE TEST PERIOD.
#
# NO TEST-BASED SIGNAL SELECTION.
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

SUMMARY_PATH = (
    DATA_DIR
    / "audit_overlap_scale_time_summary.csv"
)

TICKER_PATH = (
    DATA_DIR
    / "audit_overlap_scale_time_per_ticker.csv"
)

NONOVERLAP_PATH = (
    DATA_DIR
    / "audit_nonoverlap_test_sample.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20

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

def safe_auc(
    y,
    score,
):
    """
    Safe ROC AUC.
    """

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


def choose_direction_validation(
    df,
    target_col,
    signal_col,
):
    """
    Choose orientation using validation only.
    """

    valid = df[
        df[target_col].notna()
        &
        df[signal_col].notna()
    ].copy()

    auc_positive = safe_auc(
        valid[target_col],
        valid[signal_col],
    )

    auc_negative = safe_auc(
        valid[target_col],
        -valid[signal_col],
    )

    if (
        not np.isfinite(auc_positive)
        and
        not np.isfinite(auc_negative)
    ):
        return (
            np.nan,
            np.nan,
        )

    if (
        np.isfinite(auc_positive)
        and
        (
            not np.isfinite(auc_negative)
            or
            auc_positive >= auc_negative
        )
    ):
        return (
            1.0,
            auc_positive,
        )

    return (
        -1.0,
        auc_negative,
    )


def add_splits(
    df,
):
    """
    Exact 60/20/20 chronological split per ticker.
    """

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

    split_values = []

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

        validation_end = int(
            np.floor(
                (
                    TRAIN_FRAC
                    +
                    VALIDATION_FRAC
                )
                * n
            )
        )

        labels = np.array(
            ["test"] * n,
            dtype=object,
        )

        labels[
            :train_end
        ] = "train"

        labels[
            train_end:
            validation_end
        ] = "validation"

        split_values.extend(
            labels.tolist()
        )

    df["split"] = split_values

    return df


def make_nonoverlap_mask(
    df,
    horizon,
):
    """
    Keep non-overlapping forecast origins within each
    ticker/split group.

    Example:
        horizon = 5
        positions 0, 5, 10, 15, ...

    The function returns a boolean Series aligned to the
    ORIGINAL dataframe index.
    """

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

        # These are the ORIGINAL dataframe indices.
        indices = group.index.to_numpy()

        # Select every h-th observation.
        selected = indices[
            ::horizon
        ]

        # Assign using the original index labels.
        mask.loc[
            selected
        ] = True

    return mask


def causal_zscore(
    df,
    signal_col,
):
    """
    Causal within-ticker standardization.

    At t:

        z_t =
            (signal_t - mean(signal_1,...,signal_{t-1}))
            /
            std(signal_1,...,signal_{t-1})

    Current signal is NOT included in its own reference distribution.
    """

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):

        x = pd.to_numeric(
            group[signal_col],
            errors="coerce",
        )

        prior = (
            x.shift(1)
        )

        expanding_mean = (
            prior
            .expanding(
                min_periods=40
            )
            .mean()
        )

        expanding_std = (
            prior
            .expanding(
                min_periods=40
            )
            .std()
        )

        z = (
            x
            -
            expanding_mean
        ) / expanding_std

        result.loc[
            group.index
        ] = z

    return result


def add_time_index(
    df,
):
    """
    Within-ticker normalized time index.

        0 at first observation
        1 at last observation
    """

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):

        n = len(group)

        if n <= 1:

            values = np.zeros(
                n,
                dtype=float,
            )

        else:

            values = np.linspace(
                0.0,
                1.0,
                n,
            )

        result.loc[
            group.index
        ] = values

    return result


def remove_training_time_trend(
    df,
    signal_col,
):
    """
    Remove a quadratic time trend using TRAINING observations only.

    For each ticker:

        signal_t =
            beta0
            + beta1*time
            + beta2*time^2
            + residual_t

    Coefficients are estimated only on training observations.

    The same coefficients are then used to obtain validation/test residuals.

    This is deliberately conservative: no information from validation/test
    determines the trend coefficients.
    """

    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):

        group = (
            group
            .sort_values(
                "forecast_date"
            )
        )

        positions = np.arange(
            len(group),
            dtype=float,
        )

        if len(group) <= 1:

            t = np.zeros(
                len(group),
                dtype=float,
            )

        else:

            t = (
                positions
                /
                positions.max()
            )

        y = pd.to_numeric(
            group[signal_col],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        train_mask = (
            group["split"]
            .to_numpy()
            ==
            "train"
        )

        valid_train = (
            train_mask
            &
            np.isfinite(y)
        )

        if valid_train.sum() < 10:

            residual = np.full(
                len(group),
                np.nan,
                dtype=float,
            )

        else:

            X = np.column_stack(
                [
                    np.ones(
                        len(group)
                    ),
                    t,
                    t ** 2,
                ]
            )

            X_train = X[
                valid_train
            ]

            y_train = y[
                valid_train
            ]

            try:

                beta = np.linalg.lstsq(
                    X_train,
                    y_train,
                    rcond=None,
                )[0]

                fitted = (
                    X @ beta
                )

                residual = (
                    y
                    -
                    fitted
                )

            except np.linalg.LinAlgError:

                residual = np.full(
                    len(group),
                    np.nan,
                    dtype=float,
                )

        result.loc[
            group.index
        ] = residual

    return result


def ticker_auc_distribution(
    df,
    target_col,
    score_col,
):
    """
    Per-ticker test AUC distribution.
    """

    rows = []

    for ticker, group in df.groupby(
        "ticker",
        sort=True,
    ):

        valid = group[
            group[target_col].notna()
            &
            group[score_col].notna()
        ].copy()

        if (
            len(valid) < 10
            or
            valid[target_col].nunique()
            < 2
        ):
            auc = np.nan
        else:
            auc = safe_auc(
                valid[target_col],
                valid[score_col],
            )

        rows.append(
            {
                "ticker":
                    ticker,

                "auc":
                    auc,

                "n":
                    len(valid),
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# =============================================================================
# LOAD PANEL
# =============================================================================

print(
    "=" * 100
)

print(
    "OVERLAP / SCALE / TIME TREND AUDIT"
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

print(
    "\nSplit counts:"
)

print(
    df["split"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# CREATE SIGNALS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "BUILDING AUDIT SIGNALS"
)

print(
    "=" * 100
)


for model in MODELS:

    raw_signal_col = (
        f"{model}_past_mean_pinball"
    )

    z_col = (
        f"{model}_past_mean_pinball_causal_z"
    )

    residual_col = (
        f"{model}_past_mean_pinball_detrended"
    )

    print(
        f"  {model.upper()}"
    )

    # -------------------------------------------------------------------------
    # Causal within-ticker z-score.
    # -------------------------------------------------------------------------

    df[z_col] = causal_zscore(
        df,
        raw_signal_col,
    )

    # -------------------------------------------------------------------------
    # Training-only quadratic trend residual.
    # -------------------------------------------------------------------------

    df[residual_col] = (
        remove_training_time_trend(
            df,
            raw_signal_col,
        )
    )


# -------------------------------------------------------------------------
# Normalized time index.
# -------------------------------------------------------------------------

df["ticker_time_index"] = (
    add_time_index(
        df
    )
)


# =============================================================================
# STORE RESULTS
# =============================================================================

summary_rows = []
ticker_rows = []
nonoverlap_frames = []


# =============================================================================
# MAIN AUDIT LOOP
# =============================================================================

for model in MODELS:

    raw_col = (
        f"{model}_past_mean_pinball"
    )

    z_col = (
        f"{model}_past_mean_pinball_causal_z"
    )

    detrended_col = (
        f"{model}_past_mean_pinball_detrended"
    )

    signal_specs = {
        "raw_past_mean":
            raw_col,

        "causal_zscore":
            z_col,

        "time_trend":
            "ticker_time_index",

        "detrended_past_mean":
            detrended_col,
    }

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

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        future_loss_col = (
            f"{model}_future_pinball_{horizon}d"
        )

        # =====================================================================
        # NON-OVERLAPPING SAMPLE
        # =====================================================================

        nonoverlap_mask = (
            make_nonoverlap_mask(
                df,
                horizon,
            )
        )

        nonoverlap = (
            df[
                nonoverlap_mask
            ]
            .copy()
        )

        # Keep only observations whose h-day target exists.
        nonoverlap = nonoverlap[
            nonoverlap[target_col].notna()
        ].copy()

        nonoverlap_frames.append(
            nonoverlap[
                [
                    "ticker",
                    "forecast_date",
                    "realized_date",
                    "split",
                    target_col,
                    future_loss_col,
                    raw_col,
                    z_col,
                    detrended_col,
                    "ticker_time_index",
                ]
            ].copy()
        )

        validation = df[
            df["split"] == "validation"
        ].copy()

        test = df[
            df["split"] == "test"
        ].copy()

        validation_nonoverlap = (
            validation[
                make_nonoverlap_mask(
                    validation,
                    horizon,
                )
            ]
            .copy()
        )

        # ---------------------------------------------------------------------
        # Need test sample corresponding to non-overlap rule.
        # ---------------------------------------------------------------------

        test_nonoverlap = (
            test[
                make_nonoverlap_mask(
                    test,
                    horizon,
                )
            ]
            .copy()
        )

        test_nonoverlap = test_nonoverlap[
            test_nonoverlap[
                target_col
            ].notna()
        ].copy()

        print(
            f"\n{horizon}D "
            f"| overlapping test rows="
            f"{len(test_nonoverlap) * horizon:,}"
            f" approx"
            f" | non-overlap rows="
            f"{len(test_nonoverlap):,}"
        )

        # =====================================================================
        # SIGNAL LOOP
        # =====================================================================

        for signal_name, signal_col in signal_specs.items():

            # ---------------------------------------------------------------
            # Choose direction using VALIDATION ONLY.
            # ---------------------------------------------------------------

            validation_valid = (
                validation_nonoverlap[
                    validation_nonoverlap[
                        target_col
                    ].notna()
                    &
                    validation_nonoverlap[
                        signal_col
                    ].notna()
                ]
                .copy()
            )

            direction, validation_auc = (
                choose_direction_validation(
                    validation_valid,
                    target_col,
                    signal_col,
                )
            )

            if not np.isfinite(
                direction
            ):

                print(
                    f"  {signal_name}: no valid orientation"
                )

                continue

            # ---------------------------------------------------------------
            # Test score.
            # ---------------------------------------------------------------

            test_valid = (
                test_nonoverlap[
                    test_nonoverlap[
                        target_col
                    ].notna()
                    &
                    test_nonoverlap[
                        signal_col
                    ].notna()
                ]
                .copy()
            )

            test_score = (
                direction
                *
                test_valid[
                    signal_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y_test = (
                test_valid[
                    target_col
                ]
                .to_numpy(
                    dtype=int
                )
            )

            test_auc = safe_auc(
                y_test,
                test_score,
            )

            # ---------------------------------------------------------------
            # Continuous future-loss relationship.
            # ---------------------------------------------------------------

            future_loss = (
                test_valid[
                    future_loss_col
                ]
                .to_numpy(
                    dtype=float
                )
            )

            valid_cont = (
                np.isfinite(
                    future_loss
                )
                &
                np.isfinite(
                    test_score
                )
            )

            if (
                valid_cont.sum()
                > 10
            ):

                continuous_spearman = (
                    pd.Series(
                        future_loss[
                            valid_cont
                        ]
                    )
                    .corr(
                        pd.Series(
                            test_score[
                                valid_cont
                            ]
                        ),
                        method="spearman",
                    )
                )

                continuous_pearson = (
                    pd.Series(
                        future_loss[
                            valid_cont
                        ]
                    )
                    .corr(
                        pd.Series(
                            test_score[
                                valid_cont
                            ]
                        ),
                        method="pearson",
                    )
                )

            else:

                continuous_spearman = np.nan
                continuous_pearson = np.nan

            # ---------------------------------------------------------------
            # Per-ticker distribution.
            # ---------------------------------------------------------------

            temp = test_valid[
                [
                    "ticker",
                    target_col,
                ]
            ].copy()

            temp[
                "score"
            ] = test_score

            ticker_dist = (
                ticker_auc_distribution(
                    temp,
                    target_col,
                    "score",
                )
            )

            ticker_dist = ticker_dist[
                ticker_dist["auc"].notna()
            ].copy()

            if len(ticker_dist) > 0:

                macro_auc = (
                    ticker_dist["auc"]
                    .mean()
                )

                median_auc = (
                    ticker_dist["auc"]
                    .median()
                )

                q25 = (
                    ticker_dist["auc"]
                    .quantile(0.25)
                )

                q75 = (
                    ticker_dist["auc"]
                    .quantile(0.75)
                )

                pct_above_50 = (
                    (
                        ticker_dist["auc"]
                        > 0.50
                    )
                    .mean()
                )

                pct_above_60 = (
                    (
                        ticker_dist["auc"]
                        > 0.60
                    )
                    .mean()
                )

                pct_above_70 = (
                    (
                        ticker_dist["auc"]
                        > 0.70
                    )
                    .mean()
                )

            else:

                macro_auc = np.nan
                median_auc = np.nan
                q25 = np.nan
                q75 = np.nan
                pct_above_50 = np.nan
                pct_above_60 = np.nan
                pct_above_70 = np.nan

            # ---------------------------------------------------------------
            # Store aggregate summary.
            # ---------------------------------------------------------------

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

                    "validation_auc":
                        validation_auc,

                    "test_rows_nonoverlap":
                        len(test_valid),

                    "test_positive_rate":
                        float(
                            y_test.mean()
                        )
                        if len(y_test)
                        else np.nan,

                    "test_auc":
                        test_auc,

                    "test_macro_ticker_auc":
                        macro_auc,

                    "test_median_ticker_auc":
                        median_auc,

                    "test_ticker_auc_q25":
                        q25,

                    "test_ticker_auc_q75":
                        q75,

                    "pct_tickers_auc_gt_050":
                        pct_above_50,

                    "pct_tickers_auc_gt_060":
                        pct_above_60,

                    "pct_tickers_auc_gt_070":
                        pct_above_70,

                    "future_loss_spearman":
                        continuous_spearman,

                    "future_loss_pearson":
                        continuous_pearson,
                }
            )

            # ---------------------------------------------------------------
            # Store per-ticker values.
            # ---------------------------------------------------------------

            for _, row in ticker_dist.iterrows():

                ticker_rows.append(
                    {
                        "model":
                            model.upper(),

                        "horizon":
                            horizon,

                        "signal":
                            signal_name,

                        "ticker":
                            row["ticker"],

                        "auc":
                            row["auc"],

                        "n_nonoverlap":
                            row["n"],
                    }
                )

            print(
                f"  {signal_name:25s}"
                f" | val={validation_auc:.4f}"
                f" | test={test_auc:.4f}"
                f" | macro={macro_auc:.4f}"
                f" | median={median_auc:.4f}"
            )


# =============================================================================
# BUILD DATAFRAMES
# =============================================================================

summary = pd.DataFrame(
    summary_rows
)

per_ticker = pd.DataFrame(
    ticker_rows
)

nonoverlap_sample = pd.concat(
    nonoverlap_frames,
    ignore_index=True,
)


# =============================================================================
# NON-OVERLAP SAMPLE INFORMATION
# =============================================================================

nonoverlap_sample = (
    nonoverlap_sample
    .drop_duplicates(
        [
            "ticker",
            "forecast_date",
            "split",
        ]
    )
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

nonoverlap_sample.to_csv(
    NONOVERLAP_PATH,
    index=False,
)


# =============================================================================
# PRINT CORE SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "CORE NON-OVERLAP AUDIT RESULTS"
)

print(
    "=" * 100
)

print(
    summary[
        [
            "model",
            "horizon",
            "signal",
            "validation_auc",
            "test_auc",
            "test_macro_ticker_auc",
            "test_median_ticker_auc",
            "pct_tickers_auc_gt_050",
            "pct_tickers_auc_gt_060",
            "pct_tickers_auc_gt_070",
            "future_loss_spearman",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# DIRECT RAW-SIGNAL RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "RAW PAST-MEAN: NON-OVERLAP TEST RESULTS"
)

print(
    "=" * 100
)

raw = summary[
    summary["signal"]
    ==
    "raw_past_mean"
].copy()

print(
    raw[
        [
            "model",
            "horizon",
            "test_rows_nonoverlap",
            "test_auc",
            "test_macro_ticker_auc",
            "test_median_ticker_auc",
            "pct_tickers_auc_gt_050",
            "pct_tickers_auc_gt_060",
            "pct_tickers_auc_gt_070",
            "future_loss_spearman",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# STANDARDIZED SIGNAL RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "CAUSAL WITHIN-TICKER Z-SCORE"
)

print(
    "=" * 100
)

z = summary[
    summary["signal"]
    ==
    "causal_zscore"
].copy()

print(
    z[
        [
            "model",
            "horizon",
            "test_auc",
            "test_macro_ticker_auc",
            "test_median_ticker_auc",
            "pct_tickers_auc_gt_050",
            "pct_tickers_auc_gt_060",
            "pct_tickers_auc_gt_070",
            "future_loss_spearman",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# TIME TREND RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "PURE TIME TREND"
)

print(
    "=" * 100
)

trend = summary[
    summary["signal"]
    ==
    "time_trend"
].copy()

print(
    trend[
        [
            "model",
            "horizon",
            "validation_auc",
            "test_auc",
            "test_macro_ticker_auc",
            "test_median_ticker_auc",
            "pct_tickers_auc_gt_050",
            "pct_tickers_auc_gt_060",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# DETRENDED RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "PAST-MEAN AFTER REMOVING TRAINING-ONLY QUADRATIC TIME TREND"
)

print(
    "=" * 100
)

detrended = summary[
    summary["signal"]
    ==
    "detrended_past_mean"
].copy()

print(
    detrended[
        [
            "model",
            "horizon",
            "validation_auc",
            "test_auc",
            "test_macro_ticker_auc",
            "test_median_ticker_auc",
            "pct_tickers_auc_gt_050",
            "pct_tickers_auc_gt_060",
            "pct_tickers_auc_gt_070",
            "future_loss_spearman",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# SAVE
# =============================================================================

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

per_ticker.to_csv(
    TICKER_PATH,
    index=False,
)


# =============================================================================
# FINAL DIAGNOSTIC COMPARISONS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "DIAGNOSTIC COMPARISONS"
)

print(
    "=" * 100
)

for model in MODELS:

    for horizon in HORIZONS:

        block = summary[
            (
                summary["model"]
                ==
                model.upper()
            )
            &
            (
                summary["horizon"]
                ==
                horizon
            )
        ].set_index(
            "signal"
        )

        if (
            "raw_past_mean"
            not in block.index
        ):
            continue

        raw_auc = (
            block.loc[
                "raw_past_mean",
                "test_auc",
            ]
        )

        z_auc = (
            block.loc[
                "causal_zscore",
                "test_auc",
            ]
        )

        trend_auc = (
            block.loc[
                "time_trend",
                "test_auc",
            ]
        )

        detrended_auc = (
            block.loc[
                "detrended_past_mean",
                "test_auc",
            ]
        )

        print(
            f"{model.upper():5s} {horizon:2d}d | "
            f"raw={raw_auc:.4f} | "
            f"z={z_auc:.4f} | "
            f"time={trend_auc:.4f} | "
            f"detrended={detrended_auc:.4f}"
        )


# =============================================================================
# OUTPUT PATHS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)

print(
    f"\nSummary:"
)

print(
    SUMMARY_PATH
)

print(
    f"\nPer-ticker:"
)

print(
    TICKER_PATH
)

print(
    f"\nNon-overlap sample:"
)

print(
    NONOVERLAP_PATH
)


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "OVERLAP / SCALE / TIME AUDIT COMPLETE"
)

print(
    "=" * 100
)