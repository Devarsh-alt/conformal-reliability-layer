# ============================================================
# 28_test_new_reliability_signals.py
#
# Purpose:
#   Test the four PRE-SPECIFIED new reliability signals against
#   future QRF VaR deterioration.
#
# Signals:
#
#   1. Cross-sectional breach stress
#   2. QRF training-regime extremeness
#   3. QRF training out-of-range indicator
#   4. QRF-EWMA model disagreement
#
# Baseline:
#   breach_lag1
#
# Targets:
#   deterioration_5d
#   deterioration_10d
#   deterioration_20d
#
# Method:
#   - Chronological 60/40 split within each ticker
#   - Univariate logistic regression
#   - Exact common sample between baseline and signal
#   - Pooled OOS AUC / PR-AUC
#   - Mean ticker AUC
#   - Within-ticker paired moving-block bootstrap
#   - Cross-sectional ticker bootstrap
#
# IMPORTANT:
#   This is a PRE-SPECIFIED confirmatory-style experiment.
#   Do not add new signals after inspecting these results.
#
# Inputs:
#   data/monitoring/QRF_25ticker_new_signals.csv
#   data/monitoring/QRF_25ticker_reliability_panel_with_targets.csv
#
# Outputs:
#   data/monitoring/new_signal_test_results.csv
#   data/monitoring/new_signal_test_by_ticker.csv
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

TRAIN_FRACTION = 0.60

MIN_TRAIN = 100
MIN_TEST = 50

BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 20

RANDOM_SEED = 42

BASELINE_COL = "breach_lag1"

HORIZONS = [5, 10, 20]

TARGET_TEMPLATE = "deterioration_{}d"


# ============================================================
# NEW SIGNALS
# ============================================================

NEW_SIGNALS = {
    "Cross-Sectional Breach Stress":
        "cs_breach_stress_5d",

    "Training-Regime Extremeness":
        "train_regime_extremeness",

    "Training Out-of-Range":
        "train_out_of_range_any",

    "QRF-EWMA Disagreement":
        "qrf_ewma_disagreement",
}


# ============================================================
# TICKERS
# ============================================================

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

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MONITORING_DIR = (
    ROOT
    / "data"
    / "monitoring"
)

PANEL_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

SIGNALS_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_new_signals.csv"
)

OUTPUT_RESULTS = (
    MONITORING_DIR
    / "new_signal_test_results.csv"
)

OUTPUT_TICKER = (
    MONITORING_DIR
    / "new_signal_test_by_ticker.csv"
)


# ============================================================
# HELPERS
# ============================================================

def make_logistic():
    """
    Standardized univariate logistic regression.

    We use the same structure as the earlier signal experiments
    so the results remain comparable.
    """

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def safe_auc(y, score):
    """
    Return ROC-AUC or NaN when only one class is present.
    """

    y = np.asarray(y)

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(
        y,
        score,
    )


def safe_prauc(y, score):
    """
    Return PR-AUC or NaN when only one class is present.
    """

    y = np.asarray(y)

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return average_precision_score(
        y,
        score,
    )


def moving_block_indices(
    n,
    block_length,
    rng,
):
    """
    Moving-block bootstrap indices for a single ticker.

    Blocks remain entirely inside the ticker's own time series.
    """

    if n <= 0:
        return np.array([], dtype=int)

    if n == 1:
        return np.array([0], dtype=int)

    block_length = min(
        block_length,
        n,
    )

    starts = np.arange(
        0,
        n - block_length + 1,
    )

    n_blocks = int(
        np.ceil(
            n / block_length
        )
    )

    result = []

    for _ in range(n_blocks):

        start = rng.choice(
            starts
        )

        result.extend(
            range(
                start,
                start + block_length,
            )
        )

    return np.asarray(
        result[:n],
        dtype=int,
    )


def paired_block_bootstrap_delta(
    ticker_data,
    baseline_score_col,
    signal_score_col,
    reps=1000,
    block_length=20,
    seed=42,
):
    """
    Paired moving-block bootstrap.

    For every ticker:
      - resample contiguous blocks
      - retain y, baseline score and signal score as pairs

    Then pool across tickers and calculate:

        AUC(signal) - AUC(baseline)

    This avoids the earlier mistake of treating the end of one
    stock's time series as adjacent to another stock's time
    series.
    """

    rng = np.random.default_rng(
        seed
    )

    deltas = []

    for _ in range(reps):

        all_y = []
        all_baseline = []
        all_signal = []

        for ticker, data in ticker_data.items():

            if len(data) == 0:
                continue

            y = data[
                "target"
            ].to_numpy()

            baseline = data[
                baseline_score_col
            ].to_numpy()

            signal = data[
                signal_score_col
            ].to_numpy()

            idx = moving_block_indices(
                n=len(data),
                block_length=block_length,
                rng=rng,
            )

            all_y.append(
                y[idx]
            )

            all_baseline.append(
                baseline[idx]
            )

            all_signal.append(
                signal[idx]
            )

        if not all_y:
            continue

        y_boot = np.concatenate(
            all_y
        )

        baseline_boot = np.concatenate(
            all_baseline
        )

        signal_boot = np.concatenate(
            all_signal
        )

        if len(np.unique(y_boot)) < 2:
            continue

        baseline_auc = roc_auc_score(
            y_boot,
            baseline_boot,
        )

        signal_auc = roc_auc_score(
            y_boot,
            signal_boot,
        )

        deltas.append(
            signal_auc
            - baseline_auc
        )

    deltas = np.asarray(
        deltas
    )

    if len(deltas) == 0:

        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "prob_positive": np.nan,
            "p_approx": np.nan,
            "valid_reps": 0,
        }

    ci_lower = np.percentile(
        deltas,
        2.5,
    )

    ci_upper = np.percentile(
        deltas,
        97.5,
    )

    prob_positive = np.mean(
        deltas > 0
    )

    p_left = np.mean(
        deltas <= 0
    )

    p_right = np.mean(
        deltas >= 0
    )

    p_approx = min(
        1.0,
        2.0 * min(
            p_left,
            p_right,
        ),
    )

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "prob_positive": prob_positive,
        "p_approx": p_approx,
        "valid_reps": len(deltas),
    }


def ticker_bootstrap_mean_delta(
    ticker_results,
    reps=1000,
    seed=42,
):
    """
    Bootstrap the mean ticker-level Delta AUC.

    This treats the ticker as the cross-sectional sampling unit.
    """

    clean = ticker_results.dropna(
        subset=[
            "baseline_auc",
            "signal_auc",
        ]
    ).copy()

    if len(clean) < 3:

        return {
            "macro_ci_lower": np.nan,
            "macro_ci_upper": np.nan,
        }

    deltas = (
        clean["signal_auc"].to_numpy()
        -
        clean["baseline_auc"].to_numpy()
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(deltas)

    boot_means = []

    for _ in range(reps):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        boot_means.append(
            np.mean(
                deltas[idx]
            )
        )

    boot_means = np.asarray(
        boot_means
    )

    return {
        "macro_ci_lower":
            np.percentile(
                boot_means,
                2.5,
            ),

        "macro_ci_upper":
            np.percentile(
                boot_means,
                97.5,
            ),
    }


# ============================================================
# LOAD
# ============================================================

print("=" * 100)
print("TEST PRE-SPECIFIED NEW QRF RELIABILITY SIGNALS")
print("=" * 100)


if not PANEL_FILE.exists():

    raise FileNotFoundError(
        f"\nPanel not found:\n{PANEL_FILE}"
    )

if not SIGNALS_FILE.exists():

    raise FileNotFoundError(
        f"\nNew-signal file not found:\n{SIGNALS_FILE}"
    )


panel = pd.read_csv(
    PANEL_FILE
)

signals = pd.read_csv(
    SIGNALS_FILE
)


print(
    f"\nReliability panel rows:"
    f" {len(panel):,}"
)

print(
    f"New-signal rows:"
    f" {len(signals):,}"
)


# ============================================================
# NORMALIZE DATES
# ============================================================

for frame in [
    panel,
    signals,
]:

    if "realized_date" in frame.columns:

        frame["realized_date"] = (
            pd.to_datetime(
                frame["realized_date"],
                errors="coerce",
            )
        )

    if "forecast_date" in frame.columns:

        frame["forecast_date"] = (
            pd.to_datetime(
                frame["forecast_date"],
                errors="coerce",
            )
        )


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_panel = [
    "ticker",
    "realized_date",
    "forecast_date",
    BASELINE_COL,
]

for horizon in HORIZONS:

    required_panel.append(
        TARGET_TEMPLATE.format(
            horizon
        )
    )


missing_panel = [
    c
    for c in required_panel
    if c not in panel.columns
]

if missing_panel:

    raise ValueError(
        "\nMissing panel columns:"
        "\n"
        + "\n".join(
            missing_panel
        )
    )


required_signals = [
    "ticker",
    "realized_date",
    "forecast_date",
]

required_signals.extend(
    NEW_SIGNALS.values()
)

missing_signals = [
    c
    for c in required_signals
    if c not in signals.columns
]

if missing_signals:

    raise ValueError(
        "\nMissing signal columns:"
        "\n"
        + "\n".join(
            missing_signals
        )
    )


# ============================================================
# MERGE
# ============================================================

merge_keys = [
    "ticker",
    "forecast_date",
    "realized_date",
]

signal_columns_to_merge = [
    "ticker",
    "forecast_date",
    "realized_date",
]

signal_columns_to_merge.extend(
    NEW_SIGNALS.values()
)


signals_small = signals[
    signal_columns_to_merge
].copy()


# Make sure one row exists per key.
duplicate_signal_keys = (
    signals_small
    .duplicated(
        subset=merge_keys,
        keep=False,
    )
    .sum()
)

if duplicate_signal_keys > 0:

    raise ValueError(
        "\nDuplicate signal rows found:"
        f" {duplicate_signal_keys}"
    )


duplicate_panel_keys = (
    panel
    .duplicated(
        subset=merge_keys,
        keep=False,
    )
    .sum()
)

if duplicate_panel_keys > 0:

    raise ValueError(
        "\nDuplicate panel rows found:"
        f" {duplicate_panel_keys}"
    )


df = panel.merge(
    signals_small,
    on=merge_keys,
    how="left",
    validate="one_to_one",
)


print(
    f"\nMerged rows:"
    f" {len(df):,}"
)


# ============================================================
# SORT
# ============================================================

df = (
    df
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


# ============================================================
# SIGNAL AVAILABILITY
# ============================================================

print("\n" + "=" * 100)
print("NEW SIGNAL AVAILABILITY")
print("=" * 100)


for signal_name, signal_col in NEW_SIGNALS.items():

    available = int(
        df[signal_col]
        .notna()
        .sum()
    )

    pct = (
        100.0
        * available
        / len(df)
    )

    ticker_count = (
        df.loc[
            df[signal_col].notna(),
            "ticker",
        ]
        .nunique()
    )

    print(
        f"{signal_name:35s}"
        f" {available:6d}/{len(df):6d}"
        f" ({pct:6.2f}%)"
        f" tickers={ticker_count}"
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

summary_rows = []

ticker_rows = []


for horizon in HORIZONS:

    target_col = TARGET_TEMPLATE.format(
        horizon
    )

    print("\n" + "=" * 100)
    print(
        f"HORIZON = {horizon} DAYS"
    )
    print("=" * 100)


    for signal_name, signal_col in NEW_SIGNALS.items():

        print(
            f"\nTesting:"
            f" {signal_name}"
        )

        # ----------------------------------------------------
        # Pooled containers
        # ----------------------------------------------------

        pooled_y = []

        pooled_baseline = []

        pooled_signal = []

        # ----------------------------------------------------
        # Bootstrap containers
        # ----------------------------------------------------

        bootstrap_ticker_data = {}

        # ----------------------------------------------------
        # Ticker results
        # ----------------------------------------------------

        per_ticker = []

        eligible_tickers = 0


        # ====================================================
        # TICKER LOOP
        # ====================================================

        for ticker, group in df.groupby(
            "ticker",
            sort=True,
        ):

            g = group.sort_values(
                "forecast_date"
            ).copy()


            # ------------------------------------------------
            # Exact common sample
            # ------------------------------------------------

            cols = [
                "forecast_date",
                "realized_date",
                BASELINE_COL,
                signal_col,
                target_col,
            ]

            g = g[
                cols
            ].replace(
                [np.inf, -np.inf],
                np.nan,
            )


            g = g.dropna(
                subset=[
                    BASELINE_COL,
                    signal_col,
                    target_col,
                ]
            )


            # ------------------------------------------------
            # Minimum data
            # ------------------------------------------------

            if len(g) < (
                MIN_TRAIN
                + MIN_TEST
            ):

                continue


            # ------------------------------------------------
            # Chronological split
            # ------------------------------------------------

            split = int(
                len(g)
                * TRAIN_FRACTION
            )

            train = g.iloc[
                :split
            ].copy()

            test = g.iloc[
                split:
            ].copy()


            if len(train) < MIN_TRAIN:
                continue

            if len(test) < MIN_TEST:
                continue


            # ------------------------------------------------
            # Both target classes needed
            # ------------------------------------------------

            if (
                train[target_col].nunique()
                < 2
            ):

                continue

            if (
                test[target_col].nunique()
                < 2
            ):

                continue


            # =================================================
            # BASELINE
            # =================================================

            baseline_model = make_logistic()

            baseline_model.fit(
                train[
                    [BASELINE_COL]
                ],
                train[
                    target_col
                ],
            )

            baseline_score = (
                baseline_model
                .predict_proba(
                    test[
                        [BASELINE_COL]
                    ]
                )[:, 1]
            )


            # =================================================
            # NEW SIGNAL
            # =================================================

            signal_model = make_logistic()

            signal_model.fit(
                train[
                    [signal_col]
                ],
                train[
                    target_col
                ],
            )

            signal_score = (
                signal_model
                .predict_proba(
                    test[
                        [signal_col]
                    ]
                )[:, 1]
            )


            y_test = test[
                target_col
            ].to_numpy()


            # =================================================
            # METRICS
            # =================================================

            baseline_auc = safe_auc(
                y_test,
                baseline_score,
            )

            signal_auc = safe_auc(
                y_test,
                signal_score,
            )

            baseline_prauc = safe_prauc(
                y_test,
                baseline_score,
            )

            signal_prauc = safe_prauc(
                y_test,
                signal_score,
            )

            delta_auc = (
                signal_auc
                - baseline_auc
            )


            # =================================================
            # TICKER RESULT
            # =================================================

            per_ticker.append(
                {
                    "horizon":
                        horizon,

                    "signal_name":
                        signal_name,

                    "signal_column":
                        signal_col,

                    "ticker":
                        ticker,

                    "n_train":
                        len(train),

                    "n_test":
                        len(test),

                    "target_positive_rate":
                        y_test.mean(),

                    "baseline_auc":
                        baseline_auc,

                    "signal_auc":
                        signal_auc,

                    "delta_auc":
                        delta_auc,

                    "baseline_prauc":
                        baseline_prauc,

                    "signal_prauc":
                        signal_prauc,
                }
            )


            # =================================================
            # POOLED ARRAYS
            # =================================================

            pooled_y.append(
                y_test
            )

            pooled_baseline.append(
                baseline_score
            )

            pooled_signal.append(
                signal_score
            )


            # =================================================
            # BOOTSTRAP STORAGE
            # =================================================

            bootstrap_ticker_data[ticker] = (
                pd.DataFrame(
                    {
                        "target":
                            y_test,

                        "baseline_score":
                            baseline_score,

                        "signal_score":
                            signal_score,
                    }
                )
            )


            eligible_tickers += 1


        # ====================================================
        # CHECK
        # ====================================================

        if eligible_tickers == 0:

            print(
                "  No eligible tickers."
            )

            continue


        # ====================================================
        # POOLED METRICS
        # ====================================================

        y_all = np.concatenate(
            pooled_y
        )

        baseline_all = np.concatenate(
            pooled_baseline
        )

        signal_all = np.concatenate(
            pooled_signal
        )


        baseline_pooled_auc = safe_auc(
            y_all,
            baseline_all,
        )

        signal_pooled_auc = safe_auc(
            y_all,
            signal_all,
        )

        baseline_pooled_prauc = safe_prauc(
            y_all,
            baseline_all,
        )

        signal_pooled_prauc = safe_prauc(
            y_all,
            signal_all,
        )

        pooled_delta_auc = (
            signal_pooled_auc
            - baseline_pooled_auc
        )


        # ====================================================
        # TICKER MACRO METRICS
        # ====================================================

        ticker_result_df = pd.DataFrame(
            per_ticker
        )

        baseline_macro_auc = (
            ticker_result_df[
                "baseline_auc"
            ].mean()
        )

        signal_macro_auc = (
            ticker_result_df[
                "signal_auc"
            ].mean()
        )

        macro_delta_auc = (
            ticker_result_df[
                "delta_auc"
            ].mean()
        )

        better_tickers = int(
            (
                ticker_result_df[
                    "delta_auc"
                ]
                > 0
            ).sum()
        )

        worse_tickers = int(
            (
                ticker_result_df[
                    "delta_auc"
                ]
                < 0
            ).sum()
        )


        # ====================================================
        # WITHIN-TICKER BLOCK BOOTSTRAP
        # ====================================================

        bootstrap = (
            paired_block_bootstrap_delta(
                ticker_data=
                    bootstrap_ticker_data,

                baseline_score_col=
                    "baseline_score",

                signal_score_col=
                    "signal_score",

                reps=
                    BOOTSTRAP_REPS,

                block_length=
                    BLOCK_LENGTH,

                seed=
                    RANDOM_SEED
                    + horizon,
            )
        )


        # ====================================================
        # CROSS-SECTIONAL TICKER BOOTSTRAP
        # ====================================================

        macro_bootstrap = (
            ticker_bootstrap_mean_delta(
                ticker_results=
                    ticker_result_df,

                reps=
                    BOOTSTRAP_REPS,

                seed=
                    RANDOM_SEED
                    + horizon
                    + 1000,
            )
        )


        # ====================================================
        # APPEND SUMMARY
        # ====================================================

        summary_rows.append(
            {
                "horizon":
                    horizon,

                "signal_name":
                    signal_name,

                "signal_column":
                    signal_col,

                "n_tickers":
                    eligible_tickers,

                "n_test_total":
                    len(y_all),

                "baseline_pooled_auc":
                    baseline_pooled_auc,

                "signal_pooled_auc":
                    signal_pooled_auc,

                "delta_pooled_auc":
                    pooled_delta_auc,

                "baseline_pooled_prauc":
                    baseline_pooled_prauc,

                "signal_pooled_prauc":
                    signal_pooled_prauc,

                "baseline_mean_ticker_auc":
                    baseline_macro_auc,

                "signal_mean_ticker_auc":
                    signal_macro_auc,

                "delta_mean_ticker_auc":
                    macro_delta_auc,

                "signal_better_tickers":
                    better_tickers,

                "signal_worse_tickers":
                    worse_tickers,

                "block_ci_lower":
                    bootstrap[
                        "ci_lower"
                    ],

                "block_ci_upper":
                    bootstrap[
                        "ci_upper"
                    ],

                "block_prob_positive":
                    bootstrap[
                        "prob_positive"
                    ],

                "block_p_approx":
                    bootstrap[
                        "p_approx"
                    ],

                "block_valid_reps":
                    bootstrap[
                        "valid_reps"
                    ],

                "macro_ci_lower":
                    macro_bootstrap[
                        "macro_ci_lower"
                    ],

                "macro_ci_upper":
                    macro_bootstrap[
                        "macro_ci_upper"
                    ],
            }
        )


        ticker_rows.extend(
            per_ticker
        )


        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"  Eligible tickers:"
            f" {eligible_tickers}"
        )

        print(
            f"  Test observations:"
            f" {len(y_all):,}"
        )

        print(
            f"  Persistence AUC:"
            f" {baseline_pooled_auc:.4f}"
        )

        print(
            f"  Signal AUC:"
            f" {signal_pooled_auc:.4f}"
        )

        print(
            f"  Delta AUC:"
            f" {pooled_delta_auc:+.4f}"
        )

        print(
            f"  Block CI:"
            f" [{bootstrap['ci_lower']:+.4f}, "
            f"{bootstrap['ci_upper']:+.4f}]"
        )

        print(
            f"  Mean ticker Delta:"
            f" {macro_delta_auc:+.4f}"
        )

        print(
            f"  Better tickers:"
            f" {better_tickers}/"
            f"{eligible_tickers}"
        )


# ============================================================
# DATAFRAMES
# ============================================================

results_df = pd.DataFrame(
    summary_rows
)

ticker_df = pd.DataFrame(
    ticker_rows
)


if len(results_df) == 0:

    raise RuntimeError(
        "No new-signal results were generated."
    )


results_df = results_df.sort_values(
    [
        "horizon",
        "delta_pooled_auc",
    ],
    ascending=[
        True,
        False,
    ],
).reset_index(drop=True)


ticker_df = ticker_df.sort_values(
    [
        "horizon",
        "signal_name",
        "ticker",
    ]
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False,
)

ticker_df.to_csv(
    OUTPUT_TICKER,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL NEW-SIGNAL RESULTS")
print("=" * 100)


for horizon in HORIZONS:

    print(
        f"\n{'-' * 100}"
    )

    print(
        f"{horizon}-DAY"
    )

    print(
        f"{'-' * 100}"
    )

    sub = results_df[
        results_df["horizon"]
        == horizon
    ].copy()

    print(
        sub[
            [
                "signal_name",
                "baseline_pooled_auc",
                "signal_pooled_auc",
                "delta_pooled_auc",
                "block_ci_lower",
                "block_ci_upper",
                "delta_mean_ticker_auc",
                "signal_better_tickers",
                "n_tickers",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# ROBUSTLY POSITIVE BLOCK CI
# ============================================================

print("\n" + "=" * 100)
print(
    "SIGNALS WITH WITHIN-TICKER BLOCK CI "
    "ENTIRELY ABOVE ZERO"
)
print("=" * 100)


for horizon in HORIZONS:

    sub = results_df[
        results_df["horizon"]
        == horizon
    ]

    positive = sub[
        sub["block_ci_lower"] > 0
    ].sort_values(
        "delta_pooled_auc",
        ascending=False,
    )

    print(
        f"\n{horizon}-DAY:"
    )

    if len(positive) == 0:

        print(
            "  None"
        )

    else:

        for _, row in positive.iterrows():

            print(
                f"  {row['signal_name']}: "
                f"ΔAUC={row['delta_pooled_auc']:+.4f}, "
                f"CI=["
                f"{row['block_ci_lower']:+.4f}, "
                f"{row['block_ci_upper']:+.4f}]"
            )


# ============================================================
# CROSS-SECTIONAL POSITIVE CI
# ============================================================

print("\n" + "=" * 100)
print(
    "SIGNALS WITH CROSS-SECTIONAL TICKER CI "
    "ENTIRELY ABOVE ZERO"
)
print("=" * 100)


for horizon in HORIZONS:

    sub = results_df[
        results_df["horizon"]
        == horizon
    ]

    positive = sub[
        sub["macro_ci_lower"] > 0
    ].sort_values(
        "delta_mean_ticker_auc",
        ascending=False,
    )

    print(
        f"\n{horizon}-DAY:"
    )

    if len(positive) == 0:

        print(
            "  None"
        )

    else:

        for _, row in positive.iterrows():

            print(
                f"  {row['signal_name']}: "
                f"mean ΔAUC="
                f"{row['delta_mean_ticker_auc']:+.4f}, "
                f"CI=["
                f"{row['macro_ci_lower']:+.4f}, "
                f"{row['macro_ci_upper']:+.4f}]"
            )


# ============================================================
# RANK SIGNALS
# ============================================================

print("\n" + "=" * 100)
print("RANKING BY POOLED DELTA AUC")
print("=" * 100)


ranking = results_df[
    [
        "horizon",
        "signal_name",
        "delta_pooled_auc",
        "delta_mean_ticker_auc",
        "block_ci_lower",
        "block_ci_upper",
        "signal_better_tickers",
        "n_tickers",
    ]
].copy()

print(
    ranking.to_string(
        index=False
    )
)


# ============================================================
# OUTPUT PATHS
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_TICKER
)

print("\n" + "=" * 100)
print(
    "NEW RELIABILITY SIGNAL TEST COMPLETE"
)
print("=" * 100)