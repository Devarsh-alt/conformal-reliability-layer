# ============================================================
# signals_vs_persistence.py
#
# Purpose:
#   Compare each QRF reliability signal against the simple
#   breach-persistence baseline.
#
# Main statistic:
#   Delta AUC = AUC(signal) - AUC(breach persistence)
#
# Evaluation:
#   - Chronological 60/40 train-test split within each ticker
#   - Univariate logistic regression
#   - Exact common sample for baseline vs candidate signal
#   - Pooled OOS AUC
#   - Mean ticker AUC
#   - Within-ticker moving-block bootstrap for Delta AUC
#   - Cross-sectional ticker bootstrap for mean Delta AUC
#
# Input:
#   data/monitoring/QRF_25ticker_reliability_panel_with_targets.csv
#
# Outputs:
#   data/monitoring/signal_vs_persistence_comparison.csv
#   data/monitoring/signal_vs_persistence_ticker.csv
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ============================================================
# CONFIG
# ============================================================

TRAIN_FRACTION = 0.60

BOOTSTRAP_REPS = 1000
BLOCK_LENGTH = 20
RANDOM_SEED = 42

MIN_TRAIN = 100
MIN_TEST = 50

# Simple baseline
BASELINE_COL = "breach_lag1"

# Candidate monitoring signals
SIGNALS = {
    "Breach Streak": "breach_streak_prior",

    "5-Day Breach Rate": "breach_rate_5d",
    "20-Day Breach Rate": "breach_rate_20d",
    "60-Day Breach Rate": "breach_rate_60d",
    "EWMA Breach Rate": "ewma_breach_rate",

    "Recent 5-Day Pinball Loss": "pinball_mean_5d_prior",
    "Recent 20-Day Pinball Loss": "pinball_mean_20d_prior",
    "Recent 60-Day Pinball Loss": "pinball_mean_60d_prior",
    "Pinball Loss Acceleration": "pinball_acceleration",

    "20-Day Realized Volatility": "realized_vol_20d",
    "60-Day Realized Volatility": "realized_vol_60d",
    "Volatility Ratio": "vol_ratio_20_60",
    "KS Distribution Drift": "ks_drift_60_vs_250",

    "VaR Quantile Change": "q05_change_1d",
    "VaR Magnitude Change": "VaR_change_1d",
    "VaR Ratio to Prior": "VaR_ratio_to_prior",
    "VaR Ratio to 20-Day Mean": "VaR_ratio_to_20d_mean",
}

HORIZONS = [5, 10, 20]

TARGET_TEMPLATE = "deterioration_{}d"


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "monitoring"

OUTPUT_COMPARISON = (
    OUTPUT_DIR
    / "signal_vs_persistence_comparison.csv"
)

OUTPUT_TICKER = (
    OUTPUT_DIR
    / "signal_vs_persistence_ticker.csv"
)


# ============================================================
# HELPERS
# ============================================================

def make_logistic():
    """
    Standardized univariate logistic regression.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
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
    Return ROC-AUC or NaN if only one class is present.
    """
    y = np.asarray(y)

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(y, score)


def safe_prauc(y, score):
    """
    Return PR-AUC or NaN if only one class is present.
    """
    y = np.asarray(y)

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return average_precision_score(y, score)


def moving_block_indices(
    n,
    block_length,
    rng,
):
    """
    Generate moving-block bootstrap indices.

    Blocks are sampled from all possible contiguous blocks.
    """

    if n <= 0:
        return np.array([], dtype=int)

    if n == 1:
        return np.array([0], dtype=int)

    block_length = min(
        block_length,
        n,
    )

    possible_starts = np.arange(
        0,
        n - block_length + 1,
    )

    n_blocks = int(
        np.ceil(n / block_length)
    )

    sampled = []

    for _ in range(n_blocks):

        start = rng.choice(
            possible_starts
        )

        sampled.extend(
            range(
                start,
                start + block_length,
            )
        )

    return np.asarray(
        sampled[:n],
        dtype=int,
    )


def block_bootstrap_delta_auc(
    ticker_test_data,
    baseline_col,
    signal_col,
    reps=1000,
    block_length=20,
    seed=42,
):
    """
    Proper paired within-ticker moving-block bootstrap.

    IMPORTANT:
    We bootstrap each ticker independently so observations from
    different assets are never treated as temporally adjacent.

    For every replicate:
      - resample blocks within each ticker
      - keep target/baseline/signal paired
      - pool the resampled observations
      - calculate:
            AUC(signal) - AUC(baseline)
    """

    rng = np.random.default_rng(seed)

    ticker_names = list(
        ticker_test_data.keys()
    )

    bootstrap_deltas = []

    for _ in range(reps):

        pooled_y = []
        pooled_baseline = []
        pooled_signal = []

        for ticker in ticker_names:

            data = ticker_test_data[ticker]

            y = data["target"].to_numpy()

            baseline = data[
                baseline_col
            ].to_numpy()

            signal = data[
                signal_col
            ].to_numpy()

            n = len(data)

            if n == 0:
                continue

            idx = moving_block_indices(
                n=n,
                block_length=block_length,
                rng=rng,
            )

            pooled_y.append(
                y[idx]
            )

            pooled_baseline.append(
                baseline[idx]
            )

            pooled_signal.append(
                signal[idx]
            )

        if len(pooled_y) == 0:
            continue

        y_boot = np.concatenate(
            pooled_y
        )

        baseline_boot = np.concatenate(
            pooled_baseline
        )

        signal_boot = np.concatenate(
            pooled_signal
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

        bootstrap_deltas.append(
            signal_auc - baseline_auc
        )

    bootstrap_deltas = np.asarray(
        bootstrap_deltas
    )

    if len(bootstrap_deltas) == 0:

        return {
            "delta_ci_lower": np.nan,
            "delta_ci_upper": np.nan,
            "bootstrap_prob_signal_better": np.nan,
            "bootstrap_p_approx": np.nan,
            "bootstrap_reps_valid": 0,
        }

    ci_lower = np.percentile(
        bootstrap_deltas,
        2.5,
    )

    ci_upper = np.percentile(
        bootstrap_deltas,
        97.5,
    )

    prob_signal_better = np.mean(
        bootstrap_deltas > 0
    )

    p_left = np.mean(
        bootstrap_deltas <= 0
    )

    p_right = np.mean(
        bootstrap_deltas >= 0
    )

    p_approx = min(
        1.0,
        2.0 * min(
            p_left,
            p_right,
        ),
    )

    return {
        "delta_ci_lower": ci_lower,
        "delta_ci_upper": ci_upper,
        "bootstrap_prob_signal_better":
            prob_signal_better,
        "bootstrap_p_approx":
            p_approx,
        "bootstrap_reps_valid":
            len(bootstrap_deltas),
    }


def ticker_bootstrap_mean_delta(
    ticker_results,
    reps=1000,
    seed=42,
):
    """
    Bootstrap the mean Delta AUC across tickers.

    This answers:

        Does the candidate signal beat persistence
        across assets, on average?

    Tickers are treated as the cross-sectional sampling unit.
    """

    clean = ticker_results.dropna(
        subset=[
            "baseline_auc",
            "signal_auc",
        ]
    ).copy()

    if len(clean) < 3:

        return {
            "macro_delta_ci_lower": np.nan,
            "macro_delta_ci_upper": np.nan,
        }

    deltas = (
        clean["signal_auc"].to_numpy()
        - clean["baseline_auc"].to_numpy()
    )

    n = len(deltas)

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = []

    for _ in range(reps):

        idx = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        bootstrap_means.append(
            np.mean(
                deltas[idx]
            )
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    return {
        "macro_delta_ci_lower":
            np.percentile(
                bootstrap_means,
                2.5,
            ),

        "macro_delta_ci_upper":
            np.percentile(
                bootstrap_means,
                97.5,
            ),
    }


# ============================================================
# START
# ============================================================

print("=" * 100)
print("SIGNAL VS BREACH-PERSISTENCE COMPARISON")
print("=" * 100)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading:")
print(INPUT_FILE)

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows:    {len(df):,}"
)

print(
    f"Tickers: {df['ticker'].nunique()}"
)


# ============================================================
# DATE COLUMN
# ============================================================
#
# Reliability panel uses realized_date.
#
# We normalize it internally to "Date" so the rest of the
# script can use a consistent variable.
# ============================================================

if "realized_date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["realized_date"],
        errors="coerce",
    )

elif "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
    )

else:

    raise ValueError(
        "Input file contains neither "
        "'realized_date' nor 'Date'."
    )


if df["Date"].isna().any():

    bad_dates = int(
        df["Date"].isna().sum()
    )

    raise ValueError(
        f"{bad_dates} rows have invalid dates."
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "ticker",
        "Date",
    ]
).reset_index(
    drop=True
)


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "ticker",
    "Date",
    BASELINE_COL,
]

required_columns.extend(
    SIGNALS.values()
)

for horizon in HORIZONS:

    required_columns.append(
        TARGET_TEMPLATE.format(
            horizon
        )
    )


missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(
            missing_columns
        )
    )


# ============================================================
# STORAGE
# ============================================================

comparison_rows = []

ticker_rows = []


# ============================================================
# MAIN EXPERIMENT
# ============================================================

for horizon in HORIZONS:

    target_col = TARGET_TEMPLATE.format(
        horizon
    )

    print("\n" + "=" * 100)
    print(
        f"HORIZON = {horizon} DAYS"
    )
    print("=" * 100)

    for signal_name, signal_col in SIGNALS.items():

        print(
            f"\nTesting: {signal_name}"
        )

        pooled_y = []

        pooled_baseline_scores = []

        pooled_signal_scores = []

        ticker_test_data = {}

        per_ticker_results = []

        eligible_tickers = 0

        # ====================================================
        # TICKER LOOP
        # ====================================================

        for ticker, group in df.groupby(
            "ticker",
            sort=True,
        ):

            g = group.sort_values(
                "Date"
            ).copy()

            # ------------------------------------------------
            # Keep exact common sample
            # ------------------------------------------------

            cols = [
                "Date",
                "ticker",
                BASELINE_COL,
                signal_col,
                target_col,
            ]

            g = g[cols].copy()

            # Replace infinities
            g = g.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            # Exact common sample
            g = g.dropna(
                subset=[
                    BASELINE_COL,
                    signal_col,
                    target_col,
                ]
            )

            # ------------------------------------------------
            # Minimum size
            # ------------------------------------------------

            if len(g) < (
                MIN_TRAIN
                + MIN_TEST
            ):
                continue

            # ------------------------------------------------
            # Chronological split
            # ------------------------------------------------

            split_idx = int(
                len(g)
                * TRAIN_FRACTION
            )

            train = g.iloc[
                :split_idx
            ].copy()

            test = g.iloc[
                split_idx:
            ].copy()

            if len(train) < MIN_TRAIN:
                continue

            if len(test) < MIN_TEST:
                continue

            # ------------------------------------------------
            # Need both target classes
            # ------------------------------------------------

            if train[target_col].nunique() < 2:
                continue

            if test[target_col].nunique() < 2:
                continue

            # =================================================
            # BASELINE MODEL
            # =================================================

            baseline_model = make_logistic()

            baseline_model.fit(
                train[
                    [BASELINE_COL]
                ],
                train[target_col],
            )

            baseline_score = (
                baseline_model.predict_proba(
                    test[
                        [BASELINE_COL]
                    ]
                )[:, 1]
            )

            # =================================================
            # SIGNAL MODEL
            # =================================================

            signal_model = make_logistic()

            signal_model.fit(
                train[
                    [signal_col]
                ],
                train[target_col],
            )

            signal_score = (
                signal_model.predict_proba(
                    test[
                        [signal_col]
                    ]
                )[:, 1]
            )

            y_test = test[
                target_col
            ].to_numpy()

            # =================================================
            # TICKER METRICS
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

            per_ticker_results.append(
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
            # POOLED OOS DATA
            # =================================================

            pooled_y.append(
                y_test
            )

            pooled_baseline_scores.append(
                baseline_score
            )

            pooled_signal_scores.append(
                signal_score
            )

            # =================================================
            # BOOTSTRAP DATA
            # =================================================

            ticker_test_data[ticker] = pd.DataFrame(
                {
                    "target":
                        y_test,

                    BASELINE_COL:
                        baseline_score,

                    signal_col:
                        signal_score,
                }
            )

            eligible_tickers += 1

        # ====================================================
        # NO ELIGIBLE TICKERS
        # ====================================================

        if eligible_tickers == 0:

            print(
                "  No eligible tickers."
            )

            continue

        # ====================================================
        # POOLED ARRAYS
        # ====================================================

        pooled_y_arr = np.concatenate(
            pooled_y
        )

        pooled_baseline_arr = np.concatenate(
            pooled_baseline_scores
        )

        pooled_signal_arr = np.concatenate(
            pooled_signal_scores
        )

        # ====================================================
        # POOLED AUC
        # ====================================================

        pooled_baseline_auc = safe_auc(
            pooled_y_arr,
            pooled_baseline_arr,
        )

        pooled_signal_auc = safe_auc(
            pooled_y_arr,
            pooled_signal_arr,
        )

        pooled_baseline_prauc = safe_prauc(
            pooled_y_arr,
            pooled_baseline_arr,
        )

        pooled_signal_prauc = safe_prauc(
            pooled_y_arr,
            pooled_signal_arr,
        )

        pooled_delta = (
            pooled_signal_auc
            - pooled_baseline_auc
        )

        # ====================================================
        # TICKER RESULTS
        # ====================================================

        ticker_results = pd.DataFrame(
            per_ticker_results
        )

        mean_baseline_auc = (
            ticker_results[
                "baseline_auc"
            ].mean()
        )

        mean_signal_auc = (
            ticker_results[
                "signal_auc"
            ].mean()
        )

        mean_delta_auc = (
            ticker_results[
                "delta_auc"
            ].mean()
        )

        n_signal_better = int(
            (
                ticker_results[
                    "delta_auc"
                ] > 0
            ).sum()
        )

        n_signal_worse = int(
            (
                ticker_results[
                    "delta_auc"
                ] < 0
            ).sum()
        )

        n_equal = int(
            (
                ticker_results[
                    "delta_auc"
                ] == 0
            ).sum()
        )

        # ====================================================
        # WITHIN-TICKER BLOCK BOOTSTRAP
        # ====================================================

        block_bootstrap = (
            block_bootstrap_delta_auc(
                ticker_test_data=
                    ticker_test_data,

                baseline_col=
                    BASELINE_COL,

                signal_col=
                    signal_col,

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
                    ticker_results,

                reps=
                    BOOTSTRAP_REPS,

                seed=
                    RANDOM_SEED
                    + horizon
                    + 1000,
            )
        )

        # ====================================================
        # SAVE SUMMARY
        # ====================================================

        comparison_rows.append(
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
                    len(pooled_y_arr),

                "baseline_pooled_auc":
                    pooled_baseline_auc,

                "signal_pooled_auc":
                    pooled_signal_auc,

                "delta_pooled_auc":
                    pooled_delta,

                "baseline_pooled_prauc":
                    pooled_baseline_prauc,

                "signal_pooled_prauc":
                    pooled_signal_prauc,

                "baseline_mean_ticker_auc":
                    mean_baseline_auc,

                "signal_mean_ticker_auc":
                    mean_signal_auc,

                "delta_mean_ticker_auc":
                    mean_delta_auc,

                "signal_better_tickers":
                    n_signal_better,

                "signal_worse_tickers":
                    n_signal_worse,

                "equal_tickers":
                    n_equal,

                # Proper within-ticker block bootstrap
                "delta_ci_lower":
                    block_bootstrap[
                        "delta_ci_lower"
                    ],

                "delta_ci_upper":
                    block_bootstrap[
                        "delta_ci_upper"
                    ],

                "bootstrap_prob_signal_better":
                    block_bootstrap[
                        "bootstrap_prob_signal_better"
                    ],

                "bootstrap_p_approx":
                    block_bootstrap[
                        "bootstrap_p_approx"
                    ],

                "bootstrap_reps_valid":
                    block_bootstrap[
                        "bootstrap_reps_valid"
                    ],

                # Cross-sectional ticker bootstrap
                "macro_delta_ci_lower":
                    macro_bootstrap[
                        "macro_delta_ci_lower"
                    ],

                "macro_delta_ci_upper":
                    macro_bootstrap[
                        "macro_delta_ci_upper"
                    ],
            }
        )

        # ====================================================
        # SAVE TICKER RESULTS
        # ====================================================

        ticker_rows.extend(
            per_ticker_results
        )

        # ====================================================
        # PRINT RESULTS
        # ====================================================

        ci_low = block_bootstrap[
            "delta_ci_lower"
        ]

        ci_high = block_bootstrap[
            "delta_ci_upper"
        ]

        print(
            f"  Eligible tickers = "
            f"{eligible_tickers}"
        )

        print(
            f"  Test observations = "
            f"{len(pooled_y_arr):,}"
        )

        print(
            f"  Persistence AUC = "
            f"{pooled_baseline_auc:.4f}"
        )

        print(
            f"  Signal AUC      = "
            f"{pooled_signal_auc:.4f}"
        )

        print(
            f"  Delta AUC       = "
            f"{pooled_delta:+.4f}"
        )

        print(
            f"  Block Δ CI      = "
            f"[{ci_low:+.4f}, "
            f"{ci_high:+.4f}]"
        )

        print(
            f"  Mean ticker Δ   = "
            f"{mean_delta_auc:+.4f}"
        )

        print(
            f"  Better tickers  = "
            f"{n_signal_better}/"
            f"{eligible_tickers}"
        )


# ============================================================
# SAVE DATAFRAMES
# ============================================================

comparison_df = pd.DataFrame(
    comparison_rows
)

ticker_df = pd.DataFrame(
    ticker_rows
)

if len(comparison_df) == 0:

    raise RuntimeError(
        "No comparison results were generated."
    )


comparison_df = comparison_df.sort_values(
    [
        "horizon",
        "delta_pooled_auc",
    ],
    ascending=[
        True,
        False,
    ],
)


ticker_df = ticker_df.sort_values(
    [
        "horizon",
        "signal_name",
        "ticker",
    ]
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


comparison_df.to_csv(
    OUTPUT_COMPARISON,
    index=False,
)


ticker_df.to_csv(
    OUTPUT_TICKER,
    index=False,
)


# ============================================================
# PRINT FULL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("SUMMARY — SIGNAL VS PERSISTENCE")
print("=" * 100)


for horizon in HORIZONS:

    print(
        f"\n{horizon}-DAY"
    )

    sub = comparison_df[
        comparison_df["horizon"]
        == horizon
    ].copy()

    sub = sub.sort_values(
        "delta_pooled_auc",
        ascending=False,
    )

    display_cols = [
        "signal_name",
        "baseline_pooled_auc",
        "signal_pooled_auc",
        "delta_pooled_auc",
        "delta_ci_lower",
        "delta_ci_upper",
        "delta_mean_ticker_auc",
        "signal_better_tickers",
        "n_tickers",
    ]

    print(
        sub[display_cols].to_string(
            index=False
        )
    )


# ============================================================
# SIGNALS WITH POSITIVE BLOCK-BOOTSTRAP CI
# ============================================================

print("\n" + "=" * 100)
print(
    "SIGNALS WHOSE WITHIN-TICKER BLOCK CI "
    "IS ENTIRELY ABOVE ZERO"
)
print("=" * 100)


for horizon in HORIZONS:

    sub = comparison_df[
        comparison_df["horizon"]
        == horizon
    ].copy()

    significant = sub[
        sub["delta_ci_lower"] > 0
    ].sort_values(
        "delta_pooled_auc",
        ascending=False,
    )

    print(
        f"\n{horizon}-DAY:"
    )

    if len(significant) == 0:

        print(
            "  None"
        )

    else:

        for _, row in significant.iterrows():

            print(
                f"  {row['signal_name']}: "
                f"ΔAUC="
                f"{row['delta_pooled_auc']:+.4f}, "
                f"CI=["
                f"{row['delta_ci_lower']:+.4f}, "
                f"{row['delta_ci_upper']:+.4f}]"
            )


# ============================================================
# SIGNALS WHOSE MACRO TICKER CI IS ABOVE ZERO
# ============================================================

print("\n" + "=" * 100)
print(
    "SIGNALS WHOSE CROSS-SECTIONAL TICKER CI "
    "IS ENTIRELY ABOVE ZERO"
)
print("=" * 100)


for horizon in HORIZONS:

    sub = comparison_df[
        comparison_df["horizon"]
        == horizon
    ].copy()

    significant = sub[
        sub["macro_delta_ci_lower"] > 0
    ].sort_values(
        "delta_mean_ticker_auc",
        ascending=False,
    )

    print(
        f"\n{horizon}-DAY:"
    )

    if len(significant) == 0:

        print(
            "  None"
        )

    else:

        for _, row in significant.iterrows():

            print(
                f"  {row['signal_name']}: "
                f"Mean ΔAUC="
                f"{row['delta_mean_ticker_auc']:+.4f}, "
                f"CI=["
                f"{row['macro_delta_ci_lower']:+.4f}, "
                f"{row['macro_delta_ci_upper']:+.4f}]"
            )


# ============================================================
# OUTPUT PATHS
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_COMPARISON
)

print(
    OUTPUT_TICKER
)


print("\n" + "=" * 100)
print(
    "SIGNAL VS PERSISTENCE COMPARISON COMPLETE"
)
print("=" * 100)