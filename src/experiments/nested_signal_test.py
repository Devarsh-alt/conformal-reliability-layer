from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "monitoring"
    / "pooled_hs_reliability_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "nested_signal_oos_predictions.csv"
)

RESULTS_OUTPUT = (
    OUTPUT_DIR
    / "nested_signal_results.csv"
)

BOOTSTRAP_OUTPUT = (
    OUTPUT_DIR
    / "nested_signal_block_bootstrap.csv"
)

ALPHA = 0.05

# ------------------------------------------------------------
# Target definition
# ------------------------------------------------------------
#
# We use future pinball loss, but normalize it by the
# current HS VaR magnitude so the target is less dominated
# by inherently more volatile tickers.
#
# Target:
#
#     future_pinball_h / |HS_VaR_t|
#
# Then "failure" = unusually high relative future loss.
#
# The 75th percentile threshold is estimated separately
# for EACH TICKER using TRAINING DATA ONLY in each fold.
# ------------------------------------------------------------

TARGET_QUANTILE = 0.75

# ------------------------------------------------------------
# Walk-forward years
# ------------------------------------------------------------

TEST_YEARS = list(range(2018, 2027))

# Number of observations to embargo at the end of training
# for a given horizon.
#
# This prevents training labels whose future window overlaps
# the first test observations.
#
# Maximum horizon = 20.
# ------------------------------------------------------------

N_BOOTSTRAP = 250

BLOCK_LENGTH = 20

RANDOM_SEED = 42


# ============================================================
# Feature definitions
# ============================================================

# ------------------------------------------------------------
# Persistence baseline
# ------------------------------------------------------------

PERSISTENCE = [
    "breach_rate_20d"
]

# ------------------------------------------------------------
# Strong simple baselines
# ------------------------------------------------------------

EWMA_ONLY = [
    "ewma_breach_rate"
]

CUSUM_ONLY = [
    "cusum_breach"
]

VOLATILITY_ONLY = [
    "realized_vol_20d"
]

# ------------------------------------------------------------
# Nested reliability comparisons
# ------------------------------------------------------------

PERSISTENCE_EWMA = [
    "breach_rate_20d",
    "ewma_breach_rate",
]

PERSISTENCE_CUSUM = [
    "breach_rate_20d",
    "cusum_breach",
]

PERSISTENCE_VOL = [
    "breach_rate_20d",
    "realized_vol_20d",
]

PERSISTENCE_KS = [
    "breach_rate_20d",
    "ks_stat",
]

PERSISTENCE_DISAGREEMENT = [
    "breach_rate_20d",
    "bootstrap_var_disagreement_pct",
]

PERSISTENCE_KS_DISAGREEMENT = [
    "breach_rate_20d",
    "ks_stat",
    "bootstrap_var_disagreement_pct",
]

# ------------------------------------------------------------
# Full predefined comparison.
#
# IMPORTANT:
# We do NOT select a winner based on test performance.
# Every model below is evaluated exactly as specified.
# ------------------------------------------------------------

MODEL_FEATURES = {
    "persistence": PERSISTENCE,

    "ewma": EWMA_ONLY,

    "cusum": CUSUM_ONLY,

    "volatility": VOLATILITY_ONLY,

    "persistence_plus_ewma":
        PERSISTENCE_EWMA,

    "persistence_plus_cusum":
        PERSISTENCE_CUSUM,

    "persistence_plus_volatility":
        PERSISTENCE_VOL,

    "persistence_plus_ks":
        PERSISTENCE_KS,

    "persistence_plus_disagreement":
        PERSISTENCE_DISAGREEMENT,

    "persistence_plus_ks_disagreement":
        PERSISTENCE_KS_DISAGREEMENT,
}


# ============================================================
# Helpers
# ============================================================

def make_logistic():
    """
    Standardized logistic regression.

    We use this only for multi-feature combinations.
    """

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                C=1.0,
                random_state=42
            )
        )
    ])


def get_training_rows(
    data,
    test_year,
    horizon,
):
    """
    Expanding-window training set.

    For every ticker, remove the final `horizon` observations
    before the test-year boundary.

    This is a row-based embargo within ticker, rather than
    a calendar-day approximation.
    """

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    train_parts = []

    for ticker, group in data.groupby(
        "ticker",
        sort=False
    ):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        before_test = group[
            group["date"] < test_start
        ].copy()

        if len(before_test) <= horizon:
            continue

        # Embargo final horizon rows.
        before_test = before_test.iloc[
            :-horizon
        ].copy()

        train_parts.append(
            before_test
        )

    if not train_parts:
        return data.iloc[0:0].copy()

    return pd.concat(
        train_parts,
        ignore_index=True
    )


def get_test_rows(
    data,
    test_year,
):
    """
    All observations belonging to the given calendar year.
    """

    start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    end = pd.Timestamp(
        f"{test_year + 1}-01-01"
    )

    return data[
        (data["date"] >= start)
        & (data["date"] < end)
    ].copy()


def add_relative_target(
    train,
    test,
    horizon,
):
    """
    Construct an out-of-sample binary failure target.

    First:

        relative future loss =
            future_pinball / current |VaR|

    Then:

        threshold_ticker =
            75th percentile of relative future loss
            using TRAINING DATA ONLY.

    Finally:

        target = 1 if future relative loss > threshold.
    """

    target_raw = (
        f"future_pinball_{horizon}d"
    )

    train = train.copy()
    test = test.copy()

    train["_relative_future_loss"] = (
        train[target_raw]
        / np.maximum(
            train["var_abs"],
            1e-8
        )
    )

    test["_relative_future_loss"] = (
        test[target_raw]
        / np.maximum(
            test["var_abs"],
            1e-8
        )
    )

    thresholds = {}

    train_targets = []
    test_targets = []

    # --------------------------------------------------------
    # Threshold separately by ticker.
    # --------------------------------------------------------

    for ticker in sorted(
        train["ticker"].unique()
    ):

        train_ticker = train[
            train["ticker"] == ticker
        ]

        threshold = (
            train_ticker[
                "_relative_future_loss"
            ]
            .quantile(
                TARGET_QUANTILE
            )
        )

        thresholds[ticker] = threshold

    # --------------------------------------------------------
    # Apply thresholds.
    # --------------------------------------------------------

    train["_threshold"] = (
        train["ticker"]
        .map(thresholds)
    )

    test["_threshold"] = (
        test["ticker"]
        .map(thresholds)
    )

    train["target"] = (
        train["_relative_future_loss"]
        > train["_threshold"]
    ).astype(int)

    test["target"] = (
        test["_relative_future_loss"]
        > test["_threshold"]
    ).astype(int)

    return train, test, thresholds


def auc_score(y, score):
    """
    Safe AUC helper.
    """

    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(
        y,
        score
    )


def block_bootstrap_dates(
    predictions,
    delta_models,
    n_bootstrap,
    block_length,
    seed,
):
    """
    Calendar-date moving block bootstrap.

    This preserves:

    1. temporal dependence
    2. cross-sectional dependence across tickers

    because every selected date block brings along all tickers
    available on those dates.
    """

    rng = np.random.default_rng(
        seed
    )

    work = predictions.copy()

    work["date"] = pd.to_datetime(
        work["date"]
    )

    unique_dates = np.array(
        sorted(
            work["date"].unique()
        )
    )

    n_dates = len(unique_dates)

    if n_dates < block_length:
        raise ValueError(
            "Not enough dates for block bootstrap."
        )

    # --------------------------------------------------------
    # Create overlapping blocks.
    # --------------------------------------------------------

    blocks = []

    for start in range(
        0,
        n_dates - block_length + 1
    ):

        block_dates = unique_dates[
            start:
            start + block_length
        ]

        blocks.append(
            block_dates
        )

    # --------------------------------------------------------
    # Run bootstrap.
    # --------------------------------------------------------

    records = []

    n_blocks_needed = int(
        np.ceil(
            n_dates / block_length
        )
    )

    for b in range(
        n_bootstrap
    ):

        sampled_blocks = rng.integers(
            low=0,
            high=len(blocks),
            size=n_blocks_needed
        )

        sampled_dates = []

        for idx in sampled_blocks:

            sampled_dates.extend(
                blocks[idx]
            )

        sampled_dates = np.array(
            sampled_dates[:n_dates]
        )

        boot = work[
            work["date"]
            .isin(sampled_dates)
        ].copy()

        # ----------------------------------------------------
        # Because selected dates are unique after isin(),
        # repeated blocks collapse. To retain bootstrap
        # multiplicity, construct row copies directly.
        # ----------------------------------------------------

        pieces = []

        for idx in sampled_blocks:

            block_dates = blocks[idx]

            piece = work[
                work["date"]
                .isin(block_dates)
            ].copy()

            pieces.append(
                piece
            )

        boot = pd.concat(
            pieces,
            ignore_index=True
        )

        for model_name in delta_models:

            challenger = (
                model_name
            )

            baseline = (
                "persistence"
            )

            y = boot["target"].to_numpy()

            auc_base = auc_score(
                y,
                boot[
                    baseline
                ].to_numpy()
            )

            auc_challenger = auc_score(
                y,
                boot[
                    challenger
                ].to_numpy()
            )

            if (
                np.isfinite(auc_base)
                and np.isfinite(
                    auc_challenger
                )
            ):

                records.append({
                    "bootstrap":
                        b,

                    "comparison":
                        f"{challenger} - {baseline}",

                    "auc_delta":
                        auc_challenger
                        - auc_base,
                })

    boot_df = pd.DataFrame(
        records
    )

    return boot_df


# ============================================================
# Load panel
# ============================================================

df = pd.read_csv(
    INPUT
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values(
        ["date", "ticker"]
    )
    .reset_index(drop=True)
)


print("=" * 80)
print("NESTED HS-VaR RELIABILITY SIGNAL TEST")
print("=" * 80)

print(
    "Rows:",
    len(df)
)

print(
    "Tickers:",
    df["ticker"].nunique()
)

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)


# ============================================================
# Run expanding walk-forward
# ============================================================

all_predictions = []
fold_results = []

for horizon in [5, 10, 20]:

    print("\n" + "=" * 80)
    print(
        f"HORIZON = {horizon} DAYS"
    )
    print("=" * 80)

    for test_year in TEST_YEARS:

        train_raw = get_training_rows(
            df,
            test_year,
            horizon
        )

        test_raw = get_test_rows(
            df,
            test_year
        )

        if (
            len(train_raw) == 0
            or len(test_raw) == 0
        ):
            continue

        # ----------------------------------------------------
        # Construct target using train-only thresholds.
        # ----------------------------------------------------

        train, test, thresholds = (
            add_relative_target(
                train_raw,
                test_raw,
                horizon
            )
        )

        # ----------------------------------------------------
        # Keep only tickers for which a training threshold
        # exists and both target classes are observed.
        # ----------------------------------------------------

        usable_tickers = []

        for ticker in sorted(
            test["ticker"].unique()
        ):

            tr = train[
                train["ticker"] == ticker
            ]

            te = test[
                test["ticker"] == ticker
            ]

            if len(tr) < 250:
                continue

            if len(
                tr["target"].unique()
            ) < 2:
                continue

            if len(
                te["target"].unique()
            ) < 2:
                continue

            usable_tickers.append(
                ticker
            )

        train = train[
            train["ticker"].isin(
                usable_tickers
            )
        ].copy()

        test = test[
            test["ticker"].isin(
                usable_tickers
            )
        ].copy()

        if len(test) < 100:
            continue

        print(
            f"{test_year}: "
            f"train={len(train):,} "
            f"test={len(test):,} "
            f"tickers={len(usable_tickers)}"
        )

        # ----------------------------------------------------
        # Evaluate every predefined model.
        # ----------------------------------------------------

        fold_prediction = test[
            [
                "date",
                "ticker",
                "target",
            ]
        ].copy()

        # ----------------------------------------------------
        # Persistence score.
        # No fitting needed.
        # ----------------------------------------------------

        fold_prediction[
            "persistence"
        ] = test[
            "breach_rate_20d"
        ].to_numpy()

        # ----------------------------------------------------
        # Simple standalone baselines.
        # ----------------------------------------------------

        fold_prediction[
            "ewma"
        ] = test[
            "ewma_breach_rate"
        ].to_numpy()

        fold_prediction[
            "cusum"
        ] = test[
            "cusum_breach"
        ].to_numpy()

        fold_prediction[
            "volatility"
        ] = test[
            "realized_vol_20d"
        ].to_numpy()

        # ----------------------------------------------------
        # Multi-feature models.
        # ----------------------------------------------------

        for model_name, features in (
            MODEL_FEATURES.items()
        ):

            if model_name in [
                "persistence",
                "ewma",
                "cusum",
                "volatility",
            ]:
                continue

            # -----------------------------------------------
            # Training matrix
            # -----------------------------------------------

            X_train = train[
                features
            ]

            y_train = train[
                "target"
            ]

            X_test = test[
                features
            ]

            # -----------------------------------------------
            # Fit logistic model
            # -----------------------------------------------

            model = make_logistic()

            model.fit(
                X_train,
                y_train
            )

            probability = (
                model
                .predict_proba(
                    X_test
                )[:, 1]
            )

            fold_prediction[
                model_name
            ] = probability

        # ----------------------------------------------------
        # Store predictions.
        # ----------------------------------------------------

        fold_prediction[
            "horizon"
        ] = horizon

        fold_prediction[
            "test_year"
        ] = test_year

        all_predictions.append(
            fold_prediction
        )

        # ----------------------------------------------------
        # Fold metrics.
        # ----------------------------------------------------

        for model_name in MODEL_FEATURES:

            if model_name not in (
                fold_prediction.columns
            ):
                continue

            score = auc_score(
                fold_prediction[
                    "target"
                ],
                fold_prediction[
                    model_name
                ]
            )

            fold_results.append({
                "horizon":
                    horizon,

                "test_year":
                    test_year,

                "model":
                    model_name,

                "n":
                    len(
                        fold_prediction
                    ),

                "positive_rate":
                    fold_prediction[
                        "target"
                    ].mean(),

                "auc":
                    score,
            })


# ============================================================
# Combine OOS predictions
# ============================================================

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

fold_results_df = pd.DataFrame(
    fold_results
)


# ============================================================
# Pooled OOS results
# ============================================================

pooled_results = []

for horizon in [5, 10, 20]:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ].copy()

    for model_name in MODEL_FEATURES:

        if model_name not in (
            subset.columns
        ):
            continue

        auc = auc_score(
            subset["target"],
            subset[model_name]
        )

        pooled_results.append({
            "horizon":
                horizon,

            "model":
                model_name,

            "n":
                len(subset),

            "positive_rate":
                subset["target"].mean(),

            "pooled_auc":
                auc,
        })


pooled_results_df = pd.DataFrame(
    pooled_results
)


# ============================================================
# Compute deltas against persistence
# ============================================================

for horizon in [5, 10, 20]:

    mask = (
        pooled_results_df["horizon"]
        == horizon
    )

    baseline_auc = (
        pooled_results_df.loc[
            mask
            & (
                pooled_results_df[
                    "model"
                ]
                == "persistence"
            ),
            "pooled_auc"
        ].iloc[0]
    )

    pooled_results_df.loc[
        mask,
        "delta_vs_persistence"
    ] = (
        pooled_results_df.loc[
            mask,
            "pooled_auc"
        ]
        - baseline_auc
    )


# ============================================================
# Per-ticker OOS AUC
# ============================================================

ticker_results = []

for horizon in [5, 10, 20]:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ]

    for ticker, group in subset.groupby(
        "ticker"
    ):

        for model_name in MODEL_FEATURES:

            if model_name not in (
                group.columns
            ):
                continue

            auc = auc_score(
                group["target"],
                group[model_name]
            )

            ticker_results.append({
                "horizon":
                    horizon,

                "ticker":
                    ticker,

                "model":
                    model_name,

                "n":
                    len(group),

                "auc":
                    auc,
            })


ticker_results_df = pd.DataFrame(
    ticker_results
)


# ============================================================
# Per-ticker deltas
# ============================================================

baseline_ticker = (
    ticker_results_df[
        ticker_results_df["model"]
        == "persistence"
    ][
        [
            "horizon",
            "ticker",
            "auc",
        ]
    ]
    .rename(
        columns={
            "auc":
                "baseline_auc"
        }
    )
)

ticker_results_df = (
    ticker_results_df
    .merge(
        baseline_ticker,
        on=[
            "horizon",
            "ticker",
        ],
        how="left"
    )
)

ticker_results_df[
    "delta_vs_persistence"
] = (
    ticker_results_df["auc"]
    - ticker_results_df["baseline_auc"]
)


# ============================================================
# Block bootstrap for pooled AUC deltas
# ============================================================

print("\n" + "=" * 80)
print("BLOCK BOOTSTRAP")
print("=" * 80)

bootstrap_results = []

challengers = [
    "ewma",
    "cusum",
    "volatility",
    "persistence_plus_ewma",
    "persistence_plus_cusum",
    "persistence_plus_volatility",
    "persistence_plus_ks",
    "persistence_plus_disagreement",
    "persistence_plus_ks_disagreement",
]

for horizon in [5, 10, 20]:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ].copy()

    if len(subset) < 500:
        continue

    valid_challengers = [
        c
        for c in challengers
        if c in subset.columns
    ]

    boot = block_bootstrap_dates(
        subset,
        valid_challengers,
        N_BOOTSTRAP,
        BLOCK_LENGTH,
        RANDOM_SEED + horizon
    )

    for comparison, group in (
        boot.groupby(
            "comparison"
        )
    ):

        values = (
            group["auc_delta"]
            .dropna()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        ci_low = np.quantile(
            values,
            0.025
        )

        ci_high = np.quantile(
            values,
            0.975
        )

        probability_positive = (
            np.mean(
                values > 0
            )
        )

        bootstrap_results.append({
            "horizon":
                horizon,

            "comparison":
                comparison,

            "n_bootstrap":
                len(values),

            "delta_mean":
                values.mean(),

            "ci_2.5":
                ci_low,

            "ci_97.5":
                ci_high,

            "probability_delta_gt_0":
                probability_positive,
        })

        print(
            f"{horizon:2d}d  "
            f"{comparison:45s} "
            f"mean={values.mean():+.4f} "
            f"CI=[{ci_low:+.4f}, "
            f"{ci_high:+.4f}]"
        )


bootstrap_results_df = (
    pd.DataFrame(
        bootstrap_results
    )
)


# ============================================================
# Save outputs
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

predictions.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

# Pooled results + fold information.
pooled_results_df.to_csv(
    RESULTS_OUTPUT,
    index=False
)

bootstrap_results_df.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False
)


# ============================================================
# Print final pooled table
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\n" + "=" * 80)
print("POOLED OOS RESULTS")
print("=" * 80)

print(
    pooled_results_df[
        [
            "horizon",
            "model",
            "n",
            "positive_rate",
            "pooled_auc",
            "delta_vs_persistence",
        ]
    ]
    .sort_values(
        [
            "horizon",
            "pooled_auc",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Print fold results
# ============================================================

print("\n" + "=" * 80)
print("WALK-FORWARD FOLD RESULTS")
print("=" * 80)

print(
    fold_results_df[
        [
            "horizon",
            "test_year",
            "model",
            "n",
            "positive_rate",
            "auc",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Macro per-ticker summary
# ============================================================

macro = (
    ticker_results_df
    .groupby(
        [
            "horizon",
            "model",
        ]
    )
    .agg(
        mean_auc=(
            "auc",
            "mean"
        ),

        median_auc=(
            "auc",
            "median"
        ),

        mean_delta=(
            "delta_vs_persistence",
            "mean"
        ),

        median_delta=(
            "delta_vs_persistence",
            "median"
        ),

        tickers=(
            "ticker",
            "nunique"
        ),
    )
    .reset_index()
)

print("\n" + "=" * 80)
print("PER-TICKER MACRO RESULTS")
print("=" * 80)

print(
    macro.to_string(
        index=False
    )
)


# ============================================================
# Positive-delta frequency
# ============================================================

print("\n" + "=" * 80)
print("TICKERS WHERE CHALLENGER BEATS PERSISTENCE")
print("=" * 80)

for horizon in [5, 10, 20]:

    subset = ticker_results_df[
        ticker_results_df["horizon"]
        == horizon
    ]

    print(
        f"\n{horizon}d"
    )

    for model_name in MODEL_FEATURES:

        if model_name == "persistence":
            continue

        model_subset = subset[
            subset["model"]
            == model_name
        ]

        if len(model_subset) == 0:
            continue

        positive = (
            model_subset[
                "delta_vs_persistence"
            ]
            > 0
        ).sum()

        total = len(
            model_subset
        )

        print(
            f"  {model_name:40s} "
            f"{positive}/{total} "
            f"({positive / total:.1%})"
        )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    "Predictions:",
    PREDICTIONS_OUTPUT
)

print(
    "Results:",
    RESULTS_OUTPUT
)

print(
    "Bootstrap:",
    BOOTSTRAP_OUTPUT
)