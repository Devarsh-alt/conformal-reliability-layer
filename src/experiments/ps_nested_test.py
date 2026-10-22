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

# ------------------------------------------------------------
# Main HS-VaR panel
# ------------------------------------------------------------

HS_PANEL = (
    ROOT
    / "data"
    / "monitoring"
    / "pooled_hs_reliability_panel.csv"
)

# ------------------------------------------------------------
# Existing PS dataset from the original project
# ------------------------------------------------------------

PS_FILE = (
    ROOT
    / "data"
    / "reliability"
    / "perturbation_sensitivity"
    / "candidate_dataset_with_ps.csv"
)

# ------------------------------------------------------------
# Outputs
# ------------------------------------------------------------

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
)

PREDICTIONS_OUTPUT = (
    OUTPUT_DIR
    / "ps_nested_oos_predictions.csv"
)

RESULTS_OUTPUT = (
    OUTPUT_DIR
    / "ps_nested_results.csv"
)

BOOTSTRAP_OUTPUT = (
    OUTPUT_DIR
    / "ps_nested_block_bootstrap.csv"
)

# ============================================================
# Target / evaluation settings
# ============================================================

TARGET_QUANTILE = 0.75

HORIZONS = [5, 10, 20]

TEST_YEARS = list(
    range(2024, 2027)
)

# Block bootstrap
N_BOOTSTRAP = 500
BLOCK_LENGTH = 20

RANDOM_SEED = 42


# ============================================================
# Helper functions
# ============================================================

def make_logistic():
    """
    Standardized logistic regression.
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


def auc_score(y, score):
    """
    Safe ROC-AUC calculation.
    """

    y = np.asarray(y)
    score = np.asarray(score)

    if len(np.unique(y)) < 2:
        return np.nan

    return roc_auc_score(
        y,
        score
    )


def construct_target(
    train,
    test,
    horizon
):
    """
    Build a ticker-specific future deterioration target.

    Relative future loss:

        future pinball loss / current |VaR|

    Failure threshold:

        75th percentile within each ticker,
        estimated from TRAINING DATA ONLY.
    """

    column = (
        f"future_pinball_{horizon}d"
    )

    train = train.copy()
    test = test.copy()

    train["_relative_loss"] = (
        train[column]
        / np.maximum(
            train["var_abs"],
            1e-8
        )
    )

    test["_relative_loss"] = (
        test[column]
        / np.maximum(
            test["var_abs"],
            1e-8
        )
    )

    thresholds = (
        train
        .groupby("ticker")[
            "_relative_loss"
        ]
        .quantile(
            TARGET_QUANTILE
        )
        .to_dict()
    )

    train["_threshold"] = (
        train["ticker"]
        .map(thresholds)
    )

    test["_threshold"] = (
        test["ticker"]
        .map(thresholds)
    )

    train["target"] = (
        train["_relative_loss"]
        > train["_threshold"]
    ).astype(int)

    test["target"] = (
        test["_relative_loss"]
        > test["_threshold"]
    ).astype(int)

    return train, test


def block_bootstrap_delta(
    predictions,
    baseline_column,
    challenger_column,
    n_bootstrap,
    block_length,
    seed
):
    """
    Moving-block bootstrap over calendar dates.

    Every date brings all available tickers with it,
    preserving both temporal and cross-sectional dependence.
    """

    rng = np.random.default_rng(
        seed
    )

    data = predictions.copy()

    data["date"] = pd.to_datetime(
        data["date"]
    )

    dates = np.array(
        sorted(
            data["date"].unique()
        )
    )

    n_dates = len(dates)

    if n_dates < block_length:
        raise ValueError(
            "Not enough dates for block bootstrap."
        )

    # --------------------------------------------------------
    # Overlapping blocks
    # --------------------------------------------------------

    blocks = []

    for start in range(
        n_dates - block_length + 1
    ):

        blocks.append(
            dates[
                start:
                start + block_length
            ]
        )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    deltas = []

    n_blocks = int(
        np.ceil(
            n_dates
            / block_length
        )
    )

    for _ in range(
        n_bootstrap
    ):

        sampled_indices = rng.integers(
            0,
            len(blocks),
            size=n_blocks
        )

        pieces = []

        for idx in sampled_indices:

            piece = data[
                data["date"].isin(
                    blocks[idx]
                )
            ].copy()

            pieces.append(
                piece
            )

        boot = pd.concat(
            pieces,
            ignore_index=True
        )

        base_auc = auc_score(
            boot["target"],
            boot[baseline_column]
        )

        challenger_auc = auc_score(
            boot["target"],
            boot[challenger_column]
        )

        if (
            np.isfinite(base_auc)
            and np.isfinite(
                challenger_auc
            )
        ):

            deltas.append(
                challenger_auc
                - base_auc
            )

    return np.asarray(
        deltas
    )


# ============================================================
# Load HS panel
# ============================================================

print("=" * 80)
print("PS NESTED RELIABILITY TEST")
print("=" * 80)

hs = pd.read_csv(
    HS_PANEL
)

hs["date"] = pd.to_datetime(
    hs["date"]
)

hs = (
    hs
    .sort_values(
        ["date", "ticker"]
    )
    .reset_index(drop=True)
)

print(
    "HS panel rows:",
    len(hs)
)

print(
    "HS tickers:",
    hs["ticker"].nunique()
)

print(
    "HS date range:",
    hs["date"].min().date(),
    "to",
    hs["date"].max().date()
)


# ============================================================
# Load existing PS
# ============================================================

print("\n" + "=" * 80)
print("LOADING EXISTING PS")
print("=" * 80)

ps = pd.read_csv(
    PS_FILE
)

ps["prediction_date"] = pd.to_datetime(
    ps["prediction_date"]
)

ps = (
    ps[
        [
            "prediction_date",
            "ticker",
            "ps_raw",
            "ps_pct",
        ]
    ]
    .rename(
        columns={
            "prediction_date": "date"
        }
    )
)

print(
    "PS rows:",
    len(ps)
)

print(
    "PS tickers:",
    ps["ticker"].nunique()
)

print(
    "PS date range:",
    ps["date"].min().date(),
    "to",
    ps["date"].max().date()
)


# ============================================================
# Check duplicate keys
# ============================================================

duplicates = (
    ps
    .duplicated(
        ["date", "ticker"]
    )
    .sum()
)

print(
    "Duplicate date/ticker PS rows:",
    duplicates
)


# ============================================================
# Merge
# ============================================================

merged = hs.merge(
    ps,
    on=[
        "date",
        "ticker",
    ],
    how="inner"
)

merged = (
    merged
    .sort_values(
        ["date", "ticker"]
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 80)
print("OVERLAP")
print("=" * 80)

print(
    "Merged rows:",
    len(merged)
)

print(
    "Merged tickers:",
    merged["ticker"].nunique()
)

print(
    "Merged date range:",
    merged["date"].min().date(),
    "to",
    merged["date"].max().date()
)

print(
    "PS missing:",
    merged[
        "ps_pct"
    ].isna().sum()
)

print(
    "PS descriptive statistics:"
)

print(
    merged[
        [
            "ps_raw",
            "ps_pct",
        ]
    ]
    .describe()
    .to_string()
)


# ============================================================
# Important alignment check
# ============================================================

print("\n" + "=" * 80)
print("TICKER OVERLAP")
print("=" * 80)

hs_tickers = set(
    hs["ticker"].unique()
)

ps_tickers = set(
    ps["ticker"].unique()
)

print(
    "Only in HS panel:",
    sorted(
        hs_tickers - ps_tickers
    )
)

print(
    "Only in PS:",
    sorted(
        ps_tickers - hs_tickers
    )
)


# ============================================================
# Model definitions
# ============================================================

# ------------------------------------------------------------
# Strong baseline from previous experiment
# ------------------------------------------------------------

BASELINE_FEATURES = [
    "breach_rate_20d",
    "ewma_breach_rate",
]

# ------------------------------------------------------------
# Add PS
# ------------------------------------------------------------

BASELINE_PLUS_PS = [
    "breach_rate_20d",
    "ewma_breach_rate",
    "ps_pct",
]

# ------------------------------------------------------------
# Add PS only to persistence
# ------------------------------------------------------------

PERSISTENCE_ONLY = [
    "breach_rate_20d",
]

PERSISTENCE_PLUS_PS = [
    "breach_rate_20d",
    "ps_pct",
]


MODEL_FEATURES = {
    "persistence":
        PERSISTENCE_ONLY,

    "persistence_plus_ps":
        PERSISTENCE_PLUS_PS,

    "persistence_plus_ewma":
        BASELINE_FEATURES,

    "persistence_plus_ewma_plus_ps":
        BASELINE_PLUS_PS,
}


# ============================================================
# Walk-forward experiment
# ============================================================

oos_predictions = []
fold_results = []


for horizon in HORIZONS:

    print("\n" + "=" * 80)
    print(
        f"HORIZON = {horizon} DAYS"
    )
    print("=" * 80)

    for test_year in TEST_YEARS:

        test_start = pd.Timestamp(
            f"{test_year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{test_year + 1}-01-01"
        )

        # ----------------------------------------------------
        # Test data
        # ----------------------------------------------------

        test = merged[
            (
                merged["date"]
                >= test_start
            )
            &
            (
                merged["date"]
                < test_end
            )
        ].copy()

        # ----------------------------------------------------
        # Training data
        #
        # Must end at least `horizon` observations before
        # test start, separately within each ticker.
        # ----------------------------------------------------

        train_parts = []

        for ticker, group in merged.groupby(
            "ticker",
            sort=False
        ):

            group = (
                group
                .sort_values("date")
                .reset_index(drop=True)
            )

            before_test = group[
                group["date"]
                < test_start
            ].copy()

            if len(before_test) <= horizon:
                continue

            before_test = (
                before_test
                .iloc[:-horizon]
                .copy()
            )

            train_parts.append(
                before_test
            )

        if not train_parts:
            continue

        train = pd.concat(
            train_parts,
            ignore_index=True
        )

        # ----------------------------------------------------
        # Build target.
        # ----------------------------------------------------

        train, test = construct_target(
            train,
            test,
            horizon
        )

        # ----------------------------------------------------
        # Keep observations with PS present.
        # ----------------------------------------------------

        required = list(
            set(
                BASELINE_PLUS_PS
                + [
                    "target"
                ]
            )
        )

        train = train.dropna(
            subset=required
        ).copy()

        test = test.dropna(
            subset=required
        ).copy()

        # ----------------------------------------------------
        # Require both target classes in training/testing.
        # ----------------------------------------------------

        if (
            len(train) < 500
            or len(test) < 100
        ):
            continue

        if (
            train["target"]
            .nunique() < 2
        ):
            continue

        if (
            test["target"]
            .nunique() < 2
        ):
            continue

        print(
            f"{test_year}: "
            f"train={len(train):,} "
            f"test={len(test):,} "
            f"tickers={test['ticker'].nunique()}"
        )

        # ----------------------------------------------------
        # Prediction dataframe
        # ----------------------------------------------------

        pred = test[
            [
                "date",
                "ticker",
                "target",
            ]
        ].copy()

        # ----------------------------------------------------
        # Raw persistence
        # ----------------------------------------------------

        pred["persistence"] = (
            test[
                "breach_rate_20d"
            ].to_numpy()
        )

        # ----------------------------------------------------
        # Fit multi-feature models
        # ----------------------------------------------------

        for model_name, features in (
            MODEL_FEATURES.items()
        ):

            # persistence is direct, not fitted
            if model_name == "persistence":
                continue

            model = make_logistic()

            model.fit(
                train[features],
                train["target"]
            )

            probability = (
                model
                .predict_proba(
                    test[features]
                )[:, 1]
            )

            pred[
                model_name
            ] = probability

        pred["horizon"] = horizon
        pred["test_year"] = test_year

        oos_predictions.append(
            pred
        )

        # ----------------------------------------------------
        # Fold-level AUC
        # ----------------------------------------------------

        for model_name in MODEL_FEATURES:

            if model_name not in pred.columns:
                continue

            fold_auc = auc_score(
                pred["target"],
                pred[model_name]
            )

            fold_results.append({
                "horizon":
                    horizon,

                "test_year":
                    test_year,

                "model":
                    model_name,

                "n":
                    len(pred),

                "positive_rate":
                    pred["target"].mean(),

                "auc":
                    fold_auc,
            })


# ============================================================
# Combine OOS results
# ============================================================

predictions = pd.concat(
    oos_predictions,
    ignore_index=True
)

fold_results_df = pd.DataFrame(
    fold_results
)


# ============================================================
# Pooled OOS AUC
# ============================================================

pooled_results = []

for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ].copy()

    baseline_auc = auc_score(
        subset["target"],
        subset["persistence"]
    )

    for model_name in MODEL_FEATURES:

        if model_name == "persistence":

            auc = baseline_auc

        else:

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

            "delta_vs_persistence":
                auc - baseline_auc,
        })


pooled_results_df = pd.DataFrame(
    pooled_results
)


# ============================================================
# PS incremental comparisons
# ============================================================

comparison_pairs = {
    "PS_vs_persistence":
        (
            "persistence",
            "persistence_plus_ps"
        ),

    "PS_vs_persistence_plus_EWMA":
        (
            "persistence_plus_ewma",
            "persistence_plus_ewma_plus_ps"
        ),
}


bootstrap_records = []


print("\n" + "=" * 80)
print("BLOCK BOOTSTRAP — PS INCREMENTAL VALUE")
print("=" * 80)


for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ].copy()

    for comparison, (
        baseline,
        challenger,
    ) in comparison_pairs.items():

        if (
            baseline
            not in subset.columns
            or challenger
            not in subset.columns
        ):
            continue

        deltas = block_bootstrap_delta(
            subset,
            baseline,
            challenger,
            N_BOOTSTRAP,
            BLOCK_LENGTH,
            RANDOM_SEED
            + horizon
        )

        if len(deltas) == 0:
            continue

        mean_delta = (
            deltas.mean()
        )

        ci_low = np.quantile(
            deltas,
            0.025
        )

        ci_high = np.quantile(
            deltas,
            0.975
        )

        probability_positive = (
            np.mean(
                deltas > 0
            )
        )

        bootstrap_records.append({
            "horizon":
                horizon,

            "comparison":
                comparison,

            "n_bootstrap":
                len(deltas),

            "mean_delta":
                mean_delta,

            "ci_2.5":
                ci_low,

            "ci_97.5":
                ci_high,

            "probability_delta_gt_0":
                probability_positive,
        })

        print(
            f"{horizon:2d}d  "
            f"{comparison:40s} "
            f"mean={mean_delta:+.4f} "
            f"CI=[{ci_low:+.4f}, "
            f"{ci_high:+.4f}] "
            f"P(delta>0)="
            f"{probability_positive:.3f}"
        )


bootstrap_df = pd.DataFrame(
    bootstrap_records
)


# ============================================================
# Per-ticker comparison
# ============================================================

ticker_records = []

for horizon in HORIZONS:

    subset = predictions[
        predictions["horizon"]
        == horizon
    ]

    for ticker, group in subset.groupby(
        "ticker"
    ):

        base_auc = auc_score(
            group["target"],
            group["persistence"]
        )

        for model_name in [
            "persistence_plus_ps",
            "persistence_plus_ewma",
            "persistence_plus_ewma_plus_ps",
        ]:

            if model_name not in group.columns:
                continue

            model_auc = auc_score(
                group["target"],
                group[model_name]
            )

            ticker_records.append({
                "horizon":
                    horizon,

                "ticker":
                    ticker,

                "model":
                    model_name,

                "auc":
                    model_auc,

                "baseline_auc":
                    base_auc,

                "delta":
                    model_auc
                    - base_auc,

                "n":
                    len(group),
            })


ticker_results_df = pd.DataFrame(
    ticker_records
)


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

predictions.to_csv(
    PREDICTIONS_OUTPUT,
    index=False
)

pooled_results_df.to_csv(
    RESULTS_OUTPUT,
    index=False
)

bootstrap_df.to_csv(
    BOOTSTRAP_OUTPUT,
    index=False
)


# ============================================================
# Print final results
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)

print("\n" + "=" * 80)
print("POOLED OOS RESULTS")
print("=" * 80)

print(
    pooled_results_df
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


print("\n" + "=" * 80)
print("PS INCREMENTAL RESULTS")
print("=" * 80)

print(
    bootstrap_df.to_string(
        index=False
    )
)


# ============================================================
# Per-ticker macro summary
# ============================================================

if len(ticker_results_df) > 0:

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
                "delta",
                "mean"
            ),

            median_delta=(
                "delta",
                "median"
            ),

            tickers=(
                "ticker",
                "nunique"
            ),
        )
        .reset_index()
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PER-TICKER MACRO"
    )

    print(
        "=" * 80
    )

    print(
        macro.to_string(
            index=False
        )
    )


# ============================================================
# Final interpretation helper
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "PS GO / NO-GO CHECK"
)

print(
    "=" * 80
)

for _, row in bootstrap_df.iterrows():

    print(
        f"{row['horizon']:2.0f}d "
        f"{row['comparison']:40s} "
        f"delta={row['mean_delta']:+.4f} "
        f"CI=[{row['ci_2.5']:+.4f}, "
        f"{row['ci_97.5']:+.4f}]"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "OUTPUT FILES"
)

print(
    "=" * 80
)

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