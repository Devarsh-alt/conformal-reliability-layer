from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 32 — BREACH HISTORY STATE ANALYSIS
# =============================================================================
#
# Research question:
#
#   Does recent VaR breach history contain information about future
#   deterioration in VaR forecast quality?
#
# We characterize breach history through:
#
#   1. Prior breach count over 5 observations
#   2. Prior breach count over 10 observations
#   3. Prior breach count over 20 observations
#   4. Prior breach streak
#   5. Days since most recent observed breach
#
# Important:
# All breach-history predictors are constructed using ONLY observations
# available before the forecast date.
#
# Primary deterioration targets:
#
#   deterioration_5d
#   deterioration_10d
#   deterioration_20d
#
# Evaluation:
#
#   - chronological 60/40 split within ticker
#   - OOS logistic AUC
#   - OOS conditional deterioration rates by breach-history state
#   - lift relative to the overall OOS deterioration rate
#
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

OUTPUT_STATE = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_state_analysis.csv"
)

OUTPUT_AUC = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_oos_auc.csv"
)

OUTPUT_TICKER = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_state_by_ticker.csv"
)

FIG_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "figures"
)

TRAIN_FRACTION = 0.60
RANDOM_STATE = 42

HORIZONS = [5, 10, 20]

MIN_TICKER_OBSERVATIONS = 100


# =============================================================================
# HELPERS
# =============================================================================

def get_primary_target(df, horizon):
    """
    Explicitly identify the causal 75th-percentile primary deterioration target.
    """

    target = f"deterioration_{horizon}d"

    if target not in df.columns:
        raise ValueError(
            f"Required target not found: {target}"
        )

    return target


def fit_univariate_logistic(train_x, train_y, test_x):
    """
    Fit a train-only standardized univariate logistic model.
    """

    train_x = np.asarray(
        train_x,
        dtype=float
    )

    train_y = np.asarray(
        train_y,
        dtype=int
    )

    test_x = np.asarray(
        test_x,
        dtype=float
    )

    if len(np.unique(train_y)) < 2:
        return None

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        train_x
    )

    x_test = scaler.transform(
        test_x
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        x_train,
        train_y
    )

    return model.predict_proba(
        x_test
    )[:, 1]


def safe_auc(y_true, scores):

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        scores
    )


def assign_count_state(x):

    """
    Fixed, non-data-dependent state definitions.
    """

    if pd.isna(x):
        return np.nan

    x = int(x)

    if x == 0:
        return "0"

    if x == 1:
        return "1"

    if x == 2:
        return "2"

    return "3+"


def assign_streak_state(x):

    """
    Fixed breach-streak states.
    """

    if pd.isna(x):
        return np.nan

    x = int(x)

    if x == 0:
        return "0"

    if x == 1:
        return "1"

    if x == 2:
        return "2"

    return "3+"


def assign_recency_state(x):

    """
    Fixed days-since-last-breach states.

    1       = breach on immediately preceding observation
    2       = two observations since last breach
    3-5     = moderate recency
    6-10    = older breach
    11+     = very old breach
    """

    if pd.isna(x):
        return "No prior breach"

    x = int(x)

    if x <= 1:
        return "1"

    if x == 2:
        return "2"

    if x <= 5:
        return "3-5"

    if x <= 10:
        return "6-10"

    return "11+"


def calculate_days_since_last_breach(group):

    """
    For every forecast date t, calculate the number of observations since
    the most recent BREACH observed strictly before t.

    Example:

        prior breach at t-1  -> 1
        prior breach at t-2  -> 2
        prior breach at t-5  -> 5

    No breach in the available history -> NaN.

    The calculation is performed sequentially within each ticker.
    """

    breaches = (
        group["breach"]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    result = np.full(
        len(group),
        np.nan
    )

    last_breach_position = None

    for i in range(len(group)):

        # At forecast date i, only positions < i are known.
        if last_breach_position is not None:
            result[i] = (
                i
                - last_breach_position
            )

        # After calculating today's value, update history with today's
        # realized breach for use at the next forecast date.
        if breaches[i] == 1:
            last_breach_position = i

    return result


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 32 — BREACH HISTORY STATE ANALYSIS")
print("=" * 100)

print("\nROOT:")
print(ROOT)

print("\nPANEL:")
print(PANEL_FILE)

print("\nConfiguration:")
print(
    f"Train fraction:       {TRAIN_FRACTION:.2f}"
)

print(
    f"Random state:         {RANDOM_STATE}"
)

print(
    f"Minimum ticker rows:   {MIN_TICKER_OBSERVATIONS}"
)


# =============================================================================
# LOAD
# =============================================================================

print("\n" + "=" * 100)
print("LOADING PANEL")
print("=" * 100)

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Panel not found:\n{PANEL_FILE}"
    )

df = pd.read_csv(
    PANEL_FILE
)

print("\nPanel shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))


# =============================================================================
# BASIC VALIDATION
# =============================================================================

required_columns = [
    "ticker",
    "forecast_date",
    "breach",
    "breach_lag1",
    "breach_streak_prior",
    "breach_rate_5d",
    "breach_rate_20d"
]

missing = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.strip()
)

df["breach"] = (
    df["breach"]
    .fillna(0)
    .astype(int)
)


# =============================================================================
# DUPLICATE CHECK
# =============================================================================

duplicates = df.duplicated(
    subset=[
        "ticker",
        "forecast_date"
    ]
).sum()

print("\nDuplicate ticker-date rows:")
print(duplicates)

if duplicates > 0:
    raise ValueError(
        "Duplicate ticker-date rows detected."
    )


# =============================================================================
# SORT
# =============================================================================

df = (
    df
    .sort_values(
        [
            "ticker",
            "forecast_date"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# CONSTRUCT CAUSAL BREACH HISTORY
# =============================================================================

print("\n" + "=" * 100)
print("CONSTRUCTING CAUSAL BREACH HISTORY")
print("=" * 100)

all_groups = []

for ticker, g in df.groupby(
    "ticker",
    sort=False
):

    g = (
        g
        .sort_values(
            "forecast_date"
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # Shift breach history by one observation.
    #
    # Therefore all these quantities refer strictly to information
    # available BEFORE the current forecast date.
    # -------------------------------------------------------------------------

    past_breach = (
        g["breach"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    g["prior_breach_count_5"] = (
        past_breach
        .rolling(
            5,
            min_periods=5
        )
        .sum()
    )

    g["prior_breach_count_10"] = (
        past_breach
        .rolling(
            10,
            min_periods=10
        )
        .sum()
    )

    g["prior_breach_count_20"] = (
        past_breach
        .rolling(
            20,
            min_periods=20
        )
        .sum()
    )

    # -------------------------------------------------------------------------
    # Days since previous breach.
    # -------------------------------------------------------------------------

    g["days_since_last_breach"] = (
        calculate_days_since_last_breach(
            g
        )
    )

    # -------------------------------------------------------------------------
    # Fixed categorical states.
    # -------------------------------------------------------------------------

    g["breach_count_5_state"] = (
        g["prior_breach_count_5"]
        .apply(assign_count_state)
    )

    g["breach_count_10_state"] = (
        g["prior_breach_count_10"]
        .apply(assign_count_state)
    )

    g["breach_count_20_state"] = (
        g["prior_breach_count_20"]
        .apply(assign_count_state)
    )

    g["streak_state"] = (
        g["breach_streak_prior"]
        .apply(assign_streak_state)
    )

    g["recency_state"] = (
        g["days_since_last_breach"]
        .apply(assign_recency_state)
    )

    all_groups.append(
        g
    )


df = pd.concat(
    all_groups,
    ignore_index=True
)


# =============================================================================
# SANITY CHECKS FOR CAUSALITY
# =============================================================================

print("\n" + "=" * 100)
print("CAUSALITY SANITY CHECKS")
print("=" * 100)

print("\nRows:")
print(len(df))

print(
    "\nMissing prior 5-day breach count:",
    df["prior_breach_count_5"].isna().sum()
)

print(
    "Missing prior 10-day breach count:",
    df["prior_breach_count_10"].isna().sum()
)

print(
    "Missing prior 20-day breach count:",
    df["prior_breach_count_20"].isna().sum()
)

print(
    "Missing breach streak:",
    df["breach_streak_prior"].isna().sum()
)

print(
    "Missing days-since-breach:",
    df["days_since_last_breach"].isna().sum()
)

print("\nExample breach-history rows:")

print(
    df[
        [
            "ticker",
            "forecast_date",
            "breach",
            "breach_lag1",
            "breach_streak_prior",
            "prior_breach_count_5",
            "prior_breach_count_10",
            "prior_breach_count_20",
            "days_since_last_breach"
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# =============================================================================
# TICKER LIST
# =============================================================================

tickers = sorted(
    df["ticker"]
    .dropna()
    .unique()
)

print(
    "\nTickers:",
    len(tickers)
)

print(
    tickers
)


# =============================================================================
# STORAGE
# =============================================================================

state_results = []
auc_results = []
ticker_results = []


# =============================================================================
# HORIZON LOOP
# =============================================================================

for horizon in HORIZONS:

    target = get_primary_target(
        df,
        horizon
    )

    print(
        "\n\n"
        + "=" * 100
    )

    print(
        f"HORIZON = {horizon} DAYS"
    )

    print(
        f"TARGET  = {target}"
    )

    print(
        "=" * 100
    )


    # =========================================================================
    # BUILD OOS TEST SET
    # =========================================================================

    test_tables = []

    eligible_tickers = 0

    for ticker in tickers:

        g = (
            df[
                df["ticker"] == ticker
            ]
            .sort_values(
                "forecast_date"
            )
            .copy()
        )

        if len(g) < MIN_TICKER_OBSERVATIONS:
            continue

        # Require enough history for the 20-observation state definition.
        g = g[
            g["prior_breach_count_20"].notna()
        ].copy()

        if len(g) < MIN_TICKER_OBSERVATIONS:
            continue

        split_idx = int(
            np.floor(
                len(g)
                * TRAIN_FRACTION
            )
        )

        if (
            split_idx <= 20
            or
            split_idx >= len(g) - 20
        ):
            continue

        train = g.iloc[
            :split_idx
        ].copy()

        test = g.iloc[
            split_idx:
        ].copy()

        train = train.dropna(
            subset=[target]
        )

        test = test.dropna(
            subset=[target]
        )

        if len(train) < 50:
            continue

        if len(test) < 20:
            continue

        if (
            len(
                np.unique(
                    train[target]
                )
            )
            < 2
        ):
            continue

        if (
            len(
                np.unique(
                    test[target]
                )
            )
            < 2
        ):
            continue


        # ---------------------------------------------------------------------
        # FEATURES TO TEST
        # ---------------------------------------------------------------------

        feature_map = {

            "breach_lag1":
                "breach_lag1",

            "breach_count_5":
                "prior_breach_count_5",

            "breach_count_10":
                "prior_breach_count_10",

            "breach_count_20":
                "prior_breach_count_20",

            "breach_streak":
                "breach_streak_prior",

            "days_since_breach":
                "days_since_last_breach"
        }


        predictions = {
            "ticker": ticker,
            "forecast_date":
                test["forecast_date"].to_numpy(),
            "target":
                test[target].astype(int).to_numpy()
        }


        # ---------------------------------------------------------------------
        # UNIVARIATE OOS MODELS
        # ---------------------------------------------------------------------

        for feature_name, feature_column in feature_map.items():

            train_feature = (
                train[
                    [feature_column]
                ]
                .dropna()
            )

            # Align target to feature rows
            train_aligned = train.loc[
                train_feature.index
            ]

            test_feature = (
                test[
                    [feature_column]
                ]
                .dropna()
            )

            test_aligned = test.loc[
                test_feature.index
            ]

            if len(train_aligned) < 50:
                continue

            if len(test_aligned) < 20:
                continue

            if (
                len(
                    np.unique(
                        train_aligned[target]
                    )
                )
                < 2
            ):
                continue

            pred = fit_univariate_logistic(
                train_feature,
                train_aligned[target],
                test_feature
            )

            if pred is None:
                continue

            auc = safe_auc(
                test_aligned[target],
                pred
            )

            if np.isnan(auc):
                continue

            auc_results.append(
                {
                    "horizon": horizon,
                    "ticker": ticker,
                    "feature": feature_name,
                    "test_observations": len(test_aligned),
                    "auc": auc
                }
            )


        # ---------------------------------------------------------------------
        # ADD TEST DATA FOR STATE ANALYSIS
        # ---------------------------------------------------------------------

        test_tables.append(
            test[
                [
                    "ticker",
                    "forecast_date",
                    target,
                    "prior_breach_count_5",
                    "prior_breach_count_10",
                    "prior_breach_count_20",
                    "breach_streak_prior",
                    "days_since_last_breach",
                    "breach_count_5_state",
                    "breach_count_10_state",
                    "breach_count_20_state",
                    "streak_state",
                    "recency_state"
                ]
            ].copy()
        )

        eligible_tickers += 1


    if not test_tables:

        print(
            "\nNo eligible test data."
        )

        continue


    # =========================================================================
    # POOLED OOS DATA
    # =========================================================================

    test_all = pd.concat(
        test_tables,
        ignore_index=True
    )

    test_all["target"] = (
        test_all[target]
        .astype(int)
    )

    overall_rate = (
        test_all["target"]
        .mean()
    )

    print(
        "\nEligible tickers:",
        eligible_tickers
    )

    print(
        "Test observations:",
        f"{len(test_all):,}"
    )

    print(
        "Overall OOS deterioration rate:",
        f"{overall_rate:.4%}"
    )


    # =========================================================================
    # STATE ANALYSIS FUNCTION
    # =========================================================================

    def analyze_state(
        data,
        state_column,
        state_type
    ):

        grouped = (
            data
            .groupby(
                state_column,
                dropna=False
            )
            .agg(
                observations=(
                    "target",
                    "size"
                ),
                deterioration_events=(
                    "target",
                    "sum"
                ),
                deterioration_rate=(
                    "target",
                    "mean"
                )
            )
            .reset_index()
        )

        grouped["lift_vs_overall"] = (
            grouped["deterioration_rate"]
            / overall_rate
        )

        grouped["rate_difference_vs_overall"] = (
            grouped["deterioration_rate"]
            - overall_rate
        )

        grouped["horizon"] = horizon
        grouped["state_type"] = state_type

        grouped = grouped.rename(
            columns={
                state_column:
                    "state"
            }
        )

        return grouped[
            [
                "horizon",
                "state_type",
                "state",
                "observations",
                "deterioration_events",
                "deterioration_rate",
                "lift_vs_overall",
                "rate_difference_vs_overall"
            ]
        ]


    # =========================================================================
    # COUNT STATES
    # =========================================================================

    states_to_analyze = [

        (
            "breach_count_5_state",
            "Prior 5-observation breach count"
        ),

        (
            "breach_count_10_state",
            "Prior 10-observation breach count"
        ),

        (
            "breach_count_20_state",
            "Prior 20-observation breach count"
        ),

        (
            "streak_state",
            "Prior breach streak"
        ),

        (
            "recency_state",
            "Days since most recent breach"
        )
    ]


    for column, label in states_to_analyze:

        result = analyze_state(
            test_all,
            column,
            label
        )

        state_results.append(
            result
        )


        # ---------------------------------------------------------------------
        # PRINT
        # ---------------------------------------------------------------------

        print(
            "\n"
            + "-" * 100
        )

        print(
            label
        )

        print(
            "-" * 100
        )

        print(
            result[
                [
                    "state",
                    "observations",
                    "deterioration_events",
                    "deterioration_rate",
                    "lift_vs_overall"
                ]
            ]
            .to_string(
                index=False,
                formatters={
                    "deterioration_rate":
                        lambda x:
                            f"{x:.4%}",

                    "lift_vs_overall":
                        lambda x:
                            f"{x:.3f}x"
                }
            )
        )


    # =========================================================================
    # OOS AUC SUMMARY FOR THIS HORIZON
    # =========================================================================

    auc_h = pd.DataFrame(
        [
            row
            for row in auc_results
            if row["horizon"] == horizon
        ]
    )

    if not auc_h.empty:

        pooled_auc_rows = []

        feature_columns = {

            "breach_lag1":
                "breach_lag1",

            "breach_count_5":
                "prior_breach_count_5",

            "breach_count_10":
                "prior_breach_count_10",

            "breach_count_20":
                "prior_breach_count_20",

            "breach_streak":
                "breach_streak_prior",

            "days_since_breach":
                "days_since_last_breach"
        }


        # ---------------------------------------------------------------------
        # POOLED AUC USING TEST-SET STANDARDIZED TRAIN MODELS
        #
        # We construct this by fitting one model per ticker above and then
        # aggregate ticker predictions through the ticker-level AUC results.
        # The pooled AUC itself requires stored predictions, so below we use
        # the direct relationship between each feature and the target only
        # as a descriptive ticker summary.
        # ---------------------------------------------------------------------

        auc_summary = (
            auc_h
            .groupby("feature")
            .agg(
                mean_ticker_auc=(
                    "auc",
                    "mean"
                ),
                median_ticker_auc=(
                    "auc",
                    "median"
                ),
                tickers_tested=(
                    "ticker",
                    "nunique"
                )
            )
            .reset_index()
        )

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"OOS AUC — {horizon} DAY TARGET"
        )

        print(
            "-" * 100
        )

        print(
            auc_summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}"
            )
        )


    # =========================================================================
    # TICKER-LEVEL STATE RESULTS
    # =========================================================================

    state_features = [
        (
            "prior_breach_count_5",
            "5-day breach count"
        ),
        (
            "prior_breach_count_10",
            "10-day breach count"
        ),
        (
            "prior_breach_count_20",
            "20-day breach count"
        ),
        (
            "breach_streak_prior",
            "breach streak"
        )
    ]

    for feature_column, feature_label in state_features:

        for ticker in sorted(
            test_all["ticker"].unique()
        ):

            tg = test_all[
                test_all["ticker"] == ticker
            ].copy()

            tg = tg.dropna(
                subset=[
                    feature_column,
                    "target"
                ]
            )

            if len(tg) < 20:
                continue

            # Compare high breach state to no breach state.
            if feature_column == "breach_streak_prior":

                high = tg[
                    tg[feature_column] >= 2
                ]["target"]

                low = tg[
                    tg[feature_column] == 0
                ]["target"]

            else:

                high = tg[
                    tg[feature_column] >= 3
                ]["target"]

                low = tg[
                    tg[feature_column] == 0
                ]["target"]

            if len(high) == 0 or len(low) == 0:
                continue

            ticker_results.append(
                {
                    "horizon": horizon,
                    "ticker": ticker,
                    "state_measure":
                        feature_label,
                    "high_state_observations":
                        len(high),
                    "low_state_observations":
                        len(low),
                    "high_state_deterioration_rate":
                        high.mean(),
                    "low_state_deterioration_rate":
                        low.mean(),
                    "high_minus_low":
                        high.mean() - low.mean()
                }
            )


# =============================================================================
# COMBINE STATE RESULTS
# =============================================================================

if state_results:

    state_df = pd.concat(
        state_results,
        ignore_index=True
    )

else:

    state_df = pd.DataFrame()


auc_df = pd.DataFrame(
    auc_results
)

ticker_df = pd.DataFrame(
    ticker_results
)


# =============================================================================
# FINAL STATE SUMMARY
# =============================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL STATE ANALYSIS SUMMARY"
)

print(
    "=" * 100
)

if not state_df.empty:

    for horizon in HORIZONS:

        print(
            f"\n\n### {horizon}-DAY DETERIORATION ###"
        )

        h = state_df[
            state_df["horizon"] == horizon
        ]

        for state_type in h[
            "state_type"
        ].unique():

            sub = h[
                h["state_type"]
                == state_type
            ]

            print(
                f"\n{state_type}"
            )

            print(
                sub[
                    [
                        "state",
                        "observations",
                        "deterioration_rate",
                        "lift_vs_overall"
                    ]
                ].to_string(
                    index=False,
                    formatters={
                        "deterioration_rate":
                            lambda x:
                                f"{x:.4%}",

                        "lift_vs_overall":
                            lambda x:
                                f"{x:.3f}x"
                    }
                )
            )


# =============================================================================
# FINAL AUC SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "OOS AUC SUMMARY"
)

print(
    "=" * 100
)

if not auc_df.empty:

    auc_summary_all = (
        auc_df
        .groupby(
            [
                "horizon",
                "feature"
            ]
        )
        .agg(
            mean_ticker_auc=(
                "auc",
                "mean"
            ),
            median_ticker_auc=(
                "auc",
                "median"
            ),
            ticker_count=(
                "ticker",
                "nunique"
            )
        )
        .reset_index()
    )

    print(
        auc_summary_all.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# =============================================================================
# CREATE FIGURES
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "CREATING FIGURES"
)

print(
    "=" * 100
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


if not state_df.empty:

    for horizon in HORIZONS:

        h = state_df[
            state_df["horizon"] == horizon
        ].copy()

        # ---------------------------------------------------------------------
        # Figure 1 — 20-day breach count vs deterioration
        # ---------------------------------------------------------------------

        count20 = h[
            h["state_type"]
            == "Prior 20-observation breach count"
        ].copy()

        order = [
            "0",
            "1",
            "2",
            "3+"
        ]

        count20["order"] = (
            count20["state"]
            .map(
                {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3+": 3
                }
            )
        )

        count20 = (
            count20
            .sort_values("order")
        )

        plt.figure()

        plt.plot(
            range(len(count20)),
            count20[
                "deterioration_rate"
            ],
            marker="o"
        )

        plt.xticks(
            range(len(count20)),
            count20["state"]
        )

        plt.xlabel(
            "Prior 20-observation breach count"
        )

        plt.ylabel(
            "Future deterioration rate"
        )

        plt.title(
            f"Future VaR Deterioration vs Prior Breach Count ({horizon}d)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        path = (
            FIG_DIR
            / f"breach_count_20_vs_deterioration_{horizon}d.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()


        # ---------------------------------------------------------------------
        # Figure 2 — streak vs deterioration
        # ---------------------------------------------------------------------

        streak = h[
            h["state_type"]
            == "Prior breach streak"
        ].copy()

        streak["order"] = (
            streak["state"]
            .map(
                {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3+": 3
                }
            )
        )

        streak = (
            streak
            .sort_values("order")
        )

        plt.figure()

        plt.plot(
            range(len(streak)),
            streak[
                "deterioration_rate"
            ],
            marker="o"
        )

        plt.xticks(
            range(len(streak)),
            streak["state"]
        )

        plt.xlabel(
            "Prior breach streak"
        )

        plt.ylabel(
            "Future deterioration rate"
        )

        plt.title(
            f"Future VaR Deterioration vs Breach Streak ({horizon}d)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        path = (
            FIG_DIR
            / f"breach_streak_vs_deterioration_{horizon}d.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()


        # ---------------------------------------------------------------------
        # Figure 3 — breach recency vs deterioration
        # ---------------------------------------------------------------------

        recency = h[
            h["state_type"]
            == "Days since most recent breach"
        ].copy()

        recency_order = {
            "1": 0,
            "2": 1,
            "3-5": 2,
            "6-10": 3,
            "11+": 4,
            "No prior breach": 5
        }

        recency["order"] = (
            recency["state"]
            .map(recency_order)
        )

        recency = (
            recency
            .sort_values("order")
        )

        plt.figure()

        plt.plot(
            range(len(recency)),
            recency[
                "deterioration_rate"
            ],
            marker="o"
        )

        plt.xticks(
            range(len(recency)),
            recency["state"]
        )

        plt.xlabel(
            "Observations since most recent breach"
        )

        plt.ylabel(
            "Future deterioration rate"
        )

        plt.title(
            f"Future VaR Deterioration vs Breach Recency ({horizon}d)"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        path = (
            FIG_DIR
            / f"breach_recency_vs_deterioration_{horizon}d.png"
        )

        plt.savefig(
            path,
            dpi=200
        )

        plt.close()


# =============================================================================
# SAVE RESULTS
# =============================================================================

state_df.to_csv(
    OUTPUT_STATE,
    index=False
)

auc_df.to_csv(
    OUTPUT_AUC,
    index=False
)

ticker_df.to_csv(
    OUTPUT_TICKER,
    index=False
)


# =============================================================================
# PRINT SAVE PATHS
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SAVED"
)

print(
    "=" * 100
)

print(
    OUTPUT_STATE
)

print(
    OUTPUT_AUC
)

print(
    OUTPUT_TICKER
)

print(
    FIG_DIR
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 32 COMPLETE"
)

print(
    "=" * 100
)