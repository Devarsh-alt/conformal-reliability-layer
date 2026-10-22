from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 35 — INCREMENTAL VALUE OF BREACH HISTORY
# =============================================================================
#
# GO / NO-GO EXPERIMENT
#
# Core question:
#
# Does breach history provide predictive information about future VaR
# deterioration beyond lagged forecast loss and volatility?
#
#
# BASELINE MODEL
# ----------------
# lagged pinball loss
# + volatility ratio
# + ticker fixed effects
#
#
# BREACH MODEL
# ----------------
# baseline
# + prior 5-observation breach count
#
#
# BREACH-JOINT MODEL
# ----------------
# baseline
# + prior 5-observation breach count
# + prior breach streak
#
#
# Targets:
#   deterioration_5d
#   deterioration_10d
#   deterioration_20d
#
#
# Evaluation:
#   - chronological 60/40 split within ticker
#   - pooled OOS prediction
#   - ticker fixed effects
#   - date-clustered moving-block bootstrap
#   - AUC
#   - incremental AUC over baseline
#
#
# IMPORTANT:
#
# No future information enters predictors.
#
# All breach-history variables are calculated strictly from observations
# occurring before the forecast date.
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

OUTPUT_RESULTS = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_incremental_value_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_incremental_value_predictions.csv"
)

OUTPUT_COEFFICIENTS = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_incremental_value_coefficients.csv"
)

TRAIN_FRACTION = 0.60

BLOCK_LENGTH_DATES = 20

N_BOOTSTRAPS = 1000

RANDOM_STATE = 42

HORIZONS = [5, 10, 20]

MIN_TICKER_ROWS = 100


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

MODEL_FEATURES = {

    "baseline":
        [
            "pinball_lag1",
            "vol_ratio_20_60"
        ],

    "breach_count":
        [
            "pinball_lag1",
            "vol_ratio_20_60",
            "prior_breach_count_5"
        ],

    "breach_count_plus_streak":
        [
            "pinball_lag1",
            "vol_ratio_20_60",
            "prior_breach_count_5",
            "breach_streak_prior"
        ]
}


# =============================================================================
# TARGET
# =============================================================================

def get_target(df, horizon):

    target = f"deterioration_{horizon}d"

    if target not in df.columns:

        raise ValueError(
            f"Target not found: {target}"
        )

    return target


# =============================================================================
# MOVING DATE-BLOCK BOOTSTRAP
# =============================================================================

def moving_date_block_bootstrap_indices(
    dates,
    block_length,
    rng
):
    """
    Construct a bootstrap sample of DATE CLUSTERS.

    Every observation belonging to a sampled date is included together.

    Blocks of consecutive dates are sampled with replacement.

    This preserves:
        - cross-sectional dependence within dates
        - short-range temporal dependence across dates
    """

    unique_dates = np.array(
        sorted(
            pd.to_datetime(
                pd.Series(dates)
            ).dt.normalize().unique()
        )
    )

    n_dates = len(unique_dates)

    if n_dates == 0:
        return np.array([], dtype=int)

    block_length = min(
        block_length,
        n_dates
    )

    possible_starts = np.arange(
        0,
        n_dates - block_length + 1
    )

    sampled_date_positions = []

    while len(sampled_date_positions) < n_dates:

        start = rng.choice(
            possible_starts
        )

        block = np.arange(
            start,
            start + block_length
        )

        sampled_date_positions.extend(
            block.tolist()
        )

    sampled_date_positions = np.asarray(
        sampled_date_positions[:n_dates],
        dtype=int
    )

    return sampled_date_positions


# =============================================================================
# DATE-BLOCK BOOTSTRAP FOR AUC DIFFERENCES
# =============================================================================

def date_block_bootstrap(
    prediction_df,
    baseline_column,
    augmented_columns,
    n_bootstrap=1000,
    block_length=20,
    random_state=42
):
    """
    Bootstrap AUC and incremental AUC.

    Resampling unit:
        DATE CLUSTERS

    Every ticker observation occurring on a sampled date travels together.

    Returns a dictionary for each augmented model.
    """

    rng = np.random.default_rng(
        random_state
    )

    work = prediction_df.copy()

    work["date_cluster"] = (
        pd.to_datetime(
            work["forecast_date"]
        )
        .dt.normalize()
    )

    unique_dates = np.array(
        sorted(
            work["date_cluster"]
            .unique()
        )
    )

    if len(unique_dates) < block_length:
        block_length = len(unique_dates)

    if len(unique_dates) == 0:
        raise ValueError(
            "No dates available for bootstrap."
        )

    observed = {}

    for model in [baseline_column] + augmented_columns:

        sub = work.dropna(
            subset=[
                "target",
                model
            ]
        )

        observed_auc = roc_auc_score(
            sub["target"].astype(int),
            sub[model].astype(float)
        )

        observed[model] = observed_auc

    bootstrap_storage = {

        model: {
            "auc": [],
            "delta": []
        }

        for model in augmented_columns
    }

    baseline_auc_samples = []

    for _ in range(
        n_bootstrap
    ):

        sampled_date_positions = (
            moving_date_block_bootstrap_indices(
                dates=unique_dates,
                block_length=block_length,
                rng=rng
            )
        )

        sampled_dates = unique_dates[
            sampled_date_positions
        ]

        # Each sampled date cluster contributes ALL observations
        # associated with that date.
        pieces = []

        for date in sampled_dates:

            day = work[
                work["date_cluster"] == date
            ]

            if not day.empty:

                pieces.append(
                    day
                )

        if not pieces:
            continue

        boot = pd.concat(
            pieces,
            ignore_index=True
        )

        # The bootstrap can contain duplicated dates because blocks
        # are sampled with replacement. That is intended.

        base_sub = boot.dropna(
            subset=[
                "target",
                baseline_column
            ]
        )

        if len(
            np.unique(
                base_sub["target"]
            )
        ) < 2:
            continue

        base_auc = roc_auc_score(
            base_sub["target"].astype(int),
            base_sub[baseline_column].astype(float)
        )

        baseline_auc_samples.append(
            base_auc
        )

        for model in augmented_columns:

            sub = boot.dropna(
                subset=[
                    "target",
                    model
                ]
            )

            if len(
                np.unique(
                    sub["target"]
                )
            ) < 2:
                continue

            auc = roc_auc_score(
                sub["target"].astype(int),
                sub[model].astype(float)
            )

            delta = (
                auc - base_auc
            )

            bootstrap_storage[
                model
            ]["auc"].append(
                auc
            )

            bootstrap_storage[
                model
            ]["delta"].append(
                delta
            )

    results = {}

    for model in augmented_columns:

        auc_values = np.asarray(
            bootstrap_storage[
                model
            ]["auc"]
        )

        delta_values = np.asarray(
            bootstrap_storage[
                model
            ]["delta"]
        )

        if len(auc_values) == 0:

            results[model] = {

                "auc_bootstrap_mean":
                    np.nan,

                "auc_ci_low":
                    np.nan,

                "auc_ci_high":
                    np.nan,

                "delta_bootstrap_mean":
                    np.nan,

                "delta_ci_low":
                    np.nan,

                "delta_ci_high":
                    np.nan
            }

            continue

        results[model] = {

            "auc_bootstrap_mean":
                np.mean(auc_values),

            "auc_ci_low":
                np.percentile(
                    auc_values,
                    2.5
                ),

            "auc_ci_high":
                np.percentile(
                    auc_values,
                    97.5
                ),

            "delta_bootstrap_mean":
                np.mean(delta_values),

            "delta_ci_low":
                np.percentile(
                    delta_values,
                    2.5
                ),

            "delta_ci_high":
                np.percentile(
                    delta_values,
                    97.5
                )
        }

    return (
        observed,
        results
    )


# =============================================================================
# TICKER FIXED EFFECT DESIGN MATRIX
# =============================================================================

def build_design_matrices(
    train,
    test,
    features
):
    """
    Build:

        standardized numeric predictors
        +
        ticker fixed effects

    Standardization is learned only from train.
    Ticker categories are learned only from train.
    """

    scaler = StandardScaler()

    train_numeric = scaler.fit_transform(
        train[
            features
        ].astype(float)
    )

    test_numeric = scaler.transform(
        test[
            features
        ].astype(float)
    )

    # -------------------------------------------------------------------------
    # Ticker fixed effects
    # -------------------------------------------------------------------------

    train_ticker = pd.get_dummies(
        train["ticker"],
        prefix="ticker",
        dtype=float
    )

    test_ticker = pd.get_dummies(
        test["ticker"],
        prefix="ticker",
        dtype=float
    )

    test_ticker = test_ticker.reindex(
        columns=train_ticker.columns,
        fill_value=0.0
    )

    # Drop one fixed effect to avoid perfect collinearity.
    if train_ticker.shape[1] > 1:

        first_column = (
            train_ticker.columns[0]
        )

        train_ticker = (
            train_ticker.drop(
                columns=[first_column]
            )
        )

        test_ticker = (
            test_ticker.drop(
                columns=[first_column],
                errors="ignore"
            )
        )

    x_train = np.hstack(
        [
            train_numeric,
            train_ticker.to_numpy()
        ]
    )

    x_test = np.hstack(
        [
            test_numeric,
            test_ticker.to_numpy()
        ]
    )

    return (
        x_train,
        x_test,
        scaler
    )


# =============================================================================
# COEFFICIENT NAMES
# =============================================================================

def get_coefficient_names(
    features,
    train
):

    numeric_names = [
        f"standardized_{x}"
        for x in features
    ]

    ticker_columns = [
        f"ticker_{x}"
        for x in sorted(
            train["ticker"].unique()
        )
    ]

    # Drop reference category.
    if len(ticker_columns) > 1:

        ticker_columns = (
            ticker_columns[1:]
        )

    return (
        numeric_names
        + ticker_columns
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 35 — INCREMENTAL VALUE OF BREACH HISTORY")
print("=" * 100)

print("\nROOT:")
print(ROOT)

print("\nPANEL:")
print(PANEL_FILE)

print("\nConfiguration:")
print(
    f"Train fraction:          {TRAIN_FRACTION:.2f}"
)

print(
    f"Date block length:       {BLOCK_LENGTH_DATES}"
)

print(
    f"Date-block bootstraps:   {N_BOOTSTRAPS}"
)

print(
    f"Random state:            {RANDOM_STATE}"
)

print(
    "\nPRIMARY BASELINE:"
)

print(
    "  lagged pinball loss + volatility ratio + ticker fixed effects"
)

print(
    "\nPRIMARY BREACH ADDITION:"
)

print(
    "  prior 5-observation breach count"
)

print(
    "\nSECONDARY BREACH ADDITION:"
)

print(
    "  prior 5-observation breach count + prior breach streak"
)


# =============================================================================
# LOAD
# =============================================================================

if not PANEL_FILE.exists():

    raise FileNotFoundError(
        f"Panel not found:\n{PANEL_FILE}"
    )

df = pd.read_csv(
    PANEL_FILE
)

print(
    "\nPanel shape:"
)

print(
    df.shape
)


# =============================================================================
# TYPES
# =============================================================================

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.strip()
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["breach"] = (
    df["breach"]
    .fillna(0)
    .astype(int)
)


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
# VALIDATE REQUIRED INPUTS
# =============================================================================

required = [
    "ticker",
    "forecast_date",
    "breach",
    "pinball_lag1",
    "vol_ratio_20_60",
    "breach_streak_prior"
]

missing = [
    c
    for c in required
    if c not in df.columns
]

if missing:

    raise ValueError(
        f"Missing required columns: {missing}"
    )


# =============================================================================
# CONSTRUCT CAUSAL PRIOR 5-DAY BREACH COUNT
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "CONSTRUCTING CAUSAL BREACH HISTORY"
)

print(
    "=" * 100
)


groups = []

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

    prior_breach = (
        g["breach"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    g["prior_breach_count_5"] = (
        prior_breach
        .rolling(
            5,
            min_periods=5
        )
        .sum()
    )

    groups.append(
        g
    )


df = pd.concat(
    groups,
    ignore_index=True
)


print(
    "\nMissing prior 5-day breach count:",
    df[
        "prior_breach_count_5"
    ].isna().sum()
)

print(
    "Missing pinball_lag1:",
    df[
        "pinball_lag1"
    ].isna().sum()
)

print(
    "Missing vol_ratio_20_60:",
    df[
        "vol_ratio_20_60"
    ].isna().sum()
)

print(
    "Missing breach streak:",
    df[
        "breach_streak_prior"
    ].isna().sum()
)


# =============================================================================
# VERIFY TICKERS
# =============================================================================

tickers = sorted(
    df["ticker"].unique()
)

print(
    "\nNumber of tickers:",
    len(tickers)
)

print(
    tickers
)


# =============================================================================
# STORAGE
# =============================================================================

results = []

all_predictions = []

all_coefficients = []


# =============================================================================
# HORIZON LOOP
# =============================================================================

for horizon in HORIZONS:

    target = get_target(
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
    # CHRONOLOGICAL SPLIT
    # =========================================================================
    #
    # IMPORTANT:
    # The split is made on each ticker's full available history FIRST.
    # Missing observations are removed only after the split.
    #
    # This keeps the test boundary identical across models.
    # =========================================================================

    train_parts = []
    test_parts = []

    eligible_tickers = 0

    for ticker in tickers:

        g = (
            df[
                df["ticker"] == ticker
            ]
            .sort_values(
                "forecast_date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(g) < MIN_TICKER_ROWS:

            continue

        split_idx = int(
            np.floor(
                len(g)
                * TRAIN_FRACTION
            )
        )

        if (
            split_idx < 50
            or
            split_idx >= len(g) - 20
        ):

            continue

        train = (
            g
            .iloc[
                :split_idx
            ]
            .copy()
        )

        test = (
            g
            .iloc[
                split_idx:
            ]
            .copy()
        )

        train_parts.append(
            train
        )

        test_parts.append(
            test
        )

        eligible_tickers += 1


    train_all = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_all = pd.concat(
        test_parts,
        ignore_index=True
    )


    # =========================================================================
    # MODEL LOOP
    # =========================================================================

    horizon_model_predictions = {}

    for model_name, features in MODEL_FEATURES.items():

        # ---------------------------------------------------------------------
        # Missing-value filtering happens AFTER chronological splitting.
        # ---------------------------------------------------------------------

        train = (
            train_all
            .dropna(
                subset=
                    features
                    + [target]
            )
            .copy()
        )

        test = (
            test_all
            .dropna(
                subset=
                    features
                    + [target]
            )
            .copy()
        )

        if len(train) < 100:

            raise ValueError(
                f"Too few training observations for "
                f"{model_name}, horizon {horizon}"
            )

        if len(test) < 50:

            raise ValueError(
                f"Too few test observations for "
                f"{model_name}, horizon {horizon}"
            )

        if (
            len(
                np.unique(
                    train[target]
                )
            )
            < 2
        ):

            raise ValueError(
                f"Training target has only one class for "
                f"{model_name}, horizon {horizon}"
            )

        if (
            len(
                np.unique(
                    test[target]
                )
            )
            < 2
        ):

            raise ValueError(
                f"Test target has only one class for "
                f"{model_name}, horizon {horizon}"
            )


        # ---------------------------------------------------------------------
        # DESIGN MATRIX
        # ---------------------------------------------------------------------

        (
            x_train,
            x_test,
            scaler
        ) = build_design_matrices(
            train,
            test,
            features
        )


        # ---------------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------------

        model = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        )

        model.fit(
            x_train,
            train[target].astype(int)
        )


        # ---------------------------------------------------------------------
        # OOS PREDICTIONS
        # ---------------------------------------------------------------------

        probabilities = model.predict_proba(
            x_test
        )[:, 1]


        # ---------------------------------------------------------------------
        # AUC
        # ---------------------------------------------------------------------

        auc = roc_auc_score(
            test[target].astype(int),
            probabilities
        )


        horizon_model_predictions[
            model_name
        ] = pd.DataFrame(
            {
                "ticker":
                    test["ticker"].to_numpy(),

                "forecast_date":
                    test[
                        "forecast_date"
                    ].to_numpy(),

                "target":
                    test[target]
                    .astype(int)
                    .to_numpy(),

                "prediction":
                    probabilities
            }
        )


        # ---------------------------------------------------------------------
        # COEFFICIENTS
        # ---------------------------------------------------------------------

        coefficient_names = (
            get_coefficient_names(
                features,
                train
            )
        )

        coefficients = model.coef_[0]

        if len(coefficient_names) != len(coefficients):

            # Fall back to generic names if something unexpected occurs.
            coefficient_names = [
                f"feature_{i}"
                for i in range(
                    len(coefficients)
                )
            ]

        for name, value in zip(
            coefficient_names,
            coefficients
        ):

            all_coefficients.append(
                {
                    "horizon":
                        horizon,

                    "model":
                        model_name,

                    "feature":
                        name,

                    "coefficient":
                        value
                }
            )


        # ---------------------------------------------------------------------
        # PRINT
        # ---------------------------------------------------------------------

        print(
            f"\n{model_name}"
        )

        print(
            f"  Features:             "
            f"{features}"
        )

        print(
            f"  Train observations:   "
            f"{len(train):,}"
        )

        print(
            f"  Test observations:    "
            f"{len(test):,}"
        )

        print(
            f"  OOS AUC:              "
            f"{auc:.4f}"
        )


    # =========================================================================
    # ALIGN MODEL PREDICTIONS
    # =========================================================================

    base = (
        horizon_model_predictions[
            "baseline"
        ]
        .rename(
            columns={
                "prediction":
                    "baseline_prediction"
            }
        )
    )

    breach = (
        horizon_model_predictions[
            "breach_count"
        ]
        .rename(
            columns={
                "prediction":
                    "breach_count_prediction"
            }
        )
    )

    joint = (
        horizon_model_predictions[
            "breach_count_plus_streak"
        ]
        .rename(
            columns={
                "prediction":
                    "breach_joint_prediction"
            }
        )
    )


    merged_predictions = base.merge(
        breach[
            [
                "ticker",
                "forecast_date",
                "target",
                "breach_count_prediction"
            ]
        ],
        on=[
            "ticker",
            "forecast_date",
            "target"
        ],
        how="inner",
        validate="one_to_one"
    )

    merged_predictions = merged_predictions.merge(
        joint[
            [
                "ticker",
                "forecast_date",
                "target",
                "breach_joint_prediction"
            ]
        ],
        on=[
            "ticker",
            "forecast_date",
            "target"
        ],
        how="inner",
        validate="one_to_one"
    )

    merged_predictions["horizon"] = (
        horizon
    )


    # =========================================================================
    # POOLED OOS AUC
    # =========================================================================

    baseline_auc = roc_auc_score(
        merged_predictions["target"],
        merged_predictions[
            "baseline_prediction"
        ]
    )

    breach_auc = roc_auc_score(
        merged_predictions["target"],
        merged_predictions[
            "breach_count_prediction"
        ]
    )

    joint_auc = roc_auc_score(
        merged_predictions["target"],
        merged_predictions[
            "breach_joint_prediction"
        ]
    )


    # =========================================================================
    # DATE-BLOCK BOOTSTRAP
    # =========================================================================

    (
        observed_auc,
        bootstrap_results
    ) = date_block_bootstrap(

        prediction_df=
            merged_predictions,

        baseline_column=
            "baseline_prediction",

        augmented_columns=[
            "breach_count_prediction",
            "breach_joint_prediction"
        ],

        n_bootstrap=
            N_BOOTSTRAPS,

        block_length=
            BLOCK_LENGTH_DATES,

        random_state=
            RANDOM_STATE
            + horizon
    )


    # =========================================================================
    # PRINT BOOTSTRAP
    # =========================================================================

    print(
        "\n"
        + "-" * 100
    )

    print(
        "DATE-CLUSTERED MOVING-BLOCK BOOTSTRAP"
    )

    print(
        "-" * 100
    )

    print(
        f"Baseline AUC:          "
        f"{baseline_auc:.4f}"
    )

    print(
    "Baseline bootstrap distribution generated "
    "jointly with the augmented models."
    )

    for model_name, display_name in [
        (
            "breach_count_prediction",
            "Baseline + breach count"
        ),
        (
            "breach_joint_prediction",
            "Baseline + breach count + streak"
        )
    ]:

        br = bootstrap_results[
            model_name
        ]

        observed_model_auc = (
            observed_auc[
                model_name
            ]
        )

        print(
            f"\n{display_name}"
        )

        print(
            f"  OOS AUC:            "
            f"{observed_model_auc:.4f}"
        )

        print(
            f"  Bootstrap AUC CI:   "
            f"["
            f"{br['auc_ci_low']:.4f}, "
            f"{br['auc_ci_high']:.4f}"
            f"]"
        )

        print(
            f"  Delta vs baseline:  "
            f"{observed_model_auc - baseline_auc:+.4f}"
        )

        print(
            f"  Bootstrap delta CI: "
            f"["
            f"{br['delta_ci_low']:+.4f}, "
            f"{br['delta_ci_high']:+.4f}"
            f"]"
        )


    # =========================================================================
    # STORE RESULT ROWS
    # =========================================================================

    results.append(
        {
            "horizon":
                horizon,

            "eligible_tickers":
                eligible_tickers,

            "test_observations":
                len(merged_predictions),

            "baseline_auc":
                baseline_auc,

            "breach_count_auc":
                breach_auc,

            "breach_count_delta":
                breach_auc
                - baseline_auc,

            "breach_count_delta_ci_low":
                bootstrap_results[
                    "breach_count_prediction"
                ][
                    "delta_ci_low"
                ],

            "breach_count_delta_ci_high":
                bootstrap_results[
                    "breach_count_prediction"
                ][
                    "delta_ci_high"
                ],

            "breach_joint_auc":
                joint_auc,

            "breach_joint_delta":
                joint_auc
                - baseline_auc,

            "breach_joint_delta_ci_low":
                bootstrap_results[
                    "breach_joint_prediction"
                ][
                    "delta_ci_low"
                ],

            "breach_joint_delta_ci_high":
                bootstrap_results[
                    "breach_joint_prediction"
                ][
                    "delta_ci_high"
                ]
        }
    )


    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    all_predictions.append(
        merged_predictions
    )


# =============================================================================
# COMBINE OUTPUTS
# =============================================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

coefficients_df = pd.DataFrame(
    all_coefficients
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL EXPERIMENT 35 SUMMARY"
)

print(
    "=" * 100
)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# =============================================================================
# GO / NO-GO DECISION
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "GO / NO-GO DECISION"
)

print(
    "=" * 100
)


go_horizons = results_df[
    results_df[
        "breach_count_delta_ci_low"
    ] > 0
]["horizon"].tolist()


print(
    "\nPrimary breach-count specification:"
)

print(
    "  Horizons with a strictly positive "
    "date-block bootstrap incremental CI:"
)

print(
    f"  {go_horizons}"
)


if go_horizons:

    print(
        "\nPRELIMINARY DECISION: GO"
    )

    print(
        "Breach history adds statistically robust "
        "incremental information beyond lagged loss "
        "and volatility at at least one horizon."
    )

else:

    print(
        "\nPRELIMINARY DECISION: NO-GO"
    )

    print(
        "Breach history does not show robust incremental "
        "information beyond lagged loss and volatility "
        "under the primary specification."
    )


# =============================================================================
# SAVE
# =============================================================================

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

coefficients_df.to_csv(
    OUTPUT_COEFFICIENTS,
    index=False
)


# =============================================================================
# SAVE MESSAGE
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
    OUTPUT_RESULTS
)

print(
    OUTPUT_PREDICTIONS
)

print(
    OUTPUT_COEFFICIENTS
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 35 COMPLETE"
)

print(
    "=" * 100
)