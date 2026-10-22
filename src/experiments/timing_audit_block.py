from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 36
# TIMING / CAUSALITY AUDIT + BLOCK-LENGTH SENSITIVITY
# =============================================================================
#
# Purpose:
#
#   1. Audit the timing of predictors used in Experiment 35.
#   2. Verify that lagged pinball loss is truly lagged.
#   3. Verify the volatility ratio construction.
#   4. Check that future realized returns are not directly embedded in
#      baseline predictors.
#   5. Re-run the incremental breach-history comparison using multiple
#      date-block bootstrap lengths.
#
# Main models:
#
# BASELINE:
#     pinball_lag1
#     + vol_ratio_20_60
#     + ticker fixed effects
#
# BREACH:
#     BASELINE
#     + prior 5-observation breach count
#
# BREACH + STREAK:
#     BASELINE
#     + prior 5-observation breach count
#     + prior breach streak
#
# Bootstrap block lengths:
#     10, 20, 40, 60 dates
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

OUTPUT_AUDIT = (
    ROOT
    / "data"
    / "monitoring"
    / "timing_causality_audit.csv"
)

OUTPUT_SENSITIVITY = (
    ROOT
    / "data"
    / "monitoring"
    / "block_length_sensitivity_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "monitoring"
    / "block_length_sensitivity_predictions.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

TRAIN_FRACTION = 0.60

BLOCK_LENGTHS = [10, 20, 40, 60]

N_BOOTSTRAPS = 1000

RANDOM_STATE = 42

HORIZONS = [5, 10, 20]


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
            f"Missing target: {target}"
        )

    return target


# =============================================================================
# CAUSAL BREACH COUNT
# =============================================================================

def construct_prior_breach_count(group):

    prior_breach = (
        group["breach"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    return (
        prior_breach
        .rolling(
            5,
            min_periods=5
        )
        .sum()
    )


# =============================================================================
# DATE BLOCK BOOTSTRAP
# =============================================================================

def date_block_bootstrap(
    prediction_df,
    baseline_column,
    augmented_columns,
    block_length,
    n_bootstrap,
    random_state
):

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

    dates = np.array(
        sorted(
            work["date_cluster"]
            .unique()
        )
    )

    n_dates = len(dates)

    if n_dates == 0:

        raise ValueError(
            "No dates available."
        )

    block_length = min(
        block_length,
        n_dates
    )

    possible_starts = np.arange(
        0,
        n_dates - block_length + 1
    )

    storage = {

        model: {
            "auc": [],
            "delta": []
        }

        for model in augmented_columns
    }

    for _ in range(
        n_bootstrap
    ):

        selected_positions = []

        while len(selected_positions) < n_dates:

            start = rng.choice(
                possible_starts
            )

            block = np.arange(
                start,
                start + block_length
            )

            selected_positions.extend(
                block.tolist()
            )

        selected_positions = np.asarray(
            selected_positions[:n_dates]
        )

        sampled_dates = dates[
            selected_positions
        ]

        pieces = []

        for date in sampled_dates:

            day = work[
                work["date_cluster"] == date
            ]

            if not day.empty:

                pieces.append(day)

        if not pieces:

            continue

        boot = pd.concat(
            pieces,
            ignore_index=True
        )

        # ---------------------------------------------------------------------
        # Baseline
        # ---------------------------------------------------------------------

        baseline = boot.dropna(
            subset=[
                "target",
                baseline_column
            ]
        )

        if len(
            np.unique(
                baseline["target"]
            )
        ) < 2:

            continue

        baseline_auc = roc_auc_score(
            baseline["target"].astype(int),
            baseline[baseline_column].astype(float)
        )

        # ---------------------------------------------------------------------
        # Augmented models
        # ---------------------------------------------------------------------

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

            storage[
                model
            ]["auc"].append(
                auc
            )

            storage[
                model
            ]["delta"].append(
                auc - baseline_auc
            )

    output = {}

    for model in augmented_columns:

        auc_values = np.asarray(
            storage[model]["auc"],
            dtype=float
        )

        delta_values = np.asarray(
            storage[model]["delta"],
            dtype=float
        )

        if len(auc_values) == 0:

            output[model] = {

                "auc_mean":
                    np.nan,

                "auc_low":
                    np.nan,

                "auc_high":
                    np.nan,

                "delta_mean":
                    np.nan,

                "delta_low":
                    np.nan,

                "delta_high":
                    np.nan,

                "n_bootstrap":
                    0
            }

            continue

        output[model] = {

            "auc_mean":
                np.mean(auc_values),

            "auc_low":
                np.percentile(
                    auc_values,
                    2.5
                ),

            "auc_high":
                np.percentile(
                    auc_values,
                    97.5
                ),

            "delta_mean":
                np.mean(delta_values),

            "delta_low":
                np.percentile(
                    delta_values,
                    2.5
                ),

            "delta_high":
                np.percentile(
                    delta_values,
                    97.5
                ),

            "n_bootstrap":
                len(delta_values)
        }

    return output


# =============================================================================
# MODEL FITTING
# =============================================================================

def build_design_matrix(
    train,
    test,
    features
):

    # -------------------------------------------------------------------------
    # Numeric features
    # -------------------------------------------------------------------------

    scaler = StandardScaler()

    x_train_num = scaler.fit_transform(
        train[
            features
        ].astype(float)
    )

    x_test_num = scaler.transform(
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

    # Remove reference ticker
    if train_ticker.shape[1] > 1:

        reference = (
            train_ticker.columns[0]
        )

        train_ticker = (
            train_ticker.drop(
                columns=[reference]
            )
        )

        test_ticker = (
            test_ticker.drop(
                columns=[reference],
                errors="ignore"
            )
        )

    x_train = np.hstack(
        [
            x_train_num,
            train_ticker.to_numpy()
        ]
    )

    x_test = np.hstack(
        [
            x_test_num,
            test_ticker.to_numpy()
        ]
    )

    return (
        x_train,
        x_test
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 36 — TIMING / CAUSALITY AUDIT + BLOCK SENSITIVITY")
print("=" * 100)

print("\nROOT:")
print(ROOT)

print("\nPANEL:")
print(PANEL_FILE)

print("\nTrain fraction:")
print(TRAIN_FRACTION)

print("\nBootstrap block lengths:")
print(BLOCK_LENGTHS)

print("\nBootstrap repetitions:")
print(N_BOOTSTRAPS)


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
    "\nPanel shape:",
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

if "realized_date" in df.columns:

    df["realized_date"] = pd.to_datetime(
        df["realized_date"]
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
# BASIC COLUMN CHECKS
# =============================================================================

required_columns = [

    "ticker",

    "forecast_date",

    "pinball_loss",

    "pinball_lag1",

    "realized_vol_20d",

    "realized_vol_60d",

    "vol_ratio_20_60",

    "breach",

    "breach_streak_prior"
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


# =============================================================================
# CAUSAL BREACH COUNT
# =============================================================================

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

    g[
        "prior_breach_count_5"
    ] = construct_prior_breach_count(
        g
    )

    groups.append(
        g
    )

df = pd.concat(
    groups,
    ignore_index=True
)


# =============================================================================
# TIMING AUDIT
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TIMING / CAUSALITY AUDIT"
)

print(
    "=" * 100
)


audit_rows = []


# =============================================================================
# AUDIT 1 — PINBALL LAG
# =============================================================================

pinball_errors = []

for ticker, g in df.groupby(
    "ticker"
):

    g = (
        g
        .sort_values(
            "forecast_date"
        )
        .copy()
    )

    expected = (
        g["pinball_loss"]
        .shift(1)
    )

    actual = (
        g["pinball_lag1"]
    )

    mask = (
        expected.notna()
        &
        actual.notna()
    )

    if mask.any():

        error = (
            actual[mask]
            - expected[mask]
        ).abs()

        pinball_errors.extend(
            error.tolist()
        )


pinball_errors = np.asarray(
    pinball_errors,
    dtype=float
)

pinball_max_error = (
    np.max(pinball_errors)
    if len(pinball_errors)
    else np.nan
)

pinball_match_rate = (
    np.mean(
        pinball_errors < 1e-12
    )
    if len(pinball_errors)
    else np.nan
)

print(
    "\nPINBALL_LAG1 CHECK"
)

print(
    "Maximum absolute mismatch:",
    pinball_max_error
)

print(
    "Exact-match rate:",
    pinball_match_rate
)

print(
    "Status:",
    "PASS"
    if (
        np.isfinite(pinball_max_error)
        and
        pinball_max_error < 1e-12
    )
    else "CHECK"
)

audit_rows.append(
    {
        "audit":
            "pinball_lag1 equals previous pinball_loss",

        "value":
            pinball_max_error,

        "status":
            (
                "PASS"
                if (
                    np.isfinite(
                        pinball_max_error
                    )
                    and
                    pinball_max_error < 1e-12
                )
                else "CHECK"
            )
    }
)


# =============================================================================
# AUDIT 2 — VOLATILITY RATIO
# =============================================================================

expected_vol_ratio = (
    df["realized_vol_20d"]
    /
    df["realized_vol_60d"]
)

actual_vol_ratio = (
    df["vol_ratio_20_60"]
)

mask = (
    expected_vol_ratio.notna()
    &
    actual_vol_ratio.notna()
)

vol_errors = (
    (
        expected_vol_ratio[mask]
        -
        actual_vol_ratio[mask]
    )
    .abs()
)

vol_max_error = (
    vol_errors.max()
    if len(vol_errors)
    else np.nan
)

vol_match_rate = (
    np.mean(
        vol_errors < 1e-10
    )
    if len(vol_errors)
    else np.nan
)

print(
    "\nVOL_RATIO_20_60 CHECK"
)

print(
    "Maximum absolute mismatch:",
    vol_max_error
)

print(
    "Exact/near-exact match rate:",
    vol_match_rate
)

print(
    "Status:",
    "PASS"
    if (
        np.isfinite(vol_max_error)
        and
        vol_max_error < 1e-10
    )
    else "CHECK"
)

audit_rows.append(
    {
        "audit":
            "vol_ratio_20_60 equals realized_vol_20d / realized_vol_60d",

        "value":
            vol_max_error,

        "status":
            (
                "PASS"
                if (
                    np.isfinite(
                        vol_max_error
                    )
                    and
                    vol_max_error < 1e-10
                )
                else "CHECK"
            )
    }
)


# =============================================================================
# AUDIT 3 — FUTURE RETURN CORRELATION
# =============================================================================
#
# A predictor should not contain the current realized future return.
#
# We cannot establish causality from correlation alone, but a mechanical
# identity or extremely high correlation would be suspicious.
#
# Check:
#
#     predictor(t) vs actual_return(t)
#
# for predictors used by the model.
# =============================================================================

predictor_columns = [
    "pinball_lag1",
    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",
    "prior_breach_count_5",
    "breach_streak_prior"
]

future_return_correlations = []

for column in predictor_columns:

    sub = df[
        [
            column,
            "actual_return"
        ]
    ].dropna()

    if len(sub) < 10:

        corr = np.nan

    else:

        corr = (
            sub[column]
            .corr(
                sub["actual_return"]
            )
        )

    future_return_correlations.append(
        {
            "predictor":
                column,

            "correlation_with_actual_return":
                corr
        }
    )


future_corr_df = pd.DataFrame(
    future_return_correlations
)

print(
    "\nPREDICTOR / REALIZED-RETURN CORRELATIONS"
)

print(
    future_corr_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# =============================================================================
# AUDIT 4 — FORECAST DATE VS REALIZED DATE
# =============================================================================

if "realized_date" in df.columns:

    date_gap = (
        df["realized_date"]
        -
        df["forecast_date"]
    )

    print(
        "\nFORECAST / REALIZATION TIMING"
    )

    print(
        "Date gap distribution:"
    )

    print(
        date_gap.value_counts()
        .sort_index()
        .head(20)
    )

    print(
        "Minimum date gap:",
        date_gap.min()
    )

    print(
        "Median date gap:",
        date_gap.median()
    )

    print(
        "Maximum date gap:",
        date_gap.max()
    )


# =============================================================================
# AUDIT 5 — BREACH COUNT DOES NOT INCLUDE CURRENT BREACH
# =============================================================================

breach_count_errors = []

for ticker, g in df.groupby(
    "ticker"
):

    g = (
        g
        .sort_values(
            "forecast_date"
        )
        .copy()
    )

    expected = (
        g["breach"]
        .shift(1)
        .rolling(
            5,
            min_periods=5
        )
        .sum()
    )

    actual = (
        g["prior_breach_count_5"]
    )

    mask = (
        expected.notna()
        &
        actual.notna()
    )

    if mask.any():

        errors = (
            expected[mask]
            -
            actual[mask]
        ).abs()

        breach_count_errors.extend(
            errors.tolist()
        )


breach_count_errors = np.asarray(
    breach_count_errors,
    dtype=float
)

breach_count_max_error = (
    breach_count_errors.max()
    if len(breach_count_errors)
    else np.nan
)

print(
    "\nPRIOR 5-DAY BREACH COUNT CHECK"
)

print(
    "Maximum mismatch:",
    breach_count_max_error
)

print(
    "Status:",
    "PASS"
    if (
        np.isfinite(breach_count_max_error)
        and
        breach_count_max_error < 1e-12
    )
    else "CHECK"
)

audit_rows.append(
    {
        "audit":
            "prior_breach_count_5 excludes current breach",

        "value":
            breach_count_max_error,

        "status":
            (
                "PASS"
                if (
                    np.isfinite(
                        breach_count_max_error
                    )
                    and
                    breach_count_max_error < 1e-12
                )
                else "CHECK"
            )
    }
)


# =============================================================================
# SAVE AUDIT
# =============================================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False
)


# =============================================================================
# STOP IF BASIC TIMING CHECKS FAIL
# =============================================================================

basic_audit_pass = (
    (
        audit_df[
            "audit"
        ]
        ==
        "pinball_lag1 equals previous pinball_loss"
    )
    &
    (
        audit_df[
            "status"
        ]
        ==
        "PASS"
    )
).any()

vol_audit_pass = (
    (
        audit_df[
            "audit"
        ]
        ==
        "vol_ratio_20_60 equals realized_vol_20d / realized_vol_60d"
    )
    &
    (
        audit_df[
            "status"
        ]
        ==
        "PASS"
    )
).any()

breach_audit_pass = (
    (
        audit_df[
            "audit"
        ]
        ==
        "prior_breach_count_5 excludes current breach"
    )
    &
    (
        audit_df[
            "status"
        ]
        ==
        "PASS"
    )
).any()


if not (
    basic_audit_pass
    and
    vol_audit_pass
    and
    breach_audit_pass
):

    print(
        "\n"
        + "=" * 100
    )

    print(
        "TIMING AUDIT FAILED / NEEDS REVIEW"
    )

    print(
        "=" * 100
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

    raise SystemExit(
        "Stopping before sensitivity testing because a basic timing "
        "construction check did not pass."
    )


# =============================================================================
# PREPARE CHRONOLOGICAL SPLITS
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TIMING AUDIT PASSED"
)

print(
    "=" * 100
)

print(
    "Proceeding to date-block sensitivity testing."
)


# =============================================================================
# RESULTS STORAGE
# =============================================================================

sensitivity_results = []

prediction_results = []


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
    # SPLIT EACH TICKER CHRONOLOGICALLY
    # =========================================================================

    train_parts = []
    test_parts = []

    eligible_tickers = 0

    for ticker in sorted(
        df["ticker"].unique()
    ):

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

        if len(g) < 100:

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

        train_parts.append(
            g.iloc[
                :split_idx
            ].copy()
        )

        test_parts.append(
            g.iloc[
                split_idx:
            ].copy()
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
    # MODEL PREDICTIONS
    # =========================================================================

    model_prediction_tables = {}

    for model_name, features in MODEL_FEATURES.items():

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
            continue

        if len(test) < 50:
            continue

        if len(
            np.unique(
                train[target]
            )
        ) < 2:
            continue

        if len(
            np.unique(
                test[target]
            )
        ) < 2:
            continue


        (
            x_train,
            x_test
        ) = build_design_matrix(
            train,
            test,
            features
        )


        model = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE
        )

        model.fit(
            x_train,
            train[target].astype(int)
        )


        prediction = model.predict_proba(
            x_test
        )[:, 1]


        model_prediction_tables[
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
                    prediction
            }
        )


    if not model_prediction_tables:

        continue


    # =========================================================================
    # ALIGN ALL MODELS
    # =========================================================================

    base = (
        model_prediction_tables[
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
        model_prediction_tables[
            "breach_count"
        ]
        .rename(
            columns={
                "prediction":
                    "breach_prediction"
            }
        )
    )

    joint = (
        model_prediction_tables[
            "breach_count_plus_streak"
        ]
        .rename(
            columns={
                "prediction":
                    "joint_prediction"
            }
        )
    )


    pred = base[
        [
            "ticker",
            "forecast_date",
            "target",
            "baseline_prediction"
        ]
    ].merge(
        breach[
            [
                "ticker",
                "forecast_date",
                "target",
                "breach_prediction"
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


    pred = pred.merge(
        joint[
            [
                "ticker",
                "forecast_date",
                "target",
                "joint_prediction"
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


    pred["horizon"] = horizon


    prediction_results.append(
        pred.copy()
    )


    # =========================================================================
    # OBSERVED AUC
    # =========================================================================

    baseline_auc = roc_auc_score(
        pred["target"],
        pred[
            "baseline_prediction"
        ]
    )

    breach_auc = roc_auc_score(
        pred["target"],
        pred[
            "breach_prediction"
        ]
    )

    joint_auc = roc_auc_score(
        pred["target"],
        pred[
            "joint_prediction"
        ]
    )


    print(
        "\nEligible tickers:",
        eligible_tickers
    )

    print(
        "Pooled test observations:",
        f"{len(pred):,}"
    )

    print(
        f"Baseline AUC:                  "
        f"{baseline_auc:.4f}"
    )

    print(
        f"Baseline + breach count AUC:   "
        f"{breach_auc:.4f}"
    )

    print(
        f"Baseline + count + streak AUC: "
        f"{joint_auc:.4f}"
    )


    # =========================================================================
    # BOOTSTRAP BY BLOCK LENGTH
    # =========================================================================

    for block_length in BLOCK_LENGTHS:

        print(
            "\n"
            + "-" * 100
        )

        print(
            f"BLOCK LENGTH = {block_length} DATES"
        )

        print(
            "-" * 100
        )


        bootstrap = date_block_bootstrap(

            prediction_df=pred,

            baseline_column=
                "baseline_prediction",

            augmented_columns=[
                "breach_prediction",
                "joint_prediction"
            ],

            block_length=
                block_length,

            n_bootstrap=
                N_BOOTSTRAPS,

            random_state=
                RANDOM_STATE
                + horizon
                + block_length
        )


        for model_name, label in [
            (
                "breach_prediction",
                "Baseline + breach count"
            ),
            (
                "joint_prediction",
                "Baseline + breach count + streak"
            )
        ]:

            br = bootstrap[
                model_name
            ]

            if model_name == "breach_prediction":

                observed = breach_auc

            else:

                observed = joint_auc


            delta_observed = (
                observed
                - baseline_auc
            )


            print(
                f"\n{label}"
            )

            print(
                f"  Observed AUC:      "
                f"{observed:.4f}"
            )

            print(
                f"  Observed delta:    "
                f"{delta_observed:+.4f}"
            )

            print(
                f"  Bootstrap delta CI:"
                f" "
                f"[{br['delta_low']:+.4f}, "
                f"{br['delta_high']:+.4f}]"
            )

            print(
                f"  Robust positive?:   "
                f"{br['delta_low'] > 0}"
            )


            sensitivity_results.append(
                {
                    "horizon":
                        horizon,

                    "block_length_dates":
                        block_length,

                    "model":
                        model_name,

                    "eligible_tickers":
                        eligible_tickers,

                    "test_observations":
                        len(pred),

                    "baseline_auc":
                        baseline_auc,

                    "augmented_auc":
                        observed,

                    "observed_delta":
                        delta_observed,

                    "bootstrap_auc_mean":
                        br["auc_mean"],

                    "bootstrap_auc_ci_low":
                        br["auc_low"],

                    "bootstrap_auc_ci_high":
                        br["auc_high"],

                    "bootstrap_delta_mean":
                        br["delta_mean"],

                    "bootstrap_delta_ci_low":
                        br["delta_low"],

                    "bootstrap_delta_ci_high":
                        br["delta_high"],

                    "robust_positive":
                        br["delta_low"] > 0
                }
            )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

results_df = pd.DataFrame(
    sensitivity_results
)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)


print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL BLOCK-LENGTH SENSITIVITY SUMMARY"
)

print(
    "=" * 100
)

if not results_df.empty:

    print(
        results_df[
            [
                "horizon",
                "block_length_dates",
                "model",
                "baseline_auc",
                "augmented_auc",
                "observed_delta",
                "bootstrap_delta_ci_low",
                "bootstrap_delta_ci_high",
                "robust_positive"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# =============================================================================
# GO / NO-GO SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "GO / NO-GO SENSITIVITY SUMMARY"
)

print(
    "=" * 100
)


for model in [
    "breach_prediction",
    "joint_prediction"
]:

    sub = results_df[
        results_df[
            "model"
        ] == model
    ]

    print(
        f"\n{model}"
    )

    for horizon in HORIZONS:

        h = sub[
            sub["horizon"]
            == horizon
        ]

        positive_blocks = h[
            h["robust_positive"]
        ][
            "block_length_dates"
        ].tolist()

        print(
            f"  {horizon}d -> "
            f"positive at block lengths: "
            f"{positive_blocks}"
        )


# =============================================================================
# AUDIT SAVE
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

results_df.to_csv(
    OUTPUT_SENSITIVITY,
    index=False
)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

print(
    OUTPUT_AUDIT
)

print(
    OUTPUT_SENSITIVITY
)

print(
    OUTPUT_PREDICTIONS
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 36 COMPLETE"
)

print(
    "=" * 100
)