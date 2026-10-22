from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 33 — FORMAL BREACH-HISTORY PREDICTIVE TEST
# =============================================================================
#
# Research question:
#
#   Does recent VaR breach history predict future deterioration in VaR
#   forecast quality?
#
# Primary dimensions of breach history:
#
#   1. Breach frequency:
#        - prior 5-observation breach count
#        - prior 10-observation breach count
#        - prior 20-observation breach count
#
#   2. Breach persistence:
#        - prior breach streak
#
#   3. Breach recency:
#        - observations since most recent breach
#
# Main joint model:
#
#   prior 5-day breach count
#   + prior breach streak
#   + days since most recent breach
#
# Evaluation:
#
#   - 60/40 chronological split within each ticker
#   - train-only standardization
#   - train-only logistic regression
#   - pooled OOS AUC
#   - AUC bootstrap CI using moving blocks within ticker
#   - incremental AUC versus persistence
#   - cross-sectional ticker bootstrap
#
# Primary targets:
#
#   deterioration_5d
#   deterioration_10d
#   deterioration_20d
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

OUTPUT_MAIN = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_formal_test_results.csv"
)

OUTPUT_TICKER = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_formal_test_by_ticker.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_formal_test_predictions.csv"
)

TRAIN_FRACTION = 0.60

BLOCK_LENGTH = 20

WITHIN_TICKER_BOOTSTRAPS = 1000

TICKER_BOOTSTRAPS = 2000

RANDOM_STATE = 42

HORIZONS = [5, 10, 20]

MIN_TICKER_OBSERVATIONS = 100


# =============================================================================
# HELPER — PRIMARY TARGET
# =============================================================================

def get_primary_target(df, horizon):

    target = f"deterioration_{horizon}d"

    if target not in df.columns:

        raise ValueError(
            f"Primary target not found: {target}"
        )

    return target


# =============================================================================
# HELPER — LOGISTIC MODEL
# =============================================================================

def fit_logistic(
    train_x,
    train_y,
    test_x
):

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

    train_scaled = scaler.fit_transform(
        train_x
    )

    test_scaled = scaler.transform(
        test_x
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        train_scaled,
        train_y
    )

    return model.predict_proba(
        test_scaled
    )[:, 1]


# =============================================================================
# HELPER — SAFE AUC
# =============================================================================

def safe_auc(
    y,
    scores
):

    y = np.asarray(
        y,
        dtype=int
    )

    scores = np.asarray(
        scores,
        dtype=float
    )

    if len(np.unique(y)) < 2:

        return np.nan

    return roc_auc_score(
        y,
        scores
    )


# =============================================================================
# HELPER — MOVING BLOCK BOOTSTRAP INDICES
# =============================================================================

def moving_block_indices(
    n,
    block_length,
    rng
):

    if n <= 0:

        return np.array(
            [],
            dtype=int
        )

    block_length = min(
        block_length,
        n
    )

    starts = np.arange(
        0,
        n - block_length + 1
    )

    indices = []

    while len(indices) < n:

        start = rng.choice(
            starts
        )

        block = np.arange(
            start,
            start + block_length
        )

        indices.extend(
            block.tolist()
        )

    return np.asarray(
        indices[:n],
        dtype=int
    )


# =============================================================================
# HELPER — DAYS SINCE LAST BREACH
# =============================================================================

def calculate_days_since_last_breach(group):

    """
    Causal calculation.

    At date t, the value is based only on breaches strictly before t.

    Example:

        breach at t-1 -> 1
        breach at t-2 -> 2
        breach at t-5 -> 5
        no historical breach -> NaN
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

    last_breach = None

    for i in range(len(group)):

        if last_breach is not None:

            result[i] = (
                i - last_breach
            )

        # Today's realized breach becomes available
        # only for the next observation.
        if breaches[i] == 1:

            last_breach = i

    return result


# =============================================================================
# HELPER — BOOTSTRAP POOLED AUC AND DELTA
# =============================================================================

def bootstrap_auc_and_delta(
    prediction_df,
    signal_prediction_column,
    baseline_prediction_column=None,
    n_bootstrap=1000,
    block_length=20,
    random_state=42
):

    """
    Moving-block bootstrap independently within each ticker.

    Returns:

        signal_auc_mean
        signal_auc_low
        signal_auc_high

        delta_mean
        delta_low
        delta_high

    When baseline_prediction_column is None,
    delta outputs are NaN.
    """

    rng = np.random.default_rng(
        random_state
    )

    required = [
        "ticker",
        "target",
        signal_prediction_column
    ]

    if baseline_prediction_column is not None:

        required.append(
            baseline_prediction_column
        )

    for column in required:

        if column not in prediction_df.columns:

            raise ValueError(
                f"Missing bootstrap column: {column}"
            )

    ticker_arrays = []

    for ticker, g in prediction_df.groupby(
        "ticker"
    ):

        cols = [
            "target",
            signal_prediction_column
        ]

        if baseline_prediction_column is not None:

            cols.append(
                baseline_prediction_column
            )

        g = g.dropna(
            subset=cols
        )

        if len(g) < 20:

            continue

        y = (
            g["target"]
            .astype(int)
            .to_numpy()
        )

        signal = (
            g[signal_prediction_column]
            .astype(float)
            .to_numpy()
        )

        if baseline_prediction_column is not None:

            baseline = (
                g[baseline_prediction_column]
                .astype(float)
                .to_numpy()
            )

        else:

            baseline = None

        if len(np.unique(y)) < 2:

            continue

        ticker_arrays.append(
            (
                ticker,
                y,
                signal,
                baseline
            )
        )

    if not ticker_arrays:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    signal_aucs = []
    deltas = []

    for _ in range(
        n_bootstrap
    ):

        pooled_y = []
        pooled_signal = []
        pooled_baseline = []

        for (
            ticker,
            y,
            signal,
            baseline
        ) in ticker_arrays:

            idx = moving_block_indices(
                n=len(y),
                block_length=block_length,
                rng=rng
            )

            pooled_y.append(
                y[idx]
            )

            pooled_signal.append(
                signal[idx]
            )

            if baseline is not None:

                pooled_baseline.append(
                    baseline[idx]
                )

        y_all = np.concatenate(
            pooled_y
        )

        signal_all = np.concatenate(
            pooled_signal
        )

        if len(np.unique(y_all)) < 2:

            continue

        signal_auc = roc_auc_score(
            y_all,
            signal_all
        )

        signal_aucs.append(
            signal_auc
        )

        if baseline_prediction_column is not None:

            baseline_all = np.concatenate(
                pooled_baseline
            )

            baseline_auc = roc_auc_score(
                y_all,
                baseline_all
            )

            deltas.append(
                signal_auc - baseline_auc
            )

    if not signal_aucs:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    signal_aucs = np.asarray(
        signal_aucs
    )

    if deltas:

        deltas = np.asarray(
            deltas
        )

        delta_mean = (
            np.mean(deltas)
        )

        delta_low = (
            np.percentile(
                deltas,
                2.5
            )
        )

        delta_high = (
            np.percentile(
                deltas,
                97.5
            )
        )

    else:

        delta_mean = np.nan
        delta_low = np.nan
        delta_high = np.nan

    return (
        np.mean(signal_aucs),
        np.percentile(
            signal_aucs,
            2.5
        ),
        np.percentile(
            signal_aucs,
            97.5
        ),
        delta_mean,
        delta_low,
        delta_high
    )


# =============================================================================
# HELPER — CROSS-SECTIONAL TICKER BOOTSTRAP
# =============================================================================

def ticker_bootstrap_ci(
    values,
    n_bootstrap=2000,
    random_state=42
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    rng = np.random.default_rng(
        random_state
    )

    means = []

    for _ in range(
        n_bootstrap
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        means.append(
            np.mean(sample)
        )

    means = np.asarray(
        means
    )

    return (
        np.mean(values),
        np.percentile(
            means,
            2.5
        ),
        np.percentile(
            means,
            97.5
        )
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 33 — FORMAL BREACH-HISTORY PREDICTIVE TEST")
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
    f"Block length:         {BLOCK_LENGTH}"
)

print(
    f"Within-ticker boots:  {WITHIN_TICKER_BOOTSTRAPS}"
)

print(
    f"Ticker boots:         {TICKER_BOOTSTRAPS}"
)

print(
    f"Random state:         {RANDOM_STATE}"
)


# =============================================================================
# LOAD PANEL
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

print("\nDuplicate ticker-date rows:")

print(
    df.duplicated(
        subset=[
            "ticker",
            "forecast_date"
        ]
    ).sum()
)


# =============================================================================
# BASIC TYPES
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
# REQUIRED ORIGINAL FEATURES
# =============================================================================

required_columns = [
    "ticker",
    "forecast_date",
    "breach",
    "breach_streak_prior"
]

missing_columns = [
    c
    for c in required_columns
    if c not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================================
# CONSTRUCT CAUSAL BREACH COUNTS + RECENCY
# =============================================================================

print("\n" + "=" * 100)
print("CONSTRUCTING CAUSAL BREACH-HISTORY VARIABLES")
print("=" * 100)

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

    # Strictly lagged breach sequence
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

    g["days_since_last_breach"] = (
        calculate_days_since_last_breach(
            g
        )
    )

    groups.append(
        g
    )


df = pd.concat(
    groups,
    ignore_index=True
)


# =============================================================================
# CAUSALITY CHECK
# =============================================================================

print("\nRows:")
print(len(df))

print(
    "\nMissing prior 5-day count:",
    df["prior_breach_count_5"].isna().sum()
)

print(
    "Missing prior 10-day count:",
    df["prior_breach_count_10"].isna().sum()
)

print(
    "Missing prior 20-day count:",
    df["prior_breach_count_20"].isna().sum()
)

print(
    "Missing prior breach streak:",
    df["breach_streak_prior"].isna().sum()
)

print(
    "Missing breach recency:",
    df["days_since_last_breach"].isna().sum()
)


# =============================================================================
# TICKERS
# =============================================================================

tickers = sorted(
    df["ticker"]
    .dropna()
    .unique()
)

print(
    "\nTickers found:",
    len(tickers)
)

print(
    tickers
)


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

MODEL_FEATURES = {

    "breach_count_5":
        [
            "prior_breach_count_5"
        ],

    "breach_count_10":
        [
            "prior_breach_count_10"
        ],

    "breach_count_20":
        [
            "prior_breach_count_20"
        ],

    "breach_streak":
        [
            "breach_streak_prior"
        ],

    "breach_recency":
        [
            "days_since_last_breach"
        ],

    # Main joint model
    "joint_breach_history":
        [
            "prior_breach_count_5",
            "breach_streak_prior",
            "days_since_last_breach"
        ]
}


# =============================================================================
# RESULTS STORAGE
# =============================================================================

results = []

ticker_results = []

prediction_results = []


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
    # STORE TEST PREDICTIONS FOR THIS HORIZON
    # =========================================================================

    horizon_prediction_tables = []

    eligible_tickers = 0


    # =========================================================================
    # TICKER LOOP
    # =========================================================================

    for ticker in tickers:

        # ---------------------------------------------------------------------
        # USE FULL TICKER HISTORY FOR SPLIT
        # ---------------------------------------------------------------------

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

        if len(g) < MIN_TICKER_OBSERVATIONS:

            continue

        # ---------------------------------------------------------------------
        # CRITICAL:
        # Split FIRST.
        #
        # Only afterwards do we remove missing observations.
        #
        # This keeps the chronological split consistent across every model.
        # ---------------------------------------------------------------------

        split_idx = int(
            np.floor(
                len(g)
                * TRAIN_FRACTION
            )
        )

        if (
            split_idx <= 30
            or
            split_idx >= len(g) - 30
        ):

            continue

        train_raw = (
            g.iloc[
                :split_idx
            ]
            .copy()
        )

        test_raw = (
            g.iloc[
                split_idx:
            ]
            .copy()
        )


        # =====================================================================
        # FIT EACH MODEL
        # =====================================================================

        predictions = {
            "ticker":
                ticker,

            "forecast_date":
                test_raw[
                    "forecast_date"
                ].to_numpy(),

            "target":
                test_raw[
                    target
                ]
                .to_numpy()
        }


        ticker_model_aucs = {}


        for model_name, features in MODEL_FEATURES.items():

            required = (
                features
                + [target]
            )

            train = (
                train_raw
                .dropna(
                    subset=required
                )
                .copy()
            )

            test = (
                test_raw
                .dropna(
                    subset=required
                )
                .copy()
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


            # -----------------------------------------------------------------
            # FIT
            # -----------------------------------------------------------------

            pred = fit_logistic(
                train_x=
                    train[features],

                train_y=
                    train[target],

                test_x=
                    test[features]
            )

            if pred is None:

                continue


            auc = safe_auc(
                test[target],
                pred
            )

            if np.isnan(auc):

                continue


            ticker_model_aucs[
                model_name
            ] = auc


            # -----------------------------------------------------------------
            # Store aligned test prediction data
            # -----------------------------------------------------------------

            temp = test[
                [
                    "ticker",
                    "forecast_date",
                    target
                ]
            ].copy()

            temp["target_value"] = (
                temp[target]
                .astype(int)
            )

            temp["horizon"] = horizon

            temp["model"] = model_name

            temp["prediction"] = pred

            horizon_prediction_tables.append(
                temp[
                    [
                        "ticker",
                        "forecast_date",
                        "target_value",
                        "horizon",
                        "model",
                        "prediction"
                    ]
                ]
            )


        # =====================================================================
        # TICKER-LEVEL AUC TABLE
        # =====================================================================

        if ticker_model_aucs:

            eligible_tickers += 1

            row = {
                "horizon":
                    horizon,

                "ticker":
                    ticker
            }

            for model_name in MODEL_FEATURES:

                row[
                    f"{model_name}_auc"
                ] = ticker_model_aucs.get(
                    model_name,
                    np.nan
                )

            ticker_results.append(
                row
            )


    # =========================================================================
    # POOLED TEST PREDICTIONS
    # =========================================================================

    if not horizon_prediction_tables:

        print(
            "\nNo eligible predictions."
        )

        continue


    pred_long = pd.concat(
        horizon_prediction_tables,
        ignore_index=True
    )


    # =========================================================================
    # PIVOT TO WIDE
    # =========================================================================

    pred_wide = (
        pred_long
        .pivot_table(
            index=[
                "ticker",
                "forecast_date",
                "horizon"
            ],
            columns="model",
            values="prediction",
            aggfunc="first"
        )
        .reset_index()
    )

    target_values = (
        pred_long[
            [
                "ticker",
                "forecast_date",
                "horizon",
                "target_value"
            ]
        ]
        .drop_duplicates(
            subset=[
                "ticker",
                "forecast_date",
                "horizon"
            ]
        )
    )

    pred_wide = pred_wide.merge(
        target_values,
        on=[
            "ticker",
            "forecast_date",
            "horizon"
        ],
        how="left",
        validate="one_to_one"
    )


    # =========================================================================
    # POOLED OOS AUC
    # =========================================================================

    pooled_auc = {}

    for model_name in MODEL_FEATURES:

        if model_name not in pred_wide.columns:

            pooled_auc[
                model_name
            ] = np.nan

            continue

        sub = pred_wide.dropna(
            subset=[
                "target_value",
                model_name
            ]
        )

        if len(
            np.unique(
                sub["target_value"]
            )
        ) < 2:

            pooled_auc[
                model_name
            ] = np.nan

        else:

            pooled_auc[
                model_name
            ] = roc_auc_score(
                sub["target_value"],
                sub[model_name]
            )


    # =========================================================================
    # BASELINE = BREACH STREAK / PERSISTENCE
    # =========================================================================

    baseline_model = "breach_streak"

    baseline_auc = pooled_auc.get(
        baseline_model,
        np.nan
    )


    # =========================================================================
    # PRINT POOLED RESULTS
    # =========================================================================

    print(
        f"\nEligible tickers:       "
        f"{eligible_tickers}"
    )

    print(
        f"Pooled test rows:       "
        f"{len(pred_wide):,}"
    )

    print(
        "\nPooled OOS AUC:"
    )

    for model_name in MODEL_FEATURES:

        print(
            f"  {model_name:<25}"
            f"{pooled_auc.get(model_name, np.nan):.4f}"
        )


    # =========================================================================
    # BLOCK BOOTSTRAP
    # =========================================================================

    bootstrap_metrics = {}

    for model_name in MODEL_FEATURES:

        if model_name == baseline_model:

            (
                auc_mean,
                auc_low,
                auc_high,
                _,
                _,
                _
            ) = bootstrap_auc_and_delta(

                prediction_df=
                    pred_wide.rename(
                        columns={
                            "target_value":
                                "target"
                        }
                    ),

                signal_prediction_column=
                    model_name,

                baseline_prediction_column=None,

                n_bootstrap=
                    WITHIN_TICKER_BOOTSTRAPS,

                block_length=
                    BLOCK_LENGTH,

                random_state=
                    RANDOM_STATE
                    + horizon
            )

        else:

            (
                auc_mean,
                auc_low,
                auc_high,
                delta_mean,
                delta_low,
                delta_high
            ) = bootstrap_auc_and_delta(

                prediction_df=
                    pred_wide.rename(
                        columns={
                            "target_value":
                                "target"
                        }
                    ),

                signal_prediction_column=
                    model_name,

                baseline_prediction_column=
                    baseline_model,

                n_bootstrap=
                    WITHIN_TICKER_BOOTSTRAPS,

                block_length=
                    BLOCK_LENGTH,

                random_state=
                    RANDOM_STATE
                    + horizon
                    + (
                        100 * list(
                            MODEL_FEATURES.keys()
                        ).index(model_name)
                    )
            )

        bootstrap_metrics[
            model_name
        ] = {
            "auc_mean":
                auc_mean,

            "auc_ci_low":
                auc_low,

            "auc_ci_high":
                auc_high,

            "delta_mean":
                delta_mean,

            "delta_ci_low":
                delta_low,

            "delta_ci_high":
                delta_high
        }


    # =========================================================================
    # CROSS-SECTIONAL TICKER DELTAS
    # =========================================================================

    ticker_df_h = pd.DataFrame(
        [
            row
            for row in ticker_results
            if row["horizon"] == horizon
        ]
    )

    cross_sectional_metrics = {}

    for model_name in MODEL_FEATURES:

        if model_name == baseline_model:

            continue

        delta_column = (
            f"{model_name}_auc"
        )

        baseline_column = (
            f"{baseline_model}_auc"
        )

        available = ticker_df_h.dropna(
            subset=[
                delta_column,
                baseline_column
            ]
        ).copy()

        if available.empty:

            continue

        available["delta"] = (
            available[
                delta_column
            ]
            -
            available[
                baseline_column
            ]
        )

        (
            mean_delta,
            ticker_low,
            ticker_high
        ) = ticker_bootstrap_ci(

            available[
                "delta"
            ]
            .to_numpy(),

            n_bootstrap=
                TICKER_BOOTSTRAPS,

            random_state=
                RANDOM_STATE
                + 500
                + horizon
        )

        cross_sectional_metrics[
            model_name
        ] = {
            "mean_delta":
                mean_delta,

            "ci_low":
                ticker_low,

            "ci_high":
                ticker_high,

            "better_tickers":
                int(
                    (
                        available[
                            "delta"
                        ]
                        > 0
                    ).sum()
                ),

            "ticker_count":
                len(available)
        }


    # =========================================================================
    # STORE RESULTS
    # =========================================================================

    for model_name in MODEL_FEATURES:

        if model_name == baseline_model:

            cs = {
                "mean_delta":
                    np.nan,

                "ci_low":
                    np.nan,

                "ci_high":
                    np.nan,

                "better_tickers":
                    np.nan,

                "ticker_count":
                    np.nan
            }

        else:

            cs = (
                cross_sectional_metrics
                .get(
                    model_name,
                    {
                        "mean_delta":
                            np.nan,

                        "ci_low":
                            np.nan,

                        "ci_high":
                            np.nan,

                        "better_tickers":
                            np.nan,

                        "ticker_count":
                            np.nan
                    }
                )
            )


        bm = (
            bootstrap_metrics
            .get(
                model_name,
                {}
            )
        )

        results.append(
            {
                "horizon":
                    horizon,

                "model":
                    model_name,

                "eligible_tickers":
                    eligible_tickers,

                "pooled_test_rows":
                    len(pred_wide),

                "pooled_auc":
                    pooled_auc.get(
                        model_name,
                        np.nan
                    ),

                "bootstrap_auc_mean":
                    bm.get(
                        "auc_mean",
                        np.nan
                    ),

                "bootstrap_auc_ci_low":
                    bm.get(
                        "auc_ci_low",
                        np.nan
                    ),

                "bootstrap_auc_ci_high":
                    bm.get(
                        "auc_ci_high",
                        np.nan
                    ),

                "delta_vs_persistence":
                    (
                        pooled_auc.get(
                            model_name,
                            np.nan
                        )
                        - baseline_auc
                        if model_name
                        != baseline_model
                        else np.nan
                    ),

                "block_delta_mean":
                    bm.get(
                        "delta_mean",
                        np.nan
                    ),

                "block_delta_ci_low":
                    bm.get(
                        "delta_ci_low",
                        np.nan
                    ),

                "block_delta_ci_high":
                    bm.get(
                        "delta_ci_high",
                        np.nan
                    ),

                "mean_ticker_delta_vs_persistence":
                    cs[
                        "mean_delta"
                    ],

                "ticker_ci_low":
                    cs[
                        "ci_low"
                    ],

                "ticker_ci_high":
                    cs[
                        "ci_high"
                    ],

                "better_tickers":
                    cs[
                        "better_tickers"
                    ],

                "ticker_count":
                    cs[
                        "ticker_count"
                    ]
            }
        )


    # =========================================================================
    # PRINT BOOTSTRAP RESULTS
    # =========================================================================

    print(
        "\nBootstrap results:"
    )

    for model_name in MODEL_FEATURES:

        bm = bootstrap_metrics[
            model_name
        ]

        print(
            f"\n{model_name}"
        )

        print(
            f"  OOS AUC:            "
            f"{pooled_auc[model_name]:.4f}"
        )

        print(
            f"  Bootstrap AUC CI:   "
            f"[{bm['auc_ci_low']:.4f}, "
            f"{bm['auc_ci_high']:.4f}]"
        )

        if model_name != baseline_model:

            print(
                f"  Delta vs persistence:"
                f" {pooled_auc[model_name] - baseline_auc:+.4f}"
            )

            print(
                f"  Block delta CI:     "
                f"[{bm['delta_ci_low']:+.4f}, "
                f"{bm['delta_ci_high']:+.4f}]"
            )

            cs = cross_sectional_metrics[
                model_name
            ]

            print(
                f"  Mean ticker delta:   "
                f"{cs['mean_delta']:+.4f}"
            )

            print(
                f"  Ticker delta CI:     "
                f"[{cs['ci_low']:+.4f}, "
                f"{cs['ci_high']:+.4f}]"
            )

            print(
                f"  Better tickers:      "
                f"{int(cs['better_tickers'])}/"
                f"{int(cs['ticker_count'])}"
            )


    # =========================================================================
    # ADD PREDICTIONS TO FINAL OUTPUT
    # =========================================================================

    prediction_results.append(
        pred_wide.copy()
    )


# =============================================================================
# COMBINE RESULTS
# =============================================================================

results_df = pd.DataFrame(
    results
)

ticker_results_df = pd.DataFrame(
    ticker_results
)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)


# =============================================================================
# FINAL RESULTS TABLE
# =============================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL EXPERIMENT 33 SUMMARY"
)

print(
    "=" * 100
)

if not results_df.empty:

    print(
        results_df[
            [
                "horizon",
                "model",
                "pooled_auc",
                "bootstrap_auc_ci_low",
                "bootstrap_auc_ci_high",
                "delta_vs_persistence",
                "block_delta_ci_low",
                "block_delta_ci_high",
                "mean_ticker_delta_vs_persistence",
                "ticker_ci_low",
                "ticker_ci_high",
                "better_tickers",
                "ticker_count"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# =============================================================================
# FORMAL INTERPRETATION
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FORMAL INTERPRETATION"
)

print(
    "=" * 100
)

if not results_df.empty:

    for horizon in HORIZONS:

        h = results_df[
            results_df["horizon"]
            == horizon
        ]

        print(
            f"\n{horizon}-DAY TARGET"
        )

        print(
            "-" * 80
        )

        for _, row in h.iterrows():

            model = row["model"]

            auc = row["pooled_auc"]

            auc_low = (
                row[
                    "bootstrap_auc_ci_low"
                ]
            )

            auc_high = (
                row[
                    "bootstrap_auc_ci_high"
                ]
            )

            print(
                f"{model:<25}"
                f"AUC={auc:.4f}  "
                f"95% CI=[{auc_low:.4f}, "
                f"{auc_high:.4f}]"
            )

            if model == "breach_streak":

                if auc_low > 0.5:

                    print(
                        "  -> Persistence shows "
                        "positive predictive discrimination."
                    )

                else:

                    print(
                        "  -> Persistence does not have "
                        "a bootstrap CI entirely above 0.5."
                    )

            else:

                delta_low = (
                    row[
                        "block_delta_ci_low"
                    ]
                )

                delta_high = (
                    row[
                        "block_delta_ci_high"
                    ]
                )

                print(
                    f"  -> Delta vs persistence: "
                    f"{row['delta_vs_persistence']:+.4f}"
                )

                print(
                    f"  -> Block delta CI: "
                    f"[{delta_low:+.4f}, "
                    f"{delta_high:+.4f}]"
                )


# =============================================================================
# MAIN CONCLUSION
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MAIN CONCLUSION"
)

print(
    "=" * 100
)

if not results_df.empty:

    joint = results_df[
        results_df["model"]
        == "joint_breach_history"
    ].copy()

    if not joint.empty:

        print(
            "\nJoint breach-history model:"
        )

        for _, row in joint.iterrows():

            print(
                f"  {int(row['horizon'])}-day: "
                f"AUC={row['pooled_auc']:.4f}, "
                f"delta vs persistence="
                f"{row['delta_vs_persistence']:+.4f}, "
                f"block CI="
                f"[{row['block_delta_ci_low']:+.4f}, "
                f"{row['block_delta_ci_high']:+.4f}]"
            )

        robust_horizons = joint[
            joint[
                "block_delta_ci_low"
            ]
            > 0
        ]["horizon"].tolist()

        if robust_horizons:

            print(
                "\nThe joint breach-history model shows "
                "statistically positive incremental value "
                f"at horizon(s): {robust_horizons}"
            )

        else:

            print(
                "\nThe joint breach-history model does not "
                "show statistically robust incremental "
                "value over persistence."
            )


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv(
    OUTPUT_MAIN,
    index=False
)

ticker_results_df.to_csv(
    OUTPUT_TICKER,
    index=False
)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
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
    OUTPUT_MAIN
)

print(
    OUTPUT_TICKER
)

print(
    OUTPUT_PREDICTIONS
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 33 COMPLETE"
)

print(
    "=" * 100
)