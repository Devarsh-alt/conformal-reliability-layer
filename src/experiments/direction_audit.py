from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 39
#
# BASELINE DIRECTION AUDIT
# + TEST VS CHANCE
# + VALIDATION-SELECTED SIMPLE SIGNAL
# + PER-TICKER ROBUSTNESS
# + BLOCK BOOTSTRAP
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

QRF_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

EWMA_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "EWMA_25ticker_oos_predictions.csv"
)

RESULTS_DIR = (
    ROOT
    / "data"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SUMMARY_PATH = (
    RESULTS_DIR
    / "experiment_39_direction_audit.csv"
)

SIGNAL_PATH = (
    RESULTS_DIR
    / "experiment_39_signal_selection.csv"
)

TICKER_PATH = (
    RESULTS_DIR
    / "experiment_39_per_ticker.csv"
)

BOOTSTRAP_PATH = (
    RESULTS_DIR
    / "experiment_39_bootstrap.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "experiment_39_predictions.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [
    5,
    10,
    20,
]

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

MIN_TARGET_HISTORY = 50

BOOTSTRAP_REPS = 300

BLOCK_LENGTHS = [
    10,
    20,
    40,
    60,
]

RANDOM_STATE = 42


# =============================================================================
# XGBOOST
# =============================================================================

XGB_PARAMS = dict(
    n_estimators=180,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=8,
    reg_alpha=0.05,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    max_bin=64,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# =============================================================================
# FEATURES
# =============================================================================

BASE_FEATURES = [
    "pinball_lag1",
    "vol_ratio_20_60",
]


FEATURE_GROUPS = {

    "loss_history": [

        "pinball_lag1",
        "pinball_mean_5d_prior",
        "pinball_mean_20d_prior",
        "pinball_mean_60d_prior",
        "pinball_acceleration",
        "loss_ratio_5_20",
        "loss_ratio_5_60",

        "pinball_lag1_ticker_causal_z",
        "pinball_mean_5d_prior_ticker_causal_z",
        "pinball_mean_20d_prior_ticker_causal_z",
    ],


    "breach_history": [

        "prior_breach_count_5",
        "prior_breach_count_20",
        "breach_streak_prior",
        "breach_rate_5d",
        "breach_rate_20d",
        "breach_rate_60d",
        "ewma_breach_rate",

        "breach_rate_accel_5_20",
        "breach_rate_accel_20_60",

        "prior_breach_count_5_ticker_causal_z",
    ],


    "volatility": [

        "realized_vol_20d",
        "realized_vol_60d",
        "vol_ratio_20_60",
        "vol_ratio_change_1d",
        "ks_drift_60_vs_250",

        "vol_ratio_20_60_ticker_causal_z",
    ],


    "var_dynamics": [

        "q05_change_1d",
        "VaR_change_1d",
        "VaR_ratio_to_prior",
        "VaR_ratio_to_20d_mean",
        "VaR_ratio_change_1d",

        "q05_change_1d_ticker_causal_z",
        "VaR_ratio_to_prior_ticker_causal_z",
    ],


    "model_disagreement": [

        "qrf_ewma_disagreement",
        "qrf_ewma_disagreement_ticker_causal_z",
    ],


    "market_context": [

        "VIX",
        "volume_ratio",
        "csb_prior_1d",
        "csb_prior_5d",
    ],


    "interactions": [

        "loss_x_vol",
        "breach5_x_vol",
        "loss_x_breach5",
        "vol_x_vix",
    ],
}


ALL_FEATURES = []

for group_features in FEATURE_GROUPS.values():

    for f in group_features:

        if f not in ALL_FEATURES:

            ALL_FEATURES.append(f)


# =============================================================================
# FUNCTIONS
# =============================================================================

def pinball_loss(
    y,
    q,
    alpha=0.05,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    q = np.asarray(
        q,
        dtype=float,
    )

    return np.where(
        y < q,
        (alpha - 1.0) * (y - q),
        alpha * (y - q),
    )


def future_mean(
    series,
    horizon,
):
    """
    Mean of observations t+1 ... t+h.
    """

    return (
        series
        .shift(-1)
        .rolling(
            window=horizon,
            min_periods=horizon,
        )
        .mean()
        .shift(-(horizon - 1))
    )


def strictly_causal_target(
    future_loss,
    horizon,
    percentile=0.75,
    min_history=50,
):
    """
    At time t, the target threshold uses only fully observed
    historical future-loss windows.

    Latest allowable historical window:
        t-h-1
    """

    historical_loss = (
        future_loss
        .shift(horizon + 1)
    )


    threshold = (
        historical_loss
        .expanding(
            min_periods=min_history
        )
        .quantile(
            percentile
        )
    )


    target = pd.Series(
        np.nan,
        index=future_loss.index,
        dtype=float,
    )


    valid = (
        future_loss.notna()
        &
        threshold.notna()
    )


    target.loc[valid] = (
        future_loss.loc[valid]
        >
        threshold.loc[valid]
    ).astype(float)


    return target


def causal_expanding_z(series):

    prior = series.shift(1)


    prior_mean = (
        prior
        .expanding(
            min_periods=20
        )
        .mean()
    )


    prior_std = (
        prior
        .expanding(
            min_periods=20
        )
        .std(
            ddof=0
        )
    )


    z = (
        series - prior_mean
    ) / prior_std.replace(
        0,
        np.nan,
    )


    return z.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def make_splits(df):

    train = pd.Series(
        False,
        index=df.index,
    )

    val = pd.Series(
        False,
        index=df.index,
    )

    test = pd.Series(
        False,
        index=df.index,
    )


    for _, idx in df.groupby(
        "ticker",
        sort=False,
    ).groups.items():

        ordered = (
            df.loc[idx]
            .sort_values(
                "forecast_date"
            )
        )


        n = len(
            ordered
        )


        n_train = int(
            np.floor(
                n * TRAIN_FRAC
            )
        )


        n_val = int(
            np.floor(
                n * VAL_FRAC
            )
        )


        train.loc[
            ordered.index[
                :n_train
            ]
        ] = True


        val.loc[
            ordered.index[
                n_train:
                n_train + n_val
            ]
        ] = True


        test.loc[
            ordered.index[
                n_train + n_val:
            ]
        ] = True


    return (
        train,
        val,
        test,
    )


def prepare_matrix(
    df,
    features,
    train_mask,
    include_ticker_fe=True,
):

    features = [
        f
        for f in features
        if f in df.columns
    ]


    if not features:

        raise ValueError(
            "No valid features supplied."
        )


    X = (
        df[
            features
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .copy()
    )


    train_median = (
        X.loc[
            train_mask
        ]
        .median()
    )


    X = (
        X
        .fillna(
            train_median
        )
        .fillna(
            0.0
        )
    )


    scaler = StandardScaler()


    scaler.fit(
        X.loc[
            train_mask,
            features,
        ]
    )


    X.loc[:, features] = (
        scaler.transform(
            X.loc[:, features]
        )
    )


    if include_ticker_fe:

        ticker_dummies = pd.get_dummies(
            df[
                "ticker"
            ].astype(str),
            prefix="ticker",
            dtype=float,
        )


        X = pd.concat(
            [
                X,
                ticker_dummies,
            ],
            axis=1,
        )


    return X


def safe_auc(
    y,
    score,
):

    y = np.asarray(
        y,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )


    if np.unique(
        y
    ).size < 2:

        return np.nan


    return float(
        roc_auc_score(
            y,
            score,
        )
    )


def build_date_blocks(
    pred,
    block_days,
):

    dates = (
        pd.to_datetime(
            pred[
                "forecast_date"
            ]
        )
        .dt.normalize()
        .to_numpy()
    )


    unique_dates = np.array(
        sorted(
            np.unique(
                dates
            )
        )
    )


    date_to_rows = {}


    for d in unique_dates:

        date_to_rows[d] = (
            np.flatnonzero(
                dates == d
            )
        )


    blocks = []


    for start in range(
        0,
        len(unique_dates),
        block_days,
    ):

        chosen_dates = (
            unique_dates[
                start:
                start + block_days
            ]
        )


        pieces = [
            date_to_rows[d]
            for d in chosen_dates
        ]


        if pieces:

            blocks.append(
                np.concatenate(
                    pieces
                )
            )


    return blocks


def bootstrap_delta(
    pred,
    baseline_col,
    model_col,
    block_days,
    reps=300,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )


    blocks = build_date_blocks(
        pred,
        block_days,
    )


    if not blocks:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    y = pred[
        "y_true"
    ].to_numpy(
        dtype=int
    )


    baseline = pred[
        baseline_col
    ].to_numpy(
        dtype=float
    )


    model = pred[
        model_col
    ].to_numpy(
        dtype=float
    )


    deltas = []


    for _ in range(
        reps
    ):

        selected = rng.integers(
            0,
            len(blocks),
            size=len(blocks),
        )


        indices = np.concatenate(
            [
                blocks[i]
                for i in selected
            ]
        )


        yb = y[
            indices
        ]


        if np.unique(
            yb
        ).size < 2:

            continue


        a1 = roc_auc_score(
            yb,
            baseline[
                indices
            ],
        )


        a2 = roc_auc_score(
            yb,
            model[
                indices
            ],
        )


        deltas.append(
            a2 - a1
        )


    if not deltas:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    deltas = np.asarray(
        deltas
    )


    return (
        float(
            np.mean(
                deltas
            )
        ),

        float(
            np.quantile(
                deltas,
                0.025,
            )
        ),

        float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
    )


def bootstrap_auc_vs_chance(
    pred,
    score_col,
    block_days,
    reps=300,
    seed=42,
):

    rng = np.random.default_rng(
        seed
    )


    blocks = build_date_blocks(
        pred,
        block_days,
    )


    y = pred[
        "y_true"
    ].to_numpy(
        dtype=int
    )


    score = pred[
        score_col
    ].to_numpy(
        dtype=float
    )


    deltas = []


    for _ in range(
        reps
    ):

        selected = rng.integers(
            0,
            len(blocks),
            size=len(blocks),
        )


        indices = np.concatenate(
            [
                blocks[i]
                for i in selected
            ]
        )


        yb = y[
            indices
        ]


        if np.unique(
            yb
        ).size < 2:

            continue


        auc = roc_auc_score(
            yb,
            score[
                indices
            ],
        )


        deltas.append(
            auc - 0.5
        )


    if not deltas:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    deltas = np.asarray(
        deltas
    )


    return (
        float(
            np.mean(
                deltas
            )
        ),

        float(
            np.quantile(
                deltas,
                0.025,
            )
        ),

        float(
            np.quantile(
                deltas,
                0.975,
            )
        ),
    )


# =============================================================================
# LOAD DATA
# =============================================================================

print(
    "=" * 100,
    flush=True,
)

print(
    "EXPERIMENT 39 — BASELINE DIRECTION AUDIT",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


print(
    f"ROOT: {ROOT}",
    flush=True,
)


df = pd.read_csv(
    QRF_PATH
)


df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.strip()
)


df["forecast_date"] = (
    pd.to_datetime(
        df["forecast_date"]
    )
)


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


print(
    f"QRF shape: {df.shape}",
    flush=True,
)


print(
    f"Tickers: {df['ticker'].nunique()}",
    flush=True,
)


# =============================================================================
# LOAD EWMA
# =============================================================================

ewma = pd.read_csv(
    EWMA_PATH
)


ewma["ticker"] = (
    ewma["ticker"]
    .astype(str)
    .str.strip()
)


ewma["forecast_date"] = (
    pd.to_datetime(
        ewma["forecast_date"]
    )
)


if "ewma_q05" in ewma.columns:

    ewma_col = "ewma_q05"

elif "q05_forecast" in ewma.columns:

    ewma_col = "q05_forecast"

else:

    raise ValueError(
        "Could not identify EWMA q05 column."
    )


ewma_small = ewma[
    [
        "ticker",
        "forecast_date",
        ewma_col,
    ]
].copy()


ewma_small = (
    ewma_small
    .rename(
        columns={
            ewma_col:
                "ewma_q05"
        }
    )
)


ewma_small = (
    ewma_small
    .drop_duplicates(
        [
            "ticker",
            "forecast_date",
        ]
    )
)


df = df.merge(
    ewma_small,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="left",
    validate="one_to_one",
)


ewma_coverage = (
    df[
        "ewma_q05"
    ]
    .notna()
    .mean()
)


print(
    f"EWMA coverage: "
    f"{ewma_coverage:.4%}",
    flush=True,
)


# =============================================================================
# PINBALL LOSSES
# =============================================================================

df[
    "qrf_pinball"
] = pinball_loss(
    df[
        "actual_return"
    ],
    df[
        "q05_forecast"
    ],
)


df[
    "ewma_pinball"
] = pinball_loss(
    df[
        "actual_return"
    ],
    df[
        "ewma_q05"
    ],
)


# =============================================================================
# DISAGREEMENT
# =============================================================================

df[
    "qrf_ewma_disagreement"
] = (
    (
        df[
            "q05_forecast"
        ]
        -
        df[
            "ewma_q05"
        ]
    )
    /
    (
        df[
            "ewma_q05"
        ].abs()
        +
        1e-8
    )
)


# =============================================================================
# FUTURE LOSSES
# =============================================================================

print(
    "\nBuilding future losses...",
    flush=True,
)


for h in HORIZONS:

    df[
        f"future_qrf_loss_{h}d"
    ] = (
        df
        .groupby(
            "ticker"
        )[
            "qrf_pinball"
        ]
        .transform(
            lambda s:
                future_mean(
                    s,
                    h,
                )
        )
    )


# =============================================================================
# CAUSAL TARGETS
# =============================================================================

for h in HORIZONS:

    target = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    for _, idx in df.groupby(
        "ticker",
        sort=False,
    ).groups.items():

        ordered_idx = (
            df.loc[idx]
            .sort_values(
                "forecast_date"
            )
            .index
        )


        target.loc[
            ordered_idx
        ] = strictly_causal_target(
            df.loc[
                ordered_idx,
                f"future_qrf_loss_{h}d",
            ],
            horizon=h,
            percentile=0.75,
            min_history=MIN_TARGET_HISTORY,
        ).to_numpy()


    df[
        f"target_{h}d"
    ] = target


# =============================================================================
# CAUSAL FEATURES
# =============================================================================

print(
    "Building causal features...",
    flush=True,
)


grouped = df.groupby(
    "ticker",
    group_keys=False,
)


# -----------------------------------------------------------------------------
# BREACH HISTORY
# -----------------------------------------------------------------------------

df[
    "prior_breach_count_5"
] = grouped[
    "breach"
].transform(
    lambda s:
        s.shift(1)
        .rolling(
            5,
            min_periods=1,
        )
        .sum()
)


df[
    "prior_breach_count_20"
] = grouped[
    "breach"
].transform(
    lambda s:
        s.shift(1)
        .rolling(
            20,
            min_periods=1,
        )
        .sum()
)


# -----------------------------------------------------------------------------
# LOSS RATIOS
# -----------------------------------------------------------------------------

eps = 1e-8


if {
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
}.issubset(df.columns):

    df[
        "loss_ratio_5_20"
    ] = (
        df[
            "pinball_mean_5d_prior"
        ]
        /
        (
            df[
                "pinball_mean_20d_prior"
            ].abs()
            +
            eps
        )
    )


if {
    "pinball_mean_5d_prior",
    "pinball_mean_60d_prior",
}.issubset(df.columns):

    df[
        "loss_ratio_5_60"
    ] = (
        df[
            "pinball_mean_5d_prior"
        ]
        /
        (
            df[
                "pinball_mean_60d_prior"
            ].abs()
            +
            eps
        )
    )


# -----------------------------------------------------------------------------
# BREACH ACCELERATION
# -----------------------------------------------------------------------------

if {
    "breach_rate_5d",
    "breach_rate_20d",
}.issubset(df.columns):

    df[
        "breach_rate_accel_5_20"
    ] = (
        df[
            "breach_rate_5d"
        ]
        -
        df[
            "breach_rate_20d"
        ]
    )


if {
    "breach_rate_20d",
    "breach_rate_60d",
}.issubset(df.columns):

    df[
        "breach_rate_accel_20_60"
    ] = (
        df[
            "breach_rate_20d"
        ]
        -
        df[
            "breach_rate_60d"
        ]
    )


# -----------------------------------------------------------------------------
# VOLATILITY CHANGE
# -----------------------------------------------------------------------------

if "vol_ratio_20_60" in df.columns:

    df[
        "vol_ratio_change_1d"
    ] = grouped[
        "vol_ratio_20_60"
    ].transform(
        lambda s:
            s - s.shift(1)
    )


# -----------------------------------------------------------------------------
# VaR CHANGE
# -----------------------------------------------------------------------------

if "VaR_ratio_to_prior" in df.columns:

    df[
        "VaR_ratio_change_1d"
    ] = (
        df[
            "VaR_ratio_to_prior"
        ]
        -
        1.0
    )


# -----------------------------------------------------------------------------
# INTERACTIONS
# -----------------------------------------------------------------------------

if {
    "pinball_lag1",
    "vol_ratio_20_60",
}.issubset(df.columns):

    df[
        "loss_x_vol"
    ] = (
        df[
            "pinball_lag1"
        ]
        *
        df[
            "vol_ratio_20_60"
        ]
    )


if {
    "prior_breach_count_5",
    "vol_ratio_20_60",
}.issubset(df.columns):

    df[
        "breach5_x_vol"
    ] = (
        df[
            "prior_breach_count_5"
        ]
        *
        df[
            "vol_ratio_20_60"
        ]
    )


if {
    "pinball_lag1",
    "prior_breach_count_5",
}.issubset(df.columns):

    df[
        "loss_x_breach5"
    ] = (
        df[
            "pinball_lag1"
        ]
        *
        df[
            "prior_breach_count_5"
        ]
    )


if {
    "vol_ratio_20_60",
    "VIX",
}.issubset(df.columns):

    df[
        "vol_x_vix"
    ] = (
        df[
            "vol_ratio_20_60"
        ]
        *
        df[
            "VIX"
        ]
    )


# -----------------------------------------------------------------------------
# CROSS-SECTIONAL STRESS
# -----------------------------------------------------------------------------

daily_breach = (
    df
    .groupby(
        "forecast_date"
    )[
        "breach"
    ]
    .mean()
    .sort_index()
)


daily_prior = (
    daily_breach
    .shift(1)
)


daily_prior_5 = (
    daily_prior
    .rolling(
        5,
        min_periods=1,
    )
    .mean()
)


df[
    "csb_prior_1d"
] = df[
    "forecast_date"
].map(
    daily_prior
)


df[
    "csb_prior_5d"
] = df[
    "forecast_date"
].map(
    daily_prior_5
)


# -----------------------------------------------------------------------------
# CAUSAL Z-SCORES
# -----------------------------------------------------------------------------

z_sources = [

    "pinball_lag1",

    "pinball_mean_5d_prior",

    "pinball_mean_20d_prior",

    "vol_ratio_20_60",

    "prior_breach_count_5",

    "q05_change_1d",

    "VaR_ratio_to_prior",

    "qrf_ewma_disagreement",
]


for col in z_sources:

    if col not in df.columns:
        continue


    z_col = (
        f"{col}_ticker_causal_z"
    )


    df[z_col] = (
        grouped[col]
        .transform(
            causal_expanding_z
        )
    )


# =============================================================================
# SPLITS
# =============================================================================

train_mask, val_mask, test_mask = (
    make_splits(
        df
    )
)


print(
    "\nSPLITS",
    flush=True,
)


print(
    f"Train: {train_mask.sum():,}",
    flush=True,
)


print(
    f"Val:   {val_mask.sum():,}",
    flush=True,
)


print(
    f"Test:  {test_mask.sum():,}",
    flush=True,
)


# =============================================================================
# RESULTS CONTAINERS
# =============================================================================

summary_rows = []

signal_rows = []

ticker_rows = []

bootstrap_rows = []

prediction_rows = []


# =============================================================================
# HORIZON LOOP
# =============================================================================

for h in HORIZONS:

    print(
        "\n" + "=" * 100,
        flush=True,
    )

    print(
        f"HORIZON = {h} DAYS",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )


    target_col = (
        f"target_{h}d"
    )


    valid = (
        df[
            target_col
        ].notna()
    )


    train = (
        train_mask
        &
        valid
    )


    val = (
        val_mask
        &
        valid
    )


    test = (
        test_mask
        &
        valid
    )


    y_train = (
        df.loc[
            train,
            target_col,
        ]
        .astype(int)
    )


    y_val = (
        df.loc[
            val,
            target_col,
        ]
        .astype(int)
    )


    y_test = (
        df.loc[
            test,
            target_col,
        ]
        .astype(int)
    )


    print(
        f"Train: {len(y_train):,}",
        flush=True,
    )


    print(
        f"Val:   {len(y_val):,}",
        flush=True,
    )


    print(
        f"Test:  {len(y_test):,}",
        flush=True,
    )


    # =========================================================================
    # BASELINE
    # =========================================================================

    X_base = prepare_matrix(
        df,
        BASE_FEATURES,
        train_mask,
        include_ticker_fe=True,
    )


    baseline = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


    baseline.fit(
        X_base.loc[train],
        y_train,
    )


    baseline_val_score = (
        baseline
        .predict_proba(
            X_base.loc[val]
        )[:, 1]
    )


    baseline_test_score = (
        baseline
        .predict_proba(
            X_base.loc[test]
        )[:, 1]
    )


    baseline_val_auc = safe_auc(
        y_val,
        baseline_val_score,
    )


    baseline_test_auc = safe_auc(
        y_test,
        baseline_test_score,
    )


    # -------------------------------------------------------------------------
    # DIRECTION IS CHOSEN USING VALIDATION ONLY
    # -------------------------------------------------------------------------

    if (
        np.isfinite(
            baseline_val_auc
        )
        and
        baseline_val_auc < 0.5
    ):

        direction = -1

        oriented_val_score = (
            1.0
            -
            baseline_val_score
        )

        oriented_test_score = (
            1.0
            -
            baseline_test_score
        )

    else:

        direction = 1

        oriented_val_score = (
            baseline_val_score
        )

        oriented_test_score = (
            baseline_test_score
        )


    oriented_val_auc = safe_auc(
        y_val,
        oriented_val_score,
    )


    oriented_test_auc = safe_auc(
        y_test,
        oriented_test_score,
    )


    print(
        "\nBASELINE",
        flush=True,
    )


    print(
        f"  Validation AUC: "
        f"{baseline_val_auc:.4f}",
        flush=True,
    )


    print(
        f"  Test AUC raw:   "
        f"{baseline_test_auc:.4f}",
        flush=True,
    )


    print(
        f"  Direction selected from validation: "
        f"{'REVERSED' if direction == -1 else 'ORIGINAL'}",
        flush=True,
    )


    print(
        f"  Test AUC oriented: "
        f"{oriented_test_auc:.4f}",
        flush=True,
    )


    # =========================================================================
    # FULL XGB WITH TICKER FE
    # =========================================================================

    X_full = prepare_matrix(
        df,
        ALL_FEATURES,
        train_mask,
        include_ticker_fe=True,
    )


    full_xgb = XGBClassifier(
        **XGB_PARAMS
    )


    full_xgb.fit(
        X_full.loc[train],
        y_train,
        eval_set=[
            (
                X_full.loc[val],
                y_val,
            )
        ],
        verbose=False,
    )


    xgb_val_score = (
        full_xgb
        .predict_proba(
            X_full.loc[val]
        )[:, 1]
    )


    xgb_test_score = (
        full_xgb
        .predict_proba(
            X_full.loc[test]
        )[:, 1]
    )


    xgb_val_auc = safe_auc(
        y_val,
        xgb_val_score,
    )


    xgb_test_auc = safe_auc(
        y_test,
        xgb_test_score,
    )


    # =========================================================================
    # XGB WITHOUT TICKER FE
    # =========================================================================

    X_no_fe = prepare_matrix(
        df,
        ALL_FEATURES,
        train_mask,
        include_ticker_fe=False,
    )


    xgb_no_fe = XGBClassifier(
        **XGB_PARAMS
    )


    xgb_no_fe.fit(
        X_no_fe.loc[train],
        y_train,
        eval_set=[
            (
                X_no_fe.loc[val],
                y_val,
            )
        ],
        verbose=False,
    )


    xgb_no_fe_test_score = (
        xgb_no_fe
        .predict_proba(
            X_no_fe.loc[test]
        )[:, 1]
    )


    xgb_no_fe_test_auc = safe_auc(
        y_test,
        xgb_no_fe_test_score,
    )


    print(
        "\nXGBOOST",
        flush=True,
    )


    print(
        f"  Validation AUC: "
        f"{xgb_val_auc:.4f}",
        flush=True,
    )


    print(
        f"  Test AUC with FE: "
        f"{xgb_test_auc:.4f}",
        flush=True,
    )


    print(
        f"  Test AUC without FE: "
        f"{xgb_no_fe_test_auc:.4f}",
        flush=True,
    )


    print(
        f"  XGB vs raw baseline: "
        f"{xgb_test_auc - baseline_test_auc:+.4f}",
        flush=True,
    )


    print(
        f"  XGB vs oriented baseline: "
        f"{xgb_test_auc - oriented_test_auc:+.4f}",
        flush=True,
    )


    print(
        f"  XGB vs chance: "
        f"{xgb_test_auc - 0.5:+.4f}",
        flush=True,
    )


    # =========================================================================
    # VALIDATION-SELECTED SIMPLE SIGNALS
    # =========================================================================

    print(
        "\nSELECTING SIMPLE SIGNALS USING VALIDATION ONLY",
        flush=True,
    )


    signal_candidates = []


    for feature in ALL_FEATURES:

        if feature not in df.columns:
            continue


        X_single = prepare_matrix(
            df,
            [
                feature
            ],
            train_mask,
            include_ticker_fe=False,
        )


        model = LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )


        model.fit(
            X_single.loc[train],
            y_train,
        )


        val_score = (
            model
            .predict_proba(
                X_single.loc[val]
            )[:, 1]
        )


        test_score = (
            model
            .predict_proba(
                X_single.loc[test]
            )[:, 1]
        )


        val_auc = safe_auc(
            y_val,
            val_score,
        )


        test_auc = safe_auc(
            y_test,
            test_score,
        )


        signal_candidates.append({

            "horizon_days":
                h,

            "feature":
                feature,

            "validation_auc":
                val_auc,

            "test_auc":
                test_auc,

            "test_delta_vs_chance":
                test_auc - 0.5,

        })


    signal_candidates_df = pd.DataFrame(
        signal_candidates
    )


    signal_candidates_df = (
        signal_candidates_df
        .sort_values(
            "validation_auc",
            ascending=False,
        )
    )


    top_signal = (
        signal_candidates_df
        .iloc[0]
    )


    print(
        "\nTop 10 validation signals:",
        flush=True,
    )


    print(
        signal_candidates_df
        .head(10)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        ),
        flush=True,
    )


    # =========================================================================
    # TEST PREDICTION FOR BEST SIMPLE SIGNAL
    # =========================================================================

    best_feature = (
        top_signal[
            "feature"
        ]
    )


    best_signal_test_auc = (
        float(
            top_signal[
                "test_auc"
            ]
        )
    )


    print(
        f"\nBest validation-selected signal: "
        f"{best_feature}",
        flush=True,
    )


    print(
        f"Test AUC: "
        f"{best_signal_test_auc:.4f}",
        flush=True,
    )


    signal_candidates_df[
        "is_selected"
    ] = (
        signal_candidates_df[
            "feature"
        ]
        ==
        best_feature
    )


    signal_rows.extend(
        signal_candidates_df.to_dict(
            orient="records"
        )
    )


    # =========================================================================
    # PER-TICKER
    # =========================================================================

    temp = pd.DataFrame({

        "ticker":
            df.loc[
                test,
                "ticker",
            ].to_numpy(),

        "forecast_date":
            df.loc[
                test,
                "forecast_date",
            ].to_numpy(),

        "y_true":
            y_test.to_numpy(),

        "baseline_raw":
            baseline_test_score,

        "baseline_oriented":
            oriented_test_score,

        "xgb":
            xgb_test_score,

    })


    ticker_deltas = []


    print(
        "\nPER-TICKER ROBUSTNESS",
        flush=True,
    )


    for ticker, sub in temp.groupby(
        "ticker"
    ):

        yt = sub[
            "y_true"
        ].to_numpy(
            dtype=int
        )


        if np.unique(
            yt
        ).size < 2:

            auc_raw = np.nan
            auc_oriented = np.nan
            auc_xgb = np.nan

        else:

            auc_raw = roc_auc_score(
                yt,
                sub[
                    "baseline_raw"
                ],
            )


            auc_oriented = roc_auc_score(
                yt,
                sub[
                    "baseline_oriented"
                ],
            )


            auc_xgb = roc_auc_score(
                yt,
                sub[
                    "xgb"
                ],
            )


        delta = (
            auc_xgb
            -
            auc_oriented
            if
            np.isfinite(
                auc_xgb
            )
            and
            np.isfinite(
                auc_oriented
            )
            else np.nan
        )


        if np.isfinite(
            delta
        ):

            ticker_deltas.append(
                delta
            )


        ticker_rows.append({

            "horizon_days":
                h,

            "ticker":
                ticker,

            "n_test":
                len(sub),

            "positive_rate":
                yt.mean(),

            "auc_baseline_raw":
                auc_raw,

            "auc_baseline_oriented":
                auc_oriented,

            "auc_xgb":
                auc_xgb,

            "delta_xgb_vs_oriented_baseline":
                delta,

        })


    ticker_deltas = np.asarray(
        ticker_deltas,
        dtype=float,
    )


    if len(
        ticker_deltas
    ):

        print(
            f"  Mean XGB delta: "
            f"{ticker_deltas.mean():+.4f}",
            flush=True,
        )


        print(
            f"  Median XGB delta: "
            f"{np.median(ticker_deltas):+.4f}",
            flush=True,
        )


        print(
            f"  Improved tickers: "
            f"{np.sum(ticker_deltas > 0)}/"
            f"{len(ticker_deltas)}",
            flush=True,
        )


    # =========================================================================
    # BOOTSTRAP: XGB VS ORIENTED BASELINE
    # =========================================================================

    bootstrap_input = pd.DataFrame({

        "forecast_date":
            df.loc[
                test,
                "forecast_date",
            ].to_numpy(),

        "y_true":
            y_test.to_numpy(),

        "baseline_oriented":
            oriented_test_score,

        "xgb":
            xgb_test_score,
    })


    print(
        "\nBLOCK BOOTSTRAP: XGB VS ORIENTED BASELINE",
        flush=True,
    )


    for block_days in BLOCK_LENGTHS:

        (
            mean_delta,
            ci_low,
            ci_high,
        ) = bootstrap_delta(

            bootstrap_input,

            baseline_col=
                "baseline_oriented",

            model_col=
                "xgb",

            block_days=
                block_days,

            reps=
                BOOTSTRAP_REPS,

            seed=
                RANDOM_STATE
                + h
                + block_days,
        )


        robust = (
            np.isfinite(
                ci_low
            )
            and
            ci_low > 0
        )


        print(
            f"  Block {block_days:>2}d: "
            f"mean={mean_delta:+.4f} "
            f"CI=[{ci_low:+.4f}, "
            f"{ci_high:+.4f}] "
            f"robust={robust}",
            flush=True,
        )


        bootstrap_rows.append({

            "horizon_days":
                h,

            "comparison":
                "xgb_minus_oriented_baseline",

            "block_days":
                block_days,

            "observed_delta":
                (
                    xgb_test_auc
                    -
                    oriented_test_auc
                ),

            "bootstrap_mean_delta":
                mean_delta,

            "ci_lower":
                ci_low,

            "ci_upper":
                ci_high,

            "robust_positive":
                robust,

            "bootstrap_reps":
                BOOTSTRAP_REPS,

        })


    # =========================================================================
    # BOOTSTRAP: XGB VS CHANCE
    # =========================================================================

    print(
        "\nBLOCK BOOTSTRAP: XGB VS CHANCE",
        flush=True,
    )


    chance_input = pd.DataFrame({

        "forecast_date":
            df.loc[
                test,
                "forecast_date",
            ].to_numpy(),

        "y_true":
            y_test.to_numpy(),

        "xgb":
            xgb_test_score,
    })


    for block_days in BLOCK_LENGTHS:

        (
            mean_delta_chance,
            ci_low_chance,
            ci_high_chance,
        ) = bootstrap_auc_vs_chance(

            chance_input,

            score_col=
                "xgb",

            block_days=
                block_days,

            reps=
                BOOTSTRAP_REPS,

            seed=
                RANDOM_STATE
                + h
                + block_days
                + 1000,
        )


        robust_chance = (
            np.isfinite(
                ci_low_chance
            )
            and
            ci_low_chance > 0
        )


        print(
            f"  Block {block_days:>2}d: "
            f"mean AUC-0.5="
            f"{mean_delta_chance:+.4f} "
            f"CI=[{ci_low_chance:+.4f}, "
            f"{ci_high_chance:+.4f}] "
            f"robust={robust_chance}",
            flush=True,
        )


        bootstrap_rows.append({

            "horizon_days":
                h,

            "comparison":
                "xgb_minus_chance",

            "block_days":
                block_days,

            "observed_delta":
                xgb_test_auc - 0.5,

            "bootstrap_mean_delta":
                mean_delta_chance,

            "ci_lower":
                ci_low_chance,

            "ci_upper":
                ci_high_chance,

            "robust_positive":
                robust_chance,

            "bootstrap_reps":
                BOOTSTRAP_REPS,

        })


    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary_rows.append({

        "horizon_days":
            h,

        "baseline_validation_auc":
            baseline_val_auc,

        "baseline_test_auc_raw":
            baseline_test_auc,

        "baseline_direction":
            (
                "reversed"
                if direction == -1
                else "original"
            ),

        "baseline_test_auc_oriented":
            oriented_test_auc,

        "xgb_validation_auc":
            xgb_val_auc,

        "xgb_test_auc":
            xgb_test_auc,

        "xgb_no_fe_test_auc":
            xgb_no_fe_test_auc,

        "xgb_minus_raw_baseline":
            (
                xgb_test_auc
                -
                baseline_test_auc
            ),

        "xgb_minus_oriented_baseline":
            (
                xgb_test_auc
                -
                oriented_test_auc
            ),

        "xgb_minus_chance":
            (
                xgb_test_auc
                -
                0.5
            ),

        "best_validation_signal":
            best_feature,

        "best_signal_test_auc":
            best_signal_test_auc,

        "best_signal_minus_chance":
            (
                best_signal_test_auc
                -
                0.5
            ),

        "n_test":
            len(y_test),

        "test_positive_rate":
            y_test.mean(),

        "n_tickers_with_auc":
            len(ticker_deltas),

        "mean_ticker_delta":
            (
                ticker_deltas.mean()
                if
                len(ticker_deltas)
                else np.nan
            ),

        "median_ticker_delta":
            (
                np.median(
                    ticker_deltas
                )
                if
                len(ticker_deltas)
                else np.nan
            ),

        "tickers_improved":
            (
                int(
                    np.sum(
                        ticker_deltas > 0
                    )
                )
                if
                len(ticker_deltas)
                else 0
            ),

    })


    prediction_rows.append(
        pd.DataFrame({

            "ticker":
                df.loc[
                    test,
                    "ticker",
                ].to_numpy(),

            "forecast_date":
                df.loc[
                    test,
                    "forecast_date",
                ].to_numpy(),

            "horizon_days":
                h,

            "y_true":
                y_test.to_numpy(),

            "baseline_raw":
                baseline_test_score,

            "baseline_oriented":
                oriented_test_score,

            "xgb":
                xgb_test_score,

        })
    )


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

summary_df = pd.DataFrame(
    summary_rows
)


signal_df = pd.DataFrame(
    signal_rows
)


ticker_df = pd.DataFrame(
    ticker_rows
)


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


prediction_df = pd.concat(
    prediction_rows,
    ignore_index=True,
)


summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)


signal_df.to_csv(
    SIGNAL_PATH,
    index=False,
)


ticker_df.to_csv(
    TICKER_PATH,
    index=False,
)


bootstrap_df.to_csv(
    BOOTSTRAP_PATH,
    index=False,
)


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "EXPERIMENT 39 — FINAL SUMMARY",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


print(
    "\nMAIN RESULTS",
    flush=True,
)


print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    ),
    flush=True,
)


print(
    "\nBLOCK BOOTSTRAP",
    flush=True,
)


print(
    bootstrap_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    ),
    flush=True,
)


print(
    "\nOUTPUT FILES",
    flush=True,
)


print(
    SUMMARY_PATH,
    flush=True,
)


print(
    SIGNAL_PATH,
    flush=True,
)


print(
    TICKER_PATH,
    flush=True,
)


print(
    BOOTSTRAP_PATH,
    flush=True,
)


print(
    PREDICTIONS_PATH,
    flush=True,
)


print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "EXPERIMENT 39 COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)