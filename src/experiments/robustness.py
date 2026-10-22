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
# EXPERIMENT 38
# FEATURE ABLATION + PER-TICKER VALIDATION + BOOTSTRAP SENSITIVITY
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


ABLATION_PATH = (
    RESULTS_DIR
    / "experiment_38_feature_ablation.csv"
)

PER_TICKER_PATH = (
    RESULTS_DIR
    / "experiment_38_per_ticker.csv"
)

BOOTSTRAP_PATH = (
    RESULTS_DIR
    / "experiment_38_bootstrap_sensitivity.csv"
)

FE_SENSITIVITY_PATH = (
    RESULTS_DIR
    / "experiment_38_fixed_effect_sensitivity.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "experiment_38_predictions.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [5, 10, 20]

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
# FEATURE GROUPS
# =============================================================================

BASE_FEATURES = [
    "pinball_lag1",
    "vol_ratio_20_60",
]


FEATURE_GROUPS = {

    "loss_history": [

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


# =============================================================================
# PINBALL LOSS
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


# =============================================================================
# FUTURE MEAN
# =============================================================================

def future_mean(
    series,
    horizon,
):
    """
    At t:

        mean(t+1 ... t+h)
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


# =============================================================================
# STRICTLY CAUSAL TARGET
# =============================================================================

def strictly_causal_target(
    future_loss,
    horizon,
    percentile=0.75,
    min_history=50,
):
    """
    Future loss at t:

        D_t,h = average loss from t+1 to t+h

    The threshold at t may only use fully observed historical windows.

    Latest permissible historical target:

        D_(t-h-1,h)

    Therefore:

        historical = future_loss.shift(horizon + 1)
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


# =============================================================================
# CAUSAL EXPANDING Z
# =============================================================================

def causal_expanding_z(
    series,
):
    prior = series.shift(1)

    mean_prior = (
        prior
        .expanding(
            min_periods=20
        )
        .mean()
    )

    std_prior = (
        prior
        .expanding(
            min_periods=20
        )
        .std(
            ddof=0
        )
    )

    z = (
        series - mean_prior
    ) / std_prior.replace(
        0,
        np.nan,
    )

    return z.replace(
        [np.inf, -np.inf],
        np.nan,
    )


# =============================================================================
# CHRONOLOGICAL SPLIT
# =============================================================================

def make_splits(
    df,
):
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


# =============================================================================
# EWMA COLUMN
# =============================================================================

def find_ewma_column(
    df,
):
    candidates = [
        "ewma_q05",
        "q05_forecast",
        "q05",
        "Q05",
        "VaR_5pct",
    ]

    for c in candidates:

        if c in df.columns:
            return c

    raise ValueError(
        "EWMA q05 column not found.\n"
        f"Columns: {list(df.columns)}"
    )


# =============================================================================
# DESIGN MATRIX
# =============================================================================

def prepare_matrix(
    df,
    features,
    train_mask,
    include_ticker_fe=True,
):
    features = [
        c
        for c in features
        if c in df.columns
    ]


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


    train_medians = (
        X.loc[
            train_mask
        ]
        .median()
    )


    X = (
        X
        .fillna(
            train_medians
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


# =============================================================================
# AUC
# =============================================================================

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


# =============================================================================
# DATE BLOCKS
# =============================================================================

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

        chosen = unique_dates[
            start:
            start + block_days
        ]


        pieces = [
            date_to_rows[d]
            for d in chosen
        ]


        if pieces:

            blocks.append(
                np.concatenate(
                    pieces
                )
            )


    return blocks


# =============================================================================
# BLOCK BOOTSTRAP
# =============================================================================

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


    base = pred[
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

        chosen = rng.integers(
            0,
            len(blocks),
            size=len(blocks),
        )


        indices = np.concatenate(
            [
                blocks[i]
                for i in chosen
            ]
        )


        yb = y[
            indices
        ]


        if np.unique(
            yb
        ).size < 2:

            continue


        auc_base = roc_auc_score(
            yb,
            base[
                indices
            ],
        )


        auc_model = roc_auc_score(
            yb,
            model[
                indices
            ],
        )


        deltas.append(
            auc_model - auc_base
        )


    if not deltas:

        return (
            np.nan,
            np.nan,
            np.nan,
        )


    deltas = np.asarray(
        deltas,
        dtype=float,
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
    "EXPERIMENT 38 — ROBUSTNESS STUDY",
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


ewma_q = find_ewma_column(
    ewma
)


ewma_small = ewma[
    [
        "ticker",
        "forecast_date",
        ewma_q,
    ]
].copy()


ewma_small = (
    ewma_small
    .rename(
        columns={
            ewma_q:
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


coverage = (
    df[
        "ewma_q05"
    ]
    .notna()
    .mean()
)


print(
    f"EWMA coverage: {coverage:.4%}",
    flush=True,
)


if coverage < 0.99:

    raise ValueError(
        "EWMA coverage below 99%."
    )


# =============================================================================
# CURRENT PINBALL LOSSES
# =============================================================================

df["qrf_pinball"] = pinball_loss(
    df["actual_return"],
    df["q05_forecast"],
)


df["ewma_pinball"] = pinball_loss(
    df["actual_return"],
    df["ewma_q05"],
)


# =============================================================================
# DISAGREEMENT
# =============================================================================

df[
    "qrf_ewma_disagreement"
] = (
    (
        df["q05_forecast"]
        -
        df["ewma_q05"]
    )
    /
    (
        df["ewma_q05"].abs()
        +
        1e-8
    )
)


# =============================================================================
# FUTURE LOSSES
# =============================================================================

print(
    "\nBuilding future targets...",
    flush=True,
)


for h in HORIZONS:

    df[
        f"future_qrf_loss_{h}d"
    ] = (
        df
        .groupby(
            "ticker"
        )["qrf_pinball"]
        .transform(
            lambda s:
                future_mean(
                    s,
                    h,
                )
        )
    )


# =============================================================================
# STRICTLY CAUSAL TARGETS
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

        ordered = (
            df.loc[idx]
            .sort_values(
                "forecast_date"
            )
            .index
        )


        target.loc[
            ordered
        ] = strictly_causal_target(

            df.loc[
                ordered,
                f"future_qrf_loss_{h}d",
            ],

            horizon=h,

            percentile=0.75,

            min_history=
                MIN_TARGET_HISTORY,
        ).to_numpy()


    df[
        f"target_{h}d"
    ] = target


    print(
        f"{h}d positive rate: "
        f"{target.mean():.4%}",
        flush=True,
    )


# =============================================================================
# CAUSAL FEATURES
# =============================================================================

print(
    "\nBuilding causal features...",
    flush=True,
)


grouped = df.groupby(
    "ticker",
    group_keys=False,
)


# -----------------------------------------------------------------------------
# BREACH COUNTS
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
# BREACH RATE ACCELERATION
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
        df["pinball_lag1"]
        *
        df["vol_ratio_20_60"]
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
        df["pinball_lag1"]
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
        df["VIX"]
    )


# -----------------------------------------------------------------------------
# CROSS-SECTIONAL BREACH STRESS
# -----------------------------------------------------------------------------

daily_breach = (
    df
    .groupby(
        "forecast_date"
    )["breach"]
    .mean()
    .sort_index()
)


daily_breach_prior = (
    daily_breach
    .shift(1)
)


daily_breach_5 = (
    daily_breach_prior
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
    daily_breach_prior
)


df[
    "csb_prior_5d"
] = df[
    "forecast_date"
].map(
    daily_breach_5
)


# =============================================================================
# CAUSAL Z SCORES
# =============================================================================

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


    zcol = (
        f"{col}_ticker_causal_z"
    )


    df[zcol] = (
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
    "\nSPLITS:",
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
# BUILD FULL FEATURE LIST
# =============================================================================

all_extended_features = []


for group_name, features in FEATURE_GROUPS.items():

    for f in features:

        if f in df.columns:

            all_extended_features.append(
                f
            )


all_extended_features = list(
    dict.fromkeys(
        all_extended_features
    )
)


print(
    f"\nFull feature count: "
    f"{len(all_extended_features)}",
    flush=True,
)


# =============================================================================
# FEATURE ABLATION
# =============================================================================

ablation_rows = []

all_predictions = []

per_ticker_rows = []

fe_sensitivity_rows = []

bootstrap_rows = []


for h in HORIZONS:

    print(
        "\n" + "=" * 100,
        flush=True,
    )

    print(
        f"HORIZON {h} DAYS",
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
        f"Test:  {len(y_test):,}",
        flush=True,
    )


    print(
        f"Test positive rate: "
        f"{y_test.mean():.4%}",
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


    baseline_model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


    baseline_model.fit(
        X_base.loc[train],
        y_train,
    )


    baseline_score = (
        baseline_model
        .predict_proba(
            X_base.loc[test]
        )[:, 1]
    )


    baseline_auc = safe_auc(
        y_test,
        baseline_score,
    )


    print(
        f"\nBaseline AUC: "
        f"{baseline_auc:.4f}",
        flush=True,
    )


    # =========================================================================
    # BASELINE PREDICTION DATAFRAME
    # =========================================================================

    horizon_pred = pd.DataFrame({

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

        "baseline_score":
            baseline_score,
    })


    # =========================================================================
    # SINGLE-GROUP XGB ABLATION
    # =========================================================================

    for group_name, group_features in FEATURE_GROUPS.items():

        candidate_features = list(
            dict.fromkeys(
                BASE_FEATURES
                +
                [
                    f
                    for f in group_features
                    if f in df.columns
                ]
            )
        )


        X = prepare_matrix(
            df,
            candidate_features,
            train_mask,
            include_ticker_fe=True,
        )


        model = XGBClassifier(
            **XGB_PARAMS
        )


        model.fit(
            X.loc[train],
            y_train,
            eval_set=[
                (
                    X.loc[val],
                    y_val,
                )
            ],
            verbose=False,
        )


        score = (
            model
            .predict_proba(
                X.loc[test]
            )[:, 1]
        )


        auc = safe_auc(
            y_test,
            score,
        )


        delta = (
            auc
            -
            baseline_auc
        )


        print(
            f"  + {group_name:<20}"
            f" AUC={auc:.4f}"
            f"  Delta={delta:+.4f}",
            flush=True,
        )


        ablation_rows.append({

            "horizon_days":
                h,

            "feature_set":
                group_name,

            "n_features":
                len(candidate_features),

            "auc":
                auc,

            "baseline_auc":
                baseline_auc,

            "delta_vs_baseline":
                delta,

        })


    # =========================================================================
    # FULL XGB
    # =========================================================================

    print(
        "\nTraining FULL XGB...",
        flush=True,
    )


    X_full = prepare_matrix(
        df,
        all_extended_features,
        train_mask,
        include_ticker_fe=True,
    )


    full_model = XGBClassifier(
        **XGB_PARAMS
    )


    full_model.fit(
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


    full_score = (
        full_model
        .predict_proba(
            X_full.loc[test]
        )[:, 1]
    )


    full_auc = safe_auc(
        y_test,
        full_score,
    )


    full_delta = (
        full_auc
        -
        baseline_auc
    )


    print(
        f"FULL XGB AUC: "
        f"{full_auc:.4f}",
        flush=True,
    )


    print(
        f"FULL XGB delta: "
        f"{full_delta:+.4f}",
        flush=True,
    )


    ablation_rows.append({

        "horizon_days":
            h,

        "feature_set":
            "FULL",

        "n_features":
            len(all_extended_features),

        "auc":
            full_auc,

        "baseline_auc":
            baseline_auc,

        "delta_vs_baseline":
            full_delta,

    })


    # =========================================================================
    # SAVE PREDICTIONS
    # =========================================================================

    horizon_pred[
        "full_xgb_score"
    ] = full_score


    horizon_pred[
        "horizon_days"
    ] = h


    all_predictions.append(
        horizon_pred
    )


    # =========================================================================
    # PER-TICKER AUC
    # =========================================================================

    print(
        "\nPer-ticker AUC:",
        flush=True,
    )


    per_ticker_values = []


    temp = pd.DataFrame({

        "ticker":
            df.loc[
                test,
                "ticker",
            ].to_numpy(),

        "y_true":
            y_test.to_numpy(),

        "baseline":
            baseline_score,

        "full_xgb":
            full_score,
    })


    for ticker, sub in temp.groupby(
        "ticker"
    ):

        ticker_y = sub[
            "y_true"
        ].to_numpy(
            dtype=int
        )


        if np.unique(
            ticker_y
        ).size < 2:

            auc_b = np.nan
            auc_x = np.nan

        else:

            auc_b = roc_auc_score(
                ticker_y,
                sub[
                    "baseline"
                ],
            )

            auc_x = roc_auc_score(
                ticker_y,
                sub[
                    "full_xgb"
                ],
            )


        delta_t = (
            auc_x - auc_b
            if
            np.isfinite(auc_b)
            and
            np.isfinite(auc_x)
            else np.nan
        )


        per_ticker_values.append(
            delta_t
        )


        per_ticker_rows.append({

            "horizon_days":
                h,

            "ticker":
                ticker,

            "n_test":
                len(sub),

            "positive_rate":
                ticker_y.mean(),

            "auc_baseline":
                auc_b,

            "auc_full_xgb":
                auc_x,

            "delta_xgb_vs_baseline":
                delta_t,

        })


    finite_deltas = np.asarray(
        [
            x
            for x in per_ticker_values
            if np.isfinite(x)
        ],
        dtype=float,
    )


    if len(finite_deltas):

        print(
            f"  Mean ticker delta: "
            f"{finite_deltas.mean():+.4f}",
            flush=True,
        )


        print(
            f"  Median ticker delta: "
            f"{np.median(finite_deltas):+.4f}",
            flush=True,
        )


        print(
            f"  Tickers improved: "
            f"{np.sum(finite_deltas > 0)}/"
            f"{len(finite_deltas)}",
            flush=True,
        )


    # =========================================================================
    # BLOCK BOOTSTRAP SENSITIVITY
    # =========================================================================

    bootstrap_input = horizon_pred[
        [
            "forecast_date",
            "y_true",
            "baseline_score",
            "full_xgb_score",
        ]
    ].copy()


    print(
        "\nBootstrap sensitivity:",
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
                "baseline_score",

            model_col=
                "full_xgb_score",

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
            np.isfinite(ci_low)
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

            "block_days":
                block_days,

            "observed_delta":
                full_delta,

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
    # FIXED-EFFECT SENSITIVITY
    # =========================================================================

    print(
        "\nTicker fixed-effect sensitivity...",
        flush=True,
    )


    X_no_fe = prepare_matrix(
        df,
        all_extended_features,
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


    score_no_fe = (
        xgb_no_fe
        .predict_proba(
            X_no_fe.loc[test]
        )[:, 1]
    )


    auc_no_fe = safe_auc(
        y_test,
        score_no_fe,
    )


    print(
        f"  XGB with ticker FE:    "
        f"{full_auc:.4f}",
        flush=True,
    )


    print(
        f"  XGB without ticker FE: "
        f"{auc_no_fe:.4f}",
        flush=True,
    )


    print(
        f"  Difference: "
        f"{auc_no_fe - full_auc:+.4f}",
        flush=True,
    )


    fe_sensitivity_rows.append({

        "horizon_days":
            h,

        "auc_with_ticker_fe":
            full_auc,

        "auc_without_ticker_fe":
            auc_no_fe,

        "difference_no_fe_minus_fe":
            auc_no_fe - full_auc,

        "baseline_auc":
            baseline_auc,

        "delta_with_fe":
            full_auc - baseline_auc,

        "delta_without_fe":
            auc_no_fe - baseline_auc,

    })


# =============================================================================
# SAVE FILES
# =============================================================================

ablation_df = pd.DataFrame(
    ablation_rows
)


per_ticker_df = pd.DataFrame(
    per_ticker_rows
)


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


fe_df = pd.DataFrame(
    fe_sensitivity_rows
)


predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)


ablation_df.to_csv(
    ABLATION_PATH,
    index=False,
)


per_ticker_df.to_csv(
    PER_TICKER_PATH,
    index=False,
)


bootstrap_df.to_csv(
    BOOTSTRAP_PATH,
    index=False,
)


fe_df.to_csv(
    FE_SENSITIVITY_PATH,
    index=False,
)


predictions_df.to_csv(
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
    "EXPERIMENT 38 — FINAL SUMMARY",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


print(
    "\nFEATURE ABLATION",
    flush=True,
)


if not ablation_df.empty:

    print(
        ablation_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        ),
        flush=True,
    )


print(
    "\nBOOTSTRAP SENSITIVITY",
    flush=True,
)


if not bootstrap_df.empty:

    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        ),
        flush=True,
    )


print(
    "\nFIXED-EFFECT SENSITIVITY",
    flush=True,
)


if not fe_df.empty:

    print(
        fe_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        ),
        flush=True,
    )


print(
    "\nPER-TICKER SUMMARY",
    flush=True,
)


if not per_ticker_df.empty:

    for h in HORIZONS:

        sub = per_ticker_df[
            per_ticker_df[
                "horizon_days"
            ] == h
        ]


        deltas = sub[
            "delta_xgb_vs_baseline"
        ].dropna()


        if len(deltas):

            print(
                f"{h}d: "
                f"mean delta="
                f"{deltas.mean():+.4f}, "
                f"median="
                f"{deltas.median():+.4f}, "
                f"improved="
                f"{(deltas > 0).sum()}/"
                f"{len(deltas)}",
                flush=True,
            )


print(
    "\nOUTPUT FILES:",
    flush=True,
)


print(
    ABLATION_PATH,
    flush=True,
)


print(
    PER_TICKER_PATH,
    flush=True,
)


print(
    BOOTSTRAP_PATH,
    flush=True,
)


print(
    FE_SENSITIVITY_PATH,
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
    "EXPERIMENT 38 COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)