from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 37C
# STRICTLY CAUSAL TARGET AUDIT + RE-TEST
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

RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_AUDIT_PATH = (
    RESULTS_DIR
    / "37c_causal_target_audit.csv"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "37c_causal_target_results.csv"
)

BOOTSTRAP_PATH = (
    RESULTS_DIR
    / "37c_causal_target_bootstrap.csv"
)

PREDICTIONS_PATH = (
    RESULTS_DIR
    / "37c_causal_target_predictions.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

HORIZONS = [5, 10, 20]

MIN_TARGET_HISTORY = 50

BOOTSTRAP_REPS = 500
BLOCK_DAYS = 20

RANDOM_STATE = 42


# =============================================================================
# XGBOOST CONFIGURATION
# =============================================================================

XGB_CLASSIFIER_PARAMS = dict(
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


XGB_REGRESSOR_PARAMS = dict(
    n_estimators=180,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=8,
    reg_alpha=0.05,
    reg_lambda=1.5,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    max_bin=64,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

BASE_FEATURES = [
    "pinball_lag1",
    "vol_ratio_20_60",
]


EXTENDED_FEATURE_CANDIDATES = [
    "pinball_lag1",
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
    "pinball_mean_60d_prior",
    "pinball_acceleration",
    "prior_breach_count_5",
    "prior_breach_count_20",
    "breach_streak_prior",
    "breach_rate_5d",
    "breach_rate_20d",
    "breach_rate_60d",
    "ewma_breach_rate",
    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",
    "ks_drift_60_vs_250",
    "q05_change_1d",
    "VaR_change_1d",
    "VaR_ratio_to_prior",
    "VaR_ratio_to_20d_mean",
    "VIX",
    "volume_ratio",
    "qrf_ewma_disagreement",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pinball_loss(y, q, alpha=0.05):
    """
    Lower-tail pinball loss.
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)

    return np.where(
        y < q,
        (alpha - 1.0) * (y - q),
        alpha * (y - q),
    )


def future_mean(series, horizon):
    """
    For observation t:

        mean(series[t+1], ..., series[t+h])

    This is future TARGET information.
    It is never used as a predictor.
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


def future_std(series, horizon):
    """
    For observation t:

        std(series[t+1], ..., series[t+h])
    """

    return (
        series
        .shift(-1)
        .rolling(
            window=horizon,
            min_periods=horizon,
        )
        .std()
        .shift(-(horizon - 1))
    )


def strictly_causal_target(
    future_loss,
    horizon,
    percentile=0.75,
    min_history=50,
):
    """
    STRICTLY CAUSAL deterioration target.

    Future loss at t is:

        D(t,h) = loss over t+1 ... t+h

    At time t, the threshold must not use ANY D-window that contains
    t or a later observation.

    Therefore the latest allowable historical future-loss observation is:

        D(t-h-1,h)

    Hence the threshold is based on:

        future_loss.shift(h+1)

    followed by an expanding percentile.

    Returns
    -------
    target : Series
        0/1 deterioration label.

    threshold : Series
        Causal threshold used to create the label.
    """

    historical_loss = future_loss.shift(
        horizon + 1
    )

    threshold = (
        historical_loss
        .expanding(
            min_periods=min_history
        )
        .quantile(percentile)
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

    return target, threshold


def causal_expanding_z(series):
    """
    Strictly causal expanding z-score.

    At t, only values before t are used to calculate the mean/std.
    """

    prior = series.shift(1)

    prior_mean = (
        prior
        .expanding(min_periods=20)
        .mean()
    )

    prior_std = (
        prior
        .expanding(min_periods=20)
        .std(ddof=0)
    )

    z = (
        series - prior_mean
    ) / prior_std.replace(
        0,
        np.nan,
    )

    return (
        z.replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def make_splits(df):
    """
    Chronological 60/20/20 split separately for each ticker.
    """

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

        sub = (
            df.loc[idx]
            .sort_values("forecast_date")
        )

        n = len(sub)

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
            sub.index[:n_train]
        ] = True

        val.loc[
            sub.index[
                n_train:
                n_train + n_val
            ]
        ] = True

        test.loc[
            sub.index[
                n_train + n_val:
            ]
        ] = True

    return train, val, test


def prepare_matrix(
    df,
    features,
    train_mask,
):
    """
    Build numeric predictors + ticker fixed effects.

    All imputation/scaling parameters are learned from TRAIN only.
    """

    X = (
        df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .copy()
    )

    # -------------------------------------------------------------------------
    # TRAIN-ONLY MEDIAN IMPUTATION
    # -------------------------------------------------------------------------

    train_medians = (
        X.loc[train_mask]
        .median()
    )

    X = (
        X
        .fillna(train_medians)
        .fillna(0.0)
    )

    # -------------------------------------------------------------------------
    # TRAIN-ONLY SCALING
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # TICKER FIXED EFFECTS
    # -------------------------------------------------------------------------

    ticker_dummies = pd.get_dummies(
        df["ticker"].astype(str),
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


def auc_safe(y, scores):
    """
    Safe ROC-AUC.
    """

    y = np.asarray(
        y,
        dtype=int
    )

    if np.unique(y).size < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            scores,
        )
    )


def find_ewma_q_column(df):
    """
    Find the 5% EWMA lower quantile.
    """

    candidates = [
        "ewma_q05",
        "q05_forecast",
        "q05",
        "Q05",
        "quantile_05",
        "VaR_5pct",
        "VaR_5pct_forecast",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise ValueError(
        "Could not identify EWMA 5% quantile column.\n"
        f"Available columns: {list(df.columns)}"
    )


def build_date_blocks(
    prediction_df,
    block_days,
):
    """
    Precompute blocks of consecutive calendar dates.
    """

    dates = (
        pd.to_datetime(
            prediction_df[
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

    for date in unique_dates:

        date_to_rows[date] = (
            np.flatnonzero(
                dates == date
            )
        )

    blocks = []

    for start in range(
        0,
        len(unique_dates),
        block_days,
    ):

        selected_dates = (
            unique_dates[
                start:
                start + block_days
            ]
        )

        row_parts = [
            date_to_rows[d]
            for d in selected_dates
        ]

        if row_parts:

            blocks.append(
                np.concatenate(
                    row_parts
                )
            )

    return blocks


def bootstrap_auc_delta(
    prediction_df,
    baseline_column,
    augmented_column,
    reps=500,
    block_days=20,
    seed=42,
):
    """
    Moving date-block bootstrap.

    Statistic:

        AUC(augmented) - AUC(baseline)

    IMPORTANT:
    prediction_df["y_true"] contains the actual binary target.
    prediction_df["target"] is the target NAME string and must NOT
    be converted to integers.
    """

    rng = np.random.default_rng(
        seed
    )

    blocks = build_date_blocks(
        prediction_df,
        block_days,
    )

    if not blocks:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    # -------------------------------------------------------------------------
    # CORRECTED: use y_true, not target
    # -------------------------------------------------------------------------

    y = prediction_df[
        "y_true"
    ].to_numpy(
        dtype=int
    )

    baseline = prediction_df[
        baseline_column
    ].to_numpy(
        dtype=float
    )

    augmented = prediction_df[
        augmented_column
    ].to_numpy(
        dtype=float
    )

    deltas = []

    for _ in range(reps):

        selected_blocks = rng.integers(
            0,
            len(blocks),
            size=len(blocks),
        )

        indices = np.concatenate(
            [
                blocks[i]
                for i in selected_blocks
            ]
        )

        y_boot = y[
            indices
        ]

        if np.unique(
            y_boot
        ).size < 2:
            continue

        try:

            auc_base = roc_auc_score(
                y_boot,
                baseline[
                    indices
                ],
            )

            auc_aug = roc_auc_score(
                y_boot,
                augmented[
                    indices
                ],
            )

            deltas.append(
                auc_aug - auc_base
            )

        except ValueError:
            continue

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
            np.mean(deltas)
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
# HEADER
# =============================================================================

print(
    "=" * 100,
    flush=True,
)

print(
    "EXPERIMENT 37C — STRICTLY CAUSAL TARGET AUDIT + RE-TEST",
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

print(
    f"QRF:  {QRF_PATH}",
    flush=True,
)

print(
    f"EWMA: {EWMA_PATH}",
    flush=True,
)


# =============================================================================
# LOAD QRF DATA
# =============================================================================

if not QRF_PATH.exists():

    raise FileNotFoundError(
        f"QRF panel not found:\n{QRF_PATH}"
    )

if not EWMA_PATH.exists():

    raise FileNotFoundError(
        f"EWMA predictions not found:\n{EWMA_PATH}"
    )


df = pd.read_csv(
    QRF_PATH
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.strip()
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
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
    f"\nQRF panel shape: {df.shape}",
    flush=True,
)

print(
    f"Tickers: {df['ticker'].nunique()}",
    flush=True,
)


# =============================================================================
# REQUIRED QRF COLUMNS
# =============================================================================

required_columns = [
    "ticker",
    "forecast_date",
    "actual_return",
    "q05_forecast",
    "breach",
    "pinball_loss",
    "pinball_lag1",
    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",
    "breach_streak_prior",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        f"Missing QRF columns: {missing}"
    )


# =============================================================================
# LOAD EWMA DATA
# =============================================================================

ewma = pd.read_csv(
    EWMA_PATH
)

ewma["ticker"] = (
    ewma["ticker"]
    .astype(str)
    .str.strip()
)


ewma_date_column = next(
    (
        column
        for column in [
            "forecast_date",
            "Date",
            "date",
        ]
        if column in ewma.columns
    ),
    None,
)

if ewma_date_column is None:

    raise ValueError(
        "Could not identify EWMA date column."
    )


ewma["forecast_date"] = pd.to_datetime(
    ewma[ewma_date_column]
)


ewma_q_column = find_ewma_q_column(
    ewma
)


print(
    f"\nEWMA quantile column: "
    f"{ewma_q_column}",
    flush=True,
)


ewma_small = ewma[
    [
        "ticker",
        "forecast_date",
        ewma_q_column,
    ]
].copy()


ewma_small = (
    ewma_small
    .rename(
        columns={
            ewma_q_column:
                "ewma_q05"
        }
    )
)


ewma_small["ewma_q05"] = pd.to_numeric(
    ewma_small["ewma_q05"],
    errors="coerce",
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
    df["ewma_q05"]
    .notna()
    .mean()
)


print(
    f"QRF-EWMA current forecast coverage: "
    f"{ewma_coverage:.4%}",
    flush=True,
)


if ewma_coverage < 0.99:

    raise ValueError(
        "EWMA coverage is below 99%. "
        "Stop and inspect the data."
    )


# =============================================================================
# CURRENT-DAY PINBALL LOSSES
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
# QRF-EWMA CURRENT DISAGREEMENT
# =============================================================================

df["qrf_ewma_disagreement"] = (
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
# BUILD FUTURE CONTINUOUS TARGETS
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "BUILDING FUTURE CONTINUOUS TARGETS",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


for horizon in HORIZONS:

    print(
        f"  Building {horizon}d future losses...",
        flush=True,
    )


    df[
        f"future_qrf_loss_{horizon}d"
    ] = (
        df
        .groupby(
            "ticker"
        )["qrf_pinball"]
        .transform(
            lambda s:
                future_mean(
                    s,
                    horizon,
                )
        )
    )


    df[
        f"future_ewma_loss_{horizon}d"
    ] = (
        df
        .groupby(
            "ticker"
        )["ewma_pinball"]
        .transform(
            lambda s:
                future_mean(
                    s,
                    horizon,
                )
        )
    )


    df[
        f"future_return_std_{horizon}d"
    ] = (
        df
        .groupby(
            "ticker"
        )["actual_return"]
        .transform(
            lambda s:
                future_std(
                    s,
                    horizon,
                )
        )
    )


    # -------------------------------------------------------------------------
    # VOLATILITY-NORMALIZED LOSS
    # -------------------------------------------------------------------------

    df[
        f"future_qrf_loss_volnorm_{horizon}d"
    ] = (
        df[
            f"future_qrf_loss_{horizon}d"
        ]
        /
        (
            df[
                f"future_return_std_{horizon}d"
            ]
            .abs()
            +
            1e-8
        )
    )


    # -------------------------------------------------------------------------
    # QRF RELATIVE TO EWMA
    # -------------------------------------------------------------------------

    df[
        f"future_qrf_ewma_logratio_{horizon}d"
    ] = (
        np.log(
            df[
                f"future_qrf_loss_{horizon}d"
            ].clip(
                lower=1e-10
            )
        )
        -
        np.log(
            df[
                f"future_ewma_loss_{horizon}d"
            ].clip(
                lower=1e-10
            )
        )
    )


# =============================================================================
# STRICTLY CAUSAL TARGETS
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "CONSTRUCTING STRICTLY CAUSAL TARGETS",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


target_audit_rows = []


for horizon in HORIZONS:

    print(
        f"\nHorizon {horizon}d:",
        flush=True,
    )


    # =========================================================================
    # ORIGINAL FUTURE PINBALL-LOSS TARGET
    # =========================================================================

    original_target = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    original_threshold = pd.Series(
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


        target_i, threshold_i = (
            strictly_causal_target(
                df.loc[
                    ordered_idx,
                    f"future_qrf_loss_{horizon}d",
                ],
                horizon=horizon,
                percentile=0.75,
                min_history=MIN_TARGET_HISTORY,
            )
        )


        original_target.loc[
            ordered_idx
        ] = target_i.to_numpy()


        original_threshold.loc[
            ordered_idx
        ] = threshold_i.to_numpy()


    df[
        f"causal_deterioration_{horizon}d"
    ] = original_target


    df[
        f"causal_threshold_{horizon}d"
    ] = original_threshold


    # =========================================================================
    # VOLATILITY-NORMALIZED TARGET
    # =========================================================================

    volnorm_target = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    volnorm_threshold = pd.Series(
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


        target_i, threshold_i = (
            strictly_causal_target(
                df.loc[
                    ordered_idx,
                    (
                        f"future_qrf_loss_volnorm_"
                        f"{horizon}d"
                    ),
                ],
                horizon=horizon,
                percentile=0.75,
                min_history=MIN_TARGET_HISTORY,
            )
        )


        volnorm_target.loc[
            ordered_idx
        ] = target_i.to_numpy()


        volnorm_threshold.loc[
            ordered_idx
        ] = threshold_i.to_numpy()


    df[
        f"causal_volnorm_deterioration_{horizon}d"
    ] = volnorm_target


    df[
        f"causal_volnorm_threshold_{horizon}d"
    ] = volnorm_threshold


    # =========================================================================
    # RELATIVE QRF VS EWMA TARGET
    # =========================================================================

    relative_target = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )


    relative_threshold = pd.Series(
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


        target_i, threshold_i = (
            strictly_causal_target(
                df.loc[
                    ordered_idx,
                    (
                        f"future_qrf_ewma_logratio_"
                        f"{horizon}d"
                    ),
                ],
                horizon=horizon,
                percentile=0.75,
                min_history=MIN_TARGET_HISTORY,
            )
        )


        relative_target.loc[
            ordered_idx
        ] = target_i.to_numpy()


        relative_threshold.loc[
            ordered_idx
        ] = threshold_i.to_numpy()


    df[
        f"causal_relative_deterioration_{horizon}d"
    ] = relative_target


    df[
        f"causal_relative_threshold_{horizon}d"
    ] = relative_threshold


    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    original_rate = (
        df[
            f"causal_deterioration_{horizon}d"
        ]
        .mean()
    )


    volnorm_rate = (
        df[
            f"causal_volnorm_deterioration_{horizon}d"
        ]
        .mean()
    )


    relative_rate = (
        df[
            f"causal_relative_deterioration_{horizon}d"
        ]
        .mean()
    )


    print(
        f"  Original causal positive rate: "
        f"{original_rate:.4%}",
        flush=True,
    )


    print(
        f"  Vol-normalized positive rate: "
        f"{volnorm_rate:.4%}",
        flush=True,
    )


    print(
        f"  Relative-loss positive rate: "
        f"{relative_rate:.4%}",
        flush=True,
    )


    # =========================================================================
    # COMPARE OLD TARGET WITH NEW TARGET
    # =========================================================================

    old_column = (
        f"deterioration_{horizon}d"
    )


    new_column = (
        f"causal_deterioration_{horizon}d"
    )


    if old_column in df.columns:

        valid_compare = (
            df[old_column].notna()
            &
            df[new_column].notna()
        )


        if valid_compare.any():

            mismatch_rate = (
                (
                    df.loc[
                        valid_compare,
                        old_column,
                    ].astype(int)
                    !=
                    df.loc[
                        valid_compare,
                        new_column,
                    ].astype(int)
                )
                .mean()
            )

        else:

            mismatch_rate = np.nan


        print(
            f"  Old-vs-corrected mismatch: "
            f"{mismatch_rate:.4%}",
            flush=True,
        )


        target_audit_rows.append({

            "horizon":
                horizon,

            "metric":
                "old_vs_corrected_mismatch_rate",

            "value":
                mismatch_rate,
        })


    target_audit_rows.extend([

        {
            "horizon":
                horizon,

            "metric":
                "original_causal_positive_rate",

            "value":
                original_rate,
        },

        {
            "horizon":
                horizon,

            "metric":
                "volnorm_causal_positive_rate",

            "value":
                volnorm_rate,
        },

        {
            "horizon":
                horizon,

            "metric":
                "relative_causal_positive_rate",

            "value":
                relative_rate,
        },
    ])


# =============================================================================
# SAVE TARGET AUDIT
# =============================================================================

target_audit_df = pd.DataFrame(
    target_audit_rows
)


target_audit_df.to_csv(
    TARGET_AUDIT_PATH,
    index=False,
)


print(
    "\nSaved target audit:",
    TARGET_AUDIT_PATH,
    flush=True,
)


# =============================================================================
# CAUSAL FEATURE ENGINEERING
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "BUILDING CAUSAL FEATURES",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


grouped = df.groupby(
    "ticker",
    group_keys=False,
)


# =============================================================================
# PRIOR BREACH COUNTS
# =============================================================================

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


# =============================================================================
# EXTENDED FEATURES
# =============================================================================

extended_features = [
    c
    for c in EXTENDED_FEATURE_CANDIDATES
    if c in df.columns
]


eps = 1e-8


# =============================================================================
# LOSS RATIOS
# =============================================================================

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
            ]
            .abs()
            +
            eps
        )
    )

    extended_features.append(
        "loss_ratio_5_20"
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
            ]
            .abs()
            +
            eps
        )
    )

    extended_features.append(
        "loss_ratio_5_60"
    )


# =============================================================================
# BREACH-RATE ACCELERATION
# =============================================================================

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

    extended_features.append(
        "breach_rate_accel_5_20"
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

    extended_features.append(
        "breach_rate_accel_20_60"
    )


# =============================================================================
# VOLATILITY MOVEMENT
# =============================================================================

if "vol_ratio_20_60" in df.columns:

    df[
        "vol_ratio_change_1d"
    ] = grouped[
        "vol_ratio_20_60"
    ].transform(
        lambda s:
            s - s.shift(1)
    )

    extended_features.append(
        "vol_ratio_change_1d"
    )


# =============================================================================
# VaR MOVEMENT
# =============================================================================

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

    extended_features.append(
        "VaR_ratio_change_1d"
    )


# =============================================================================
# INTERACTIONS
# =============================================================================

interaction_specs = [

    (
        "loss_x_vol",
        "pinball_lag1",
        "vol_ratio_20_60",
    ),

    (
        "breach5_x_vol",
        "prior_breach_count_5",
        "vol_ratio_20_60",
    ),

    (
        "loss_x_breach5",
        "pinball_lag1",
        "prior_breach_count_5",
    ),

    (
        "vol_x_vix",
        "vol_ratio_20_60",
        "VIX",
    ),
]


for (
    name,
    a,
    b,
) in interaction_specs:

    if {
        a,
        b,
    }.issubset(df.columns):

        df[name] = (
            df[a] * df[b]
        )

        extended_features.append(
            name
        )


# =============================================================================
# PRIOR CROSS-SECTIONAL BREACH STRESS
# =============================================================================

date_breach_share = (
    df
    .groupby(
        "forecast_date"
    )["breach"]
    .mean()
    .sort_index()
)


prior_share = (
    date_breach_share
    .shift(1)
)


prior_share_5 = (
    prior_share
    .rolling(
        5,
        min_periods=1,
    )
    .mean()
)


df[
    "csb_prior_1d"
] = (
    df[
        "forecast_date"
    ]
    .map(
        prior_share
    )
)


df[
    "csb_prior_5d"
] = (
    df[
        "forecast_date"
    ]
    .map(
        prior_share_5
    )
)


extended_features.extend([
    "csb_prior_1d",
    "csb_prior_5d",
])


# =============================================================================
# CAUSAL WITHIN-TICKER Z-SCORES
# =============================================================================

z_source = [

    "pinball_lag1",

    "pinball_mean_5d_prior",

    "pinball_mean_20d_prior",

    "vol_ratio_20_60",

    "prior_breach_count_5",

    "q05_change_1d",

    "VaR_ratio_to_prior",

    "qrf_ewma_disagreement",
]


for column in z_source:

    if column not in df.columns:
        continue


    z_column = (
        f"{column}_ticker_causal_z"
    )


    df[z_column] = (
        grouped[column]
        .transform(
            causal_expanding_z
        )
    )


    extended_features.append(
        z_column
    )


extended_features = list(
    dict.fromkeys(
        extended_features
    )
)


BASE_FEATURES = [
    c
    for c in BASE_FEATURES
    if c in df.columns
]


print(
    f"Baseline features: "
    f"{len(BASE_FEATURES)}",
    flush=True,
)


print(
    f"Extended features: "
    f"{len(extended_features)}",
    flush=True,
)


# =============================================================================
# CHRONOLOGICAL SPLITS
# =============================================================================

train_mask, val_mask, test_mask = (
    make_splits(df)
)


print(
    "\nSPLIT SIZES",
    flush=True,
)


print(
    f"Train: "
    f"{int(train_mask.sum()):,}",
    flush=True,
)


print(
    f"Val:   "
    f"{int(val_mask.sum()):,}",
    flush=True,
)


print(
    f"Test:  "
    f"{int(test_mask.sum()):,}",
    flush=True,
)


# =============================================================================
# DESIGN MATRICES
# =============================================================================

print(
    "\nPreparing design matrices...",
    flush=True,
)


X_base = prepare_matrix(
    df,
    BASE_FEATURES,
    train_mask,
)


X_extended = prepare_matrix(
    df,
    extended_features,
    train_mask,
)


print(
    "Design matrices ready.",
    flush=True,
)


# =============================================================================
# MODEL EXPERIMENTS
# =============================================================================

results = []

bootstrap_results = []

prediction_results = []


target_specs = []


for horizon in HORIZONS:

    target_specs.extend([

        (
            horizon,
            "original_causal_pinball",
            (
                f"causal_deterioration_"
                f"{horizon}d"
            ),
        ),

        (
            horizon,
            "volatility_normalized",
            (
                f"causal_volnorm_deterioration_"
                f"{horizon}d"
            ),
        ),

        (
            horizon,
            "relative_qrf_vs_ewma",
            (
                f"causal_relative_deterioration_"
                f"{horizon}d"
            ),
        ),
    ])


# =============================================================================
# TARGET LOOP
# =============================================================================

for (
    horizon,
    target_name,
    target_column,
) in target_specs:

    print(
        "\n" + "=" * 100,
        flush=True,
    )

    print(
        f"HORIZON = {horizon} DAYS",
        flush=True,
    )

    print(
        f"TARGET = {target_name}",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )


    valid = (
        df[target_column]
        .notna()
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
            target_column,
        ]
        .astype(int)
    )


    y_val = (
        df.loc[
            val,
            target_column,
        ]
        .astype(int)
    )


    y_test = (
        df.loc[
            test,
            target_column,
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


    print(
        f"Test positive rate: "
        f"{y_test.mean():.4%}",
        flush=True,
    )


    if (
        y_train.nunique() < 2
        or
        y_test.nunique() < 2
    ):

        print(
            "Skipping: insufficient class variation.",
            flush=True,
        )

        continue


    # =========================================================================
    # LOGISTIC BASELINE
    # =========================================================================

    print(
        "\n[1/3] Logistic baseline...",
        flush=True,
    )


    logistic_base = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )


    logistic_base.fit(
        X_base.loc[train],
        y_train,
    )


    score_base = (
        logistic_base
        .predict_proba(
            X_base.loc[test]
        )[:, 1]
    )


    auc_base = auc_safe(
        y_test,
        score_base,
    )


    # =========================================================================
    # LOGISTIC EXTENDED
    # =========================================================================

    print(
        "[2/3] Logistic extended...",
        flush=True,
    )


    logistic_extended = (
        LogisticRegression(
            C=0.5,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    )


    logistic_extended.fit(
        X_extended.loc[train],
        y_train,
    )


    score_logistic_extended = (
        logistic_extended
        .predict_proba(
            X_extended.loc[test]
        )[:, 1]
    )


    auc_logistic_extended = (
        auc_safe(
            y_test,
            score_logistic_extended,
        )
    )


    # =========================================================================
    # XGBOOST
    # =========================================================================

    print(
        "[3/3] XGBoost extended...",
        flush=True,
    )


    xgb = XGBClassifier(
        **XGB_CLASSIFIER_PARAMS
    )


    xgb.fit(
        X_extended.loc[train],
        y_train,
        eval_set=[
            (
                X_extended.loc[val],
                y_val,
            )
        ],
        verbose=False,
    )


    score_xgb = (
        xgb
        .predict_proba(
            X_extended.loc[test]
        )[:, 1]
    )


    auc_xgb = auc_safe(
        y_test,
        score_xgb,
    )


    print(
        f"    Baseline AUC: "
        f"{auc_base:.4f}",
        flush=True,
    )


    print(
        f"    Logistic extended AUC: "
        f"{auc_logistic_extended:.4f}",
        flush=True,
    )


    print(
        f"    XGBoost extended AUC: "
        f"{auc_xgb:.4f}",
        flush=True,
    )


    print(
        f"    XGB delta vs baseline: "
        f"{auc_xgb - auc_base:+.4f}",
        flush=True,
    )


    # =========================================================================
    # CONTINUOUS FUTURE-LOSS REGRESSION
    # =========================================================================

    continuous_auc = np.nan
    continuous_mse = np.nan


    if (
        target_name
        ==
        "original_causal_pinball"
    ):

        continuous_column = (
            f"future_qrf_loss_{horizon}d"
        )


        continuous_valid = (
            df[
                continuous_column
            ].notna()
            &
            df[
                target_column
            ].notna()
        )


        train_cont = (
            train_mask
            &
            continuous_valid
        )


        val_cont = (
            val_mask
            &
            continuous_valid
        )


        test_cont = (
            test_mask
            &
            continuous_valid
        )


        print(
            "\nContinuous-loss XGB regression...",
            flush=True,
        )


        reg = XGBRegressor(
            **XGB_REGRESSOR_PARAMS
        )


        reg.fit(
            X_extended.loc[
                train_cont
            ],
            df.loc[
                train_cont,
                continuous_column,
            ].astype(float),
            eval_set=[
                (
                    X_extended.loc[
                        val_cont
                    ],
                    df.loc[
                        val_cont,
                        continuous_column,
                    ].astype(float),
                )
            ],
            verbose=False,
        )


        continuous_score = (
            reg.predict(
                X_extended.loc[
                    test_cont
                ]
            )
        )


        continuous_auc = (
            auc_safe(
                df.loc[
                    test_cont,
                    target_column,
                ].astype(int),
                continuous_score,
            )
        )


        continuous_mse = (
            mean_squared_error(
                df.loc[
                    test_cont,
                    continuous_column,
                ].astype(float),
                continuous_score,
            )
        )


        print(
            f"    Continuous-loss AUC: "
            f"{continuous_auc:.4f}",
            flush=True,
        )


        print(
            f"    Continuous-loss MSE: "
            f"{continuous_mse:.8f}",
            flush=True,
        )


    # =========================================================================
    # SAVE RESULTS
    # =========================================================================

    results.append({

        "horizon_days":
            horizon,

        "target":
            target_name,

        "n_train":
            int(train.sum()),

        "n_val":
            int(val.sum()),

        "n_test":
            int(test.sum()),

        "test_positive_rate":
            float(y_test.mean()),

        "auc_logistic_baseline":
            auc_base,

        "auc_logistic_extended":
            auc_logistic_extended,

        "auc_xgb_extended":
            auc_xgb,

        "auc_continuous_loss_rank":
            continuous_auc,

        "continuous_loss_mse":
            continuous_mse,

        "delta_logistic_extended_vs_baseline":
            (
                auc_logistic_extended
                -
                auc_base
            ),

        "delta_xgb_vs_baseline":
            (
                auc_xgb
                -
                auc_base
            ),
    })


    # =========================================================================
    # PREDICTIONS FOR BOOTSTRAP
    # =========================================================================

    pred = pd.DataFrame({

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
            horizon,

        "target":
            target_name,

        "y_true":
            y_test.to_numpy(),

        "score_logistic_baseline":
            score_base,

        "score_logistic_extended":
            score_logistic_extended,

        "score_xgb_extended":
            score_xgb,
    })


    prediction_results.append(
        pred
    )


    # =========================================================================
    # DATE-BLOCK BOOTSTRAP
    # =========================================================================

    if (
        target_name
        ==
        "original_causal_pinball"
    ):

        print(
            "\nDate-block bootstrap...",
            flush=True,
        )


        bootstrap_input = pred[
            [
                "forecast_date",
                "y_true",
                "score_logistic_baseline",
                "score_logistic_extended",
                "score_xgb_extended",
            ]
        ].copy()


        # ---------------------------------------------------------------------
        # XGB vs baseline
        # ---------------------------------------------------------------------

        (
            mean_delta_xgb,
            low_xgb,
            high_xgb,
        ) = bootstrap_auc_delta(

            bootstrap_input,

            baseline_column=
                "score_logistic_baseline",

            augmented_column=
                "score_xgb_extended",

            reps=
                BOOTSTRAP_REPS,

            block_days=
                BLOCK_DAYS,

            seed=
                RANDOM_STATE
                + horizon,
        )


        print(
            f"    XGB bootstrap CI: "
            f"[{low_xgb:+.4f}, "
            f"{high_xgb:+.4f}]",
            flush=True,
        )


        bootstrap_results.append({

            "horizon_days":
                horizon,

            "comparison":
                (
                    "XGB_extended_minus_"
                    "logistic_baseline"
                ),

            "observed_delta":
                (
                    auc_xgb
                    -
                    auc_base
                ),

            "bootstrap_mean_delta":
                mean_delta_xgb,

            "ci_lower":
                low_xgb,

            "ci_upper":
                high_xgb,

            "robust_positive":
                (
                    np.isfinite(low_xgb)
                    and
                    low_xgb > 0
                ),

            "block_days":
                BLOCK_DAYS,

            "bootstrap_reps":
                BOOTSTRAP_REPS,
        })


        # ---------------------------------------------------------------------
        # Logistic extended vs baseline
        # ---------------------------------------------------------------------

        (
            mean_delta_log,
            low_log,
            high_log,
        ) = bootstrap_auc_delta(

            bootstrap_input,

            baseline_column=
                "score_logistic_baseline",

            augmented_column=
                "score_logistic_extended",

            reps=
                BOOTSTRAP_REPS,

            block_days=
                BLOCK_DAYS,

            seed=
                RANDOM_STATE
                + horizon
                + 100,
        )


        print(
            f"    Logistic bootstrap CI: "
            f"[{low_log:+.4f}, "
            f"{high_log:+.4f}]",
            flush=True,
        )


        bootstrap_results.append({

            "horizon_days":
                horizon,

            "comparison":
                (
                    "logistic_extended_minus_"
                    "logistic_baseline"
                ),

            "observed_delta":
                (
                    auc_logistic_extended
                    -
                    auc_base
                ),

            "bootstrap_mean_delta":
                mean_delta_log,

            "ci_lower":
                low_log,

            "ci_upper":
                high_log,

            "robust_positive":
                (
                    np.isfinite(low_log)
                    and
                    low_log > 0
                ),

            "block_days":
                BLOCK_DAYS,

            "bootstrap_reps":
                BOOTSTRAP_REPS,
        })


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

results_df = pd.DataFrame(
    results
)


bootstrap_df = pd.DataFrame(
    bootstrap_results
)


if prediction_results:

    predictions_df = pd.concat(
        prediction_results,
        ignore_index=True,
    )

else:

    predictions_df = pd.DataFrame()


results_df.to_csv(
    RESULTS_PATH,
    index=False,
)


bootstrap_df.to_csv(
    BOOTSTRAP_PATH,
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
    "EXPERIMENT 37C — FINAL SUMMARY",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)


if not results_df.empty:

    summary_columns = [

        "horizon_days",

        "target",

        "auc_logistic_baseline",

        "auc_logistic_extended",

        "auc_xgb_extended",

        "auc_continuous_loss_rank",

        "delta_logistic_extended_vs_baseline",

        "delta_xgb_vs_baseline",
    ]


    print(
        results_df[
            summary_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        ),
        flush=True,
    )


print(
    "\n" + "-" * 100,
    flush=True,
)

print(
    "PRIMARY BOOTSTRAP RESULTS",
    flush=True,
)

print(
    "-" * 100,
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

else:

    print(
        "No bootstrap results.",
        flush=True,
    )


print(
    "\n" + "-" * 100,
    flush=True,
)

print(
    "OUTPUT FILES",
    flush=True,
)

print(
    "-" * 100,
    flush=True,
)


print(
    TARGET_AUDIT_PATH,
    flush=True,
)

print(
    RESULTS_PATH,
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
    "EXPERIMENT 37C COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)