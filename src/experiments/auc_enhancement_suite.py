"""
EXPERIMENT 37 — CORRECTED AUC ENHANCEMENT SUITE
================================================

QRF panel only.

This script tests five pre-specified ways to improve deterioration
discrimination without changing the information set after seeing the test set:

1. Nonlinear model:
   Logistic regression vs XGBoost.

2. Dynamic / interaction features:
   loss ratios, acceleration, volatility change, VaR change,
   breach/volatility interactions, prior cross-sectional stress.

3. Continuous future-loss prediction:
   XGBoost predicts continuous future pinball loss and the predictions
   are ranked against the binary deterioration target.

4. Relative/model-performance deterioration:
   QRF future pinball loss relative to EWMA future pinball loss.

5. Within-ticker normalization:
   causal expanding z-scores using strictly prior observations only.

IMPORTANT
---------
This is a controlled QRF-panel experiment. It does NOT replace the final
cross-model (HS/EWMA/GARCH/QRF) paper protocol.

INPUTS
------
D:\reliability_layer\data\monitoring\QRF_25ticker_reliability_panel_with_targets.csv
D:\reliability_layer\data\monitoring\EWMA_25ticker_oos_predictions.csv

OUTPUTS
-------
D:\reliability_layer\data\results\auc_enhancement_suite_corrected_results.csv
D:\reliability_layer\data\results\auc_enhancement_suite_corrected_predictions.csv
D:\reliability_layer\data\results\auc_enhancement_suite_corrected_bootstrap.csv
D:\reliability_layer\data\results\auc_enhancement_suite_corrected_relative_targets.csv

RUN
---
python auc_enhancement_suite_corrected.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

ALPHA = 0.05

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

HORIZONS = (5, 10, 20)

# Kept deliberately moderate so the experiment is practical to run locally.
XGB_N_ESTIMATORS = 180
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE = 0.80
XGB_COLSAMPLE = 0.80
XGB_MIN_CHILD_WEIGHT = 8
XGB_REG_ALPHA = 0.05
XGB_REG_LAMBDA = 1.5
XGB_MAX_BIN = 64

BOOTSTRAP_REPS = 500
BOOTSTRAP_BLOCK_DATES = 20

RANDOM_STATE = 42


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MONITORING_DIR = ROOT / "data" / "monitoring"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

QRF_PATH = (
    MONITORING_DIR /
    "QRF_25ticker_reliability_panel_with_targets.csv"
)

EWMA_PATH = (
    MONITORING_DIR /
    "EWMA_25ticker_oos_predictions.csv"
)

RESULTS_PATH = (
    RESULTS_DIR /
    "auc_enhancement_suite_corrected_results.csv"
)

PRED_PATH = (
    RESULTS_DIR /
    "auc_enhancement_suite_corrected_predictions.csv"
)

BOOT_PATH = (
    RESULTS_DIR /
    "auc_enhancement_suite_corrected_bootstrap.csv"
)

REL_TARGET_PATH = (
    RESULTS_DIR /
    "auc_enhancement_suite_corrected_relative_targets.csv"
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing required columns: {missing}"
        )


def pinball_loss(
    actual: np.ndarray,
    quantile: np.ndarray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Lower-tail pinball loss for quantile alpha.
    """
    actual = np.asarray(actual, dtype=float)
    quantile = np.asarray(quantile, dtype=float)

    return np.where(
        actual < quantile,
        (alpha - 1.0) * (actual - quantile),
        alpha * (actual - quantile),
    )


def find_q05_column(df: pd.DataFrame) -> str:
    """
    FIXED:
    The actual EWMA file uses 'ewma_q05'.

    The previous script had the candidates but forgot to return a matching
    column, which caused ewma_q_col to become None and then triggered:

        AttributeError: 'NoneType' object has no attribute 'lower'
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

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a 5% lower-quantile column in EWMA data. "
        f"Available columns: {list(df.columns)}"
    )


def normalize_q_column(
    df: pd.DataFrame,
    col: str,
) -> pd.Series:
    """
    True q05 columns are already lower quantiles.
    VaR_5pct-style columns are treated as positive loss magnitudes
    and converted to negative lower-quantile values.
    """
    if col.lower().startswith("var") or col == "VaR_5pct":
        x = pd.to_numeric(df[col], errors="coerce")
        return -np.abs(x)

    return pd.to_numeric(df[col], errors="coerce")


def auc_safe(
    y_true: pd.Series | np.ndarray,
    score: np.ndarray,
) -> float:
    y = np.asarray(y_true).astype(int)

    if np.unique(y).size < 2:
        return np.nan

    return float(roc_auc_score(y, score))


# =============================================================================
# CAUSAL / FUTURE TARGET HELPERS
# =============================================================================

def causal_expanding_z(series: pd.Series) -> pd.Series:
    """
    Current value standardized using only STRICTLY PRIOR values.
    """
    s = pd.to_numeric(series, errors="coerce")

    prior = s.shift(1)

    mean = prior.expanding(min_periods=20).mean()
    std = prior.expanding(min_periods=20).std(ddof=0)

    z = (s - mean) / std.replace(0, np.nan)

    return z.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)


def future_mean_grouped(
    df: pd.DataFrame,
    value_col: str,
    horizon: int,
) -> pd.Series:
    """
    At time t:
        mean(value[t+1], ..., value[t+h])

    This is used only for TARGET construction.
    """
    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, idx in df.groupby(
        "ticker",
        sort=False,
    ).groups.items():

        sub = (
            df.loc[idx]
            .sort_values("forecast_date")
        )

        values = (
            pd.to_numeric(
                sub[value_col],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        n = len(values)

        arr = np.full(
            n,
            np.nan,
            dtype=float,
        )

        # O(n*h), but n is only ~873 per ticker.
        for i in range(n - horizon):
            w = values[i + 1:i + horizon + 1]

            if np.all(np.isfinite(w)):
                arr[i] = float(np.mean(w))

        result.loc[sub.index] = arr

    return result


def future_std_grouped(
    df: pd.DataFrame,
    value_col: str,
    horizon: int,
) -> pd.Series:
    """
    At time t:
        std(value[t+1], ..., value[t+h])
    """
    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, idx in df.groupby(
        "ticker",
        sort=False,
    ).groups.items():

        sub = (
            df.loc[idx]
            .sort_values("forecast_date")
        )

        values = (
            pd.to_numeric(
                sub[value_col],
                errors="coerce",
            )
            .to_numpy(dtype=float)
        )

        n = len(values)

        arr = np.full(
            n,
            np.nan,
            dtype=float,
        )

        for i in range(n - horizon):
            w = values[i + 1:i + horizon + 1]

            if np.all(np.isfinite(w)) and len(w) >= 2:
                arr[i] = float(np.std(w, ddof=1))

        result.loc[sub.index] = arr

    return result


def add_causal_percentile_target(
    df: pd.DataFrame,
    continuous_col: str,
    percentile: float = 0.75,
    min_history: int = 50,
) -> pd.Series:
    """
    Binary target:
        current continuous value >
        ticker-specific expanding percentile based ONLY on prior values.
    """
    result = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    for ticker, idx in df.groupby(
        "ticker",
        sort=False,
    ).groups.items():

        sub = (
            df.loc[idx]
            .sort_values("forecast_date")
        )

        values = pd.to_numeric(
            sub[continuous_col],
            errors="coerce",
        )

        prior = values.shift(1)

        threshold = (
            prior
            .expanding(min_periods=min_history)
            .quantile(percentile)
        )

        target = (
            values > threshold
        ).astype(float)

        target[threshold.isna()] = np.nan

        result.loc[sub.index] = target

    return result


# =============================================================================
# SPLITS
# =============================================================================

def make_splits(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:

    train = pd.Series(False, index=df.index)
    val = pd.Series(False, index=df.index)
    test = pd.Series(False, index=df.index)

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
            np.floor(n * TRAIN_FRAC)
        )

        n_val = int(
            np.floor(n * VAL_FRAC)
        )

        train.loc[
            sub.index[:n_train]
        ] = True

        val.loc[
            sub.index[n_train:n_train + n_val]
        ] = True

        test.loc[
            sub.index[n_train + n_val:]
        ] = True

    return train, val, test


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:

    x = df.copy()

    g = x.groupby(
        "ticker",
        group_keys=False,
    )

    # -------------------------------------------------------------------------
    # Explicit causal breach history
    # -------------------------------------------------------------------------

    x["prior_breach_count_5"] = (
        g["breach"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                5,
                min_periods=1,
            )
            .sum()
        )
    )

    x["prior_breach_count_20"] = (
        g["breach"]
        .transform(
            lambda s:
            s.shift(1)
            .rolling(
                20,
                min_periods=1,
            )
            .sum()
        )
    )

    # -------------------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------------------

    base_features = [
        c
        for c in [
            "pinball_lag1",
            "vol_ratio_20_60",
        ]
        if c in x.columns
    ]

    # -------------------------------------------------------------------------
    # Existing signals
    # -------------------------------------------------------------------------

    candidates = [
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

    extended_features = [
        c
        for c in candidates
        if c in x.columns
    ]

    eps = 1e-8

    # -------------------------------------------------------------------------
    # Dynamic loss features
    # -------------------------------------------------------------------------

    if {
        "pinball_mean_5d_prior",
        "pinball_mean_20d_prior",
    }.issubset(x.columns):

        x["loss_ratio_5_20"] = (
            x["pinball_mean_5d_prior"] /
            (
                x["pinball_mean_20d_prior"].abs()
                + eps
            )
        )

        extended_features.append(
            "loss_ratio_5_20"
        )

    if {
        "pinball_mean_5d_prior",
        "pinball_mean_60d_prior",
    }.issubset(x.columns):

        x["loss_ratio_5_60"] = (
            x["pinball_mean_5d_prior"] /
            (
                x["pinball_mean_60d_prior"].abs()
                + eps
            )
        )

        extended_features.append(
            "loss_ratio_5_60"
        )

    # -------------------------------------------------------------------------
    # Breach acceleration
    # -------------------------------------------------------------------------

    if {
        "breach_rate_5d",
        "breach_rate_20d",
    }.issubset(x.columns):

        x["breach_rate_accel_5_20"] = (
            x["breach_rate_5d"] -
            x["breach_rate_20d"]
        )

        extended_features.append(
            "breach_rate_accel_5_20"
        )

    if {
        "breach_rate_20d",
        "breach_rate_60d",
    }.issubset(x.columns):

        x["breach_rate_accel_20_60"] = (
            x["breach_rate_20d"] -
            x["breach_rate_60d"]
        )

        extended_features.append(
            "breach_rate_accel_20_60"
        )

    # -------------------------------------------------------------------------
    # Volatility / VaR changes
    # -------------------------------------------------------------------------

    if "vol_ratio_20_60" in x.columns:

        x["vol_ratio_change_1d"] = (
            g["vol_ratio_20_60"]
            .transform(
                lambda s: s - s.shift(1)
            )
        )

        extended_features.append(
            "vol_ratio_change_1d"
        )

    if "VaR_ratio_to_prior" in x.columns:

        x["VaR_ratio_change_1d"] = (
            x["VaR_ratio_to_prior"] - 1.0
        )

        extended_features.append(
            "VaR_ratio_change_1d"
        )

    # -------------------------------------------------------------------------
    # Explicit interactions
    # -------------------------------------------------------------------------

    interactions = [
        (
            "pinball_x_vol",
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

    for new_name, a, b in interactions:

        if {
            a,
            b,
        }.issubset(x.columns):

            x[new_name] = (
                x[a] *
                x[b]
            )

            extended_features.append(
                new_name
            )

    # -------------------------------------------------------------------------
    # Causal cross-sectional stress
    # -------------------------------------------------------------------------

    date_breach_share = (
        x.groupby("forecast_date")["breach"]
        .mean()
        .sort_index()
    )

    prior_share = (
        date_breach_share.shift(1)
    )

    prior_share_5 = (
        prior_share
        .rolling(
            5,
            min_periods=1,
        )
        .mean()
    )

    x["csb_prior_1d"] = (
        x["forecast_date"]
        .map(prior_share)
    )

    x["csb_prior_5d"] = (
        x["forecast_date"]
        .map(prior_share_5)
    )

    extended_features += [
        "csb_prior_1d",
        "csb_prior_5d",
    ]

    # -------------------------------------------------------------------------
    # Causal within-ticker normalization
    # -------------------------------------------------------------------------

    z_source = [
        c
        for c in [
            "pinball_lag1",
            "pinball_mean_5d_prior",
            "pinball_mean_20d_prior",
            "vol_ratio_20_60",
            "prior_breach_count_5",
            "q05_change_1d",
            "VaR_ratio_to_prior",
            "qrf_ewma_disagreement",
        ]
        if c in x.columns
    ]

    for c in z_source:

        z_name = (
            f"{c}_ticker_causal_z"
        )

        x[z_name] = (
            x.groupby(
                "ticker",
                group_keys=False,
            )[c]
            .transform(
                causal_expanding_z
            )
        )

        extended_features.append(
            z_name
        )

    base_features = list(
        dict.fromkeys(
            base_features
        )
    )

    extended_features = list(
        dict.fromkeys(
            c
            for c in extended_features
            if c in x.columns
        )
    )

    return (
        x,
        base_features,
        extended_features,
    )


# =============================================================================
# DESIGN MATRIX
# =============================================================================

def prepare_matrix(
    df: pd.DataFrame,
    features: list[str],
    train_mask: pd.Series,
) -> pd.DataFrame:

    X = (
        df[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .copy()
    )

    # TRAIN medians only.
    medians = (
        X.loc[train_mask]
        .median()
    )

    X = (
        X
        .fillna(medians)
        .fillna(0.0)
    )

    numeric_cols = list(features)

    # TRAIN-only scaling.
    scaler = StandardScaler()

    scaler.fit(
        X.loc[
            train_mask,
            numeric_cols,
        ]
    )

    X.loc[:, numeric_cols] = (
        scaler.transform(
            X.loc[:, numeric_cols]
        )
    )

    # Ticker fixed effects.
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


# =============================================================================
# MODELS
# =============================================================================

def fit_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:

    model = LogisticRegression(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def fit_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:

    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        max_bin=XGB_MAX_BIN,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        verbose=False,
    )

    return model


def fit_xgb_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBRegressor:

    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT,
        reg_alpha=XGB_REG_ALPHA,
        reg_lambda=XGB_REG_LAMBDA,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        max_bin=XGB_MAX_BIN,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_val,
                y_val,
            )
        ],
        verbose=False,
    )

    return model


# =============================================================================
# FAST DATE-BLOCK BOOTSTRAP
# =============================================================================

def make_bootstrap_blocks(
    prediction_df: pd.DataFrame,
    block_length: int,
) -> tuple[list[np.ndarray], int]:

    dates = (
        pd.to_datetime(
            prediction_df["forecast_date"]
        )
        .dt.normalize()
        .to_numpy()
    )

    unique_dates = np.array(
        sorted(
            np.unique(dates)
        )
    )

    n_dates = len(unique_dates)

    block_length = min(
        block_length,
        n_dates,
    )

    # Map each date to row indices.
    date_to_rows = {
        d: np.flatnonzero(
            dates == d
        )
        for d in unique_dates
    }

    possible_starts = np.arange(
        0,
        n_dates - block_length + 1,
    )

    blocks = []

    for start in range(
        0,
        n_dates,
        block_length,
    ):

        block_dates = unique_dates[
            start:start + block_length
        ]

        if len(block_dates) == 0:
            continue

        rows = np.concatenate(
            [
                date_to_rows[d]
                for d in block_dates
            ]
        )

        blocks.append(rows)

    return (
        blocks,
        len(possible_starts),
    )


def block_bootstrap_auc_delta(
    prediction_df: pd.DataFrame,
    baseline_column: str,
    augmented_column: str,
    reps: int,
    block_length: int,
    seed: int,
) -> tuple[float, float, float]:

    work = prediction_df.reset_index(
        drop=True
    )

    y = work["target"].to_numpy(
        dtype=np.int8
    )

    base_score = work[
        baseline_column
    ].to_numpy(
        dtype=float
    )

    aug_score = work[
        augmented_column
    ].to_numpy(
        dtype=float
    )

    blocks, n_starts = (
        make_bootstrap_blocks(
            work,
            block_length,
        )
    )

    n_blocks = len(blocks)

    if n_blocks == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    deltas = []

    # Date-level block bootstrap.
    # Work entirely with NumPy arrays inside the loop.
    for _ in range(reps):

        sampled_block_ids = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        idx = np.concatenate(
            [
                blocks[i]
                for i in sampled_block_ids
            ]
        )

        yy = y[idx]

        if np.unique(yy).size < 2:
            continue

        a = base_score[idx]
        b = aug_score[idx]

        deltas.append(
            roc_auc_score(
                yy,
                b,
            )
            -
            roc_auc_score(
                yy,
                a,
            )
        )

    if not deltas:
        return np.nan, np.nan, np.nan

    d = np.asarray(
        deltas,
        dtype=float,
    )

    return (
        float(np.mean(d)),
        float(np.quantile(d, 0.025)),
        float(np.quantile(d, 0.975)),
    )


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 100, flush=True)
print("EXPERIMENT 37 — CORRECTED AUC ENHANCEMENT SUITE", flush=True)
print("=" * 100, flush=True)

print(f"ROOT: {ROOT}", flush=True)
print(f"QRF:  {QRF_PATH}", flush=True)
print(f"EWMA: {EWMA_PATH}", flush=True)

if not QRF_PATH.exists():
    raise FileNotFoundError(QRF_PATH)

if not EWMA_PATH.exists():
    raise FileNotFoundError(EWMA_PATH)

qrf = pd.read_csv(QRF_PATH)

qrf["ticker"] = (
    qrf["ticker"]
    .astype(str)
    .str.strip()
)

qrf["forecast_date"] = pd.to_datetime(
    qrf["forecast_date"]
)

require_columns(
    qrf,
    [
        "ticker",
        "forecast_date",
        "actual_return",
        "q05_forecast",
        "breach",
        "pinball_loss",
    ],
    "QRF panel",
)

qrf = (
    qrf
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

print(
    f"\nQRF panel shape: {qrf.shape}",
    flush=True,
)

print(
    f"Tickers: {qrf['ticker'].nunique()}",
    flush=True,
)


# =============================================================================
# LOAD EWMA — FIXED COLUMN DETECTION
# =============================================================================

ewma = pd.read_csv(EWMA_PATH)

ewma["ticker"] = (
    ewma["ticker"]
    .astype(str)
    .str.strip()
)

date_candidates = [
    "forecast_date",
    "Date",
    "date",
]

ewma_date_col = next(
    (
        c
        for c in date_candidates
        if c in ewma.columns
    ),
    None,
)

if ewma_date_col is None:
    raise ValueError(
        "Could not find EWMA forecast date column. "
        f"Available columns: {list(ewma.columns)}"
    )

ewma["forecast_date"] = pd.to_datetime(
    ewma[ewma_date_col]
)

# THIS NOW CORRECTLY RETURNS 'ewma_q05'.
ewma_q_col = find_q05_column(
    ewma
)

print(
    f"\nEWMA 5% quantile column detected: "
    f"{ewma_q_col}",
    flush=True,
)

ewma["ewma_q05"] = normalize_q_column(
    ewma,
    ewma_q_col,
)

print(
    f"EWMA rows: {len(ewma):,}",
    flush=True,
)


# =============================================================================
# MERGE QRF / EWMA CURRENT-DAY FORECASTS
# =============================================================================

ewma_small = (
    ewma[
        [
            "ticker",
            "forecast_date",
            "ewma_q05",
        ]
    ]
    .drop_duplicates(
        [
            "ticker",
            "forecast_date",
        ]
    )
)

qrf = qrf.merge(
    ewma_small,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="left",
    validate="one_to_one",
)

qrf["qrf_ewma_disagreement"] = (
    (
        qrf["q05_forecast"]
        -
        qrf["ewma_q05"]
    )
    /
    (
        qrf["ewma_q05"].abs()
        + 1e-8
    )
)

print(
    "\nQRF-EWMA current forecast coverage:",
    f"{qrf['ewma_q05'].notna().mean():.4%}",
    flush=True,
)


# =============================================================================
# BUILD FUTURE CONTINUOUS TARGETS
# =============================================================================

print(
    "\nBuilding future continuous targets...",
    flush=True,
)

# Pinball losses are already in the QRF panel.
# Use those directly to preserve the original target definition.
qrf["qrf_pinball"] = pd.to_numeric(
    qrf["pinball_loss"],
    errors="coerce",
)

# EWMA future loss requires current EWMA q05 + realized return.
qrf["ewma_pinball"] = pinball_loss(
    qrf["actual_return"].to_numpy(),
    qrf["ewma_q05"].to_numpy(),
)

for h in HORIZONS:

    print(
        f"  Building {h}d targets...",
        flush=True,
    )

    qrf[
        f"future_qrf_loss_{h}d"
    ] = future_mean_grouped(
        qrf,
        "qrf_pinball",
        h,
    )

    qrf[
        f"future_ewma_loss_{h}d"
    ] = future_mean_grouped(
        qrf,
        "ewma_pinball",
        h,
    )

    qrf[
        f"future_return_std_{h}d"
    ] = future_std_grouped(
        qrf,
        "actual_return",
        h,
    )

    # Volatility-normalized future loss.
    qrf[
        f"future_qrf_loss_volnorm_{h}d"
    ] = (
        qrf[
            f"future_qrf_loss_{h}d"
        ]
        /
        (
            qrf[
                f"future_return_std_{h}d"
            ].abs()
            + 1e-8
        )
    )

    # Relative performance:
    # log(QRF future loss / EWMA future loss)
    qrf[
        f"future_qrf_ewma_logratio_{h}d"
    ] = (
        np.log(
            qrf[
                f"future_qrf_loss_{h}d"
            ].clip(lower=1e-10)
        )
        -
        np.log(
            qrf[
                f"future_ewma_loss_{h}d"
            ].clip(lower=1e-10)
        )
    )

    # Volatility-normalized deterioration target.
    qrf[
        f"volnorm_deterioration_{h}d"
    ] = add_causal_percentile_target(
        qrf,
        f"future_qrf_loss_volnorm_{h}d",
        percentile=0.75,
        min_history=50,
    )

    # Relative model-performance deterioration target.
    qrf[
        f"relative_deterioration_{h}d"
    ] = add_causal_percentile_target(
        qrf,
        f"future_qrf_ewma_logratio_{h}d",
        percentile=0.75,
        min_history=50,
    )


# Save target audit panel.
qrf[
    [
        "ticker",
        "forecast_date",
    ]
    +
    [
        c
        for h in HORIZONS
        for c in [
            f"future_qrf_loss_{h}d",
            f"future_ewma_loss_{h}d",
            f"future_qrf_loss_volnorm_{h}d",
            f"future_qrf_ewma_logratio_{h}d",
            f"volnorm_deterioration_{h}d",
            f"relative_deterioration_{h}d",
        ]
    ]
].to_csv(
    REL_TARGET_PATH,
    index=False,
)

print(
    f"\nSaved target audit panel: "
    f"{REL_TARGET_PATH}",
    flush=True,
)


# =============================================================================
# FEATURES
# =============================================================================

qrf, base_features, extended_features = (
    build_features(qrf)
)

# Causal z-score features are already included in extended_features.

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "FEATURE SET",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    f"Baseline features ({len(base_features)}):",
    flush=True,
)

print(
    base_features,
    flush=True,
)

print(
    f"\nExtended features ({len(extended_features)}):",
    flush=True,
)

print(
    extended_features,
    flush=True,
)


# =============================================================================
# SPLITS
# =============================================================================

train_mask, val_mask, test_mask = (
    make_splits(qrf)
)

print(
    "\nSPLIT SIZES",
    flush=True,
)

print(
    f"Train: {int(train_mask.sum()):,}",
    flush=True,
)

print(
    f"Val:   {int(val_mask.sum()):,}",
    flush=True,
)

print(
    f"Test:  {int(test_mask.sum()):,}",
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
    qrf,
    base_features,
    train_mask,
)

X_ext = prepare_matrix(
    qrf,
    extended_features,
    train_mask,
)

print(
    "Design matrices ready.",
    flush=True,
)


# =============================================================================
# MAIN EXPERIMENTS
# =============================================================================

results = []
predictions = []
bootstrap_results = []

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

    target_specs = [
        (
            "original_normalized_pinball",
            f"deterioration_{h}d",
            "original",
        ),
        (
            "volatility_normalized_loss",
            f"volnorm_deterioration_{h}d",
            "volnorm",
        ),
        (
            "relative_qrf_vs_ewma_loss",
            f"relative_deterioration_{h}d",
            "relative",
        ),
    ]

    for target_name, y_col, target_type in target_specs:

        if y_col not in qrf.columns:
            print(
                f"\nSkipping {target_name}: "
                f"{y_col} not present.",
                flush=True,
            )
            continue

        valid = qrf[y_col].notna()

        train = train_mask & valid
        val = val_mask & valid
        test = test_mask & valid

        if (
            qrf.loc[train, y_col].nunique() < 2
            or
            qrf.loc[test, y_col].nunique() < 2
        ):
            print(
                f"\nSkipping {target_name}: "
                "insufficient target variation.",
                flush=True,
            )
            continue

        Xtr_b = X_base.loc[train]
        Xte_b = X_base.loc[test]

        Xtr_e = X_ext.loc[train]
        Xv_e = X_ext.loc[val]
        Xte_e = X_ext.loc[test]

        ytr = (
            qrf.loc[train, y_col]
            .astype(int)
        )

        yv = (
            qrf.loc[val, y_col]
            .astype(int)
        )

        yte = (
            qrf.loc[test, y_col]
            .astype(int)
        )

        # ---------------------------------------------------------------------
        # Logistic baseline
        # ---------------------------------------------------------------------

        print(
            f"\nTarget: {target_name}",
            flush=True,
        )

        print(
            "  1/3 Logistic baseline...",
            flush=True,
        )

        logit_base = fit_logistic(
            Xtr_b,
            ytr,
        )

        score_base = (
            logit_base
            .predict_proba(Xte_b)[:, 1]
        )

        auc_base = auc_safe(
            yte,
            score_base,
        )

        # ---------------------------------------------------------------------
        # Logistic extended
        # ---------------------------------------------------------------------

        print(
            "  2/3 Logistic extended...",
            flush=True,
        )

        logit_ext = fit_logistic(
            Xtr_e,
            ytr,
        )

        score_log_ext = (
            logit_ext
            .predict_proba(Xte_e)[:, 1]
        )

        auc_log_ext = auc_safe(
            yte,
            score_log_ext,
        )

        # ---------------------------------------------------------------------
        # XGBoost extended
        # ---------------------------------------------------------------------

        print(
            "  3/3 XGBoost extended...",
            flush=True,
        )

        xgb = fit_xgb_classifier(
            Xtr_e,
            ytr,
            Xv_e,
            yv,
        )

        score_xgb = (
            xgb
            .predict_proba(Xte_e)[:, 1]
        )

        auc_xgb = auc_safe(
            yte,
            score_xgb,
        )

        # ---------------------------------------------------------------------
        # Save model result
        # ---------------------------------------------------------------------

        results.append({
            "horizon_days": h,
            "target": target_name,
            "target_type": target_type,
            "n_train": int(train.sum()),
            "n_val": int(val.sum()),
            "n_test": int(test.sum()),
            "positive_rate_test": float(yte.mean()),
            "auc_logistic_baseline": auc_base,
            "auc_logistic_extended": auc_log_ext,
            "auc_xgb_extended": auc_xgb,
            "delta_logistic_extended_vs_baseline":
                auc_log_ext - auc_base,
            "delta_xgb_vs_baseline":
                auc_xgb - auc_base,
            "delta_xgb_vs_logistic_extended":
                auc_xgb - auc_log_ext,
        })

        # ---------------------------------------------------------------------
        # Save prediction rows
        # ---------------------------------------------------------------------

        pred = pd.DataFrame({
            "ticker":
                qrf.loc[test, "ticker"].to_numpy(),
            "forecast_date":
                qrf.loc[test, "forecast_date"].to_numpy(),
            "horizon_days":
                h,
            "target":
                target_name,
            "y_true":
                yte.to_numpy(),
            "score_logistic_baseline":
                score_base,
            "score_logistic_extended":
                score_log_ext,
            "score_xgb_extended":
                score_xgb,
        })

        predictions.append(pred)

        print(
            f"    Baseline AUC:       {auc_base:.4f}",
            flush=True,
        )

        print(
            f"    Logistic extended:  {auc_log_ext:.4f}",
            flush=True,
        )

        print(
            f"    XGB extended:       {auc_xgb:.4f}",
            flush=True,
        )

        print(
            f"    XGB delta:          "
            f"{auc_xgb - auc_base:+.4f}",
            flush=True,
        )

        # ---------------------------------------------------------------------
        # Continuous target regression
        #
        # Only for the original normalized pinball-loss target.
        # ---------------------------------------------------------------------

        if target_type == "original":

            cont_col = (
                f"future_qrf_loss_{h}d"
            )

            cont_valid = (
                qrf[cont_col].notna()
                &
                valid
            )

            tr_c = train_mask & cont_valid
            va_c = val_mask & cont_valid
            te_c = test_mask & cont_valid

            if (
                tr_c.sum() > 100
                and va_c.sum() > 50
                and te_c.sum() > 50
            ):

                print(
                    "    Continuous-loss XGB regression...",
                    flush=True,
                )

                reg = fit_xgb_regressor(
                    X_ext.loc[tr_c],
                    qrf.loc[
                        tr_c,
                        cont_col,
                    ].astype(float),
                    X_ext.loc[va_c],
                    qrf.loc[
                        va_c,
                        cont_col,
                    ].astype(float),
                )

                score_cont = reg.predict(
                    X_ext.loc[te_c]
                )

                auc_cont = auc_safe(
                    qrf.loc[
                        te_c,
                        y_col,
                    ].astype(int),
                    score_cont,
                )

                mse_cont = mean_squared_error(
                    qrf.loc[
                        te_c,
                        cont_col,
                    ].astype(float),
                    score_cont,
                )

                results[-1][
                    "auc_continuous_target_rank"
                ] = auc_cont

                results[-1][
                    "continuous_target_mse"
                ] = mse_cont

                print(
                    f"    Continuous-loss AUC: "
                    f"{auc_cont:.4f}",
                    flush=True,
                )

            else:

                results[-1][
                    "auc_continuous_target_rank"
                ] = np.nan

                results[-1][
                    "continuous_target_mse"
                ] = np.nan

        # ---------------------------------------------------------------------
        # Primary bootstrap
        #
        # We compare XGB extended against the simple baseline.
        # Also compare logistic extended against baseline.
        # ---------------------------------------------------------------------

        if target_type == "original":

            test_frame = (
                qrf.loc[
                    test,
                    [
                        "forecast_date",
                    ],
                ]
                .reset_index(drop=True)
            )

            print(
                "    Date-block bootstrap "
                f"({BOOTSTRAP_REPS} reps)...",
                flush=True,
            )

            mean_d_xgb, lo_xgb, hi_xgb = (
                block_bootstrap_auc_delta(
                    prediction_df=pd.DataFrame({
                        "forecast_date":
                            test_frame["forecast_date"],
                        "target":
                            yte.to_numpy(),
                        "baseline_prediction":
                            score_base,
                        "augmented_prediction":
                            score_xgb,
                    }),
                    baseline_column=
                        "baseline_prediction",
                    augmented_column=
                        "augmented_prediction",
                    reps=BOOTSTRAP_REPS,
                    block_length=
                        BOOTSTRAP_BLOCK_DATES,
                    seed=RANDOM_STATE + h,
                )
            )

            bootstrap_results.append({
                "horizon_days": h,
                "comparison":
                    "XGB_extended_minus_logistic_baseline",
                "observed_delta":
                    auc_xgb - auc_base,
                "bootstrap_mean_delta":
                    mean_d_xgb,
                "ci_lower":
                    lo_xgb,
                "ci_upper":
                    hi_xgb,
                "robust_positive":
                    bool(
                        np.isfinite(lo_xgb)
                        and lo_xgb > 0
                    ),
                "block_dates":
                    BOOTSTRAP_BLOCK_DATES,
                "bootstrap_reps":
                    BOOTSTRAP_REPS,
            })

            mean_d_log, lo_log, hi_log = (
                block_bootstrap_auc_delta(
                    prediction_df=pd.DataFrame({
                        "forecast_date":
                            test_frame["forecast_date"],
                        "target":
                            yte.to_numpy(),
                        "baseline_prediction":
                            score_base,
                        "augmented_prediction":
                            score_log_ext,
                    }),
                    baseline_column=
                        "baseline_prediction",
                    augmented_column=
                        "augmented_prediction",
                    reps=BOOTSTRAP_REPS,
                    block_length=
                        BOOTSTRAP_BLOCK_DATES,
                    seed=RANDOM_STATE + h + 100,
                )
            )

            bootstrap_results.append({
                "horizon_days": h,
                "comparison":
                    "logistic_extended_minus_logistic_baseline",
                "observed_delta":
                    auc_log_ext - auc_base,
                "bootstrap_mean_delta":
                    mean_d_log,
                "ci_lower":
                    lo_log,
                "ci_upper":
                    hi_log,
                "robust_positive":
                    bool(
                        np.isfinite(lo_log)
                        and lo_log > 0
                    ),
                "block_dates":
                    BOOTSTRAP_BLOCK_DATES,
                "bootstrap_reps":
                    BOOTSTRAP_REPS,
            })

            print(
                "    XGB bootstrap CI: "
                f"[{lo_xgb:+.4f}, {hi_xgb:+.4f}]",
                flush=True,
            )

            print(
                "    Logistic bootstrap CI: "
                f"[{lo_log:+.4f}, {hi_log:+.4f}]",
                flush=True,
            )


# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(results)

predictions_df = pd.concat(
    predictions,
    ignore_index=True,
) if predictions else pd.DataFrame()

bootstrap_df = pd.DataFrame(
    bootstrap_results
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
)

predictions_df.to_csv(
    PRED_PATH,
    index=False,
)

bootstrap_df.to_csv(
    BOOT_PATH,
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
    "EXPERIMENT 37 — FINAL SUMMARY",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

if results_df.empty:

    print(
        "No results were generated.",
        flush=True,
    )

else:

    summary_cols = [
        "horizon_days",
        "target",
        "auc_logistic_baseline",
        "auc_logistic_extended",
        "auc_xgb_extended",
        "auc_continuous_target_rank",
        "delta_xgb_vs_baseline",
    ]

    available_cols = [
        c
        for c in summary_cols
        if c in results_df.columns
    ]

    print(
        results_df[
            available_cols
        ].to_string(index=False),
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

if bootstrap_df.empty:

    print(
        "No bootstrap results.",
        flush=True,
    )

else:

    print(
        bootstrap_df.to_string(index=False),
        flush=True,
    )


print(
    "\n" + "-" * 100,
    flush=True,
)

print(
    "FILES",
    flush=True,
)

print(
    "-" * 100,
    flush=True,
)

print(
    f"Results:       {RESULTS_PATH}",
    flush=True,
)

print(
    f"Predictions:   {PRED_PATH}",
    flush=True,
)

print(
    f"Bootstrap:     {BOOT_PATH}",
    flush=True,
)

print(
    f"Relative:      {REL_TARGET_PATH}",
    flush=True,
)

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "EXPERIMENT 37 COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)
