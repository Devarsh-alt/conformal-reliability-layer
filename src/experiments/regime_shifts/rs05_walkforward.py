r"""
RS05a — WALK-FORWARD RELIABILITY MONITOR
========================================

Research role
-------------
RS05a estimates, strictly out of sample:

    P(VaR model deterioration over the next H days | information available at t)

It is model-specific: HS, EWMA, GARCH-t and QRF are fitted separately.

This script is intentionally NOT the final current-trust engine.
RS05a produces calibrated future-deterioration probabilities and provisional
alert/trust bands. A later RS05b stage should combine these outputs with
current/recent model state to construct the final time-varying trust signal.

Primary specification
---------------------
H = 10 days
Targets:
    T1 = deterioration_10d
    T2 = coverage_failure_10d

Models:
    Logistic Regression
    XGBoost

Main model-specific runs:
    S0 = trailing base-rate reference
    S1 = deterministic persistence rule
    S2 = persistence only
    S3 = market + own volatility
    S4 = persistence + market + own volatility
    S5 = S4 + VaR/model state
    S6 = S5 + cross-model disagreement (FULL)
    S7 = model state + disagreement

Causal safeguards
-----------------
- date-based quarterly test blocks
- expanding training
- 252-date calibration block
- purge by the SAME H-day label_end_date column
- no random train/test splitting
- all tickers of a date remain on the same side of a split
- preprocessing fitted on training only
- logistic C selected by inner forward-chaining within training
- calibration fitted on the held-out calibration block only
- alert thresholds fixed from calibration only
- no future/label/outcome/episode/calendar columns used as features

Input
-----
D:\reliability_layer\data\monitoring\regime_shifts\RS04b_causal_reliability_panel.csv

The RS04b panel already contains the validated own-volatility features.
This script DOES NOT rebuild them.

Output
------
D:\reliability_layer\data\monitoring\regime_shifts\RS05\

Examples
--------
Smoke test:
python rs05_walkforward.py --targets T1 --models HS --algos logit --specs S4 S6 --test-start 2020Q1 --test-end 2020Q2 --tag smoke

Primary H=10 run, logistic only:
python rs05_walkforward.py --horizon 10 --targets T1 T2 --algos logit --seeds 3

Primary H=10 run, Logistic + XGBoost:
python rs05_walkforward.py --horizon 10 --targets T1 T2 --algos logit xgb --seeds 3

Full robustness horizons:
python rs05_walkforward.py --horizon 5 --targets T1 T2 --algos logit xgb --seeds 3
python rs05_walkforward.py --horizon 20 --targets T1 T2 --algos logit xgb --seeds 3

Negative-control leakage test:
python rs05_walkforward.py --horizon 10 --targets T1 --algos logit --specs S4 S6 --negctrl --tag negctrl
"""

from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
except ImportError:
    xgb = None


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(r"D:\reliability_layer")

RS_DIR = (
    BASE
    / "data"
    / "monitoring"
    / "regime_shifts"
)

RS04B_PATH = RS_DIR / "RS04b_causal_reliability_panel.csv"
RS04_MANIFEST_PATH = RS_DIR / "RS04_column_manifest.json"
RS04B_MANIFEST_PATH = RS_DIR / "RS04b_feature_manifest.json"

OUT_DIR = RS_DIR / "RS05"

DATE_COL = "forecast_date"
TICKER_COL = "ticker"
MODEL_COL = "model"

MODEL_CANDIDATES = [
    "model_name",
    "model",
    "var_model",
]

EXPECTED_MODELS = [
    "HS",
    "EWMA",
    "GARCH-t",
    "QRF",
]

REGIME_LEVELS = [
    "normal",
    "transition",
    "stress",
    "recovery",
]

HEADLINE_START = pd.Timestamp("2020-01-01")

CAL_LEN = 252
MIN_FIT_DATES = 300

BUDGETS = [0.05, 0.10, 0.20]

TARGET_PREFIX = {
    "T1": "deterioration_{H}d",
    "T2": "coverage_failure_{H}d",
}

SPECS = {
    "S2": ["P"],
    "S3": ["M", "V"],
    "S4": ["P", "M", "V"],
    "S5": ["P", "M", "V", "S"],
    "S6": ["P", "M", "V", "S", "D"],
    "S7": ["S", "D"],
}

# Conservative tabular XGBoost specification.
# min_child_weight is intentionally large because market-level variables
# repeat across tickers on the same date.
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=2,
    learning_rate=0.04,
    min_child_weight=500,
    subsample=0.70,
    colsample_bytree=0.70,
    reg_lambda=10.0,
    reg_alpha=0.0,
    tree_method="hist",
    eval_metric="logloss",
    n_jobs=-1,
    verbosity=0,
)

# Strict feature exclusion.
NOT_FEATURE = re.compile(
    r"(^future_|loss_excess|label_|outcome_|^deterioration_|"
    r"^coverage_failure_|episode|duration|^year$|^month$|^day|"
    r"time_idx|^index$|^Unnamed)",
    re.I,
)


# =============================================================================
# LOGGING
# =============================================================================

LOG_LINES: list[str] = []


def emit(message: str = "") -> None:
    print(message, flush=True)
    LOG_LINES.append(str(message))


def save_log(path: Path) -> None:
    path.write_text(
        "\n".join(LOG_LINES) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# HELPERS
# =============================================================================

def detect_model_col(df: pd.DataFrame) -> str:
    for c in MODEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(
        "No model column found. "
        f"Tried {MODEL_CANDIDATES}. "
        f"Columns: {list(df.columns)}"
    )


def find_label_end_col(
    df: pd.DataFrame,
    horizon: int,
) -> str:
    exact = f"label_end_date_{horizon}d"
    if exact in df.columns:
        return exact

    candidates = [
        c
        for c in df.columns
        if (
            "label_end_date" in str(c).lower()
            and f"{horizon}d" in str(c).lower()
        )
    ]

    if not candidates:
        raise KeyError(
            f"Could not find label_end_date column for H={horizon}. "
            f"Available label columns: "
            f"{[c for c in df.columns if 'label_' in str(c).lower()]}"
        )

    return candidates[0]


def normalize_target_target_column(
    df: pd.DataFrame,
    target: str,
    H: int,
) -> str:
    col = TARGET_PREFIX[target].format(H=H)

    if col in df.columns:
        return col

    raise KeyError(
        f"Missing target column {col}."
    )


def add_persistence_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add causal persistence ratios:

        historical_Hd_mean_norm_loss /
        causal_Hd_loss_threshold

    All source columns already exist in the validated RS04 panel.
    """
    out = df.copy()

    for H in (5, 10, 20):
        a = f"historical_{H}d_mean_norm_loss"
        b = f"causal_{H}d_loss_threshold"

        if a in out.columns and b in out.columns:
            denom = pd.to_numeric(
                out[b],
                errors="coerce",
            ).replace(0, np.nan)

            out[f"hist_ratio_{H}d"] = (
                pd.to_numeric(
                    out[a],
                    errors="coerce",
                )
                / denom
            )

    return out


# =============================================================================
# FEATURE GROUPING
# =============================================================================

def group_of(column: str) -> str | None:
    lc = column.lower()

    # Own-volatility first because "own_rv10" also contains "rv".
    if column.startswith("own_"):
        return "V"

    if re.match(
        r"(past_|historical_|causal_|hist_ratio_)",
        column,
        re.I,
    ):
        return "P"

    if column.startswith("current_"):
        return "S"

    if re.match(
        r"(cross_model_|model_abs_var_)",
        column,
        re.I,
    ):
        return "D"

    if (
        "vix" in lc
        or re.search(r"(^|_)rv", lc)
        or "stress_dimensions" in lc
        or lc == "regime"
    ):
        return "M"

    return None


def build_feature_groups(
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[str]]:
    groups = {
        "P": [],
        "M": [],
        "V": [],
        "S": [],
        "D": [],
    }

    unassigned = []

    skip = {
        DATE_COL,
        "realized_date",
        TICKER_COL,
        MODEL_COL,
    }

    for c in df.columns:
        if c in skip:
            continue

        if NOT_FEATURE.search(str(c)):
            continue

        group = group_of(str(c))

        if group is None:
            unassigned.append(c)
            continue

        if c != "regime" and not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            unassigned.append(c)
            continue

        groups[group].append(c)

    return groups, unassigned


# =============================================================================
# DATA LOADING
# =============================================================================

def load_panel() -> pd.DataFrame:
    if not RS04B_PATH.exists():
        raise FileNotFoundError(
            f"RS04b panel not found:\n{RS04B_PATH}"
        )

    df = pd.read_csv(
        RS04B_PATH,
        low_memory=False,
    )

    required = [
        TICKER_COL,
        DATE_COL,
        "model_name",
        "regime",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"RS04b missing required columns: {missing}"
        )

    model_col = detect_model_col(df)

    df = df.rename(
        columns={
            model_col: MODEL_COL,
        }
    )

    df[TICKER_COL] = (
        df[TICKER_COL]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df[MODEL_COL] = (
        df[MODEL_COL]
        .astype(str)
        .str.strip()
    )

    df["regime"] = (
        df["regime"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for c in df.columns:
        if (
            "label_end_date" in str(c).lower()
            or "label_start_date" in str(c).lower()
        ):
            df[c] = pd.to_datetime(
                df[c],
                errors="coerce",
            )

    if df[DATE_COL].isna().any():
        raise ValueError(
            "RS04b contains invalid forecast_date values."
        )

    # Keep only the intended 24-stock universe and 4 VaR models.
    unexpected_tickers = sorted(
        set(df[TICKER_COL].unique())
        - set([
            "AAPL", "AMZN", "BA", "BAC", "CAT", "CVX",
            "DIS", "GME", "GOOGL", "GS", "JNJ", "JPM",
            "KO", "MCD", "MSFT", "NFLX", "NVDA", "PFE",
            "PG", "TSLA", "UNH", "V", "WMT", "XOM",
        ])
    )

    if unexpected_tickers:
        emit(
            f"WARNING: unexpected tickers present; they will be ignored: "
            f"{unexpected_tickers}"
        )

    df = df[
        df[TICKER_COL].isin([
            "AAPL", "AMZN", "BA", "BAC", "CAT", "CVX",
            "DIS", "GME", "GOOGL", "GS", "JNJ", "JPM",
            "KO", "MCD", "MSFT", "NFLX", "NVDA", "PFE",
            "PG", "TSLA", "UNH", "V", "WMT", "XOM",
        ])
    ].copy()

    unexpected_models = sorted(
        set(df[MODEL_COL].unique())
        - set(EXPECTED_MODELS)
    )

    if unexpected_models:
        emit(
            f"WARNING: unexpected model labels ignored: "
            f"{unexpected_models}"
        )

    df = df[
        df[MODEL_COL].isin(EXPECTED_MODELS)
    ].copy()

    df = (
        df.sort_values(
            [MODEL_COL, TICKER_COL, DATE_COL]
        )
        .reset_index(drop=True)
    )

    duplicates = int(
        df.duplicated(
            [TICKER_COL, DATE_COL, MODEL_COL]
        ).sum()
    )

    if duplicates:
        raise ValueError(
            f"RS04b has {duplicates} duplicate "
            "ticker/date/model rows."
        )

    if not any(
        c.startswith("own_")
        for c in df.columns
    ):
        raise AssertionError(
            "RS04b has no own_* features. "
            "Point RS04B_PATH to the validated RS04b panel."
        )

    df = add_persistence_ratios(df)

    emit(f"Loaded RS04b: {RS04B_PATH}")
    emit(f"Shape: {df.shape}")
    emit(f"Tickers: {df[TICKER_COL].nunique()}")
    emit(
        f"Models: {sorted(df[MODEL_COL].unique().tolist())}"
    )
    emit(
        f"Forecast range: {df[DATE_COL].min().date()} "
        f"to {df[DATE_COL].max().date()}"
    )

    return df


# =============================================================================
# PREPROCESSOR
# =============================================================================

class Prep:
    """
    Numeric features are:
        - winsorized using F_k only
        - median-imputed using F_k only
        - standardized using F_k only

    Regime is one-hot encoded using a fixed, pre-specified level set.
    """

    def __init__(
        self,
        columns: list[str],
    ):
        self.columns = list(columns)
        self.numeric_columns = [
            c
            for c in self.columns
            if c != "regime"
        ]
        self.use_regime = "regime" in self.columns

        self.regime_levels = [
            x.lower()
            for x in REGIME_LEVELS
        ]

        self.lo = None
        self.hi = None
        self.median = None
        self.scaler = None

    @staticmethod
    def _numeric_frame(
        data: pd.DataFrame,
        columns: list[str],
    ) -> pd.DataFrame:
        if not columns:
            return pd.DataFrame(
                index=data.index
            )

        X = data[columns].copy()

        for c in columns:
            X[c] = pd.to_numeric(
                X[c],
                errors="coerce",
            )

        return X.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    def fit(
        self,
        F: pd.DataFrame,
    ) -> "Prep":
        X = self._numeric_frame(
            F,
            self.numeric_columns,
        )

        if self.numeric_columns:
            self.lo = X.quantile(
                0.01
            )
            self.hi = X.quantile(
                0.99
            )

            X = X.clip(
                self.lo,
                self.hi,
                axis=1,
            )

            self.median = (
                X.median()
                .fillna(0.0)
            )

            Xf = X.fillna(
                self.median
            )

            self.scaler = StandardScaler()
            self.scaler.fit(
                Xf.to_numpy()
            )

        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        parts = []

        if self.numeric_columns:
            X = self._numeric_frame(
                data,
                self.numeric_columns,
            )

            X = X.clip(
                self.lo,
                self.hi,
                axis=1,
            ).fillna(
                self.median
            )

            Z = self.scaler.transform(
                X.to_numpy()
            )

            parts.append(Z)

        if self.use_regime:
            regime_values = (
                data["regime"]
                .astype(str)
                .str.lower()
                .fillna("")
                .values
            )

            R = np.column_stack(
                [
                    (
                        regime_values == level
                    ).astype(float)
                    for level in self.regime_levels
                ]
            )

            parts.append(R)

        if not parts:
            raise ValueError(
                "No usable features were provided."
            )

        return np.column_stack(parts)


# =============================================================================
# LOGISTIC C SELECTION
# =============================================================================

def choose_logistic_C(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    ends: np.ndarray,
    grid=(0.003, 0.03, 0.3, 3.0),
) -> float:
    """
    Inner forward-chaining validation inside the training block.

    Training rows must have label_end_date < validation_start.
    """

    unique_dates = np.unique(
        dates
    )

    n_dates = len(unique_dates)

    if n_dates < 200:
        return 0.03

    step = max(
        int(n_dates * 0.16),
        20,
    )

    scores = {
        float(C): []
        for C in grid
    }

    # These validation anchors remain wholly inside F_k.
    for fraction in (
        0.50,
        0.67,
        0.84,
    ):
        v0 = unique_dates[
            min(
                int(n_dates * fraction),
                n_dates - 1,
            )
        ]

        v1 = unique_dates[
            min(
                int(
                    int(n_dates * fraction)
                    + step
                ),
                n_dates - 1,
            )
        ]

        train_mask = (
            ends < v0
        )

        val_mask = (
            (dates >= v0)
            & (dates < v1)
        )

        if (
            train_mask.sum() < 1000
            or val_mask.sum() < 200
            or y[train_mask].sum() < 20
            or y[val_mask].sum() < 5
            or (
                y[train_mask].sum()
                == train_mask.sum()
            )
        ):
            continue

        for C in scores:
            model = LogisticRegression(
                C=C,
                max_iter=1000,
                solver="lbfgs",
            )

            try:
                model.fit(
                    X[train_mask],
                    y[train_mask],
                )
            except Exception:
                continue

            p = np.clip(
                model.predict_proba(
                    X[val_mask]
                )[:, 1],
                1e-6,
                1 - 1e-6,
            )

            scores[C].append(
                log_loss(
                    y[val_mask],
                    p,
                    labels=[0, 1],
                )
            )

    valid = {
        C: float(np.mean(losses))
        for C, losses in scores.items()
        if losses
    }

    if not valid:
        return 0.03

    return min(
        valid,
        key=valid.get,
    )


# =============================================================================
# CALIBRATION
# =============================================================================

def clip_probability(
    p: np.ndarray,
) -> np.ndarray:
    return np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )


def logit_score(
    p: np.ndarray,
) -> np.ndarray:
    p = clip_probability(p)
    return np.log(
        p / (1.0 - p)
    )


def fit_platt(
    calibration_score: np.ndarray,
    calibration_y: np.ndarray,
    test_score: np.ndarray,
    score_type: str = "logit",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit Platt scaling on calibration data only.

    Returns:
        calibrated_test_probability,
        calibrated_calibration_probability
    """

    y = np.asarray(
        calibration_y,
        dtype=int,
    )

    if (
        len(y) == 0
        or y.min() == y.max()
        or y.sum() < 5
        or (len(y) - y.sum()) < 5
    ):
        base = float(
            y.mean()
        ) if len(y) else 0.5

        return (
            np.full(
                len(test_score),
                np.clip(base, 1e-6, 1 - 1e-6),
            ),
            np.full(
                len(calibration_score),
                np.clip(base, 1e-6, 1 - 1e-6),
            ),
        )

    calibration_score = np.asarray(
        calibration_score,
        dtype=float,
    )
    test_score = np.asarray(
        test_score,
        dtype=float,
    )

    if score_type == "probability":
        # Probability input is deliberately converted to log-odds before
        # Platt scaling so a probability is not treated as a linear predictor.
        X_cal = logit_score(
            calibration_score
        ).reshape(-1, 1)

        X_test = logit_score(
            test_score
        ).reshape(-1, 1)

    elif score_type == "raw":
        X_cal = calibration_score.reshape(
            -1,
            1,
        )
        X_test = test_score.reshape(
            -1,
            1,
        )

    else:
        raise ValueError(
            f"Unknown score_type: {score_type}"
        )

    # Very weak regularization because calibration itself is held out.
    model = LogisticRegression(
        C=1e6,
        max_iter=1000,
        solver="lbfgs",
    )

    try:
        model.fit(
            X_cal,
            y,
        )
    except Exception:
        base = float(
            y.mean()
        )
        return (
            np.full(
                len(test_score),
                np.clip(base, 1e-6, 1 - 1e-6),
            ),
            np.full(
                len(calibration_score),
                np.clip(base, 1e-6, 1 - 1e-6),
            ),
        )

    p_test = clip_probability(
        model.predict_proba(
            X_test
        )[:, 1]
    )

    p_cal = clip_probability(
        model.predict_proba(
            X_cal
        )[:, 1]
    )

    return p_test, p_cal


# =============================================================================
# MODEL FIT / PREDICT
# =============================================================================

def fit_predict(
    algo: str,
    columns: list[str],
    F: pd.DataFrame,
    Cb: pd.DataFrame,
    T: pd.DataFrame,
    ycol: str,
    endcol: str,
    seeds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit a model on F_k, produce raw probabilities on calibration/test.

    Preprocessing is fit ONLY on F_k.
    """

    prep = Prep(
        columns
    ).fit(F)

    XF = prep.transform(F)
    XC = prep.transform(Cb)
    XT = prep.transform(T)

    yF = (
        pd.to_numeric(
            F[ycol],
            errors="coerce",
        )
        .astype(int)
        .to_numpy()
    )

    if (
        yF.min() == yF.max()
    ):
        base = float(
            yF.mean()
        )

        return (
            np.full(
                len(Cb),
                base,
            ),
            np.full(
                len(T),
                base,
            ),
        )

    if algo == "logit":
        C = choose_logistic_C(
            XF,
            yF,
            F[DATE_COL].to_numpy(),
            pd.to_datetime(
                F[endcol]
            ).to_numpy(),
        )

        model = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
        )

        model.fit(
            XF,
            yF,
        )

        pC = model.predict_proba(
            XC
        )[:, 1]

        pT = model.predict_proba(
            XT
        )[:, 1]

        return (
            clip_probability(pC),
            clip_probability(pT),
        )

    if algo == "xgb":
        if xgb is None:
            raise RuntimeError(
                "xgboost is not installed. "
                "Use --algos logit or install xgboost."
            )

        pC = np.zeros(
            len(Cb),
            dtype=float,
        )
        pT = np.zeros(
            len(T),
            dtype=float,
        )

        # Average multiple seeds for stability.
        for seed in range(seeds):
            model = xgb.XGBClassifier(
                random_state=42 + seed,
                **XGB_PARAMS,
            )

            model.fit(
                XF,
                yF,
            )

            pC += (
                model.predict_proba(
                    XC
                )[:, 1]
                / seeds
            )

            pT += (
                model.predict_proba(
                    XT
                )[:, 1]
                / seeds
            )

        return (
            clip_probability(pC),
            clip_probability(pT),
        )

    raise ValueError(
        f"Unknown algorithm: {algo}"
    )


# =============================================================================
# S1 — DETERMINISTIC PERSISTENCE
# =============================================================================

def s1_binary_score(
    data: pd.DataFrame,
    target: str,
    H: int,
) -> np.ndarray:
    """
    Strictly causal deterministic persistence baseline.

    T1:
        1 when historical mean normalized loss > causal threshold.

    T2:
        1 when the prior H-day breach rate reaches the critical
        breach-count threshold k/H.

    For H=5,10,20 the RS04 coverage targets use:
        H=5  -> k=2
        H=10 -> k=3
        H=20 -> k=4
    """

    if target == "T1":
        a = f"historical_{H}d_mean_norm_loss"
        b = f"causal_{H}d_loss_threshold"

        if a not in data.columns or b not in data.columns:
            raise KeyError(
                f"Missing S1 T1 columns: {a}, {b}"
            )

        left = pd.to_numeric(
            data[a],
            errors="coerce",
        )

        right = pd.to_numeric(
            data[b],
            errors="coerce",
        )

        return (
            (left > right)
            .astype(float)
            .to_numpy()
        )

    if target == "T2":
        breach_col = f"past_{H}d_breach_rate"

        if breach_col not in data.columns:
            raise KeyError(
                f"Missing S1 T2 column: {breach_col}"
            )

        critical_k = {
            5: 2,
            10: 3,
            20: 4,
        }[H]

        prior_rate = pd.to_numeric(
            data[breach_col],
            errors="coerce",
        )

        return (
            prior_rate
            >= critical_k / H
        ).astype(float).to_numpy()

    raise ValueError(
        f"Unknown target: {target}"
    )


# =============================================================================
# WALK-FORWARD FOLDS
# =============================================================================

def make_folds(
    date_end: pd.Series,
    test_start: str,
    test_end: str | None,
) -> list[dict]:
    """
    Build quarterly date-based test blocks.

    For each test block:
        calibration dates are the latest CAL_LEN dates whose label_end_date
        is strictly before the first test date.

        training dates are all earlier dates whose label_end_date is strictly
        before the beginning of calibration.

    This performs horizon-specific purging.
    """

    idx = (
        pd.DatetimeIndex(
            date_end.index
        )
        .sort_values()
    )

    quarter_index = pd.PeriodIndex(
        idx,
        freq="Q",
    )

    start_period = pd.Period(
        test_start,
        freq="Q",
    )

    end_period = (
        pd.Period(
            test_end,
            freq="Q",
        )
        if test_end
        else quarter_index.max()
    )

    folds = []

    for period in pd.period_range(
        start_period,
        end_period,
        freq="Q",
    ):
        test_dates = idx[
            quarter_index == period
        ]

        if len(test_dates) < 10:
            continue

        test_first = test_dates.min()

        eligible_before_test = idx[
            date_end.reindex(idx).to_numpy()
            < test_first
        ]

        if len(eligible_before_test) < (
            CAL_LEN + MIN_FIT_DATES
        ):
            continue

        cal_dates = (
            eligible_before_test[
                -CAL_LEN:
            ]
        )

        cal_first = cal_dates.min()

        fit_dates = idx[
            date_end.reindex(idx).to_numpy()
            < cal_first
        ]

        if len(fit_dates) < MIN_FIT_DATES:
            continue

        folds.append(
            {
                "q": str(period),
                "fit": fit_dates,
                "cal": cal_dates,
                "test": test_dates,
            }
        )

    return folds


# =============================================================================
# STORE PREDICTIONS
# =============================================================================

def make_alert_thresholds(
    calibration_prob: np.ndarray,
) -> dict[float, float]:
    p = clip_probability(
        calibration_prob
    )

    return {
        budget: float(
            np.quantile(
                p,
                1.0 - budget,
            )
        )
        for budget in BUDGETS
    }


def store_predictions(
    output_parts: list[pd.DataFrame],
    test_df: pd.DataFrame,
    model_name: str,
    target: str,
    H: int,
    spec: str,
    algo: str,
    fold: dict,
    p_test_calibrated: np.ndarray,
    p_calibration_calibrated: np.ndarray,
) -> None:

    thresholds = make_alert_thresholds(
        p_calibration_calibrated
    )

    output_parts.append(
        pd.DataFrame(
            {
                DATE_COL: test_df[DATE_COL].to_numpy(),
                TICKER_COL: test_df[TICKER_COL].to_numpy(),
                "model": model_name,
                "target": target,
                "H": H,
                "spec": spec,
                "algo": algo,
                "fold": fold["q"],
                "y": pd.to_numeric(
                    test_df["_y"],
                    errors="coerce",
                )
                .astype(np.int8)
                .to_numpy(),
                "p_cal": clip_probability(
                    p_test_calibrated
                ).astype(np.float32),
                "thr05": thresholds[0.05],
                "thr10": thresholds[0.10],
                "thr20": thresholds[0.20],
            }
        )
    )


# =============================================================================
# METRICS
# =============================================================================

def calibration_fit(
    p: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float]:
    p = clip_probability(
        p
    )

    if (
        len(y) < 20
        or np.min(y) == np.max(y)
    ):
        return np.nan, np.nan

    X = logit_score(
        p
    ).reshape(
        -1,
        1,
    )

    try:
        model = LogisticRegression(
            C=1e6,
            max_iter=1000,
            solver="lbfgs",
        )

        model.fit(
            X,
            y,
        )

        return (
            float(
                model.coef_[0, 0]
            ),
            float(
                model.intercept_[0]
            ),
        )

    except Exception:
        return np.nan, np.nan


def group_metrics(
    g: pd.DataFrame,
) -> pd.Series:

    y = (
        pd.to_numeric(
            g["y"],
            errors="coerce",
        )
        .astype(int)
        .to_numpy()
    )

    p = clip_probability(
        g["p_cal"]
        .astype(float)
        .to_numpy()
    )

    result = {
        "n_rows": len(g),
        "n_dates": g[DATE_COL].nunique(),
        "prevalence": float(y.mean()),
        "brier": float(
            brier_score_loss(
                y,
                p,
            )
        ),
        "logloss": float(
            log_loss(
                y,
                p,
                labels=[0, 1],
            )
        ),
    }

    if (
        len(np.unique(y)) > 1
        and np.ptp(p) > 0
    ):
        result["auc"] = float(
            roc_auc_score(
                y,
                p,
            )
        )

        result["pr_auc"] = float(
            average_precision_score(
                y,
                p,
            )
        )

        result["pr_lift"] = float(
            result["pr_auc"]
            / max(
                result["prevalence"],
                1e-12,
            )
        )
    else:
        result["auc"] = np.nan
        result["pr_auc"] = np.nan
        result["pr_lift"] = np.nan

    slope, intercept = calibration_fit(
        p,
        y,
    )

    result["cal_slope"] = slope
    result["cal_intercept"] = intercept

    # Date-aggregated relationship:
    # mean predicted probability vs fraction of tickers that deteriorate.
    date_level = (
        g.groupby(DATE_COL)
        .agg(
            p=("p_cal", "mean"),
            f=("y", "mean"),
        )
        .reset_index()
    )

    if (
        len(date_level) >= 10
        and date_level["p"].nunique() > 2
        and date_level["f"].nunique() > 2
    ):
        result["date_level_spearman"] = float(
            spearmanr(
                date_level["p"],
                date_level["f"],
            ).statistic
        )
    else:
        result["date_level_spearman"] = np.nan

    return pd.Series(
        result
    )


def evaluate_predictions(
    predictions: pd.DataFrame,
    tag: str,
) -> None:
    """
    Evaluate headline period beginning 2020-01-01.

    Saves:
      - pooled metrics
      - date-level metrics
      - alert-budget metrics
      - provisional trust-band table
      - daily monitoring series
      - fold-level metrics
    """

    P = predictions.copy()

    P[DATE_COL] = pd.to_datetime(
        P[DATE_COL]
    )

    headline = P[
        P[DATE_COL]
        >= HEADLINE_START
    ].copy()

    if headline.empty:
        raise ValueError(
            "No predictions in headline period."
        )

    keys = [
        "target",
        "model",
        "algo",
        "spec",
    ]

    # -------------------------------------------------------------------------
    # Pooled metrics
    # -------------------------------------------------------------------------
    M = (
        headline
        .groupby(keys)
        .apply(
            group_metrics,
            include_groups=False,
        )
        .reset_index()
    )

    # Compare all methods to S0 within the same target/model.
    references = (
        M[
            M["spec"] == "S0"
        ]
        .set_index(
            [
                "target",
                "model",
            ]
        )[
            [
                "brier",
                "logloss",
            ]
        ]
        .rename(
            columns={
                "brier": "brier_S0",
                "logloss": "logloss_S0",
            }
        )
    )

    M = M.join(
        references,
        on=[
            "target",
            "model",
        ],
    )

    M["brier_skill"] = (
        1.0
        - M["brier"]
        / M["brier_S0"]
    )

    M["logloss_skill"] = (
        1.0
        - M["logloss"]
        / M["logloss_S0"]
    )

    metrics_path = (
        OUT_DIR
        / f"RS05_metrics_{tag}.csv"
    )

    M.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Fold metrics
    # -------------------------------------------------------------------------
    fold_rows = []

    for fold_key, g in headline.groupby(
        [
            "target",
            "model",
            "algo",
            "spec",
            "fold",
        ]
    ):
        row = group_metrics(
            g
        ).to_dict()

        (
            target,
            model,
            algo,
            spec,
            fold,
        ) = fold_key

        row.update(
            {
                "target": target,
                "model": model,
                "algo": algo,
                "spec": spec,
                "fold": fold,
            }
        )

        fold_rows.append(
            row
        )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        OUT_DIR
        / f"RS05_fold_metrics_{tag}.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Alert budgets
    # -------------------------------------------------------------------------
    non_s0 = headline[
        headline["spec"] != "S0"
    ].copy()

    alert_rows = []

    for k, g in non_s0.groupby(
        keys
    ):
        y = g["y"].astype(int).to_numpy()

        for budget, threshold_col in zip(
            BUDGETS,
            [
                "thr05",
                "thr10",
                "thr20",
            ],
        ):
            alerted = (
                g["p_cal"].astype(float).to_numpy()
                >= g[threshold_col].astype(float).to_numpy()
            )

            tp = int(
                (
                    alerted
                    & (y == 1)
                ).sum()
            )

            fp = int(
                (
                    alerted
                    & (y == 0)
                ).sum()
            )

            positives = int(
                (y == 1).sum()
            )

            negatives = int(
                (y == 0).sum()
            )

            n_alerts = int(
                alerted.sum()
            )

            alert_rows.append(
                {
                    "target": k[0],
                    "model": k[1],
                    "algo": k[2],
                    "spec": k[3],
                    "budget": budget,
                    "realized_alert_rate": (
                        n_alerts
                        / max(len(y), 1)
                    ),
                    "recall": (
                        tp
                        / max(
                            positives,
                            1,
                        )
                    ),
                    "precision": (
                        tp
                        / max(
                            n_alerts,
                            1,
                        )
                    ),
                    "false_alarm_rate": (
                        fp
                        / max(
                            negatives,
                            1,
                        )
                    ),
                }
            )

    alert_df = pd.DataFrame(
        alert_rows
    )

    alert_df.to_csv(
        OUT_DIR
        / f"RS05_alert_metrics_{tag}.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Provisional trust bands
    #
    # These are rank/alert bands, NOT the final current-trust score.
    # -------------------------------------------------------------------------
    non_s0["provisional_trust_band"] = np.where(
        non_s0["p_cal"]
        >= non_s0["thr10"],
        "LOW_TRUST_ALERT",
        np.where(
            non_s0["p_cal"]
            >= non_s0["thr20"],
            "MODERATE_TRUST_ALERT",
            "HIGH_TRUST_ALERT",
        ),
    )

    band = (
        non_s0
        .groupby(
            keys
            + [
                "provisional_trust_band",
            ]
        )
        .agg(
            n=("y", "size"),
            realized_deterioration_freq=(
                "y",
                "mean",
            ),
            mean_predicted_probability=(
                "p_cal",
                "mean",
            ),
        )
        .reset_index()
    )

    band["share"] = (
        band["n"]
        / band.groupby(keys)["n"]
        .transform("sum")
    )

    band.to_csv(
        OUT_DIR
        / f"RS05_provisional_trust_bands_{tag}.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Date-level series
    # -------------------------------------------------------------------------
    date_series = (
        non_s0
        .groupby(
            keys
            + [
                DATE_COL,
            ]
        )
        .agg(
            mean_p_cal=(
                "p_cal",
                "mean",
            ),
            frac_low_trust=(
                "provisional_trust_band",
                lambda s: float(
                    np.mean(
                        s == "LOW_TRUST_ALERT"
                    )
                ),
            ),
            frac_moderate_or_low=(
                "provisional_trust_band",
                lambda s: float(
                    np.mean(
                        s != "HIGH_TRUST_ALERT"
                    )
                ),
            ),
            frac_realized=(
                "y",
                "mean",
            ),
            n_tickers=(
                TICKER_COL,
                "nunique",
            ),
        )
        .reset_index()
    )

    date_series.to_csv(
        OUT_DIR
        / f"RS05_date_series_{tag}.csv.gz",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Headline console table
    # -------------------------------------------------------------------------
    show_cols = [
        "target",
        "model",
        "algo",
        "spec",
        "auc",
        "pr_lift",
        "brier_skill",
        "logloss_skill",
        "cal_slope",
        "cal_intercept",
        "date_level_spearman",
        "n_rows",
        "n_dates",
    ]

    emit(
        "\n"
        + "=" * 110
    )
    emit(
        f"HEADLINE METRICS — {tag}"
    )
    emit(
        f"Headline evaluation begins: "
        f"{HEADLINE_START.date()}"
    )
    emit(
        "=" * 110
    )

    available = [
        c
        for c in show_cols
        if c in M.columns
    ]

    emit(
        M[available]
        .round(4)
        .to_string(
            index=False
        )
    )

    emit(
        f"\nSaved outputs to:\n{OUT_DIR}"
    )


# =============================================================================
# MAIN RUN
# =============================================================================

def run(args: argparse.Namespace) -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_panel()

    groups, unassigned = (
        build_feature_groups(df)
    )

    emit("\n" + "=" * 110)
    emit("FEATURE GROUPS")
    emit("=" * 110)

    for group_name in (
        "P",
        "M",
        "V",
        "S",
        "D",
    ):
        emit(
            f"{group_name} ({len(groups[group_name])}): "
            f"{groups[group_name]}"
        )

    if unassigned:
        emit(
            "\nUNASSIGNED / EXCLUDED FEATURES:"
        )
        emit(
            str(unassigned)
        )
    else:
        emit(
            "\nUNASSIGNED / EXCLUDED FEATURES: []"
        )

    regime_levels = REGIME_LEVELS

    requested_models = (
        args.models
        if args.models
        else sorted(
            df[MODEL_COL].unique()
        )
    )

    invalid_models = sorted(
        set(requested_models)
        - set(EXPECTED_MODELS)
    )

    if invalid_models:
        raise ValueError(
            f"Invalid model(s): {invalid_models}. "
            f"Expected: {EXPECTED_MODELS}"
        )

    all_predictions = []

    H = args.horizon

    for target in args.targets:

        ycol = normalize_target_target_column(
            df,
            target,
            H,
        )

        endcol = find_label_end_col(
            df,
            H,
        )

        emit(
            "\n"
            + "=" * 110
        )
        emit(
            f"TARGET {target} | horizon={H} | "
            f"target={ycol} | purge={endcol}"
        )
        emit(
            "=" * 110
        )

        for model_name in requested_models:

            model_df = (
                df[
                    df[MODEL_COL]
                    == model_name
                ]
                .dropna(
                    subset=[
                        ycol,
                        endcol,
                    ]
                )
                .copy()
            )

            model_df = (
                model_df
                .sort_values(
                    [
                        DATE_COL,
                        TICKER_COL,
                    ]
                )
                .reset_index(drop=True)
            )

            if args.negctrl:
                emit(
                    f"\n{model_name}: "
                    "applying 126-day within-ticker "
                    "negative-control label shift."
                )

                model_df[ycol] = (
                    model_df
                    .groupby(
                        TICKER_COL,
                        group_keys=False,
                    )[ycol]
                    .transform(
                        lambda s:
                        np.roll(
                            s.to_numpy(),
                            126,
                        )
                    )
                )

            model_df["_y"] = (
                pd.to_numeric(
                    model_df[ycol],
                    errors="coerce",
                )
                .astype(int)
            )

            # Date-level purge unit.
            # All tickers for the same date share the same side.
            date_end = (
                model_df
                .groupby(
                    DATE_COL
                )[endcol]
                .max()
            )

            folds = make_folds(
                date_end=date_end,
                test_start=args.test_start,
                test_end=args.test_end,
            )

            emit(
                f"\n{model_name}: "
                f"{len(folds)} quarterly folds"
            )

            if not folds:
                raise RuntimeError(
                    f"No valid folds for "
                    f"{model_name}/{target}/H={H}."
                )

            for fold in folds:

                started = time.time()

                fit_dates = fold["fit"]
                cal_dates = fold["cal"]
                test_dates = fold["test"]

                F = model_df[
                    model_df[DATE_COL]
                    .isin(fit_dates)
                ].copy()

                Cb = model_df[
                    model_df[DATE_COL]
                    .isin(cal_dates)
                ].copy()

                T = model_df[
                    model_df[DATE_COL]
                    .isin(test_dates)
                ].copy()

                yC = Cb[
                    "_y"
                ].to_numpy(
                    dtype=int
                )

                # -----------------------------------------------------------------
                # S0 — trailing base-rate reference
                # -----------------------------------------------------------------
                if "S0" in args.specs:

                    base_rate = float(
                        yC.mean()
                    )

                    store_predictions(
                        all_predictions,
                        T,
                        model_name,
                        target,
                        H,
                        "S0",
                        "base",
                        fold,
                        np.full(
                            len(T),
                            base_rate,
                        ),
                        np.full(
                            len(Cb),
                            base_rate,
                        ),
                    )

                # -----------------------------------------------------------------
                # S1 — deterministic persistence
                # -----------------------------------------------------------------
                if "S1" in args.specs:

                    score_cal = (
                        s1_binary_score(
                            Cb,
                            target,
                            H,
                        )
                    )

                    score_test = (
                        s1_binary_score(
                            T,
                            target,
                            H,
                        )
                    )

                    fill = float(
                        np.nanmedian(
                            s1_binary_score(
                                F,
                                target,
                                H,
                            )
                        )
                    )

                    score_cal = np.nan_to_num(
                        score_cal,
                        nan=fill,
                        posinf=fill,
                        neginf=fill,
                    )

                    score_test = np.nan_to_num(
                        score_test,
                        nan=fill,
                        posinf=fill,
                        neginf=fill,
                    )

                    p_test, p_cal = fit_platt(
                        score_cal,
                        yC,
                        score_test,
                        score_type="raw",
                    )

                    store_predictions(
                        all_predictions,
                        T,
                        model_name,
                        target,
                        H,
                        "S1",
                        "rule",
                        fold,
                        p_test,
                        p_cal,
                    )

                # -----------------------------------------------------------------
                # S2-S7 — learned specifications
                # -----------------------------------------------------------------
                learned_specs = [
                    s
                    for s in args.specs
                    if s in SPECS
                ]

                for spec in learned_specs:

                    selected_columns = []

                    for group_name in (
                        SPECS[spec]
                    ):
                        selected_columns.extend(
                            groups[group_name]
                        )

                    selected_columns = list(
                        dict.fromkeys(
                            selected_columns
                        )
                    )

                    if not selected_columns:
                        raise RuntimeError(
                            f"{spec} has no features."
                        )

                    for algo in args.algos:

                        raw_cal, raw_test = (
                            fit_predict(
                                algo=algo,
                                columns=selected_columns,
                                F=F,
                                Cb=Cb,
                                T=T,
                                ycol=ycol,
                                endcol=endcol,
                                seeds=args.seeds,
                            )
                        )

                        p_test, p_cal = fit_platt(
                            raw_cal,
                            yC,
                            raw_test,
                            score_type="probability",
                        )

                        store_predictions(
                            all_predictions,
                            T,
                            model_name,
                            target,
                            H,
                            spec,
                            algo,
                            fold,
                            p_test,
                            p_cal,
                        )

                elapsed = (
                    time.time()
                    - started
                )

                emit(
                    f"  {model_name} {fold['q']} "
                    f"done | fit_dates={len(F):,} "
                    f"cal_dates={len(Cb[DATE_COL].unique()):,} "
                    f"test_dates={len(T[DATE_COL].unique()):,} "
                    f"test_rows={len(T):,} "
                    f"test_prev={T['_y'].mean():.4f} "
                    f"| {elapsed:.1f}s"
                )

    if not all_predictions:
        raise RuntimeError(
            "No predictions were generated."
        )

    P = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    # Stable ordering.
    P = (
        P.sort_values(
            [
                "target",
                "model",
                "algo",
                "spec",
                DATE_COL,
                TICKER_COL,
            ]
        )
        .reset_index(drop=True)
    )

    suffix = (
        f"T{H}"
    )

    if args.tag:
        suffix += (
            f"_{args.tag}"
        )

    if args.negctrl:
        suffix += "_NEGCTRL"

    predictions_path = (
        OUT_DIR
        / f"RS05_oos_predictions_H{H}"
        f"_{args.tag or 'main'}"
        f"{'_NEGCTRL' if args.negctrl else ''}.csv.gz"
    )

    P.to_csv(
        predictions_path,
        index=False,
        compression="gzip",
    )

    meta = {
        "input_panel": str(
            RS04B_PATH
        ),
        "horizon": H,
        "targets": args.targets,
        "models": requested_models,
        "algorithms": args.algos,
        "specs": args.specs,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "headline_start": str(
            HEADLINE_START.date()
        ),
        "calibration_dates": CAL_LEN,
        "minimum_fit_dates": MIN_FIT_DATES,
        "purge_column": find_label_end_col(
            df,
            H,
        ),
        "regime_levels": regime_levels,
        "feature_groups": groups,
        "xgb_params": XGB_PARAMS,
        "seeds": args.seeds,
        "negative_control": args.negctrl,
        "notes": [
            "RS05a predicts future deterioration probability.",
            "Final current-trust signal is a separate later stage.",
            "Own-volatility features are read from validated RS04b; not rebuilt.",
            "Raw next-day outcomes are not used as features.",
            "Alert thresholds are determined from calibration only.",
        ],
    }

    meta_path = (
        OUT_DIR
        / f"RS05_run_metadata_H{H}"
        f"{'_NEGCTRL' if args.negctrl else ''}.json"
    )

    meta_path.write_text(
        json.dumps(
            meta,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    evaluate_predictions(
        P,
        suffix,
    )

    log_path = (
        OUT_DIR
        / f"RS05_walkforward_H{H}"
        f"{'_NEGCTRL' if args.negctrl else ''}.log"
    )

    save_log(
        log_path
    )

    emit(
        "\n"
        + "=" * 110
    )
    emit("RS05a WALK-FORWARD COMPLETE")
    emit("=" * 110)
    emit(
        f"Predictions: {predictions_path}"
    )
    emit(
        f"Metadata:    {meta_path}"
    )
    emit(
        f"Outputs:     {OUT_DIR}"
    )

    save_log(
        log_path
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "RS05a strictly causal, model-specific "
            "VaR reliability walk-forward monitor."
        )
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        choices=[5, 10, 20],
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        default=["T1", "T2"],
        choices=["T1", "T2"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=EXPECTED_MODELS,
    )

    parser.add_argument(
        "--algos",
        nargs="+",
        default=["logit", "xgb"],
        choices=["logit", "xgb"],
    )

    parser.add_argument(
        "--specs",
        nargs="+",
        default=[
            "S0",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5",
            "S6",
            "S7",
        ],
    )

    parser.add_argument(
        "--test-start",
        default="2019Q1",
    )

    parser.add_argument(
        "--test-end",
        default=None,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help=(
            "Number of XGBoost seeds to average. "
            "Use 3 for the main run and 5 for final robustness."
        ),
    )

    parser.add_argument(
        "--negctrl",
        action="store_true",
        help=(
            "Run the 126-day within-ticker negative-control "
            "label permutation."
        ),
    )

    parser.add_argument(
        "--tag",
        default="",
    )

    args = parser.parse_args()

    if (
        "xgb" in args.algos
        and xgb is None
    ):
        raise SystemExit(
            "xgboost is not installed. "
            "Install xgboost or run with --algos logit."
        )

    if args.seeds < 1:
        raise SystemExit(
            "--seeds must be >= 1."
        )

    run(args)
