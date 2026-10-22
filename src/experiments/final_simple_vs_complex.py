from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed in the current environment.\n"
        "Run: pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore")


# =============================================================================
# FINAL CONFIRMATORY EXPERIMENT
# =============================================================================
#
# Research question:
#
# "Do simple, interpretable signals—especially recent VaR exceptions and
# recent forecast loss—capture most of the predictable information about
# future VaR forecast deterioration, or do complex monitoring signals provide
# meaningful incremental value?"
#
# FINAL MODELS BEING MONITORED:
#   QRF
#   EWMA
#   GARCH-t
#   HS
#
# PRIMARY OUTCOME:
#   future h-day average pinball loss
#
# PRIMARY BINARY TARGET:
#   future loss > causal expanding 75th percentile
#
# VALIDATION:
#   Used ONLY for:
#       1. selecting the best individual simple signal
#       2. selecting its orientation
#       3. setting alert threshold
#
# TEST:
#   Completely untouched until final evaluation.
#
# COMPARISONS:
#
#   1. Best individual simple signal
#   2. Simple logistic combination of all simple signals
#   3. XGBoost complex monitor:
#        simple signals
#        +
#        cross-model disagreement signals
#
# MAIN METRICS:
#   pooled AUC
#   macro ticker AUC
#   median ticker AUC
#   improvement in AUC
#   fraction of tickers improved
#   block-bootstrap confidence intervals
#   alert precision / recall / rate / lift
#
# SPLIT:
#   60% train
#   20% validation
#   20% test
#
# Split is chronological separately for every ticker.
#
# NO ticker ID is used as a model feature.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

MONITORING_DIR = (
    ROOT
    / "data"
    / "monitoring"
)

PANEL_PATH = (
    MONITORING_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

VALIDATION_SELECTION_PATH = (
    MONITORING_DIR
    / "final_validation_signal_selection.csv"
)

SUMMARY_PATH = (
    MONITORING_DIR
    / "final_simple_vs_complex_results.csv"
)

PER_TICKER_PATH = (
    MONITORING_DIR
    / "final_per_ticker_simple_vs_complex.csv"
)

BOOTSTRAP_PATH = (
    MONITORING_DIR
    / "final_block_bootstrap_simple_vs_complex.csv"
)

ALERT_PATH = (
    MONITORING_DIR
    / "final_alert_metrics_simple_vs_complex.csv"
)

SUBPERIOD_PATH = (
    MONITORING_DIR
    / "final_subperiod_simple_vs_complex.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42

TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20
TEST_FRAC = 0.20

HORIZONS = [5, 10, 20]

MODELS = [
    "qrf",
    "ewma",
    "garch",
    "hs",
]

TICKERS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CAT",
    "CVX",
    "DIS",
    "GME",
    "GOOGL",
    "GS",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MSFT",
    "NFLX",
    "NVDA",
    "PFE",
    "PG",
    "TSLA",
    "UNH",
    "V",
    "WMT",
    "XOM",
]

BOOTSTRAP_REPS = 500

BLOCK_LENGTHS = [
    10,
    20,
    40,
    60,
]

ALERT_QUANTILE = 0.90


# =============================================================================
# XGBOOST PARAMETERS
# =============================================================================
#
# Fixed ex ante.
# They are NOT tuned on the test set.
# =============================================================================

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "min_child_weight": 5,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": SEED,
    "n_jobs": -1,
    "tree_method": "hist",
}


# =============================================================================
# HELPERS
# =============================================================================

def safe_auc(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
) -> float:

    y = np.asarray(y_true)
    s = np.asarray(scores)

    mask = (
        np.isfinite(s)
        &
        pd.notna(y)
    )

    y = y[mask]
    s = s[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            s,
        )
    )


def percentile_threshold(
    scores: np.ndarray,
    quantile: float = ALERT_QUANTILE,
) -> float:

    scores = np.asarray(
        scores,
        dtype=float,
    )

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:
        return np.nan

    return float(
        np.quantile(
            scores,
            quantile,
        )
    )


def alert_metrics(
    y_true,
    scores,
    threshold,
) -> dict:

    y = np.asarray(
        y_true,
        dtype=int,
    )

    s = np.asarray(
        scores,
        dtype=float,
    )

    mask = (
        np.isfinite(s)
        &
        np.isfinite(y)
    )

    y = y[mask]
    s = s[mask]

    if (
        len(y) == 0
        or not np.isfinite(threshold)
    ):
        return {
            "alert_rate": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "lift": np.nan,
        }

    alert = (
        s >= threshold
    ).astype(int)

    alert_rate = float(
        alert.mean()
    )

    precision = (
        float(
            precision_score(
                y,
                alert,
                zero_division=0,
            )
        )
    )

    recall = (
        float(
            recall_score(
                y,
                alert,
                zero_division=0,
            )
        )
    )

    base_rate = float(
        y.mean()
    )

    if alert_rate > 0 and base_rate > 0:

        lift = (
            precision
            / base_rate
        )

    else:
        lift = np.nan

    return {
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "lift": lift,
    }


def ticker_auc_table(
    df: pd.DataFrame,
    ticker_col: str,
    target_col: str,
    score_col: str,
) -> pd.DataFrame:

    rows = []

    for ticker, group in df.groupby(
        ticker_col,
        sort=True,
    ):

        valid = group[
            group[target_col].notna()
            &
            group[score_col].notna()
        ].copy()

        if (
            len(valid) == 0
            or valid[target_col].nunique() < 2
        ):
            auc = np.nan

        else:

            auc = safe_auc(
                valid[target_col],
                valid[score_col],
            )

        rows.append(
            {
                "ticker": ticker,
                "auc": auc,
                "n": len(valid),
            }
        )

    return pd.DataFrame(rows)


def add_split_labels(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        df
        .sort_values(
            [
                "ticker",
                "forecast_date",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    split_values = []

    for ticker, group in df.groupby(
        "ticker",
        sort=False,
    ):

        n = len(group)

        train_end = int(
            np.floor(
                TRAIN_FRAC * n
            )
        )

        validation_end = int(
            np.floor(
                (
                    TRAIN_FRAC
                    + VALIDATION_FRAC
                )
                * n
            )
        )

        labels = np.empty(
            n,
            dtype=object,
        )

        labels[
            :train_end
        ] = "train"

        labels[
            train_end:
            validation_end
        ] = "validation"

        labels[
            validation_end:
        ] = "test"

        split_values.extend(
            labels.tolist()
        )

    df["split"] = split_values

    return df


def orient_simple_signal(
    validation_df: pd.DataFrame,
    target_col: str,
    signal_col: str,
) -> tuple[float, float, float, float]:

    ticker_results = []

    for ticker, group in validation_df.groupby(
        "ticker",
        sort=True,
    ):

        valid = group[
            group[target_col].notna()
            &
            group[signal_col].notna()
        ].copy()

        if (
            len(valid) < 20
            or valid[target_col].nunique() < 2
        ):
            continue

        auc_positive = safe_auc(
            valid[target_col],
            valid[signal_col],
        )

        auc_negative = safe_auc(
            valid[target_col],
            -valid[signal_col],
        )

        if not np.isfinite(auc_positive):
            continue

        if not np.isfinite(auc_negative):
            continue

        direction = (
            1.0
            if auc_positive >= auc_negative
            else -1.0
        )

        oriented_auc = max(
            auc_positive,
            auc_negative,
        )

        ticker_results.append(
            {
                "ticker": ticker,
                "auc_positive": auc_positive,
                "auc_negative": auc_negative,
                "direction": direction,
                "oriented_auc": oriented_auc,
            }
        )

    if len(ticker_results) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )

    result = pd.DataFrame(
        ticker_results
    )

    # -------------------------------------------------------------------------
    # Determine global direction using validation only.
    #
    # Macro average is the primary criterion.
    # Pooled AUC is used as the tie-breaker.
    # -------------------------------------------------------------------------

    macro_positive = (
        result["auc_positive"]
        .mean()
    )

    macro_negative = (
        result["auc_negative"]
        .mean()
    )

    if macro_positive >= macro_negative:

        direction = 1.0

    else:

        direction = -1.0

    pooled = validation_df[
        validation_df[target_col].notna()
        &
        validation_df[signal_col].notna()
    ].copy()

    pooled_auc = safe_auc(
        pooled[target_col],
        direction
        * pooled[signal_col],
    )

    macro_auc = (
        result.apply(
            lambda row:
                (
                    row["auc_positive"]
                    if direction == 1
                    else row["auc_negative"]
                ),
            axis=1,
        )
        .mean()
    )

    n_tickers = len(
        result
    )

    return (
        float(direction),
        float(macro_auc),
        float(pooled_auc),
        float(n_tickers),
    )


def bootstrap_auc_difference(
    y: np.ndarray,
    simple_score: np.ndarray,
    complex_score: np.ndarray,
    tickers: np.ndarray,
    dates: np.ndarray,
    block_length: int,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, int]:

    rng = np.random.default_rng(
        seed
    )

    temp = pd.DataFrame(
        {
            "ticker": tickers,
            "date": dates,
            "y": y,
            "simple": simple_score,
            "complex": complex_score,
        }
    )

    samples = []

    for ticker, group in temp.groupby(
        "ticker",
        sort=True,
    ):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
        )

        samples.append(
            group
        )

    bootstrap_deltas = []

    for _ in range(
        repetitions
    ):

        pieces = []

        for group in samples:

            n = len(group)

            if n == 0:
                continue

            starts = list(
                range(
                    0,
                    n,
                    block_length,
                )
            )

            blocks = []

            for start in starts:

                end = min(
                    start
                    + block_length,
                    n,
                )

                blocks.append(
                    group.iloc[
                        start:end
                    ]
                )

            n_blocks = len(
                blocks
            )

            if n_blocks == 0:
                continue

            selected_ids = (
                rng.integers(
                    0,
                    n_blocks,
                    size=n_blocks,
                )
            )

            sampled = pd.concat(
                [
                    blocks[i]
                    for i in selected_ids
                ],
                ignore_index=True,
            )

            # Preserve approximately original sample size.
            sampled = sampled.iloc[
                :n
            ]

            pieces.append(
                sampled
            )

        if len(pieces) == 0:
            continue

        bootstrap = pd.concat(
            pieces,
            ignore_index=True,
        )

        y_b = (
            bootstrap["y"]
            .to_numpy()
        )

        s_b = (
            bootstrap["simple"]
            .to_numpy()
        )

        c_b = (
            bootstrap["complex"]
            .to_numpy()
        )

        if len(
            np.unique(y_b)
        ) < 2:
            continue

        auc_simple = safe_auc(
            y_b,
            s_b,
        )

        auc_complex = safe_auc(
            y_b,
            c_b,
        )

        if (
            np.isfinite(auc_simple)
            and
            np.isfinite(auc_complex)
        ):

            bootstrap_deltas.append(
                auc_complex
                -
                auc_simple
            )

    if len(
        bootstrap_deltas
    ) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
            0,
        )

    values = np.asarray(
        bootstrap_deltas
    )

    lower = float(
        np.quantile(
            values,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            values,
            0.975,
        )
    )

    mean = float(
        np.mean(values)
    )

    return (
        mean,
        lower,
        upper,
        len(values),
    )


# =============================================================================
# LOAD PANEL
# =============================================================================

print(
    "=" * 100
)

print(
    "FINAL SIMPLE VS COMPLEX CONFIRMATORY EXPERIMENT"
)

print(
    "=" * 100
)

print(
    f"Root: {ROOT}"
)

print(
    f"Panel: {PANEL_PATH}"
)

if not PANEL_PATH.exists():

    raise FileNotFoundError(
        f"Panel not found:\n{PANEL_PATH}"
    )

panel = pd.read_csv(
    PANEL_PATH
)

panel["forecast_date"] = pd.to_datetime(
    panel["forecast_date"],
    errors="coerce",
)

panel["realized_date"] = pd.to_datetime(
    panel["realized_date"],
    errors="coerce",
)

panel["ticker"] = (
    panel["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

panel = panel[
    panel["ticker"].isin(
        TICKERS
    )
].copy()

panel = (
    panel
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

print(
    f"\nRows: {len(panel):,}"
)

print(
    f"Tickers: {panel['ticker'].nunique()}"
)


# =============================================================================
# CHECK COMMON PANEL
# =============================================================================

expected_rows = (
    24 * 853
)

if len(panel) != expected_rows:

    print(
        "\nWARNING:"
    )

    print(
        f"Expected approximately {expected_rows:,} rows."
    )

    print(
        f"Observed {len(panel):,}."
    )


# =============================================================================
# ADD CHRONOLOGICAL SPLITS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "BUILDING 60 / 20 / 20 CHRONOLOGICAL SPLITS"
)

print(
    "=" * 100
)

panel = add_split_labels(
    panel
)

print(
    panel[
        "split"
    ].value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# CAUSAL STANDARDIZED RECENT LOSS
# =============================================================================
#
# Important:
#
# pinball_mean_20d_prior[t]
#
# already uses information available BEFORE t.
#
# To standardize it, the reference distribution is itself shifted by one
# observation so that row t does not use the current signal's own value in
# the reference mean/std.
#
# Thus this remains causal.
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "ADDING CAUSAL STANDARDIZED RECENT-LOSS SIGNAL"
)

print(
    "=" * 100
)

for model in MODELS:

    base_col = (
        f"{model}_pinball_mean_20d_prior"
    )

    z_col = (
        f"{model}_pinball_mean_20d_prior_z"
    )

    prior_signal = (
        panel
        .groupby(
            "ticker"
        )[base_col]
        .shift(1)
    )

    expanding_mean = (
        prior_signal
        .groupby(
            panel["ticker"]
        )
        .transform(
            lambda x:
                x.expanding(
                    min_periods=40
                ).mean()
        )
    )

    expanding_std = (
        prior_signal
        .groupby(
            panel["ticker"]
        )
        .transform(
            lambda x:
                x.expanding(
                    min_periods=40
                ).std()
        )
    )

    panel[z_col] = (
        (
            panel[base_col]
            -
            expanding_mean
        )
        /
        expanding_std
    )


# =============================================================================
# DEFINE SIMPLE FEATURES
# =============================================================================

SIMPLE_BASE_NAMES = [
    "var_abs_current",
    "var_change_1d",
    "var_ratio_to_prior",
    "var_ratio_to_20d_mean",

    "pinball_lag1",
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
    "pinball_mean_60d_prior",
    "pinball_acceleration",
    "pinball_mean_20d_prior_z",

    "breach_lag1",
    "breach_streak_prior",
    "days_since_last_breach",

    "breach_rate_5d_prior",
    "breach_rate_20d_prior",
    "breach_rate_60d_prior",

    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",

    "past_mean_pinball",
]


# =============================================================================
# CROSS-MODEL COMPLEX MONITOR FEATURES
# =============================================================================

COMPLEX_FEATURES = [
    "qrf_ewma_disagreement",
    "qrf_garch_disagreement",
    "qrf_hs_disagreement",
    "mean_absolute_model_disagreement",
]


# =============================================================================
# BUILD MODEL-SPECIFIC FEATURE LISTS
# =============================================================================

simple_features_by_model = {}

for model in MODELS:

    features = []

    for base_name in SIMPLE_BASE_NAMES:

        column = (
            f"{model}_{base_name}"
        )

        if column in panel.columns:

            features.append(
                column
            )

    simple_features_by_model[
        model
    ] = features


complex_features_by_model = {}

for model in MODELS:

    complex_features_by_model[
        model
    ] = (
        simple_features_by_model[model]
        +
        COMPLEX_FEATURES
    )


# =============================================================================
# PRINT FEATURE COUNTS
# =============================================================================

print(
    "\nSimple feature counts:"
)

for model in MODELS:

    print(
        f"  {model.upper()}: "
        f"{len(simple_features_by_model[model])}"
    )

print(
    "\nComplex feature counts:"
)

for model in MODELS:

    print(
        f"  {model.upper()}: "
        f"{len(complex_features_by_model[model])}"
    )


# =============================================================================
# VALIDATION SIGNAL SELECTION
# =============================================================================

validation_selection_rows = []

# Candidate signal names correspond to the unprefixed feature names.
signal_candidates = SIMPLE_BASE_NAMES.copy()

print(
    "\n" + "=" * 100
)

print(
    "VALIDATION-ONLY SIMPLE SIGNAL SELECTION"
)

print(
    "=" * 100
)

for model in MODELS:

    print(
        f"\nMODEL: {model.upper()}"
    )

    for horizon in HORIZONS:

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        validation_df = panel[
            panel["split"]
            ==
            "validation"
        ].copy()

        candidate_rows = []

        for base_name in signal_candidates:

            signal_col = (
                f"{model}_{base_name}"
            )

            if signal_col not in validation_df.columns:
                continue

            direction, macro_auc, pooled_auc, n_tickers = (
                orient_simple_signal(
                    validation_df,
                    target_col,
                    signal_col,
                )
            )

            if not np.isfinite(
                macro_auc
            ):
                continue

            candidate_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "signal":
                        base_name,

                    "signal_column":
                        signal_col,

                    "direction":
                        direction,

                    "validation_macro_auc":
                        macro_auc,

                    "validation_pooled_auc":
                        pooled_auc,

                    "validation_tickers":
                        int(n_tickers),
                }
            )

        if len(candidate_rows) == 0:

            raise RuntimeError(
                f"No usable validation signals for "
                f"{model.upper()} {horizon}d."
            )

        candidate_df = pd.DataFrame(
            candidate_rows
        )

        candidate_df = (
            candidate_df
            .sort_values(
                [
                    "validation_macro_auc",
                    "validation_pooled_auc",
                ],
                ascending=False,
            )
            .reset_index(drop=True)
        )

        best = candidate_df.iloc[
            0
        ]

        print(
            f"  {horizon}d best simple signal: "
            f"{best['signal']}"
            f" | validation macro AUC="
            f"{best['validation_macro_auc']:.4f}"
            f" | pooled AUC="
            f"{best['validation_pooled_auc']:.4f}"
            f" | direction="
            f"{'+' if best['direction'] > 0 else '-'}"
        )

        candidate_df[
            "selected"
        ] = 0

        candidate_df.loc[
            0,
            "selected"
        ] = 1

        validation_selection_rows.extend(
            candidate_df.to_dict(
                orient="records"
            )
        )


validation_selection = pd.DataFrame(
    validation_selection_rows
)


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

summary_rows = []
per_ticker_rows = []
bootstrap_rows = []
alert_rows = []
subperiod_rows = []


# =============================================================================
# FINAL EVALUATION
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL TEST EVALUATION"
)

print(
    "=" * 100
)


for model in MODELS:

    for horizon in HORIZONS:

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        selected = validation_selection[
            (
                validation_selection["model"]
                ==
                model.upper()
            )
            &
            (
                validation_selection["horizon"]
                ==
                horizon
            )
            &
            (
                validation_selection["selected"]
                ==
                1
            )
        ].iloc[0]

        best_signal = (
            selected["signal"]
        )

        best_signal_col = (
            selected["signal_column"]
        )

        direction = float(
            selected["direction"]
        )

        simple_cols = (
            simple_features_by_model[
                model
            ]
        )

        complex_cols = (
            complex_features_by_model[
                model
            ]
        )

        print(
            f"\n{'-' * 100}"
        )

        print(
            f"{model.upper()} | {horizon}D"
        )

        print(
            f"Best simple signal: {best_signal}"
        )

        print(
            f"Validation direction: "
            f"{'positive' if direction > 0 else 'negative'}"
        )

        # ---------------------------------------------------------------------
        # BUILD TRAIN / VALIDATION / TEST
        # ---------------------------------------------------------------------

        train_df = panel[
            panel["split"] == "train"
        ].copy()

        validation_df = panel[
            panel["split"] == "validation"
        ].copy()

        test_df = panel[
            panel["split"] == "test"
        ].copy()

        train_df = train_df[
            train_df[target_col].notna()
        ].copy()

        validation_df = validation_df[
            validation_df[target_col].notna()
        ].copy()

        test_df = test_df[
            test_df[target_col].notna()
        ].copy()

        # ---------------------------------------------------------------------
        # SIMPLE SINGLE SIGNAL
        # ---------------------------------------------------------------------

        simple_test_mask = (
            test_df[target_col].notna()
            &
            test_df[
                best_signal_col
            ].notna()
        )

        simple_test = (
            test_df.loc[
                simple_test_mask
            ]
            .copy()
        )

        simple_test_score = (
            direction
            *
            simple_test[
                best_signal_col
            ]
            .to_numpy(
                dtype=float
            )
        )

        simple_test_y = (
            simple_test[
                target_col
            ]
            .to_numpy(
                dtype=int
            )
        )

        simple_pooled_auc = safe_auc(
            simple_test_y,
            simple_test_score,
        )

        simple_threshold = percentile_threshold(
            direction
            *
            validation_df[
                best_signal_col
            ]
            .dropna()
            .to_numpy(
                dtype=float
            )
        )

        simple_alert = alert_metrics(
            simple_test_y,
            simple_test_score,
            simple_threshold,
        )

        # ---------------------------------------------------------------------
        # SIMPLE LOGISTIC COMBINATION
        # ---------------------------------------------------------------------

        logistic_train = train_df[
            [
                *simple_cols,
                target_col,
            ]
        ].copy()

        logistic_test = test_df[
            [
                *simple_cols,
                target_col,
            ]
        ].copy()

        X_train = logistic_train[
            simple_cols
        ]

        y_train = logistic_train[
            target_col
        ].astype(int)

        X_test = logistic_test[
            simple_cols
        ]

        y_test = logistic_test[
            target_col
        ].astype(int)

        if y_train.nunique() < 2:

            raise RuntimeError(
                f"Logistic training target has "
                f"<2 classes: "
                f"{model} {horizon}d"
            )

        logistic_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        logistic_pipe.fit(
            X_train,
            y_train,
        )

        logistic_test_score = (
            logistic_pipe
            .predict_proba(
                X_test
            )[:, 1]
        )

        logistic_pooled_auc = safe_auc(
            y_test,
            logistic_test_score,
        )

        # Threshold selected on validation.
        logistic_validation = validation_df[
            simple_cols
        ]

        logistic_validation_score = (
            logistic_pipe
            .predict_proba(
                logistic_validation
            )[:, 1]
        )

        logistic_validation_threshold = (
            percentile_threshold(
                logistic_validation_score
            )
        )

        logistic_alert = alert_metrics(
            y_test,
            logistic_test_score,
            logistic_validation_threshold,
        )

        # ---------------------------------------------------------------------
        # XGBOOST COMPLEX MONITOR
        # ---------------------------------------------------------------------

        xgb_train = train_df[
            [
                *complex_cols,
                target_col,
            ]
        ].copy()

        xgb_test = test_df[
            [
                *complex_cols,
                target_col,
            ]
        ].copy()

        X_train_xgb = xgb_train[
            complex_cols
        ]

        y_train_xgb = xgb_train[
            target_col
        ].astype(int)

        X_test_xgb = xgb_test[
            complex_cols
        ]

        y_test_xgb = xgb_test[
            target_col
        ].astype(int)

        xgb_imputer = SimpleImputer(
            strategy="median"
        )

        X_train_xgb_imp = (
            xgb_imputer
            .fit_transform(
                X_train_xgb
            )
        )

        X_test_xgb_imp = (
            xgb_imputer
            .transform(
                X_test_xgb
            )
        )

        xgb_model = XGBClassifier(
            **XGB_PARAMS
        )

        xgb_model.fit(
            X_train_xgb_imp,
            y_train_xgb,
        )

        xgb_test_score = (
            xgb_model
            .predict_proba(
                X_test_xgb_imp
            )[:, 1]
        )

        xgb_pooled_auc = safe_auc(
            y_test_xgb,
            xgb_test_score,
        )

        # Validation threshold for alerts.
        X_validation_xgb = (
            validation_df[
                complex_cols
            ]
        )

        X_validation_xgb_imp = (
            xgb_imputer
            .transform(
                X_validation_xgb
            )
        )

        xgb_validation_score = (
            xgb_model
            .predict_proba(
                X_validation_xgb_imp
            )[:, 1]
        )

        xgb_validation_threshold = (
            percentile_threshold(
                xgb_validation_score
            )
        )

        xgb_alert = alert_metrics(
            y_test_xgb,
            xgb_test_score,
            xgb_validation_threshold,
        )

        # ---------------------------------------------------------------------
        # COMMON TEST SAMPLE
        #
        # For direct paired comparison against the best simple signal, restrict
        # to rows where the selected simple signal is actually available.
        # ---------------------------------------------------------------------

        test_common = test_df[
            test_df[target_col].notna()
            &
            test_df[
                best_signal_col
            ].notna()
        ].copy()

        common_indices = (
            test_common.index
        )

        common_simple_scores = (
            direction
            *
            test_common[
                best_signal_col
            ].to_numpy(
                dtype=float
            )
        )

        common_xgb_scores = (
            pd.Series(
                xgb_test_score,
                index=xgb_test.index,
            )
            .loc[
                common_indices
            ]
            .to_numpy(
                dtype=float
            )
        )

        common_y = (
            test_common[
                target_col
            ]
            .to_numpy(
                dtype=int
            )
        )

        common_tickers = (
            test_common[
                "ticker"
            ]
            .to_numpy()
        )

        common_dates = (
            test_common[
                "forecast_date"
            ]
            .to_numpy()
        )

        simple_common_auc = safe_auc(
            common_y,
            common_simple_scores,
        )

        xgb_common_auc = safe_auc(
            common_y,
            common_xgb_scores,
        )

        xgb_minus_simple = (
            xgb_common_auc
            -
            simple_common_auc
        )

        # ---------------------------------------------------------------------
        # PER-TICKER AUC
        # ---------------------------------------------------------------------

        ticker_eval = pd.DataFrame(
            {
                "ticker":
                    common_tickers,

                "forecast_date":
                    common_dates,

                "target":
                    common_y,

                "simple_score":
                    common_simple_scores,

                "xgb_score":
                    common_xgb_scores,
            }
        )

        for ticker, group in ticker_eval.groupby(
            "ticker",
            sort=True,
        ):

            if group["target"].nunique() < 2:
                continue

            ticker_simple_auc = safe_auc(
                group["target"],
                group["simple_score"],
            )

            ticker_xgb_auc = safe_auc(
                group["target"],
                group["xgb_score"],
            )

            if (
                np.isfinite(
                    ticker_simple_auc
                )
                and
                np.isfinite(
                    ticker_xgb_auc
                )
            ):

                per_ticker_rows.append(
                    {
                        "model":
                            model.upper(),

                        "horizon":
                            horizon,

                        "ticker":
                            ticker,

                        "best_simple_signal":
                            best_signal,

                        "simple_auc":
                            ticker_simple_auc,

                        "xgb_auc":
                            ticker_xgb_auc,

                        "xgb_minus_simple":
                            (
                                ticker_xgb_auc
                                -
                                ticker_simple_auc
                            ),
                    }
                )

        # ---------------------------------------------------------------------
        # BLOCK BOOTSTRAP
        # ---------------------------------------------------------------------

        print(
            "  Running block bootstrap..."
        )

        for block_length in BLOCK_LENGTHS:

            mean_delta, lower, upper, n_boot = (
                bootstrap_auc_difference(
                    y=common_y,
                    simple_score=common_simple_scores,
                    complex_score=common_xgb_scores,
                    tickers=common_tickers,
                    dates=common_dates,
                    block_length=block_length,
                    repetitions=BOOTSTRAP_REPS,
                    seed=(
                        SEED
                        +
                        horizon
                        +
                        block_length
                        +
                        len(model)
                    ),
                )
            )

            bootstrap_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "block_length":
                        block_length,

                    "bootstrap_repetitions":
                        BOOTSTRAP_REPS,

                    "successful_repetitions":
                        n_boot,

                    "mean_auc_delta":
                        mean_delta,

                    "ci_lower":
                        lower,

                    "ci_upper":
                        upper,

                    "robust_positive":
                        (
                            lower > 0
                            if np.isfinite(lower)
                            else False
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # SUBPERIOD ROBUSTNESS
        # ---------------------------------------------------------------------

        median_date = (
            test_common[
                "forecast_date"
            ]
            .median()
        )

        for period_name, mask in [
            (
                "early_test",
                test_common[
                    "forecast_date"
                ]
                <= median_date,
            ),
            (
                "late_test",
                test_common[
                    "forecast_date"
                ]
                > median_date,
            ),
        ]:

            sub = test_common[
                mask
            ].copy()

            y_sub = (
                sub[target_col]
                .to_numpy(
                    dtype=int
                )
            )

            simple_sub = (
                direction
                *
                sub[
                    best_signal_col
                ].to_numpy(
                    dtype=float
                )
            )

            xgb_sub = (
                pd.Series(
                    xgb_test_score,
                    index=test_df.index,
                )
                .loc[
                    sub.index
                ]
                .to_numpy(
                    dtype=float
                )
            )

            sub_simple_auc = safe_auc(
                y_sub,
                simple_sub,
            )

            sub_xgb_auc = safe_auc(
                y_sub,
                xgb_sub,
            )

            subperiod_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "period":
                        period_name,

                    "rows":
                        len(sub),

                    "positive_rate":
                        float(
                            y_sub.mean()
                        ),

                    "simple_auc":
                        sub_simple_auc,

                    "xgb_auc":
                        sub_xgb_auc,

                    "xgb_minus_simple":
                        (
                            sub_xgb_auc
                            -
                            sub_simple_auc
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        per_ticker_subset = pd.DataFrame(
            [
                row
                for row in per_ticker_rows
                if (
                    row["model"]
                    ==
                    model.upper()
                    and
                    row["horizon"]
                    ==
                    horizon
                )
            ]
        )

        if len(
            per_ticker_subset
        ) > 0:

            mean_simple_ticker_auc = (
                per_ticker_subset[
                    "simple_auc"
                ].mean()
            )

            median_simple_ticker_auc = (
                per_ticker_subset[
                    "simple_auc"
                ].median()
            )

            mean_xgb_ticker_auc = (
                per_ticker_subset[
                    "xgb_auc"
                ].mean()
            )

            median_xgb_ticker_auc = (
                per_ticker_subset[
                    "xgb_auc"
                ].median()
            )

            improved_count = int(
                (
                    per_ticker_subset[
                        "xgb_minus_simple"
                    ]
                    > 0
                ).sum()
            )

            ticker_count = len(
                per_ticker_subset
            )

            improved_fraction = (
                improved_count
                /
                ticker_count
            )

        else:

            mean_simple_ticker_auc = np.nan
            median_simple_ticker_auc = np.nan
            mean_xgb_ticker_auc = np.nan
            median_xgb_ticker_auc = np.nan
            improved_count = 0
            ticker_count = 0
            improved_fraction = np.nan

        # ---------------------------------------------------------------------
        # SUMMARY ROW
        # ---------------------------------------------------------------------

        summary_rows.append(
            {
                "model":
                    model.upper(),

                "horizon":
                    horizon,

                "best_simple_signal":
                    best_signal,

                "validation_direction":
                    direction,

                "validation_macro_auc":
                    float(
                        selected[
                            "validation_macro_auc"
                        ]
                    ),

                "validation_pooled_auc":
                    float(
                        selected[
                            "validation_pooled_auc"
                        ]
                    ),

                "test_rows_common":
                    len(test_common),

                "test_positive_rate":
                    float(
                        common_y.mean()
                    ),

                "best_simple_test_auc":
                    simple_common_auc,

                "simple_logistic_test_auc":
                    logistic_pooled_auc,

                "xgb_test_auc":
                    xgb_common_auc,

                "xgb_minus_best_simple":
                    xgb_minus_simple,

                "simple_macro_ticker_auc":
                    mean_simple_ticker_auc,

                "simple_median_ticker_auc":
                    median_simple_ticker_auc,

                "xgb_macro_ticker_auc":
                    mean_xgb_ticker_auc,

                "xgb_median_ticker_auc":
                    median_xgb_ticker_auc,

                "xgb_improved_tickers":
                    improved_count,

                "valid_tickers":
                    ticker_count,

                "xgb_improved_fraction":
                    improved_fraction,

                "simple_alert_rate":
                    simple_alert[
                        "alert_rate"
                    ],

                "simple_precision":
                    simple_alert[
                        "precision"
                    ],

                "simple_recall":
                    simple_alert[
                        "recall"
                    ],

                "simple_lift":
                    simple_alert[
                        "lift"
                    ],

                "logistic_alert_rate":
                    logistic_alert[
                        "alert_rate"
                    ],

                "logistic_precision":
                    logistic_alert[
                        "precision"
                    ],

                "logistic_recall":
                    logistic_alert[
                        "recall"
                    ],

                "logistic_lift":
                    logistic_alert[
                        "lift"
                    ],

                "xgb_alert_rate":
                    xgb_alert[
                        "alert_rate"
                    ],

                "xgb_precision":
                    xgb_alert[
                        "precision"
                    ],

                "xgb_recall":
                    xgb_alert[
                        "recall"
                    ],

                "xgb_lift":
                    xgb_alert[
                        "lift"
                    ],
            }
        )

        # ---------------------------------------------------------------------
        # ALERT OUTPUT
        # ---------------------------------------------------------------------

        alert_rows.extend(
            [
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "method":
                        "best_simple_signal",

                    "signal":
                        best_signal,

                    **simple_alert,
                },

                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "method":
                        "simple_logistic",

                    "signal":
                        "all_simple_features",

                    **logistic_alert,
                },

                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "method":
                        "xgb_complex",

                    "signal":
                        "simple_plus_disagreement",

                    **xgb_alert,
                },
            ]
        )

        print(
            f"  Test AUC:"
            f" simple={simple_common_auc:.4f}"
            f" | logistic={logistic_pooled_auc:.4f}"
            f" | XGB={xgb_common_auc:.4f}"
            f" | XGB-simple={xgb_minus_simple:+.4f}"
        )


# =============================================================================
# DATAFRAMES
# =============================================================================

validation_selection = (
    validation_selection
    .sort_values(
        [
            "model",
            "horizon",
            "selected",
            "validation_macro_auc",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )
    .reset_index(drop=True)
)

summary = pd.DataFrame(
    summary_rows
)

per_ticker = pd.DataFrame(
    per_ticker_rows
)

bootstrap = pd.DataFrame(
    bootstrap_rows
)

alerts = pd.DataFrame(
    alert_rows
)

subperiod = pd.DataFrame(
    subperiod_rows
)


# =============================================================================
# PRINT FINAL RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL CONFIRMATORY SUMMARY"
)

print(
    "=" * 100
)

display_columns = [
    "model",
    "horizon",
    "best_simple_signal",
    "validation_macro_auc",
    "best_simple_test_auc",
    "simple_logistic_test_auc",
    "xgb_test_auc",
    "xgb_minus_best_simple",
    "xgb_improved_fraction",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# MAIN CLAIM AUDIT
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "MAIN CLAIM AUDIT"
)

print(
    "=" * 100
)

for _, row in summary.iterrows():

    model = row["model"]
    horizon = int(
        row["horizon"]
    )

    best_simple = (
        row["best_simple_test_auc"]
    )

    xgb_auc = (
        row["xgb_test_auc"]
    )

    delta = (
        row["xgb_minus_best_simple"]
    )

    print(
        f"{model} {horizon}d: "
        f"simple={best_simple:.4f}, "
        f"XGB={xgb_auc:.4f}, "
        f"delta={delta:+.4f}"
    )


# =============================================================================
# BOOTSTRAP SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "BLOCK BOOTSTRAP: XGB - BEST SIMPLE"
)

print(
    "=" * 100
)

print(
    bootstrap.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# SAVE ALL OUTPUTS
# =============================================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

validation_selection.to_csv(
    VALIDATION_SELECTION_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

per_ticker.to_csv(
    PER_TICKER_PATH,
    index=False,
)

bootstrap.to_csv(
    BOOTSTRAP_PATH,
    index=False,
)

alerts.to_csv(
    ALERT_PATH,
    index=False,
)

subperiod.to_csv(
    SUBPERIOD_PATH,
    index=False,
)


# =============================================================================
# FINAL OUTPUT PATHS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FILES SAVED"
)

print(
    "=" * 100
)

print(
    f"\nValidation signal selection:"
    f"\n{VALIDATION_SELECTION_PATH}"
)

print(
    f"\nMain confirmatory results:"
    f"\n{SUMMARY_PATH}"
)

print(
    f"\nPer-ticker results:"
    f"\n{PER_TICKER_PATH}"
)

print(
    f"\nBlock bootstrap:"
    f"\n{BOOTSTRAP_PATH}"
)

print(
    f"\nAlert metrics:"
    f"\n{ALERT_PATH}"
)

print(
    f"\nSubperiod robustness:"
    f"\n{SUBPERIOD_PATH}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL SIMPLE VS COMPLEX EXPERIMENT COMPLETE"
)

print(
    "=" * 100
)

print(
    "\nResearch direction remains locked."
)

print(
    "The next interpretation should use ONLY the untouched test results."
)

print(
    "\nPrimary comparison:"
)

print(
    "Best validation-selected simple signal"
    "  vs  "
    "XGBoost complex monitor"
)

print(
    "\nSecondary comparison:"
)

print(
    "Simple logistic combination"
    "  vs  "
    "XGBoost complex monitor"
)