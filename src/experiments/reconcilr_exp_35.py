from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# RECONCILE EXPERIMENT 35 AGAINST FINAL 24-TICKER PANEL
# =============================================================================
#
# Purpose:
#
# Determine why the current experiment produces much larger 10d/20d simple
# signal AUCs than the earlier Experiment 35 benchmark.
#
# Exact benchmark reconstructed here:
#
#   pinball_lag1
#   +
#   vol_ratio_20_60
#   +
#   ticker fixed effects
#
# Model:
#
#   Regularized logistic regression
#
# We compare:
#
#   A) Experiment-35-style two-feature logistic model
#   B) past_mean_pinball single signal
#
# Everything is recomputed from the FINAL 24-ticker panel.
#
# Split:
#   60% train
#   20% validation
#   20% test
#
# No test-based tuning.
#
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

MONITORING_DIR = (
    ROOT / "data" / "monitoring"
)

PANEL_PATH = (
    MONITORING_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_PATH = (
    MONITORING_DIR
    / "reconcile_exp35_results.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20

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


# =============================================================================
# HELPERS
# =============================================================================

def safe_auc(
    y,
    score,
):
    y = np.asarray(y)
    score = np.asarray(score)

    mask = (
        pd.notna(y)
        &
        np.isfinite(score)
    )

    y = y[mask]
    score = score[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y,
            score,
        )
    )


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

    split = []

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
                    +
                    VALIDATION_FRAC
                )
                * n
            )
        )

        labels = np.array(
            ["test"] * n,
            dtype=object,
        )

        labels[
            :train_end
        ] = "train"

        labels[
            train_end:validation_end
        ] = "validation"

        split.extend(
            labels.tolist()
        )

    df["split"] = split

    return df


def macro_ticker_auc(
    df,
    target_col,
    score_col,
):

    values = []

    for ticker, group in df.groupby(
        "ticker",
        sort=True,
    ):

        valid = group[
            group[target_col].notna()
            &
            group[score_col].notna()
        ]

        if (
            len(valid) < 10
            or valid[target_col].nunique() < 2
        ):
            continue

        auc = safe_auc(
            valid[target_col],
            valid[score_col],
        )

        if np.isfinite(auc):
            values.append(auc)

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def build_ticker_effect_logistic(
    train_df,
    test_df,
    numeric_features,
    target_col,
):

    # -------------------------------------------------------------------------
    # Numeric features
    # -------------------------------------------------------------------------

    numeric_pipeline = Pipeline(
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
        ]
    )

    # -------------------------------------------------------------------------
    # Ticker fixed effects
    # -------------------------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),

            (
                "ticker",
                categorical_pipeline,
                ["ticker"],
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    X_train = train_df[
        [
            *numeric_features,
            "ticker",
        ]
    ]

    y_train = train_df[
        target_col
    ].astype(int)

    X_test = test_df[
        [
            *numeric_features,
            "ticker",
        ]
    ]

    model.fit(
        X_train,
        y_train,
    )

    scores = (
        model
        .predict_proba(
            X_test
        )[:, 1]
    )

    return scores


# =============================================================================
# LOAD
# =============================================================================

print(
    "=" * 100
)

print(
    "RECONCILING EXPERIMENT 35"
)

print(
    "=" * 100
)

print(
    f"Panel: {PANEL_PATH}"
)

if not PANEL_PATH.exists():
    raise FileNotFoundError(
        PANEL_PATH
    )

df = pd.read_csv(
    PANEL_PATH
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df[
    df["ticker"].isin(
        TICKERS
    )
].copy()

df = (
    df
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Tickers: {df['ticker'].nunique()}"
)


# =============================================================================
# SPLIT
# =============================================================================

df = add_split_labels(
    df
)

print(
    "\nSplit counts:"
)

print(
    df["split"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# MAIN LOOP
# =============================================================================

results = []


for model in MODELS:

    print(
        "\n" + "-" * 100
    )

    print(
        f"MODEL: {model.upper()}"
    )

    print(
        "-" * 100
    )

    lag_loss_col = (
        f"{model}_pinball_lag1"
    )

    vol_ratio_col = (
        f"{model}_vol_ratio_20_60"
    )

    past_mean_col = (
        f"{model}_past_mean_pinball"
    )

    for horizon in HORIZONS:

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        print(
            f"\n{horizon}D TARGET:"
        )

        print(
            f"  {target_col}"
        )

        train = df[
            df["split"] == "train"
        ].copy()

        validation = df[
            df["split"] == "validation"
        ].copy()

        test = df[
            df["split"] == "test"
        ].copy()

        # ---------------------------------------------------------------------
        # Keep valid target rows.
        # ---------------------------------------------------------------------

        train = train[
            train[target_col].notna()
        ].copy()

        validation = validation[
            validation[target_col].notna()
        ].copy()

        test = test[
            test[target_col].notna()
        ].copy()

        # ---------------------------------------------------------------------
        # PRINT TARGET RATES BY SPLIT
        # ---------------------------------------------------------------------

        print(
            f"  Train rows: {len(train):,} "
            f"| positive={train[target_col].mean():.4%}"
        )

        print(
            f"  Validation rows: {len(validation):,} "
            f"| positive={validation[target_col].mean():.4%}"
        )

        print(
            f"  Test rows: {len(test):,} "
            f"| positive={test[target_col].mean():.4%}"
        )

        # =====================================================================
        # A) EXPERIMENT 35 STYLE
        # =====================================================================

        benchmark_features = [
            lag_loss_col,
            vol_ratio_col,
        ]

        benchmark_train = train.dropna(
            subset=benchmark_features
        ).copy()

        benchmark_test = test.dropna(
            subset=benchmark_features
        ).copy()

        benchmark_validation = validation.dropna(
            subset=benchmark_features
        ).copy()

        benchmark_scores = (
            build_ticker_effect_logistic(
                benchmark_train,
                benchmark_test,
                benchmark_features,
                target_col,
            )
        )

        benchmark_validation_scores = (
            build_ticker_effect_logistic(
                benchmark_train,
                benchmark_validation,
                benchmark_features,
                target_col,
            )
        )

        # NOTE:
        # The function fits the model independently for each call.
        # For the true final validation/test comparison, fit once below.
        #
        # Refit correctly using train only.

        numeric_features = benchmark_features

        numeric_pipeline = Pipeline(
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
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),

                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        drop="first",
                    ),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features,
                ),

                (
                    "ticker",
                    categorical_pipeline,
                    ["ticker"],
                ),
            ]
        )

        benchmark_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),

                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        X_train = benchmark_train[
            [
                *benchmark_features,
                "ticker",
            ]
        ]

        y_train = benchmark_train[
            target_col
        ].astype(int)

        X_validation = benchmark_validation[
            [
                *benchmark_features,
                "ticker",
            ]
        ]

        y_validation = benchmark_validation[
            target_col
        ].astype(int)

        X_test = benchmark_test[
            [
                *benchmark_features,
                "ticker",
            ]
        ]

        y_test = benchmark_test[
            target_col
        ].astype(int)

        benchmark_model.fit(
            X_train,
            y_train,
        )

        benchmark_validation_score = (
            benchmark_model
            .predict_proba(
                X_validation
            )[:, 1]
        )

        benchmark_test_score = (
            benchmark_model
            .predict_proba(
                X_test
            )[:, 1]
        )

        benchmark_validation_auc = (
            safe_auc(
                y_validation,
                benchmark_validation_score,
            )
        )

        benchmark_test_auc = (
            safe_auc(
                y_test,
                benchmark_test_score,
            )
        )

        # =====================================================================
        # B) SINGLE PAST-MEAN-PINBALL SIGNAL
        # =====================================================================

        past_train = train[
            train[past_mean_col].notna()
        ].copy()

        past_validation = validation[
            validation[past_mean_col].notna()
        ].copy()

        past_test = test[
            test[past_mean_col].notna()
        ].copy()

        past_validation_auc_positive = safe_auc(
            past_validation[target_col],
            past_validation[past_mean_col],
        )

        past_validation_auc_negative = safe_auc(
            past_validation[target_col],
            -past_validation[past_mean_col],
        )

        if (
            past_validation_auc_positive
            >=
            past_validation_auc_negative
        ):

            past_direction = 1.0
            past_validation_auc = (
                past_validation_auc_positive
            )

        else:

            past_direction = -1.0
            past_validation_auc = (
                past_validation_auc_negative
            )

        past_test_score = (
            past_direction
            *
            past_test[past_mean_col]
            .to_numpy(
                dtype=float
            )
        )

        past_test_auc = safe_auc(
            past_test[target_col],
            past_test_score,
        )

        past_validation_score = (
            past_direction
            *
            past_validation[
                past_mean_col
            ].to_numpy(
                dtype=float
            )
        )

        # =====================================================================
        # C) TWO-FEATURE LOGISTIC WITHOUT TICKER EFFECTS
        # =====================================================================

        no_ticker_pipeline = Pipeline(
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
                        penalty="l2",
                        max_iter=3000,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        no_ticker_pipeline.fit(
            benchmark_train[
                benchmark_features
            ],
            benchmark_train[
                target_col
            ].astype(int),
        )

        two_feature_test_score = (
            no_ticker_pipeline
            .predict_proba(
                benchmark_test[
                    benchmark_features
                ]
            )[:, 1]
        )

        two_feature_test_auc = safe_auc(
            benchmark_test[target_col],
            two_feature_test_score,
        )

        # =====================================================================
        # D) PER-TICKER AUC FOR PAST-MEAN SIGNAL
        # =====================================================================

        past_test_eval = past_test[
            [
                "ticker",
                target_col,
                past_mean_col,
            ]
        ].copy()

        past_test_eval[
            "score"
        ] = (
            past_direction
            *
            past_test_eval[
                past_mean_col
            ]
        )

        macro_past_test = (
            macro_ticker_auc(
                past_test_eval,
                target_col,
                "score",
            )
        )

        # =====================================================================
        # E) PER-TICKER AUC FOR EXP35 MODEL
        # =====================================================================

        benchmark_test_eval = benchmark_test[
            [
                "ticker",
                target_col,
            ]
        ].copy()

        benchmark_test_eval[
            "score"
        ] = benchmark_test_score

        macro_benchmark_test = (
            macro_ticker_auc(
                benchmark_test_eval,
                target_col,
                "score",
            )
        )

        # =====================================================================
        # STORE
        # =====================================================================

        results.append(
            {
                "model":
                    model.upper(),

                "horizon":
                    horizon,

                "target":
                    target_col,

                "train_rows":
                    len(train),

                "validation_rows":
                    len(validation),

                "test_rows":
                    len(test),

                "train_positive_rate":
                    float(
                        train[target_col].mean()
                    ),

                "validation_positive_rate":
                    float(
                        validation[target_col].mean()
                    ),

                "test_positive_rate":
                    float(
                        test[target_col].mean()
                    ),

                "exp35_validation_auc":
                    benchmark_validation_auc,

                "exp35_test_auc":
                    benchmark_test_auc,

                "exp35_macro_test_auc":
                    macro_benchmark_test,

                "two_feature_no_ticker_test_auc":
                    two_feature_test_auc,

                "past_mean_validation_auc":
                    past_validation_auc,

                "past_mean_direction":
                    past_direction,

                "past_mean_test_auc":
                    past_test_auc,

                "past_mean_macro_test_auc":
                    macro_past_test,

                "exp35_minus_past_mean":
                    (
                        benchmark_test_auc
                        -
                        past_test_auc
                    ),
            }
        )

        print(
            f"  Exp35-style:"
            f" validation={benchmark_validation_auc:.4f}"
            f" test={benchmark_test_auc:.4f}"
            f" macro_test={macro_benchmark_test:.4f}"
        )

        print(
            f"  Past mean:"
            f" validation={past_validation_auc:.4f}"
            f" test={past_test_auc:.4f}"
            f" macro_test={macro_past_test:.4f}"
            f" direction="
            f"{'+' if past_direction > 0 else '-'}"
        )

        print(
            f"  Two-feature logistic WITHOUT ticker:"
            f" test={two_feature_test_auc:.4f}"
        )

        print(
            f"  Exp35 - past mean:"
            f" {benchmark_test_auc - past_test_auc:+.4f}"
        )


# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =============================================================================
# PRINT FINAL TABLE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "RECONCILIATION RESULTS"
)

print(
    "=" * 100
)

print(
    results_df[
        [
            "model",
            "horizon",
            "exp35_validation_auc",
            "exp35_test_auc",
            "exp35_macro_test_auc",
            "two_feature_no_ticker_test_auc",
            "past_mean_validation_auc",
            "past_mean_test_auc",
            "past_mean_macro_test_auc",
            "exp35_minus_past_mean",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# FINAL CHECK
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "EXPERIMENT 35 RECONCILIATION COMPLETE"
)

print(
    "=" * 100
)

print(
    f"\nSaved:"
)

print(
    OUTPUT_PATH
)