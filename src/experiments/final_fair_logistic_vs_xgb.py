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

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is not installed.\n"
        "Run: pip install xgboost"
    ) from exc

warnings.filterwarnings("ignore")


# =============================================================================
# FINAL FAIR SIMPLE VS COMPLEX COMPARISON
# =============================================================================
#
# Research question:
#
# Do complex nonlinear monitoring signals provide meaningful incremental
# information over simple, interpretable monitoring signals for predicting
# future VaR forecast deterioration?
#
#
# THREE MODELS
#
# MODEL 1:
#   20 simple monitoring features
#   + ticker fixed effects
#   + regularized logistic regression
#
# MODEL 2:
#   24 features
#   + ticker fixed effects
#   + regularized logistic regression
#
# MODEL 3:
#   SAME 24 features
#   + SAME ticker fixed effects
#   + XGBoost
#
#
# Decomposition:
#
# feature_gain =
#       extended_logistic
#       -
#       simple_logistic
#
# nonlinear_gain =
#       XGB
#       -
#       extended_logistic
#
# total_gain =
#       XGB
#       -
#       simple_logistic
#
#
# Split:
#   60% train
#   20% validation
#   20% test
#
# Hyperparameters selected using validation only.
# Final models refit on train + validation.
# Test remains untouched.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "monitoring"

PANEL_PATH = (
    DATA_DIR / "final_24ticker_common_monitoring_panel.csv"
)

SUMMARY_PATH = (
    DATA_DIR / "final_fair_logistic_vs_xgb_results.csv"
)

PER_TICKER_PATH = (
    DATA_DIR / "final_fair_logistic_vs_xgb_per_ticker.csv"
)

SUBPERIOD_PATH = (
    DATA_DIR / "final_fair_logistic_vs_xgb_subperiod.csv"
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
# FIXED FEATURE DEFINITIONS
# =============================================================================

SIMPLE_FEATURE_NAMES = [
    "var_abs_current",
    "var_change_1d",
    "var_ratio_to_prior",
    "var_ratio_to_20d_mean",
    "pinball_lag1",
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
    "pinball_mean_60d_prior",
    "pinball_acceleration",
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

COMPLEX_FEATURE_NAMES = [
    "qrf_ewma_disagreement",
    "qrf_garch_disagreement",
    "qrf_hs_disagreement",
    "mean_absolute_model_disagreement",
]


# =============================================================================
# HYPERPARAMETERS
# =============================================================================

LOGISTIC_C_VALUES = [
    0.01,
    0.1,
    1.0,
    10.0,
]

XGB_CONFIGS = [
    {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 300,
    },
    {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 250,
    },
]


XGB_FIXED = {
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

def safe_auc(y, score):
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)

    mask = (
        pd.notna(y)
        & np.isfinite(score)
    )

    y = y[mask]
    score = score[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(y, score)
    )


def add_splits(df):
    """
    Exact chronological 60/20/20 split separately for each ticker.
    """

    df = (
        df
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
        .copy()
    )

    labels = []

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

        ticker_labels = np.empty(
            n,
            dtype=object,
        )

        ticker_labels[:train_end] = "train"

        ticker_labels[
            train_end:validation_end
        ] = "validation"

        ticker_labels[
            validation_end:
        ] = "test"

        labels.extend(
            ticker_labels.tolist()
        )

    df["split"] = labels

    return df


def make_logistic_pipeline(
    numeric_features,
    C,
):
    """
    Logistic regression:
        numeric features
        + ticker fixed effects
        + L2 regularization
    """

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

    ticker_pipeline = Pipeline(
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
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
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
                ticker_pipeline,
                ["ticker"],
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    max_iter=4000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def make_xgb_transformer(
    numeric_features,
):
    """
    Same numeric features + ticker fixed effects.

    This transformer is fitted ONLY on the supplied training data.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    ticker_pipeline = Pipeline(
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
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "ticker",
                ticker_pipeline,
                ["ticker"],
            ),
        ]
    )


def make_xgb(
    config,
):
    params = {
        **XGB_FIXED,
        **config,
    }

    return XGBClassifier(
        **params
    )


def per_ticker_auc(
    evaluation,
    target_col,
):
    rows = []

    for ticker, group in evaluation.groupby(
        "ticker",
        sort=True,
    ):

        row = {
            "ticker": ticker,
        }

        for score_name in [
            "simple_logistic",
            "extended_logistic",
            "xgb",
        ]:

            valid = group[
                group[target_col].notna()
                &
                group[score_name].notna()
            ]

            if (
                len(valid) < 10
                or valid[target_col].nunique() < 2
            ):
                auc = np.nan
            else:
                auc = safe_auc(
                    valid[target_col],
                    valid[score_name],
                )

            row[
                f"{score_name}_auc"
            ] = auc

        rows.append(row)

    return pd.DataFrame(rows)


def macro_mean(series):
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def median_value(series):
    values = series.dropna()

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


# =============================================================================
# LOAD
# =============================================================================

print("=" * 100)
print("FINAL FAIR SIMPLE VS COMPLEX COMPARISON")
print("=" * 100)

if not PANEL_PATH.exists():
    raise FileNotFoundError(
        f"Panel not found:\n{PANEL_PATH}"
    )

df = pd.read_csv(
    PANEL_PATH
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"],
    errors="coerce",
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df[
    df["ticker"].isin(TICKERS)
].copy()

df = add_splits(df)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Tickers: {df['ticker'].nunique()}"
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
# BUILD MODEL-SPECIFIC FEATURES
# =============================================================================

simple_features_by_model = {}
extended_features_by_model = {}

print(
    "\n" + "=" * 100
)

print(
    "VERIFYING FEATURE SET"
)

print(
    "=" * 100
)

for model in MODELS:

    simple_features = [
        f"{model}_{name}"
        for name in SIMPLE_FEATURE_NAMES
    ]

    extended_features = (
        simple_features
        + COMPLEX_FEATURE_NAMES
    )

    for column in extended_features:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing feature column:\n{column}"
            )

    simple_features_by_model[
        model
    ] = simple_features

    extended_features_by_model[
        model
    ] = extended_features

    print(
        f"{model.upper():5s}: "
        f"{len(simple_features)} simple + "
        f"{len(COMPLEX_FEATURE_NAMES)} "
        f"additional = "
        f"{len(extended_features)} total"
    )


# =============================================================================
# RESULT STORAGE
# =============================================================================

summary_rows = []
per_ticker_rows = []
subperiod_rows = []


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

for model in MODELS:

    simple_features = (
        simple_features_by_model[model]
    )

    extended_features = (
        extended_features_by_model[model]
    )

    print(
        "\n" + "=" * 100
    )

    print(
        f"MODEL UNDER MONITORING: {model.upper()}"
    )

    print(
        "=" * 100
    )

    for horizon in HORIZONS:

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        print(
            "\n" + "-" * 100
        )

        print(
            f"{model.upper()} | {horizon}D"
        )

        print(
            "-" * 100
        )

        # ---------------------------------------------------------------------
        # Split data
        # ---------------------------------------------------------------------

        train = df[
            df["split"] == "train"
        ].copy()

        validation = df[
            df["split"] == "validation"
        ].copy()

        test = df[
            df["split"] == "test"
        ].copy()

        train = train[
            train[target_col].notna()
        ].copy()

        validation = validation[
            validation[target_col].notna()
        ].copy()

        test = test[
            test[target_col].notna()
        ].copy()

        print(
            f"Train:      {len(train):,} "
            f"| positive={train[target_col].mean():.4%}"
        )

        print(
            f"Validation: {len(validation):,} "
            f"| positive={validation[target_col].mean():.4%}"
        )

        print(
            f"Test:       {len(test):,} "
            f"| positive={test[target_col].mean():.4%}"
        )

        # =====================================================================
        # MODEL 1
        # SIMPLE LOGISTIC: 20 FEATURES + TICKER FE
        # =====================================================================

        print(
            "\n[1/3] SIMPLE LOGISTIC"
        )

        best_simple_C = None
        best_simple_val_auc = -np.inf

        X_train_simple = train[
            simple_features + ["ticker"]
        ]

        y_train_simple = (
            train[target_col]
            .astype(int)
        )

        X_val_simple = validation[
            simple_features + ["ticker"]
        ]

        y_val = (
            validation[target_col]
            .astype(int)
        )

        for C in LOGISTIC_C_VALUES:

            candidate = make_logistic_pipeline(
                simple_features,
                C,
            )

            candidate.fit(
                X_train_simple,
                y_train_simple,
            )

            val_score = (
                candidate
                .predict_proba(
                    X_val_simple
                )[:, 1]
            )

            val_auc = safe_auc(
                y_val,
                val_score,
            )

            print(
                f"  C={C:<5} "
                f"validation AUC={val_auc:.4f}"
            )

            if (
                np.isfinite(val_auc)
                and
                val_auc
                > best_simple_val_auc
            ):

                best_simple_val_auc = (
                    val_auc
                )

                best_simple_C = C

        print(
            f"  Selected C={best_simple_C}"
        )

        train_validation = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        simple_final = make_logistic_pipeline(
            simple_features,
            best_simple_C,
        )

        simple_final.fit(
            train_validation[
                simple_features
                + ["ticker"]
            ],
            train_validation[
                target_col
            ].astype(int),
        )

        simple_test_score = (
            simple_final
            .predict_proba(
                test[
                    simple_features
                    + ["ticker"]
                ]
            )[:, 1]
        )

        simple_test_auc = safe_auc(
            test[target_col],
            simple_test_score,
        )

        # =====================================================================
        # MODEL 2
        # EXTENDED LOGISTIC: 24 FEATURES + TICKER FE
        # =====================================================================

        print(
            "\n[2/3] EXTENDED LOGISTIC"
        )

        best_extended_C = None
        best_extended_val_auc = -np.inf

        X_train_extended = train[
            extended_features + ["ticker"]
        ]

        X_val_extended = validation[
            extended_features + ["ticker"]
        ]

        for C in LOGISTIC_C_VALUES:

            candidate = make_logistic_pipeline(
                extended_features,
                C,
            )

            candidate.fit(
                X_train_extended,
                y_train_simple,
            )

            val_score = (
                candidate
                .predict_proba(
                    X_val_extended
                )[:, 1]
            )

            val_auc = safe_auc(
                y_val,
                val_score,
            )

            print(
                f"  C={C:<5} "
                f"validation AUC={val_auc:.4f}"
            )

            if (
                np.isfinite(val_auc)
                and
                val_auc
                > best_extended_val_auc
            ):

                best_extended_val_auc = (
                    val_auc
                )

                best_extended_C = C

        print(
            f"  Selected C={best_extended_C}"
        )

        extended_final = make_logistic_pipeline(
            extended_features,
            best_extended_C,
        )

        extended_final.fit(
            train_validation[
                extended_features
                + ["ticker"]
            ],
            train_validation[
                target_col
            ].astype(int),
        )

        extended_test_score = (
            extended_final
            .predict_proba(
                test[
                    extended_features
                    + ["ticker"]
                ]
            )[:, 1]
        )

        extended_test_auc = safe_auc(
            test[target_col],
            extended_test_score,
        )

        # =====================================================================
        # MODEL 3
        # XGBOOST: SAME 24 FEATURES + SAME TICKER FE
        # =====================================================================

        print(
            "\n[3/3] XGBOOST"
        )

        # ---------------------------------------------------------------------
        # Validation-stage transformer fitted ONLY on training data.
        # ---------------------------------------------------------------------

        xgb_validation_transformer = (
            make_xgb_transformer(
                extended_features
            )
        )

        X_train_xgb = (
            xgb_validation_transformer
            .fit_transform(
                train[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        X_val_xgb = (
            xgb_validation_transformer
            .transform(
                validation[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        y_train_xgb = (
            train[target_col]
            .astype(int)
            .to_numpy()
        )

        y_val_xgb = (
            validation[target_col]
            .astype(int)
            .to_numpy()
        )

        best_xgb_config = None
        best_xgb_val_auc = -np.inf

        for config in XGB_CONFIGS:

            candidate = make_xgb(
                config
            )

            candidate.fit(
                X_train_xgb,
                y_train_xgb,
            )

            val_score = (
                candidate
                .predict_proba(
                    X_val_xgb
                )[:, 1]
            )

            val_auc = safe_auc(
                y_val_xgb,
                val_score,
            )

            print(
                "  "
                f"depth={config['max_depth']} "
                f"lr={config['learning_rate']} "
                f"trees={config['n_estimators']}"
                f" | validation AUC={val_auc:.4f}"
            )

            if (
                np.isfinite(val_auc)
                and
                val_auc
                > best_xgb_val_auc
            ):

                best_xgb_val_auc = (
                    val_auc
                )

                best_xgb_config = dict(
                    config
                )

        print(
            f"  Selected XGB config: "
            f"{best_xgb_config}"
        )

        # ---------------------------------------------------------------------
        # Final XGB transformer fitted on TRAIN + VALIDATION.
        # TEST remains untouched.
        # ---------------------------------------------------------------------

        final_xgb_transformer = (
            make_xgb_transformer(
                extended_features
            )
        )

        X_train_validation_xgb = (
            final_xgb_transformer
            .fit_transform(
                train_validation[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        X_test_xgb = (
            final_xgb_transformer
            .transform(
                test[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        y_train_validation_xgb = (
            train_validation[
                target_col
            ]
            .astype(int)
            .to_numpy()
        )

        final_xgb = make_xgb(
            best_xgb_config
        )

        final_xgb.fit(
            X_train_validation_xgb,
            y_train_validation_xgb,
        )

        xgb_test_score = (
            final_xgb
            .predict_proba(
                X_test_xgb
            )[:, 1]
        )

        xgb_test_auc = safe_auc(
            test[target_col],
            xgb_test_score,
        )

        # =====================================================================
        # TEST EVALUATION DATAFRAME
        # =====================================================================

        evaluation = test[
            [
                "ticker",
                "forecast_date",
                target_col,
            ]
        ].copy()

        evaluation[
            "simple_logistic"
        ] = simple_test_score

        evaluation[
            "extended_logistic"
        ] = extended_test_score

        evaluation[
            "xgb"
        ] = xgb_test_score

        # =====================================================================
        # PER-TICKER AUC
        # =====================================================================

        ticker_results = per_ticker_auc(
            evaluation,
            target_col,
        )

        ticker_results = ticker_results[
            ticker_results[
                "simple_logistic_auc"
            ].notna()
            &
            ticker_results[
                "extended_logistic_auc"
            ].notna()
            &
            ticker_results[
                "xgb_auc"
            ].notna()
        ].copy()

        # ---------------------------------------------------------------------
        # Add per-ticker gains.
        # ---------------------------------------------------------------------

        if len(ticker_results) > 0:

            ticker_results[
                "feature_gain"
            ] = (
                ticker_results[
                    "extended_logistic_auc"
                ]
                -
                ticker_results[
                    "simple_logistic_auc"
                ]
            )

            ticker_results[
                "nonlinear_gain"
            ] = (
                ticker_results[
                    "xgb_auc"
                ]
                -
                ticker_results[
                    "extended_logistic_auc"
                ]
            )

            ticker_results[
                "total_xgb_gain"
            ] = (
                ticker_results[
                    "xgb_auc"
                ]
                -
                ticker_results[
                    "simple_logistic_auc"
                ]
            )

            simple_macro = macro_mean(
                ticker_results[
                    "simple_logistic_auc"
                ]
            )

            extended_macro = macro_mean(
                ticker_results[
                    "extended_logistic_auc"
                ]
            )

            xgb_macro = macro_mean(
                ticker_results[
                    "xgb_auc"
                ]
            )

            simple_median = median_value(
                ticker_results[
                    "simple_logistic_auc"
                ]
            )

            extended_median = median_value(
                ticker_results[
                    "extended_logistic_auc"
                ]
            )

            xgb_median = median_value(
                ticker_results[
                    "xgb_auc"
                ]
            )

            fraction_extended_better = float(
                (
                    ticker_results[
                        "feature_gain"
                    ]
                    > 0
                ).mean()
            )

            fraction_xgb_better = float(
                (
                    ticker_results[
                        "nonlinear_gain"
                    ]
                    > 0
                ).mean()
            )

        else:

            simple_macro = np.nan
            extended_macro = np.nan
            xgb_macro = np.nan

            simple_median = np.nan
            extended_median = np.nan
            xgb_median = np.nan

            fraction_extended_better = np.nan
            fraction_xgb_better = np.nan

        # ---------------------------------------------------------------------
        # Store ticker rows.
        # ---------------------------------------------------------------------

        for _, row in ticker_results.iterrows():

            per_ticker_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "ticker":
                        row["ticker"],

                    "simple_logistic_auc":
                        row["simple_logistic_auc"],

                    "extended_logistic_auc":
                        row["extended_logistic_auc"],

                    "xgb_auc":
                        row["xgb_auc"],

                    "feature_gain":
                        row["feature_gain"],

                    "nonlinear_gain":
                        row["nonlinear_gain"],

                    "total_xgb_gain":
                        row["total_xgb_gain"],
                }
            )

        # =====================================================================
        # EARLY / LATE TEST
        # =====================================================================

        midpoint = (
            evaluation[
                "forecast_date"
            ].median()
        )

        early = evaluation[
            evaluation[
                "forecast_date"
            ] <= midpoint
        ].copy()

        late = evaluation[
            evaluation[
                "forecast_date"
            ] > midpoint
        ].copy()

        for period_name, period_df in [
            ("early_test", early),
            ("late_test", late),
        ]:

            if len(period_df) == 0:
                continue

            simple_period_auc = safe_auc(
                period_df[target_col],
                period_df[
                    "simple_logistic"
                ],
            )

            extended_period_auc = safe_auc(
                period_df[target_col],
                period_df[
                    "extended_logistic"
                ],
            )

            xgb_period_auc = safe_auc(
                period_df[target_col],
                period_df[
                    "xgb"
                ],
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
                        len(period_df),

                    "positive_rate":
                        float(
                            period_df[
                                target_col
                            ].mean()
                        ),

                    "simple_logistic_auc":
                        simple_period_auc,

                    "extended_logistic_auc":
                        extended_period_auc,

                    "xgb_auc":
                        xgb_period_auc,

                    "feature_gain":
                        (
                            extended_period_auc
                            -
                            simple_period_auc
                        ),

                    "nonlinear_gain":
                        (
                            xgb_period_auc
                            -
                            extended_period_auc
                        ),

                    "total_xgb_gain":
                        (
                            xgb_period_auc
                            -
                            simple_period_auc
                        ),
                }
            )

        # =====================================================================
        # MAIN SUMMARY ROW
        # =====================================================================

        summary_rows.append(
            {
                "model":
                    model.upper(),

                "horizon":
                    horizon,

                "train_rows":
                    len(train),

                "validation_rows":
                    len(validation),

                "test_rows":
                    len(test),

                "train_positive_rate":
                    float(
                        train[
                            target_col
                        ].mean()
                    ),

                "validation_positive_rate":
                    float(
                        validation[
                            target_col
                        ].mean()
                    ),

                "test_positive_rate":
                    float(
                        test[
                            target_col
                        ].mean()
                    ),

                "simple_logistic_C":
                    best_simple_C,

                "simple_logistic_validation_auc":
                    best_simple_val_auc,

                "simple_logistic_test_auc":
                    simple_test_auc,

                "extended_logistic_C":
                    best_extended_C,

                "extended_logistic_validation_auc":
                    best_extended_val_auc,

                "extended_logistic_test_auc":
                    extended_test_auc,

                "xgb_validation_auc":
                    best_xgb_val_auc,

                "xgb_test_auc":
                    xgb_test_auc,

                "feature_gain_extended_minus_simple":
                    (
                        extended_test_auc
                        -
                        simple_test_auc
                    ),

                "nonlinear_gain_xgb_minus_extended":
                    (
                        xgb_test_auc
                        -
                        extended_test_auc
                    ),

                "total_xgb_gain":
                    (
                        xgb_test_auc
                        -
                        simple_test_auc
                    ),

                "simple_macro_ticker_auc":
                    simple_macro,

                "extended_macro_ticker_auc":
                    extended_macro,

                "xgb_macro_ticker_auc":
                    xgb_macro,

                "simple_median_ticker_auc":
                    simple_median,

                "extended_median_ticker_auc":
                    extended_median,

                "xgb_median_ticker_auc":
                    xgb_median,

                "fraction_tickers_extended_gt_simple":
                    fraction_extended_better,

                "fraction_tickers_xgb_gt_extended":
                    fraction_xgb_better,
            }
        )

        # =====================================================================
        # PRINT CURRENT RESULT
        # =====================================================================

        print(
            "\nTEST RESULTS:"
        )

        print(
            f"  Simple logistic (20): "
            f"{simple_test_auc:.4f}"
        )

        print(
            f"  Extended logistic (24): "
            f"{extended_test_auc:.4f}"
        )

        print(
            f"  XGBoost (24): "
            f"{xgb_test_auc:.4f}"
        )

        print(
            "\nDECOMPOSITION:"
        )

        print(
            f"  Feature gain: "
            f"{extended_test_auc - simple_test_auc:+.4f}"
        )

        print(
            f"  Nonlinear gain: "
            f"{xgb_test_auc - extended_test_auc:+.4f}"
        )

        print(
            f"  Total XGB gain: "
            f"{xgb_test_auc - simple_test_auc:+.4f}"
        )


# =============================================================================
# CONVERT TO DATAFRAMES
# =============================================================================

summary = pd.DataFrame(
    summary_rows
)

per_ticker = pd.DataFrame(
    per_ticker_rows
)

subperiod = pd.DataFrame(
    subperiod_rows
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL FAIR COMPARISON"
)

print(
    "=" * 100
)

summary_columns = [
    "model",
    "horizon",
    "simple_logistic_test_auc",
    "extended_logistic_test_auc",
    "xgb_test_auc",
    "feature_gain_extended_minus_simple",
    "nonlinear_gain_xgb_minus_extended",
    "total_xgb_gain",
    "simple_macro_ticker_auc",
    "extended_macro_ticker_auc",
    "xgb_macro_ticker_auc",
    "fraction_tickers_xgb_gt_extended",
]

print(
    summary[
        summary_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# =============================================================================
# SUBPERIOD RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "EARLY / LATE TEST ROBUSTNESS"
)

print(
    "=" * 100
)

if len(subperiod) > 0:

    print(
        subperiod.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:

    print(
        "No subperiod results."
    )


# =============================================================================
# TICKER-LEVEL RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "TICKER-LEVEL AGGREGATION"
)

print(
    "=" * 100
)

if len(per_ticker) > 0:

    ticker_summary = (
        per_ticker
        .groupby(
            [
                "model",
                "horizon",
            ]
        )
        .agg(
            ticker_count=(
                "ticker",
                "count",
            ),

            simple_mean=(
                "simple_logistic_auc",
                "mean",
            ),

            simple_median=(
                "simple_logistic_auc",
                "median",
            ),

            extended_mean=(
                "extended_logistic_auc",
                "mean",
            ),

            extended_median=(
                "extended_logistic_auc",
                "median",
            ),

            xgb_mean=(
                "xgb_auc",
                "mean",
            ),

            xgb_median=(
                "xgb_auc",
                "median",
            ),

            feature_gain_mean=(
                "feature_gain",
                "mean",
            ),

            nonlinear_gain_mean=(
                "nonlinear_gain",
                "mean",
            ),

            total_xgb_gain_mean=(
                "total_xgb_gain",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        ticker_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

else:

    print(
        "No ticker-level results."
    )


# =============================================================================
# SAVE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "SAVING RESULTS"
)

print(
    "=" * 100
)

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

per_ticker.to_csv(
    PER_TICKER_PATH,
    index=False,
)

subperiod.to_csv(
    SUBPERIOD_PATH,
    index=False,
)

print(
    f"\nSummary:\n{SUMMARY_PATH}"
)

print(
    f"\nPer ticker:\n{PER_TICKER_PATH}"
)

print(
    f"\nSubperiod:\n{SUBPERIOD_PATH}"
)


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FAIR LOGISTIC VS XGBOOST EXPERIMENT COMPLETE"
)

print(
    "=" * 100
)