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

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =============================================================================
# FINAL STATISTICAL INFERENCE
# =============================================================================
#
# This is the LAST empirical experiment.
#
# Exact model specifications are frozen from the previous fair comparison.
#
# Model 1:
#   19 simple features + ticker fixed effects
#   regularized logistic regression
#
# Model 2:
#   23 total features + ticker fixed effects
#   regularized logistic regression
#
# Model 3:
#   SAME 23 features + ticker fixed effects
#   XGBoost
#
# The purpose is inference:
#
#   extended - simple
#   XGB - extended
#   XGB - simple
#
# using:
#
#   - full test sample
#   - non-overlapping test sample
#   - block bootstrap
#   - per-ticker paired comparisons
#   - early/late robustness
#
# Multiple comparisons:
#   Holm correction across the 12 model-horizon XGB-vs-simple comparisons.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "monitoring"

PANEL_PATH = (
    DATA_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

PREDICTIONS_PATH = (
    DATA_DIR
    / "final_statistical_inference_predictions.csv"
)

SUMMARY_PATH = (
    DATA_DIR
    / "final_statistical_inference_summary.csv"
)

BOOTSTRAP_PATH = (
    DATA_DIR
    / "final_statistical_inference_bootstrap.csv"
)

TICKER_PATH = (
    DATA_DIR
    / "final_statistical_inference_per_ticker.csv"
)

SUBPERIOD_PATH = (
    DATA_DIR
    / "final_statistical_inference_subperiod.csv"
)


# =============================================================================
# CONFIG
# =============================================================================

SEED = 42

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

TRAIN_FRAC = 0.60
VALIDATION_FRAC = 0.20

BOOTSTRAP_REPS = 500

BLOCK_LENGTHS = [
    10,
    20,
    40,
    60,
]


# =============================================================================
# FROZEN SPECIFICATIONS FROM PREVIOUS FAIR EXPERIMENT
# =============================================================================

LOGISTIC_C = {

    ("qrf", 5): 0.01,
    ("qrf", 10): 10.0,
    ("qrf", 20): 10.0,

    ("ewma", 5): 1.0,
    ("ewma", 10): 10.0,
    ("ewma", 20): 10.0,

    ("garch", 5): 10.0,
    ("garch", 10): 10.0,
    ("garch", 20): 10.0,

    ("hs", 5): 0.01,
    ("hs", 10): 0.01,
    ("hs", 20): 10.0,
}


XGB_CONFIG = {

    ("qrf", 5): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("qrf", 10): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("qrf", 20): {
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 250,
    },

    ("ewma", 5): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("ewma", 10): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("ewma", 20): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("garch", 5): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("garch", 10): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("garch", 20): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("hs", 5): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },

    ("hs", 10): {
        "max_depth": 3,
        "learning_rate": 0.03,
        "n_estimators": 300,
    },

    ("hs", 20): {
        "max_depth": 2,
        "learning_rate": 0.03,
        "n_estimators": 250,
    },
}


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
# FEATURES
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
# HELPERS
# =============================================================================

def safe_auc(
    y,
    score,
):

    y = np.asarray(y)

    score = np.asarray(
        score,
        dtype=float,
    )

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


def add_splits(
    df,
):

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

        ticker_labels = np.array(
            ["test"] * n,
            dtype=object,
        )

        ticker_labels[
            :train_end
        ] = "train"

        ticker_labels[
            train_end:
            validation_end
        ] = "validation"

        labels.extend(
            ticker_labels.tolist()
        )

    df["split"] = labels

    return df


def make_logistic_pipeline(
    features,
    C,
):

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
                features,
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
    features,
):

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
                features,
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

    return XGBClassifier(
        **XGB_FIXED,
        **config,
    )


def nonoverlap_mask(
    df,
    horizon,
):

    mask = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    for (
        ticker,
        split,
    ), group in df.groupby(
        [
            "ticker",
            "split",
        ],
        sort=False,
    ):

        indices = group.index.to_numpy()

        selected = indices[
            ::horizon
        ]

        mask.loc[
            selected
        ] = True

    return mask


def block_bootstrap_difference(
    evaluation,
    horizon,
    score_a,
    score_b,
    repetitions=500,
    block_length=20,
    seed=42,
):
    """
    Paired block bootstrap.

    Computes:
        AUC(score_b) - AUC(score_a)

    Blocks are resampled independently within ticker,
    preserving temporal dependence within blocks.
    """

    rng = np.random.default_rng(
        seed
    )

    groups = []

    for ticker, group in evaluation.groupby(
        "ticker",
        sort=True,
    ):

        group = (
            group
            .sort_values(
                "forecast_date"
            )
            .reset_index(drop=True)
        )

        n = len(group)

        if n == 0:
            continue

        blocks = []

        for start in range(
            0,
            n,
            block_length,
        ):

            end = min(
                start + block_length,
                n,
            )

            blocks.append(
                group.iloc[
                    start:end
                ]
            )

        groups.append(
            blocks
        )

    deltas = []

    for _ in range(
        repetitions
    ):

        sampled_groups = []

        for blocks in groups:

            n_blocks = len(blocks)

            selected = rng.integers(
                0,
                n_blocks,
                size=n_blocks,
            )

            sampled = pd.concat(
                [
                    blocks[i]
                    for i in selected
                ],
                ignore_index=True,
            )

            sampled_groups.append(
                sampled
            )

        bootstrap = pd.concat(
            sampled_groups,
            ignore_index=True,
        )

        y = (
            bootstrap["target"]
            .to_numpy()
        )

        a = (
            bootstrap[score_a]
            .to_numpy(
                dtype=float
            )
        )

        b = (
            bootstrap[score_b]
            .to_numpy(
                dtype=float
            )
        )

        auc_a = safe_auc(
            y,
            a,
        )

        auc_b = safe_auc(
            y,
            b,
        )

        if (
            np.isfinite(auc_a)
            and
            np.isfinite(auc_b)
        ):

            deltas.append(
                auc_b - auc_a
            )

    if len(deltas) == 0:

        return {
            "mean_delta": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_bootstrap": np.nan,
            "n_success": 0,
        }

    deltas = np.asarray(
        deltas,
        dtype=float,
    )

    mean_delta = float(
        np.mean(deltas)
    )

    ci_lower = float(
        np.quantile(
            deltas,
            0.025,
        )
    )

    ci_upper = float(
        np.quantile(
            deltas,
            0.975,
        )
    )

    # Approximate two-sided bootstrap probability.
    p_left = np.mean(
        deltas <= 0
    )

    p_right = np.mean(
        deltas >= 0
    )

    p_bootstrap = min(
        1.0,
        2.0 * min(
            p_left,
            p_right,
        ),
    )

    return {
        "mean_delta":
            mean_delta,

        "ci_lower":
            ci_lower,

        "ci_upper":
            ci_upper,

        "p_bootstrap":
            float(p_bootstrap),

        "n_success":
            len(deltas),
    }


def holm_adjust(
    p_values,
):
    """
    Holm-Bonferroni adjustment.
    """

    p = np.asarray(
        p_values,
        dtype=float,
    )

    result = np.full(
        len(p),
        np.nan,
    )

    valid = np.isfinite(p)

    indices = np.where(
        valid
    )[0]

    if len(indices) == 0:
        return result

    order = indices[
        np.argsort(
            p[indices]
        )
    ]

    m = len(order)

    running_max = 0.0

    for rank, idx in enumerate(
        order
    ):

        adjusted = (
            m - rank
        ) * p[idx]

        adjusted = min(
            1.0,
            adjusted,
        )

        running_max = max(
            running_max,
            adjusted,
        )

        result[idx] = running_max

    return result


def ticker_results(
    evaluation,
    target_col,
):

    rows = []

    for ticker, group in evaluation.groupby(
        "ticker",
        sort=True,
    ):

        if group[
            target_col
        ].nunique() < 2:

            continue

        simple_auc = safe_auc(
            group[target_col],
            group["simple"],
        )

        extended_auc = safe_auc(
            group[target_col],
            group["extended"],
        )

        xgb_auc = safe_auc(
            group[target_col],
            group["xgb"],
        )

        if not all(
            np.isfinite(x)
            for x in [
                simple_auc,
                extended_auc,
                xgb_auc,
            ]
        ):

            continue

        rows.append(
            {
                "ticker":
                    ticker,

                "simple_auc":
                    simple_auc,

                "extended_auc":
                    extended_auc,

                "xgb_auc":
                    xgb_auc,

                "feature_gain":
                    extended_auc
                    -
                    simple_auc,

                "nonlinear_gain":
                    xgb_auc
                    -
                    extended_auc,

                "total_xgb_gain":
                    xgb_auc
                    -
                    simple_auc,

                "n":
                    len(group),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# LOAD
# =============================================================================

print(
    "=" * 100
)

print(
    "FINAL STATISTICAL INFERENCE"
)

print(
    "=" * 100
)

if not PANEL_PATH.exists():

    raise FileNotFoundError(
        PANEL_PATH
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
    df["ticker"].isin(
        TICKERS
    )
].copy()

df = add_splits(
    df
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Tickers: {df['ticker'].nunique()}"
)


# =============================================================================
# VERIFY FEATURES
# =============================================================================

for model in MODELS:

    simple_features = [
        f"{model}_{name}"
        for name in SIMPLE_FEATURE_NAMES
    ]

    extended_features = (
        simple_features
        +
        COMPLEX_FEATURE_NAMES
    )

    if len(simple_features) != 19:

        raise RuntimeError(
            "Simple feature set should contain "
            "19 features."
        )

    if len(extended_features) != 23:

        raise RuntimeError(
            "Extended feature set should contain "
            "23 features."
        )

    for column in extended_features:

        if column not in df.columns:

            raise RuntimeError(
                f"Missing feature: {column}"
            )


# =============================================================================
# RESULT STORAGE
# =============================================================================

summary_rows = []
bootstrap_rows = []
ticker_rows = []
subperiod_rows = []
prediction_frames = []


# =============================================================================
# MAIN LOOP
# =============================================================================

for model in MODELS:

    simple_features = [
        f"{model}_{name}"
        for name in SIMPLE_FEATURE_NAMES
    ]

    extended_features = (
        simple_features
        +
        COMPLEX_FEATURE_NAMES
    )

    for horizon in HORIZONS:

        print(
            "\n" + "-" * 100
        )

        print(
            f"{model.upper()} | {horizon}D"
        )

        print(
            "-" * 100
        )

        target_col = (
            f"{model}_deterioration_{horizon}d"
        )

        # ---------------------------------------------------------------------
        # SPLITS
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

        train_validation = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        # =====================================================================
        # SIMPLE LOGISTIC
        # =====================================================================

        simple_C = LOGISTIC_C[
            (model, horizon)
        ]

        simple_model = make_logistic_pipeline(
            simple_features,
            simple_C,
        )

        simple_model.fit(
            train_validation[
                simple_features
                + ["ticker"]
            ],
            train_validation[
                target_col
            ].astype(int),
        )

        simple_score = (
            simple_model
            .predict_proba(
                test[
                    simple_features
                    + ["ticker"]
                ]
            )[:, 1]
        )

        # =====================================================================
        # EXTENDED LOGISTIC
        # =====================================================================

        # The previous fair experiment selected its extended C.
        #
        # From the previous output:
        #
        # QRF:   5=0.01, 10=10, 20=10
        # EWMA:  5=0.01, 10=10, 20=10
        # GARCH: 5=10, 10=10, 20=10
        # HS:    5=0.01, 10=0.01, 20=10
        #
        # Use exactly the same C as the corresponding simple model,
        # except where the previous result explicitly differs.
        #
        # This map is stored separately below.
        # ---------------------------------------------------------------------

        extended_C_map = {

            ("qrf", 5): 0.01,
            ("qrf", 10): 10.0,
            ("qrf", 20): 10.0,

            ("ewma", 5): 0.01,
            ("ewma", 10): 10.0,
            ("ewma", 20): 10.0,

            ("garch", 5): 10.0,
            ("garch", 10): 10.0,
            ("garch", 20): 10.0,

            ("hs", 5): 0.01,
            ("hs", 10): 0.01,
            ("hs", 20): 10.0,
        }

        extended_C = extended_C_map[
            (model, horizon)
        ]

        extended_model = make_logistic_pipeline(
            extended_features,
            extended_C,
        )

        extended_model.fit(
            train_validation[
                extended_features
                + ["ticker"]
            ],
            train_validation[
                target_col
            ].astype(int),
        )

        extended_score = (
            extended_model
            .predict_proba(
                test[
                    extended_features
                    + ["ticker"]
                ]
            )[:, 1]
        )

        # =====================================================================
        # XGB
        # =====================================================================

        xgb_transformer = (
            make_xgb_transformer(
                extended_features
            )
        )

        X_train_validation = (
            xgb_transformer
            .fit_transform(
                train_validation[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        X_test = (
            xgb_transformer
            .transform(
                test[
                    extended_features
                    + ["ticker"]
                ]
            )
        )

        y_train_validation = (
            train_validation[
                target_col
            ]
            .astype(int)
            .to_numpy()
        )

        xgb = make_xgb(
            XGB_CONFIG[
                (model, horizon)
            ]
        )

        xgb.fit(
            X_train_validation,
            y_train_validation,
        )

        xgb_score = (
            xgb
            .predict_proba(
                X_test
            )[:, 1]
        )

        # =====================================================================
        # EVALUATION DATASET
        # =====================================================================

        evaluation = test[
            [
                "ticker",
                "forecast_date",
                target_col,
            ]
        ].copy()

        evaluation = evaluation.rename(
            columns={
                target_col:
                    "target"
            }
        )

        evaluation["simple"] = (
            simple_score
        )

        evaluation["extended"] = (
            extended_score
        )

        evaluation["xgb"] = (
            xgb_score
        )

        evaluation["model"] = (
            model.upper()
        )

        evaluation["horizon"] = (
            horizon
        )

        prediction_frames.append(
            evaluation.copy()
        )

        # =====================================================================
        # POOLED AUC
        # =====================================================================

        simple_auc = safe_auc(
            evaluation["target"],
            evaluation["simple"],
        )

        extended_auc = safe_auc(
            evaluation["target"],
            evaluation["extended"],
        )

        xgb_auc = safe_auc(
            evaluation["target"],
            evaluation["xgb"],
        )

        # =====================================================================
        # NON-OVERLAPPING TEST
        # =====================================================================

        no_mask = nonoverlap_mask(
            test,
            horizon,
        )

        test_no = test[
            no_mask
        ].copy()

        test_no["target"] = (
            test_no[target_col]
        )

        # Scores need matching test indices.
        score_frame = pd.DataFrame(
            {
                "index":
                    test.index,

                "simple":
                    simple_score,

                "extended":
                    extended_score,

                "xgb":
                    xgb_score,
            }
        )

        score_frame = score_frame.set_index(
            "index"
        )

        test_no = test_no.join(
            score_frame,
            how="inner",
        )

        no_simple_auc = safe_auc(
            test_no["target"],
            test_no["simple"],
        )

        no_extended_auc = safe_auc(
            test_no["target"],
            test_no["extended"],
        )

        no_xgb_auc = safe_auc(
            test_no["target"],
            test_no["xgb"],
        )

        # =====================================================================
        # PER-TICKER RESULTS
        # =====================================================================

        ticker_eval = ticker_results(
            evaluation,
            "target",
        )

        for _, row in ticker_eval.iterrows():

            ticker_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "ticker":
                        row["ticker"],

                    "simple_auc":
                        row["simple_auc"],

                    "extended_auc":
                        row["extended_auc"],

                    "xgb_auc":
                        row["xgb_auc"],

                    "feature_gain":
                        row["feature_gain"],

                    "nonlinear_gain":
                        row["nonlinear_gain"],

                    "total_xgb_gain":
                        row["total_xgb_gain"],

                    "n":
                        row["n"],
                }
            )

        # =====================================================================
        # SUBPERIOD
        # =====================================================================

        midpoint = (
            evaluation[
                "forecast_date"
            ].median()
        )

        for period_name, condition in [

            (
                "early_test",
                evaluation[
                    "forecast_date"
                ]
                <= midpoint,
            ),

            (
                "late_test",
                evaluation[
                    "forecast_date"
                ]
                > midpoint,
            ),
        ]:

            period = evaluation[
                condition
            ].copy()

            sub_simple = safe_auc(
                period["target"],
                period["simple"],
            )

            sub_extended = safe_auc(
                period["target"],
                period["extended"],
            )

            sub_xgb = safe_auc(
                period["target"],
                period["xgb"],
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
                        len(period),

                    "positive_rate":
                        float(
                            period["target"]
                            .mean()
                        ),

                    "simple_auc":
                        sub_simple,

                    "extended_auc":
                        sub_extended,

                    "xgb_auc":
                        sub_xgb,

                    "feature_gain":
                        sub_extended
                        -
                        sub_simple,

                    "nonlinear_gain":
                        sub_xgb
                        -
                        sub_extended,

                    "total_xgb_gain":
                        sub_xgb
                        -
                        sub_simple,
                }
            )

        # =====================================================================
        # BLOCK BOOTSTRAP
        # =====================================================================

        print(
            "  Running block bootstrap..."
        )

        for block_length in BLOCK_LENGTHS:

            xgb_boot = (
                block_bootstrap_difference(
                    evaluation,
                    horizon,
                    "simple",
                    "xgb",
                    repetitions=BOOTSTRAP_REPS,
                    block_length=block_length,
                    seed=(
                        SEED
                        + horizon
                        + block_length
                        + len(model)
                    ),
                )
            )

            extended_boot = (
                block_bootstrap_difference(
                    evaluation,
                    horizon,
                    "simple",
                    "extended",
                    repetitions=BOOTSTRAP_REPS,
                    block_length=block_length,
                    seed=(
                        SEED
                        + 100
                        + horizon
                        + block_length
                        + len(model)
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

                    "comparison":
                        "xgb_minus_simple",

                    **xgb_boot,
                }
            )

            bootstrap_rows.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "block_length":
                        block_length,

                    "comparison":
                        "extended_minus_simple",

                    **extended_boot,
                }
            )

        # =====================================================================
        # SUMMARY
        # =====================================================================

        valid_ticker = ticker_eval.dropna(
            subset=[
                "simple_auc",
                "extended_auc",
                "xgb_auc",
            ]
        )

        if len(valid_ticker) > 0:

            simple_macro = (
                valid_ticker[
                    "simple_auc"
                ].mean()
            )

            extended_macro = (
                valid_ticker[
                    "extended_auc"
                ].mean()
            )

            xgb_macro = (
                valid_ticker[
                    "xgb_auc"
                ].mean()
            )

            simple_median = (
                valid_ticker[
                    "simple_auc"
                ].median()
            )

            extended_median = (
                valid_ticker[
                    "extended_auc"
                ].median()
            )

            xgb_median = (
                valid_ticker[
                    "xgb_auc"
                ].median()
            )

            xgb_better = (
                (
                    valid_ticker[
                        "xgb_auc"
                    ]
                    >
                    valid_ticker[
                        "simple_auc"
                    ]
                ).mean()
            )

            extended_better = (
                (
                    valid_ticker[
                        "extended_auc"
                    ]
                    >
                    valid_ticker[
                        "simple_auc"
                    ]
                ).mean()
            )

        else:

            simple_macro = np.nan
            extended_macro = np.nan
            xgb_macro = np.nan

            simple_median = np.nan
            extended_median = np.nan
            xgb_median = np.nan

            xgb_better = np.nan
            extended_better = np.nan

        summary_rows.append(
            {
                "model":
                    model.upper(),

                "horizon":
                    horizon,

                "simple_auc":
                    simple_auc,

                "extended_auc":
                    extended_auc,

                "xgb_auc":
                    xgb_auc,

                "feature_gain":
                    extended_auc
                    -
                    simple_auc,

                "nonlinear_gain":
                    xgb_auc
                    -
                    extended_auc,

                "total_xgb_gain":
                    xgb_auc
                    -
                    simple_auc,

                "nonoverlap_simple_auc":
                    no_simple_auc,

                "nonoverlap_extended_auc":
                    no_extended_auc,

                "nonoverlap_xgb_auc":
                    no_xgb_auc,

                "simple_macro_auc":
                    simple_macro,

                "extended_macro_auc":
                    extended_macro,

                "xgb_macro_auc":
                    xgb_macro,

                "simple_median_auc":
                    simple_median,

                "extended_median_auc":
                    extended_median,

                "xgb_median_auc":
                    xgb_median,

                "fraction_xgb_gt_simple":
                    xgb_better,

                "fraction_extended_gt_simple":
                    extended_better,

                "test_positive_rate":
                    float(
                        evaluation[
                            "target"
                        ].mean()
                    ),
            }
        )

        print(
            f"  Test:"
            f" simple={simple_auc:.4f}"
            f" extended={extended_auc:.4f}"
            f" xgb={xgb_auc:.4f}"
        )

        print(
            f"  Gains:"
            f" features={extended_auc - simple_auc:+.4f}"
            f" nonlinear={xgb_auc - extended_auc:+.4f}"
            f" total={xgb_auc - simple_auc:+.4f}"
        )

        print(
            f"  Non-overlap:"
            f" simple={no_simple_auc:.4f}"
            f" extended={no_extended_auc:.4f}"
            f" xgb={no_xgb_auc:.4f}"
        )


# =============================================================================
# DATAFRAMES
# =============================================================================

summary = pd.DataFrame(
    summary_rows
)

bootstrap = pd.DataFrame(
    bootstrap_rows
)

per_ticker = pd.DataFrame(
    ticker_rows
)

subperiod = pd.DataFrame(
    subperiod_rows
)

predictions = pd.concat(
    prediction_frames,
    ignore_index=True,
)


# =============================================================================
# HOLM MULTIPLE COMPARISON ADJUSTMENT
# =============================================================================
#
# Use the 10-day block bootstrap results as the primary dependence setting.
#
# We adjust the 12 XGB-vs-simple comparisons together.
# =============================================================================

primary_bootstrap = bootstrap[
    (
        bootstrap["comparison"]
        ==
        "xgb_minus_simple"
    )
    &
    (
        bootstrap["block_length"]
        ==
        10
    )
].copy()

primary_bootstrap = (
    primary_bootstrap
    .sort_values(
        [
            "model",
            "horizon",
        ]
    )
    .reset_index(drop=True)
)

adjusted_p = holm_adjust(
    primary_bootstrap[
        "p_bootstrap"
    ].to_numpy()
)

primary_bootstrap[
    "holm_adjusted_p"
] = adjusted_p

primary_bootstrap[
    "holm_significant_0p05"
] = (
    primary_bootstrap[
        "holm_adjusted_p"
    ]
    <
    0.05
)


# Merge the multiplicity-adjusted primary results into summary.
summary = summary.merge(
    primary_bootstrap[
        [
            "model",
            "horizon",
            "p_bootstrap",
            "holm_adjusted_p",
            "holm_significant_0p05",
        ]
    ],
    on=[
        "model",
        "horizon",
    ],
    how="left",
    validate="one_to_one",
)


# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "FINAL STATISTICAL SUMMARY"
)

print(
    "=" * 100
)

print(
    summary[
        [
            "model",
            "horizon",
            "simple_auc",
            "extended_auc",
            "xgb_auc",
            "feature_gain",
            "nonlinear_gain",
            "total_xgb_gain",
            "nonoverlap_simple_auc",
            "nonoverlap_xgb_auc",
            "fraction_xgb_gt_simple",
            "p_bootstrap",
            "holm_adjusted_p",
            "holm_significant_0p05",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# PRIMARY BOOTSTRAP RESULTS
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "PRIMARY XGB - SIMPLE BLOCK BOOTSTRAP"
)

print(
    "=" * 100
)

print(
    primary_bootstrap[
        [
            "model",
            "horizon",
            "block_length",
            "mean_delta",
            "ci_lower",
            "ci_upper",
            "p_bootstrap",
            "holm_adjusted_p",
            "holm_significant_0p05",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# EXTENDED FEATURE EFFECT
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "ADDITIONAL FEATURE EFFECT: EXTENDED - SIMPLE"
)

print(
    "=" * 100
)

feature_bootstrap = bootstrap[
    (
        bootstrap["comparison"]
        ==
        "extended_minus_simple"
    )
    &
    (
        bootstrap["block_length"]
        ==
        10
    )
].copy()

print(
    feature_bootstrap[
        [
            "model",
            "horizon",
            "mean_delta",
            "ci_lower",
            "ci_upper",
            "p_bootstrap",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# TICKER-LEVEL FINAL SUMMARY
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "PER-TICKER SUMMARY"
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
                "simple_auc",
                "mean",
            ),

            simple_median=(
                "simple_auc",
                "median",
            ),

            extended_mean=(
                "extended_auc",
                "mean",
            ),

            extended_median=(
                "extended_auc",
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
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# =============================================================================
# EARLY / LATE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "EARLY / LATE ROBUSTNESS"
)

print(
    "=" * 100
)

print(
    subperiod.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# SAVE
# =============================================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

predictions.to_csv(
    PREDICTIONS_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

bootstrap.to_csv(
    BOOTSTRAP_PATH,
    index=False,
)

per_ticker.to_csv(
    TICKER_PATH,
    index=False,
)

subperiod.to_csv(
    SUBPERIOD_PATH,
    index=False,
)


# =============================================================================
# FINAL COUNTS
# =============================================================================

robust_positive = (
    primary_bootstrap[
        "ci_lower"
    ]
    > 0
).sum()

robust_negative = (
    primary_bootstrap[
        "ci_upper"
    ]
    < 0
).sum()

holm_positive = (
    primary_bootstrap[
        "holm_significant_0p05"
    ]
    .sum()
)

print(
    "\n" + "=" * 100
)

print(
    "FINAL INFERENCE COUNTS"
)

print(
    "=" * 100
)

print(
    f"XGB-vs-simple comparisons: "
    f"{len(primary_bootstrap)}"
)

print(
    f"10-day-block CI entirely > 0: "
    f"{robust_positive}"
)

print(
    f"10-day-block CI entirely < 0: "
    f"{robust_negative}"
)

print(
    f"Holm-adjusted significant XGB gains: "
    f"{holm_positive}"
)

print(
    "\nSaved:"
)

print(
    PREDICTIONS_PATH
)

print(
    SUMMARY_PATH
)

print(
    BOOTSTRAP_PATH
)

print(
    TICKER_PATH
)

print(
    SUBPERIOD_PATH
)


print(
    "\n" + "=" * 100
)

print(
    "FINAL STATISTICAL INFERENCE COMPLETE"
)

print(
    "=" * 100
)