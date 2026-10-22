# ============================================================
# FINAL DECOMPOSITION AUDIT
# ============================================================
#
# Purpose
# -------
# Determine how much of the apparent pooled early-warning skill
# comes from:
#
#   1. Ticker identity / persistent ticker-level deterioration
#   2. Common market-stress / date effects
#   3. Genuine within-ticker discrimination
#
# This is an AUDIT.
#
# It does NOT search for new features or new models.
#
# Models:
#   QRF, EWMA, GARCH, HS
#
# Horizons:
#   5d, 10d, 20d
#
# Split:
#   60% train / 20% validation / 20% test
#   within each ticker
#
# Simple monitor:
#   19 causal/interpretable monitoring features
#   + ticker fixed effects
#   + L2 logistic regression
#
# C is selected on validation only.
#
# Main outputs:
#   - Simple pooled AUC
#   - Simple centered pooled AUC
#   - Ticker-identity baseline AUC
#   - Market-stress baseline AUC
#   - Macro per-ticker AUC
#   - Median per-ticker AUC
#   - Fraction of tickers with AUC > 0.50
#   - Per-ticker AUC
#   - Date-block bootstrap confidence intervals
#
# ============================================================

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\reliability_layer")

PANEL_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "decomposition_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

MODELS = [
    "QRF",
    "EWMA",
    "GARCH",
    "HS",
]

HORIZONS = [
    5,
    10,
    20,
]

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

RANDOM_SEED = 42

N_BOOTSTRAPS = 500
BLOCK_LENGTH = 10

# Regularization candidates.
# Selected using validation only.
C_GRID = [
    0.01,
    0.1,
    1.0,
    10.0,
]


# ============================================================
# EXACT FINAL 19 SIMPLE FEATURES
# ============================================================
#
# IMPORTANT:
# The 19th historical-loss feature is:
#
#     past_mean_pinball
#
# not pinball_expanding_mean.
#
# The actual panel contains:
#
#     qrf_past_mean_pinball
#     ewma_past_mean_pinball
#     garch_past_mean_pinball
#     hs_past_mean_pinball
#
# ============================================================

SIMPLE_FEATURE_BASES = [

    # --------------------------------------------------------
    # VaR dynamics
    # --------------------------------------------------------

    "var_abs_current",
    "var_change_1d",
    "var_ratio_to_prior",
    "var_ratio_to_20d_mean",

    # --------------------------------------------------------
    # Historical pinball-loss information
    # --------------------------------------------------------

    "pinball_lag1",
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
    "pinball_mean_60d_prior",
    "pinball_acceleration",
    "past_mean_pinball",

    # --------------------------------------------------------
    # Breach history
    # --------------------------------------------------------

    "breach_lag1",
    "breach_streak_prior",
    "days_since_last_breach",
    "breach_rate_5d_prior",
    "breach_rate_20d_prior",
    "breach_rate_60d_prior",

    # --------------------------------------------------------
    # Volatility / market state
    # --------------------------------------------------------

    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",
]


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_name(value):
    """
    Normalize a column name for robust matching.
    """
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).lower(),
    ).strip("_")


# ============================================================
# MODEL FEATURE RESOLUTION
# ============================================================

def find_model_column(
    df_columns,
    model,
    base_name,
):
    """
    Resolve model-specific feature names.

    Example:
        QRF + var_abs_current

    can resolve to:
        qrf_var_abs_current
    """

    columns = list(df_columns)

    model_l = model.lower()
    model_u = model.upper()

    candidates = [

        f"{model_l}_{base_name}",
        f"{model_u}_{base_name}",

        f"{base_name}_{model_l}",
        f"{base_name}_{model_u}",

        base_name,
    ]

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate in columns:
            return candidate

    # --------------------------------------------------------
    # Normalized exact match
    # --------------------------------------------------------

    normalized_map = {
        normalize_name(c): c
        for c in columns
    }

    for candidate in candidates:

        normalized_candidate = normalize_name(
            candidate
        )

        if normalized_candidate in normalized_map:
            return normalized_map[
                normalized_candidate
            ]

    # --------------------------------------------------------
    # Flexible model + feature search
    # --------------------------------------------------------

    model_token = normalize_name(model)
    base_token = normalize_name(base_name)

    matches = []

    for column in columns:

        normalized_column = normalize_name(
            column
        )

        if (
            model_token in normalized_column
            and base_token in normalized_column
        ):
            matches.append(column)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:

        print(
            f"WARNING: Multiple matches for "
            f"{model} / {base_name}:"
        )

        for match in matches:
            print(f"    {match}")

    return None


# ============================================================
# TARGET RESOLUTION
# ============================================================

def find_target_column(
    df_columns,
    model,
    horizon,
):
    """
    Resolve the FINAL primary deterioration target.

    Final panel naming convention:

        qrf_deterioration_5d
        qrf_deterioration_10d
        qrf_deterioration_20d

        ewma_deterioration_5d
        ...

        garch_deterioration_5d
        ...

        hs_deterioration_5d
        ...

    IMPORTANT:
    Do not select:
        *_severe_deterioration_*
        *_deterioration_fixed_1p5_*
    """

    columns = list(df_columns)

    model_l = model.lower()

    target = (
        f"{model_l}_deterioration_{horizon}d"
    )

    # Exact final-panel naming convention
    if target in columns:
        return target

    # Normalized fallback
    normalized_map = {
        normalize_name(c): c
        for c in columns
    }

    normalized_target = normalize_name(
        target
    )

    if normalized_target in normalized_map:
        return normalized_map[
            normalized_target
        ]

    raise RuntimeError(
        f"Could not resolve final primary "
        f"deterioration target: {target}"
    )

# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def split_ticker(
    df,
):
    """
    Chronological 60/20/20 split for one ticker.
    """

    g = (
        df
        .sort_values("forecast_date")
        .reset_index(drop=True)
        .copy()
    )

    n = len(g)

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

    g["split"] = "test"

    g.loc[
        :n_train - 1,
        "split"
    ] = "train"

    g.loc[
        n_train:n_train + n_val - 1,
        "split"
    ] = "validation"

    return g


# ============================================================
# SAFE AUC
# ============================================================

def safe_auc(
    y_true,
    score,
):
    """
    Return NaN if AUC cannot be calculated.
    """

    y_true = np.asarray(
        y_true
    )

    score = np.asarray(
        score
    )

    mask = (
        pd.notna(y_true)
        & np.isfinite(score)
    )

    y_true = y_true[mask]
    score = score[mask]

    if len(y_true) == 0:
        return np.nan

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        score,
    )


# ============================================================
# PER-TICKER AUC
# ============================================================

def calculate_per_ticker_auc(
    frame,
    target_col,
    score_col,
):
    """
    AUC separately within each ticker.
    """

    rows = []

    for ticker, g in frame.groupby(
        "ticker",
        sort=True,
    ):

        auc = safe_auc(
            g[target_col],
            g[score_col],
        )

        rows.append(
            {
                "ticker": ticker,
                "auc": auc,
                "n": len(g),
                "positive_rate": (
                    g[target_col].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PER-TICKER SUMMARY
# ============================================================

def summarize_per_ticker(
    per_ticker,
):
    """
    Macro / median / coverage.
    """

    aucs = (
        per_ticker["auc"]
        .dropna()
    )

    if len(aucs) == 0:

        return {
            "macro_auc": np.nan,
            "median_auc": np.nan,
            "fraction_auc_gt_050": np.nan,
            "n_valid": 0,
        }

    return {
        "macro_auc": aucs.mean(),
        "median_auc": aucs.median(),
        "fraction_auc_gt_050": (
            aucs > 0.50
        ).mean(),
        "n_valid": len(aucs),
    }


# ============================================================
# MARKET-STRESS BASELINE
# ============================================================

def build_market_stress_state(
    work,
    train_dates,
):
    """
    Build a common date-level stress score using only
    information available at the forecast date.

    Components:

        median realized_vol_20d
        median vol_ratio_20_60

    Standardization parameters are calculated from
    TRAINING DATES ONLY.
    """

    date_state = (
        work
        .groupby("forecast_date")
        .agg(
            market_median_vol20=(
                "market_vol20_source",
                "median",
            ),

            market_median_vol_ratio=(
                "market_vol_ratio_source",
                "median",
            ),
        )
        .reset_index()
    )

    date_state[
        "forecast_date"
    ] = pd.to_datetime(
        date_state["forecast_date"]
    )

    train_dates = pd.to_datetime(
        train_dates
    )

    train_state = date_state[
        date_state["forecast_date"].isin(
            train_dates
        )
    ].copy()

    if train_state.empty:
        raise RuntimeError(
            "No training dates available "
            "for market-stress calibration."
        )

    # --------------------------------------------------------
    # Training-only moments
    # --------------------------------------------------------

    vol_mean = (
        train_state[
            "market_median_vol20"
        ]
        .mean()
    )

    vol_std = (
        train_state[
            "market_median_vol20"
        ]
        .std()
    )

    ratio_mean = (
        train_state[
            "market_median_vol_ratio"
        ]
        .mean()
    )

    ratio_std = (
        train_state[
            "market_median_vol_ratio"
        ]
        .std()
    )

    if (
        not np.isfinite(vol_std)
        or vol_std == 0
    ):
        vol_std = 1.0

    if (
        not np.isfinite(ratio_std)
        or ratio_std == 0
    ):
        ratio_std = 1.0

    # --------------------------------------------------------
    # Ex-ante stress score
    # --------------------------------------------------------

    date_state[
        "market_stress_score"
    ] = (

        (
            date_state[
                "market_median_vol20"
            ]
            - vol_mean
        )
        / vol_std

        +

        (
            date_state[
                "market_median_vol_ratio"
            ]
            - ratio_mean
        )
        / ratio_std
    )

    return date_state


# ============================================================
# LOGISTIC SIMPLE MONITOR
# ============================================================

def fit_simple_monitor(
    train,
    validation,
    feature_cols,
):
    """
    Train regularized logistic model.

    Input:
        19 simple causal features
        + ticker fixed effect

    C chosen on VALIDATION ONLY.

    Final model refit:
        TRAIN + VALIDATION
    """

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ticker fixed effects
    # --------------------------------------------------------

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
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # --------------------------------------------------------
    # Combined design matrix
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_cols,
            ),

            (
                "ticker",
                ticker_pipeline,
                ["ticker"],
            ),
        ]
    )

    best_auc = -np.inf
    best_c = None

    X_train = train[
        feature_cols
        + ["ticker"]
    ]

    y_train = (
        train["target"]
        .astype(int)
    )

    X_val = validation[
        feature_cols
        + ["ticker"]
    ]

    y_val = (
        validation["target"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Validation selection
    # --------------------------------------------------------

    for c_value in C_GRID:

        model = Pipeline(
            steps=[

                (
                    "preprocess",
                    preprocessor,
                ),

                (
                    "logistic",
                    LogisticRegression(
                        C=c_value,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=4000,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        model.fit(
            X_train,
            y_train,
        )

        validation_score = (
            model
            .predict_proba(X_val)[:, 1]
        )

        validation_auc = safe_auc(
            y_val,
            validation_score,
        )

        print(
            f"      C={c_value:<5} "
            f"validation pooled AUC="
            f"{validation_auc:.4f}"
        )

        if (
            np.isfinite(validation_auc)
            and validation_auc > best_auc
        ):
            best_auc = validation_auc
            best_c = c_value

    if best_c is None:
        raise RuntimeError(
            "Could not select a valid C."
        )

    # --------------------------------------------------------
    # Refit on train + validation
    # --------------------------------------------------------

    train_validation = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    X_train_validation = (
        train_validation[
            feature_cols
            + ["ticker"]
        ]
    )

    y_train_validation = (
        train_validation["target"]
        .astype(int)
    )

    final_model = Pipeline(
        steps=[

            (
                "preprocess",
                preprocessor,
            ),

            (
                "logistic",
                LogisticRegression(
                    C=best_c,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=4000,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )

    final_model.fit(
        X_train_validation,
        y_train_validation,
    )

    return (
        final_model,
        best_c,
        best_auc,
    )


# ============================================================
# DATE-BLOCK BOOTSTRAP
# ============================================================

def make_date_blocks(
    dates,
    block_length,
):
    """
    Divide sorted dates into consecutive blocks.
    """

    unique_dates = np.array(
        sorted(
            pd.to_datetime(
                dates
            ).unique()
        )
    )

    blocks = []

    for start in range(
        0,
        len(unique_dates),
        block_length,
    ):

        block = unique_dates[
            start:start + block_length
        ]

        if len(block) > 0:
            blocks.append(block)

    return blocks


def sample_bootstrap_dates(
    dates,
    block_length,
    rng,
):
    """
    Resample common forecast dates by consecutive blocks.

    This preserves cross-sectional dependence among tickers
    observed on the same market date.
    """

    dates = np.array(
        sorted(
            pd.to_datetime(
                dates
            ).unique()
        )
    )

    blocks = make_date_blocks(
        dates,
        block_length,
    )

    sampled = []

    while len(sampled) < len(dates):

        block_index = rng.integers(
            0,
            len(blocks),
        )

        sampled.extend(
            blocks[block_index]
        )

    sampled = sampled[
        :len(dates)
    ]

    return pd.to_datetime(
        sampled
    )


def bootstrap_pooled_auc(
    frame,
    target_col,
    score_col,
    block_length,
    n_bootstraps,
    seed,
):
    """
    Date-block bootstrap for pooled AUC.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = (
        pd.to_datetime(
            frame["forecast_date"]
        )
        .drop_duplicates()
        .sort_values()
        .values
    )

    estimates = []

    for _ in range(
        n_bootstraps
    ):

        sampled_dates = (
            sample_bootstrap_dates(
                dates,
                block_length,
                rng,
            )
        )

        parts = []

        for date in sampled_dates:

            block = frame[
                pd.to_datetime(
                    frame[
                        "forecast_date"
                    ]
                )
                == date
            ]

            if not block.empty:
                parts.append(block)

        if not parts:
            continue

        boot = pd.concat(
            parts,
            ignore_index=True,
        )

        auc = safe_auc(
            boot[target_col],
            boot[score_col],
        )

        if np.isfinite(auc):
            estimates.append(
                auc
            )

    estimates = np.asarray(
        estimates
    )

    if len(estimates) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    lower = np.percentile(
        estimates,
        2.5,
    )

    upper = np.percentile(
        estimates,
        97.5,
    )

    return (
        estimates.mean(),
        lower,
        upper,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "FINAL DECOMPOSITION AUDIT"
    )
    print("=" * 100)

    print(
        f"ROOT:       {ROOT}"
    )

    print(
        f"PANEL:      {PANEL_PATH}"
    )

    print(
        f"OUTPUT DIR: {OUTPUT_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not PANEL_PATH.exists():

        raise FileNotFoundError(
            f"Panel not found:\n"
            f"{PANEL_PATH}"
        )

    df = pd.read_csv(
        PANEL_PATH
    )

    print(
        f"Loaded panel: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    print()

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    required = [
        "ticker",
        "forecast_date",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing required columns: "
            f"{missing}"
        )

    df[
        "forecast_date"
    ] = pd.to_datetime(
        df["forecast_date"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
    )

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

    # --------------------------------------------------------
    # Result containers
    # --------------------------------------------------------

    summary_rows = []
    per_ticker_rows = []
    bootstrap_rows = []

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for model_name in MODELS:

        print()
        print(
            "#" * 100
        )
        print(
            f"MODEL: {model_name}"
        )
        print(
            "#" * 100
        )

        # ----------------------------------------------------
        # Resolve 19 simple features
        # ----------------------------------------------------

        resolved_features = []

        for base_name in (
            SIMPLE_FEATURE_BASES
        ):

            column = find_model_column(
                df.columns,
                model_name,
                base_name,
            )

            if column is None:

                print(
                    f"WARNING: Could not resolve "
                    f"'{base_name}' for "
                    f"{model_name}"
                )

            else:

                resolved_features.append(
                    column
                )

        print()

        print(
            f"Resolved "
            f"{len(resolved_features)}/19 "
            f"simple features"
        )

        if len(
            resolved_features
        ) != 19:

            print(
                "Resolved feature columns:"
            )

            for column in (
                resolved_features
            ):
                print(
                    f"    {column}"
                )

            raise RuntimeError(
                f"Could not resolve all "
                f"19 simple features for "
                f"{model_name}."
            )

        # ----------------------------------------------------
        # Market stress source columns
        # ----------------------------------------------------

        vol20_col = (
            find_model_column(
                df.columns,
                model_name,
                "realized_vol_20d",
            )
        )

        volratio_col = (
            find_model_column(
                df.columns,
                model_name,
                "vol_ratio_20_60",
            )
        )

        if (
            vol20_col is None
            or volratio_col is None
        ):

            raise RuntimeError(
                f"Could not resolve market "
                f"stress columns for "
                f"{model_name}."
            )

        print(
            f"Market stress vol20: "
            f"{vol20_col}"
        )

        print(
            f"Market stress ratio: "
            f"{volratio_col}"
        )

        # ----------------------------------------------------
        # Working dataset
        # ----------------------------------------------------

        work = df[
            [
                "ticker",
                "forecast_date",
                *resolved_features,
            ]
        ].copy()

        # Keep common market-state columns.
        work[
            "market_vol20_source"
        ] = df[
            vol20_col
        ].to_numpy()

        work[
            "market_vol_ratio_source"
        ] = df[
            volratio_col
        ].to_numpy()

        # ----------------------------------------------------
        # Create fixed chronological split per ticker
        # ----------------------------------------------------

        split_parts = []

        for ticker, g in (
            work
            .groupby(
                "ticker",
                sort=True,
            )
        ):

            split_parts.append(
                split_ticker(g)
            )

        work = pd.concat(
            split_parts,
            ignore_index=True,
        )

        # Training forecast dates.
        train_dates = (
            work.loc[
                work["split"] == "train",
                "forecast_date",
            ]
            .drop_duplicates()
            .sort_values()
            .values
        )

        # ----------------------------------------------------
        # Common market-stress state
        # ----------------------------------------------------

        market_state = (
            build_market_stress_state(
                work,
                train_dates,
            )
        )

        work = work.merge(
            market_state[
                [
                    "forecast_date",
                    "market_median_vol20",
                    "market_median_vol_ratio",
                    "market_stress_score",
                ]
            ],
            on="forecast_date",
            how="left",
            validate="many_to_one",
        )

        print(
            f"Common stress dates: "
            f"{market_state['forecast_date'].nunique():,}"
        )

        # ====================================================
        # HORIZON LOOP
        # ====================================================

        for horizon in HORIZONS:

            print()
            print(
                "-" * 100
            )

            print(
                f"{model_name} | "
                f"{horizon}D"
            )

            print(
                "-" * 100
            )

            # ------------------------------------------------
            # Target
            # ------------------------------------------------

            target_col = (
                find_target_column(
                    df.columns,
                    model_name,
                    horizon,
                )
            )

            if target_col is None:

                raise RuntimeError(
                    f"Could not resolve primary "
                    f"deterioration target for "
                    f"{model_name} {horizon}d."
                )

            print(
                f"Target: {target_col}"
            )

            # ------------------------------------------------
            # Merge target
            # ------------------------------------------------

            target_data = df[
                [
                    "ticker",
                    "forecast_date",
                    target_col,
                ]
            ].copy()

            target_data[
                "target"
            ] = pd.to_numeric(
                target_data[
                    target_col
                ],
                errors="coerce",
            )

            current = work.merge(
                target_data[
                    [
                        "ticker",
                        "forecast_date",
                        "target",
                    ]
                ],
                on=[
                    "ticker",
                    "forecast_date",
                ],
                how="left",
                validate="one_to_one",
            )

            current = current[
                current["target"].notna()
            ].copy()

            current[
                "target"
            ] = (
                current["target"]
                .astype(int)
            )

            # ------------------------------------------------
            # Split
            # ------------------------------------------------

            train = current[
                current["split"] == "train"
            ].copy()

            validation = current[
                current["split"] == "validation"
            ].copy()

            test = current[
                current["split"] == "test"
            ].copy()

            print(
                f"Train:      {len(train):,}"
            )

            print(
                f"Validation: {len(validation):,}"
            )

            print(
                f"Test:       {len(test):,}"
            )

            print(
                f"Test positive rate: "
                f"{test['target'].mean():.4f}"
            )

            # =================================================
            # SIMPLE LOGISTIC
            # =================================================

            print()
            print(
                "[1] SIMPLE LOGISTIC"
            )

            (
                simple_model,
                best_c,
                validation_auc,
            ) = fit_simple_monitor(
                train,
                validation,
                resolved_features,
            )

            # ------------------------------------------------
            # Test scores
            # ------------------------------------------------

            test_X = test[
                resolved_features
                + ["ticker"]
            ]

            test[
                "simple_score"
            ] = (
                simple_model
                .predict_proba(test_X)[
                    :, 1
                ]
            )

            # =================================================
            # TICKER-CENTERED SCORE
            # =================================================
            #
            # Important:
            # subtract the TRAINING-period mean score
            # for each ticker only.
            #
            # This removes persistent ticker-level score
            # differences while retaining within-ticker
            # movements.
            #
            # =================================================

            train_X = train[
                resolved_features
                + ["ticker"]
            ]

            train[
                "_simple_score"
            ] = (
                simple_model
                .predict_proba(train_X)[
                    :, 1
                ]
            )

            ticker_score_mean = (
                train
                .groupby("ticker")[
                    "_simple_score"
                ]
                .mean()
                .rename(
                    "ticker_train_mean_score"
                )
            )

            test = test.merge(
                ticker_score_mean,
                left_on="ticker",
                right_index=True,
                how="left",
            )

            test[
                "simple_centered_score"
            ] = (
                test["simple_score"]
                -
                test[
                    "ticker_train_mean_score"
                ]
            )

            # =================================================
            # TICKER-IDENTITY BASELINE
            # =================================================
            #
            # For each ticker:
            #
            #   training deterioration rate
            #
            # becomes the fixed prediction for its test rows.
            #
            # No test labels are used.
            #
            # =================================================

            ticker_deterioration_rate = (
                train
                .groupby("ticker")[
                    "target"
                ]
                .mean()
                .rename(
                    "ticker_train_deterioration_rate"
                )
            )

            test = test.merge(
                ticker_deterioration_rate,
                left_on="ticker",
                right_index=True,
                how="left",
            )

            # =================================================
            # POOLED METRICS
            # =================================================

            simple_pooled_auc = safe_auc(
                test["target"],
                test["simple_score"],
            )

            centered_pooled_auc = safe_auc(
                test["target"],
                test[
                    "simple_centered_score"
                ],
            )

            ticker_identity_auc = safe_auc(
                test["target"],
                test[
                    "ticker_train_deterioration_rate"
                ],
            )

            market_stress_auc = safe_auc(
                test["target"],
                test[
                    "market_stress_score"
                ],
            )

            # =================================================
            # PER-TICKER AUC
            # =================================================

            per_ticker = (
                calculate_per_ticker_auc(
                    test,
                    "target",
                    "simple_score",
                )
            )

            ticker_summary = (
                summarize_per_ticker(
                    per_ticker
                )
            )

            per_ticker[
                "model"
            ] = model_name

            per_ticker[
                "horizon"
            ] = horizon

            per_ticker_rows.extend(
                per_ticker[
                    [
                        "model",
                        "horizon",
                        "ticker",
                        "auc",
                        "n",
                        "positive_rate",
                    ]
                ].to_dict(
                    "records"
                )
            )

            # =================================================
            # WITHIN-TICKER CENTERED AUC
            # =================================================

            centered_per_ticker = (
                calculate_per_ticker_auc(
                    test,
                    "target",
                    "simple_centered_score",
                )
            )

            centered_aucs = (
                centered_per_ticker[
                    "auc"
                ]
                .dropna()
            )

            if len(centered_aucs) > 0:

                centered_macro_auc = (
                    centered_aucs.mean()
                )

                centered_median_auc = (
                    centered_aucs.median()
                )

                centered_fraction = (
                    (
                        centered_aucs
                        > 0.50
                    )
                    .mean()
                )

            else:

                centered_macro_auc = np.nan
                centered_median_auc = np.nan
                centered_fraction = np.nan

            # =================================================
            # SUMMARY ROW
            # =================================================

            summary_rows.append(
                {

                    "model": model_name,

                    "horizon": horizon,

                    "n_test": len(test),

                    "n_tickers": (
                        test["ticker"]
                        .nunique()
                    ),

                    "test_positive_rate": (
                        test["target"]
                        .mean()
                    ),

                    # ---------------------------------------
                    # Pooled
                    # ---------------------------------------

                    "simple_pooled_auc": (
                        simple_pooled_auc
                    ),

                    "simple_centered_pooled_auc": (
                        centered_pooled_auc
                    ),

                    # ---------------------------------------
                    # Baselines
                    # ---------------------------------------

                    "ticker_identity_auc": (
                        ticker_identity_auc
                    ),

                    "market_stress_auc": (
                        market_stress_auc
                    ),

                    # ---------------------------------------
                    # Per ticker
                    # ---------------------------------------

                    "simple_macro_ticker_auc": (
                        ticker_summary[
                            "macro_auc"
                        ]
                    ),

                    "simple_median_ticker_auc": (
                        ticker_summary[
                            "median_auc"
                        ]
                    ),

                    "simple_fraction_ticker_auc_gt_050": (
                        ticker_summary[
                            "fraction_auc_gt_050"
                        ]
                    ),

                    # ---------------------------------------
                    # Centered
                    # ---------------------------------------

                    "centered_macro_ticker_auc": (
                        centered_macro_auc
                    ),

                    "centered_median_ticker_auc": (
                        centered_median_auc
                    ),

                    "centered_fraction_ticker_auc_gt_050": (
                        centered_fraction
                    ),

                    # ---------------------------------------
                    # Gaps
                    # ---------------------------------------

                    "pooled_minus_centered_auc": (
                        simple_pooled_auc
                        - centered_pooled_auc
                    ),

                    "pooled_minus_ticker_identity_auc": (
                        simple_pooled_auc
                        - ticker_identity_auc
                    ),

                    "pooled_minus_market_stress_auc": (
                        simple_pooled_auc
                        - market_stress_auc
                    ),

                    # ---------------------------------------
                    # Model-selection information
                    # ---------------------------------------

                    "selected_C": best_c,

                    "validation_auc": (
                        validation_auc
                    ),
                }
            )

            # =================================================
            # BOOTSTRAP
            # =================================================

            print()
            print(
                f"Running {N_BOOTSTRAPS} "
                f"date-block bootstraps "
                f"(block={BLOCK_LENGTH})..."
            )

            # -------------------------------------------------
            # Simple pooled
            # -------------------------------------------------

            (
                boot_mean,
                ci_low,
                ci_high,
            ) = bootstrap_pooled_auc(
                test,
                "target",
                "simple_score",
                BLOCK_LENGTH,
                N_BOOTSTRAPS,
                RANDOM_SEED,
            )

            bootstrap_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "metric": (
                        "simple_pooled_auc"
                    ),
                    "bootstrap_mean": boot_mean,
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                }
            )

            # -------------------------------------------------
            # Centered pooled
            # -------------------------------------------------

            (
                boot_mean,
                ci_low,
                ci_high,
            ) = bootstrap_pooled_auc(
                test,
                "target",
                "simple_centered_score",
                BLOCK_LENGTH,
                N_BOOTSTRAPS,
                RANDOM_SEED + 1,
            )

            bootstrap_rows.append(
                {
                    "model": model_name,
                    "horizon": horizon,
                    "metric": (
                        "simple_centered_pooled_auc"
                    ),
                    "bootstrap_mean": boot_mean,
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                }
            )

            # =================================================
            # PRINT RESULTS
            # =================================================

            print()
            print(
                "HEADLINE RESULTS"
            )

            print(
                f"  Simple pooled AUC:          "
                f"{simple_pooled_auc:.4f}"
            )

            print(
                f"  Centered pooled AUC:        "
                f"{centered_pooled_auc:.4f}"
            )

            print(
                f"  Ticker-identity AUC:        "
                f"{ticker_identity_auc:.4f}"
            )

            print(
                f"  Market-stress AUC:          "
                f"{market_stress_auc:.4f}"
            )

            print(
                f"  Macro ticker AUC:            "
                f"{ticker_summary['macro_auc']:.4f}"
            )

            print(
                f"  Median ticker AUC:           "
                f"{ticker_summary['median_auc']:.4f}"
            )

            print(
                f"  Fraction ticker AUC > .50:  "
                f"{ticker_summary['fraction_auc_gt_050']:.4f}"
            )

            print(
                f"  Centered macro ticker AUC:  "
                f"{centered_macro_auc:.4f}"
            )

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            if (
                np.isfinite(
                    simple_pooled_auc
                )
                and np.isfinite(
                    centered_pooled_auc
                )
                and (
                    simple_pooled_auc
                    - centered_pooled_auc
                ) > 0.05
            ):

                print(
                    "  WARNING: large pooled-vs-centered gap."
                )

            if (
                np.isfinite(
                    ticker_identity_auc
                )
                and np.isfinite(
                    simple_pooled_auc
                )
                and ticker_identity_auc
                >= simple_pooled_auc - 0.02
            ):

                print(
                    "  WARNING: ticker identity explains "
                    "much of pooled AUC."
                )

            if (
                np.isfinite(
                    market_stress_auc
                )
                and np.isfinite(
                    simple_pooled_auc
                )
                and market_stress_auc
                >= simple_pooled_auc - 0.02
            ):

                print(
                    "  WARNING: market stress explains "
                    "much of pooled AUC."
                )

            if (
                np.isfinite(
                    ticker_summary[
                        "macro_auc"
                    ]
                )
                and ticker_summary[
                    "macro_auc"
                ] < 0.50
            ):

                print(
                    "  WARNING: average within-ticker "
                    "discrimination is below chance."
                )

            # =================================================
            # SAVE TEST SCORES
            # =================================================

            scores_path = (
                OUTPUT_DIR
                / (
                    f"{model_name}_"
                    f"{horizon}d_test_scores.csv"
                )
            )

            save_columns = [

                "ticker",
                "forecast_date",

                "target",

                "simple_score",

                "simple_centered_score",

                "ticker_train_deterioration_rate",

                "market_stress_score",

                "market_median_vol20",

                "market_median_vol_ratio",
            ]

            test[
                save_columns
            ].to_csv(
                scores_path,
                index=False,
            )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    per_ticker_df = pd.DataFrame(
        per_ticker_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "decomposition_summary.csv"
    )

    per_ticker_path = (
        OUTPUT_DIR
        / "decomposition_per_ticker.csv"
    )

    bootstrap_path = (
        OUTPUT_DIR
        / "decomposition_bootstrap.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    per_ticker_df.to_csv(
        per_ticker_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print()
    print()
    print(
        "=" * 120
    )
    print(
        "FINAL DECOMPOSITION TABLE"
    )
    print(
        "=" * 120
    )

    display_columns = [

        "model",
        "horizon",

        "simple_pooled_auc",

        "simple_centered_pooled_auc",

        "ticker_identity_auc",

        "market_stress_auc",

        "simple_macro_ticker_auc",

        "simple_median_ticker_auc",

        "simple_fraction_ticker_auc_gt_050",

        "centered_macro_ticker_auc",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # BOOTSTRAP TABLE
    # ========================================================

    print()
    print()
    print(
        "=" * 120
    )
    print(
        "BOOTSTRAP 95% CIs"
    )
    print(
        "=" * 120
    )

    print(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # ========================================================
    # SAVE FULL SUMMARY TEXT
    # ========================================================

    report_path = (
        OUTPUT_DIR
        / "decomposition_audit_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "FINAL DECOMPOSITION AUDIT\n"
        )

        f.write(
            "=" * 100
            + "\n\n"
        )

        f.write(
            "Purpose:\n"
        )

        f.write(
            "Decompose pooled early-warning AUC "
            "into ticker identity, market stress, "
            "and within-ticker discrimination.\n\n"
        )

        f.write(
            "Final 19 simple features:\n"
        )

        for feature in SIMPLE_FEATURE_BASES:

            f.write(
                f"  - {feature}\n"
            )

        f.write(
            "\n\nFINAL TABLE\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            summary_df[
                display_columns
            ].to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

        f.write(
            "\n\n\nBOOTSTRAP RESULTS\n"
        )

        f.write(
            "=" * 100
            + "\n"
        )

        f.write(
            bootstrap_df.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # ========================================================
    # FINAL FILE LOCATIONS
    # ========================================================

    print()
    print()
    print(
        "=" * 120
    )
    print(
        "FILES SAVED"
    )
    print(
        "=" * 120
    )

    print(
        summary_path
    )

    print(
        per_ticker_path
    )

    print(
        bootstrap_path
    )

    print(
        report_path
    )

    print()
    print(
        "=" * 120
    )

    print(
        "DECOMPOSITION AUDIT COMPLETE"
    )

    print(
        "=" * 120
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()