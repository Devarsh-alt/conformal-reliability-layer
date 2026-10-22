from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 34 — TICKER-FIXED-EFFECT BREACH-HISTORY TEST
# =============================================================================
#
# Research question:
#
#   Does breach history predict future VaR deterioration after accounting
#   for ticker-specific baseline differences?
#
# Model:
#
#   logit(P(D_t,h = 1))
#       = ticker fixed effect + breach-history signal
#
# Specifications:
#
#   1. prior 5-day breach count
#   2. prior 10-day breach count
#   3. prior 20-day breach count
#   4. prior breach streak
#   5. days since most recent breach
#   6. joint breach history
#
# Evaluation:
#
#   - split chronologically within ticker
#   - 60% train / 40% test
#   - train-only standardization
#   - ticker fixed effects learned only from training
#   - pooled OOS AUC
#   - moving-block bootstrap within ticker
#
# =============================================================================


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

OUTPUT_MAIN = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_fixed_effects_results.csv"
)

OUTPUT_PREDICTIONS = (
    ROOT
    / "data"
    / "monitoring"
    / "breach_history_fixed_effects_predictions.csv"
)

TRAIN_FRACTION = 0.60

BLOCK_LENGTH = 20

N_BOOTSTRAPS = 1000

RANDOM_STATE = 42

HORIZONS = [5, 10, 20]

MIN_TRAIN_ROWS = 100


# =============================================================================
# PRIMARY TARGET
# =============================================================================

def get_primary_target(df, horizon):

    target = f"deterioration_{horizon}d"

    if target not in df.columns:
        raise ValueError(
            f"Missing target: {target}"
        )

    return target


# =============================================================================
# MOVING BLOCK INDICES
# =============================================================================

def moving_block_indices(
    n,
    block_length,
    rng
):

    if n <= 0:
        return np.array([], dtype=int)

    block_length = min(
        block_length,
        n
    )

    starts = np.arange(
        0,
        n - block_length + 1
    )

    indices = []

    while len(indices) < n:

        start = rng.choice(
            starts
        )

        block = np.arange(
            start,
            start + block_length
        )

        indices.extend(
            block.tolist()
        )

    return np.asarray(
        indices[:n],
        dtype=int
    )


# =============================================================================
# BOOTSTRAP POOLED AUC
# =============================================================================

def bootstrap_auc(
    prediction_df,
    score_column,
    n_bootstrap=1000,
    block_length=20,
    random_state=42
):

    rng = np.random.default_rng(
        random_state
    )

    arrays = []

    for ticker, g in prediction_df.groupby(
        "ticker"
    ):

        g = g.dropna(
            subset=[
                "target",
                score_column
            ]
        )

        if len(g) < 20:
            continue

        y = (
            g["target"]
            .astype(int)
            .to_numpy()
        )

        score = (
            g[score_column]
            .astype(float)
            .to_numpy()
        )

        if len(np.unique(y)) < 2:
            continue

        arrays.append(
            (
                ticker,
                y,
                score
            )
        )

    if not arrays:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    boot_aucs = []

    for _ in range(n_bootstrap):

        pooled_y = []
        pooled_score = []

        for ticker, y, score in arrays:

            idx = moving_block_indices(
                len(y),
                block_length,
                rng
            )

            pooled_y.append(
                y[idx]
            )

            pooled_score.append(
                score[idx]
            )

        y_all = np.concatenate(
            pooled_y
        )

        score_all = np.concatenate(
            pooled_score
        )

        if len(np.unique(y_all)) < 2:
            continue

        boot_aucs.append(
            roc_auc_score(
                y_all,
                score_all
            )
        )

    if not boot_aucs:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    boot_aucs = np.asarray(
        boot_aucs
    )

    return (
        np.mean(boot_aucs),
        np.percentile(
            boot_aucs,
            2.5
        ),
        np.percentile(
            boot_aucs,
            97.5
        )
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 34 — TICKER-FIXED-EFFECT BREACH-HISTORY TEST")
print("=" * 100)

print("\nROOT:")
print(ROOT)

print("\nPANEL:")
print(PANEL_FILE)

print("\nConfiguration:")
print(
    f"Train fraction:      {TRAIN_FRACTION:.2f}"
)

print(
    f"Block length:        {BLOCK_LENGTH}"
)

print(
    f"Bootstrap samples:   {N_BOOTSTRAPS}"
)

print(
    f"Random state:        {RANDOM_STATE}"
)


# =============================================================================
# LOAD DATA
# =============================================================================

if not PANEL_FILE.exists():

    raise FileNotFoundError(
        f"Panel not found:\n{PANEL_FILE}"
    )

df = pd.read_csv(
    PANEL_FILE
)

print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

print("\nShape:")
print(df.shape)


# =============================================================================
# TYPES
# =============================================================================

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.strip()
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["breach"] = (
    df["breach"]
    .fillna(0)
    .astype(int)
)

df = (
    df
    .sort_values(
        [
            "ticker",
            "forecast_date"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# CONSTRUCT CAUSAL BREACH HISTORY
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "CONSTRUCTING CAUSAL BREACH HISTORY"
)

print(
    "=" * 100
)


def calculate_days_since_last_breach(group):

    breaches = (
        group["breach"]
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    result = np.full(
        len(group),
        np.nan
    )

    last_breach = None

    for i in range(len(group)):

        if last_breach is not None:

            result[i] = (
                i - last_breach
            )

        if breaches[i] == 1:

            last_breach = i

    return result


groups = []

for ticker, g in df.groupby(
    "ticker",
    sort=False
):

    g = (
        g
        .sort_values(
            "forecast_date"
        )
        .copy()
    )

    past_breach = (
        g["breach"]
        .shift(1)
        .fillna(0)
        .astype(int)
    )

    g["prior_breach_count_5"] = (
        past_breach
        .rolling(
            5,
            min_periods=5
        )
        .sum()
    )

    g["prior_breach_count_10"] = (
        past_breach
        .rolling(
            10,
            min_periods=10
        )
        .sum()
    )

    g["prior_breach_count_20"] = (
        past_breach
        .rolling(
            20,
            min_periods=20
        )
        .sum()
    )

    g["days_since_last_breach"] = (
        calculate_days_since_last_breach(
            g
        )
    )

    groups.append(
        g
    )


df = pd.concat(
    groups,
    ignore_index=True
)


# =============================================================================
# MODEL SPECIFICATIONS
# =============================================================================

MODEL_FEATURES = {

    "breach_count_5":
        [
            "prior_breach_count_5"
        ],

    "breach_count_10":
        [
            "prior_breach_count_10"
        ],

    "breach_count_20":
        [
            "prior_breach_count_20"
        ],

    "breach_streak":
        [
            "breach_streak_prior"
        ],

    "breach_recency":
        [
            "days_since_last_breach"
        ],

    "joint_breach_history":
        [
            "prior_breach_count_5",
            "breach_streak_prior",
            "days_since_last_breach"
        ]
}


# =============================================================================
# STORAGE
# =============================================================================

results = []

prediction_tables = []


# =============================================================================
# HORIZON LOOP
# =============================================================================

for horizon in HORIZONS:

    target = get_primary_target(
        df,
        horizon
    )

    print(
        "\n\n"
        + "=" * 100
    )

    print(
        f"HORIZON = {horizon} DAYS"
    )

    print(
        f"TARGET  = {target}"
    )

    print(
        "=" * 100
    )


    # =========================================================================
    # CREATE CHRONOLOGICAL TRAIN / TEST DATA
    # =========================================================================

    train_parts = []
    test_parts = []

    eligible_tickers = 0

    for ticker in sorted(
        df["ticker"].unique()
    ):

        g = (
            df[
                df["ticker"] == ticker
            ]
            .sort_values(
                "forecast_date"
            )
            .copy()
        )

        if len(g) < MIN_TRAIN_ROWS:
            continue

        # IMPORTANT:
        # Split the full history FIRST.
        split_idx = int(
            np.floor(
                len(g)
                * TRAIN_FRACTION
            )
        )

        if (
            split_idx < 50
            or
            split_idx >= len(g) - 20
        ):
            continue

        train = (
            g.iloc[
                :split_idx
            ]
            .copy()
        )

        test = (
            g.iloc[
                split_idx:
            ]
            .copy()
        )

        train["ticker"] = ticker
        test["ticker"] = ticker

        train_parts.append(
            train
        )

        test_parts.append(
            test
        )

        eligible_tickers += 1


    train_all = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_all_raw = pd.concat(
        test_parts,
        ignore_index=True
    )


    # =========================================================================
    # FIT EACH MODEL
    # =========================================================================

    horizon_predictions = []

    for model_name, features in MODEL_FEATURES.items():

        train = (
            train_all
            .dropna(
                subset=
                    features
                    + [target]
            )
            .copy()
        )

        test = (
            test_all_raw
            .dropna(
                subset=
                    features
                    + [target]
            )
            .copy()
        )

        if len(train) < 100:
            continue

        if len(test) < 50:
            continue

        if (
            len(
                np.unique(
                    train[target]
                )
            )
            < 2
        ):
            continue

        if (
            len(
                np.unique(
                    test[target]
                )
            )
            < 2
        ):
            continue


        # =====================================================================
        # NUMERIC TRANSFORM
        # =====================================================================

        scaler = StandardScaler()

        x_train_numeric = scaler.fit_transform(
            train[features]
            .astype(float)
        )

        x_test_numeric = scaler.transform(
            test[features]
            .astype(float)
        )


        # =====================================================================
        # TICKER FIXED EFFECTS
        #
        # Fit one-hot encoding ONLY on training tickers.
        # =====================================================================

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False
        )

        x_train_ticker = encoder.fit_transform(
            train[
                ["ticker"]
            ]
        )

        x_test_ticker = encoder.transform(
            test[
                ["ticker"]
            ]
        )


        # =====================================================================
        # COMBINE FEATURES
        # =====================================================================

        x_train = np.hstack(
            [
                x_train_numeric,
                x_train_ticker
            ]
        )

        x_test = np.hstack(
            [
                x_test_numeric,
                x_test_ticker
            ]
        )


        # =====================================================================
        # LOGISTIC REGRESSION
        # =====================================================================

        model = LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE
        )

        model.fit(
            x_train,
            train[target].astype(int)
        )


        # =====================================================================
        # OOS PROBABILITY
        # =====================================================================

        test_pred = model.predict_proba(
            x_test
        )[:, 1]


        # =====================================================================
        # TEST AUC
        # =====================================================================

        test_auc = roc_auc_score(
            test[target].astype(int),
            test_pred
        )


        # =====================================================================
        # STORE PREDICTIONS
        # =====================================================================

        temp = test[
            [
                "ticker",
                "forecast_date"
            ]
        ].copy()

        temp["target"] = (
            test[target]
            .astype(int)
            .to_numpy()
        )

        temp["prediction"] = (
            test_pred
        )

        temp["model"] = model_name

        temp["horizon"] = horizon

        horizon_predictions.append(
            temp
        )


        # =====================================================================
        # PRINT
        # =====================================================================

        print(
            f"\n{model_name}"
        )

        print(
            f"  Train observations: "
            f"{len(train):,}"
        )

        print(
            f"  Test observations:  "
            f"{len(test):,}"
        )

        print(
            f"  OOS AUC:            "
            f"{test_auc:.4f}"
        )


    # =========================================================================
    # HORIZON PREDICTIONS
    # =========================================================================

    if not horizon_predictions:

        continue

    horizon_pred = pd.concat(
        horizon_predictions,
        ignore_index=True
    )

    prediction_tables.append(
        horizon_pred
    )


    # =========================================================================
    # BOOTSTRAP FOR EACH MODEL
    # =========================================================================

    for model_name in MODEL_FEATURES:

        sub = horizon_pred[
            horizon_pred["model"]
            == model_name
        ].copy()

        if sub.empty:
            continue

        (
            bootstrap_mean,
            ci_low,
            ci_high
        ) = bootstrap_auc(
            prediction_df=sub,
            score_column="prediction",
            n_bootstrap=N_BOOTSTRAPS,
            block_length=BLOCK_LENGTH,
            random_state=(
                RANDOM_STATE
                + horizon
                + 100
                * list(
                    MODEL_FEATURES.keys()
                ).index(model_name)
            )
        )


        pooled_auc = roc_auc_score(
            sub["target"],
            sub["prediction"]
        )


        results.append(
            {
                "horizon":
                    horizon,

                "model":
                    model_name,

                "eligible_tickers":
                    eligible_tickers,

                "test_observations":
                    len(sub),

                "pooled_auc":
                    pooled_auc,

                "bootstrap_auc_mean":
                    bootstrap_mean,

                "bootstrap_ci_low":
                    ci_low,

                "bootstrap_ci_high":
                    ci_high
            }
        )


# =============================================================================
# FINAL RESULTS
# =============================================================================

results_df = pd.DataFrame(
    results
)

predictions_df = pd.concat(
    prediction_tables,
    ignore_index=True
)


# =============================================================================
# PRINT SUMMARY
# =============================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL EXPERIMENT 34 SUMMARY"
)

print(
    "=" * 100
)

if not results_df.empty:

    print(
        results_df[
            [
                "horizon",
                "model",
                "eligible_tickers",
                "test_observations",
                "pooled_auc",
                "bootstrap_auc_mean",
                "bootstrap_ci_low",
                "bootstrap_ci_high"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# =============================================================================
# INTERPRETATION
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "INTERPRETATION"
)

print(
    "=" * 100
)

if not results_df.empty:

    for horizon in HORIZONS:

        h = results_df[
            results_df["horizon"]
            == horizon
        ]

        print(
            f"\n{horizon}-DAY DETERIORATION"
        )

        print(
            "-" * 80
        )

        for _, row in h.iterrows():

            auc = row[
                "pooled_auc"
            ]

            low = row[
                "bootstrap_ci_low"
            ]

            high = row[
                "bootstrap_ci_high"
            ]

            print(
                f"{row['model']:<25}"
                f"AUC={auc:.4f}  "
                f"95% CI=[{low:.4f}, "
                f"{high:.4f}]"
            )

            if low > 0.5:

                print(
                    "  -> Positive OOS discrimination "
                    "after controlling for ticker fixed effects."
                )

            else:

                print(
                    "  -> CI includes 0.5."
                )


# =============================================================================
# SAVE
# =============================================================================

results_df.to_csv(
    OUTPUT_MAIN,
    index=False
)

predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


print(
    "\n"
    + "=" * 100
)

print(
    "SAVED"
)

print(
    "=" * 100
)

print(
    OUTPUT_MAIN
)

print(
    OUTPUT_PREDICTIONS
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 34 COMPLETE"
)

print(
    "=" * 100
)