from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# EXPERIMENT 31 — QRF–EWMA DISAGREEMENT VS BREACH PERSISTENCE
# =============================================================================
#
# Research question:
# Does disagreement between the QRF and EWMA VaR forecasts provide
# incremental information about future VaR deterioration beyond
# breach persistence?
#
# Primary target:
#   deterioration_{horizon}d
#
# Baseline:
#   breach persistence / prior breach streak
#
# Signal:
#   QRF-EWMA disagreement
#
# Incremental model:
#   breach persistence + QRF-EWMA disagreement
#
# Evaluation:
#   - chronological 60/40 split within ticker
#   - pooled test AUC
#   - within-ticker moving block bootstrap
#   - cross-sectional ticker bootstrap
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

EWMA_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "EWMA_25ticker_oos_predictions.csv"
)

OUTPUT_MAIN = (
    ROOT
    / "data"
    / "monitoring"
    / "qrf_ewma_disagreement_test_results.csv"
)

OUTPUT_TICKER = (
    ROOT
    / "data"
    / "monitoring"
    / "qrf_ewma_disagreement_test_by_ticker.csv"
)

TRAIN_FRACTION = 0.60
BLOCK_LENGTH = 20

WITHIN_TICKER_BOOTSTRAPS = 1000
TICKER_BOOTSTRAPS = 2000

RANDOM_STATE = 42

HORIZONS = [5, 10, 20]


# =============================================================================
# HELPER — FIND COLUMN
# =============================================================================

def find_column(df, candidates, description):
    """
    Return the first matching column name from candidates.

    Matching is case-insensitive.
    """
    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        f"\nCould not find {description}.\n"
        f"Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# HELPER — EXPLICIT PRIMARY TARGET
# =============================================================================

def get_primary_target(df, horizon):
    """
    Explicitly select the causal 75th-percentile deterioration target.

    Important:
    Do NOT automatically search for target columns because the dataset
    also contains continuous historical deterioration thresholds.
    """

    target_col = f"deterioration_{horizon}d"

    if target_col not in df.columns:
        raise ValueError(
            f"Required primary target not found: {target_col}"
        )

    return target_col


# =============================================================================
# HELPER — LOGISTIC REGRESSION
# =============================================================================

def fit_logistic(train_x, train_y, test_x):
    """
    Train-only standardization followed by logistic regression.

    Returns:
        test probabilities
    """

    train_x = np.asarray(train_x, dtype=float)
    train_y = np.asarray(train_y, dtype=int)
    test_x = np.asarray(test_x, dtype=float)

    # Cannot fit if only one class exists
    if len(np.unique(train_y)) < 2:
        return None

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(train_x)
    x_test_scaled = scaler.transform(test_x)

    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        x_train_scaled,
        train_y
    )

    return model.predict_proba(
        x_test_scaled
    )[:, 1]


# =============================================================================
# HELPER — AUC
# =============================================================================

def compute_auc(y_true, scores):
    """
    Safe ROC-AUC calculation.
    """

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return roc_auc_score(
        y_true,
        scores
    )


# =============================================================================
# HELPER — MOVING BLOCK BOOTSTRAP INDICES
# =============================================================================

def make_moving_block_indices(
    n,
    block_length,
    rng
):
    """
    Generate moving-block bootstrap indices.

    Blocks are sampled with replacement until n observations
    are obtained.
    """

    if n <= 0:
        return np.array([], dtype=int)

    block_length = min(
        block_length,
        n
    )

    possible_starts = np.arange(
        0,
        n - block_length + 1
    )

    sampled_indices = []

    while len(sampled_indices) < n:

        start = rng.choice(
            possible_starts
        )

        block = np.arange(
            start,
            start + block_length
        )

        sampled_indices.extend(
            block.tolist()
        )

    return np.asarray(
        sampled_indices[:n],
        dtype=int
    )


# =============================================================================
# HELPER — WITHIN-TICKER BLOCK BOOTSTRAP
# =============================================================================

def bootstrap_pooled_delta(
    ticker_test_data,
    signal_column,
    n_bootstrap=1000,
    block_length=20,
    random_state=42
):
    """
    Moving block bootstrap within each ticker.

    Computes:

        AUC(signal) - AUC(persistence)

    using the already-generated out-of-sample prediction scores.

    We do NOT refit models inside each bootstrap replication.
    The chronological train/test models have already been fixed.

    This isolates uncertainty in the test-set AUC difference while
    preserving temporal dependence within each ticker.
    """

    rng = np.random.default_rng(
        random_state
    )

    ticker_arrays = []

    required = [
        "ticker",
        "target",
        "persistence_score",
        signal_column
    ]

    missing = [
        c for c in required
        if c not in ticker_test_data.columns
    ]

    if missing:
        raise ValueError(
            f"Bootstrap input is missing columns: {missing}"
        )

    for ticker, g in ticker_test_data.groupby(
        "ticker"
    ):

        g = g.dropna(
            subset=[
                "target",
                "persistence_score",
                signal_column
            ]
        ).copy()

        if len(g) < 20:
            continue

        y = g["target"].astype(int).to_numpy()

        persistence = (
            g["persistence_score"]
            .astype(float)
            .to_numpy()
        )

        signal = (
            g[signal_column]
            .astype(float)
            .to_numpy()
        )

        if len(np.unique(y)) < 2:
            continue

        ticker_arrays.append(
            (
                ticker,
                y,
                persistence,
                signal
            )
        )

    if not ticker_arrays:
        return np.nan, np.nan, np.nan

    bootstrap_deltas = []

    for _ in range(n_bootstrap):

        pooled_y = []
        pooled_persistence = []
        pooled_signal = []

        for (
            ticker,
            y,
            persistence,
            signal
        ) in ticker_arrays:

            idx = make_moving_block_indices(
                n=len(y),
                block_length=block_length,
                rng=rng
            )

            pooled_y.append(
                y[idx]
            )

            pooled_persistence.append(
                persistence[idx]
            )

            pooled_signal.append(
                signal[idx]
            )

        y_all = np.concatenate(
            pooled_y
        )

        persistence_all = np.concatenate(
            pooled_persistence
        )

        signal_all = np.concatenate(
            pooled_signal
        )

        if len(np.unique(y_all)) < 2:
            continue

        auc_persistence = roc_auc_score(
            y_all,
            persistence_all
        )

        auc_signal = roc_auc_score(
            y_all,
            signal_all
        )

        bootstrap_deltas.append(
            auc_signal - auc_persistence
        )

    if not bootstrap_deltas:
        return np.nan, np.nan, np.nan

    bootstrap_deltas = np.asarray(
        bootstrap_deltas
    )

    return (
        np.mean(bootstrap_deltas),
        np.percentile(
            bootstrap_deltas,
            2.5
        ),
        np.percentile(
            bootstrap_deltas,
            97.5
        )
    )


# =============================================================================
# HELPER — CROSS-SECTIONAL TICKER BOOTSTRAP
# =============================================================================

def bootstrap_ticker_delta(
    ticker_summary,
    delta_column,
    n_bootstrap=2000,
    random_state=42
):
    """
    Bootstrap over tickers.

    The unit of resampling is the ticker rather than the observation,
    which tests whether the average ticker-level improvement is robust
    across assets.
    """

    rng = np.random.default_rng(
        random_state
    )

    values = (
        ticker_summary[delta_column]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    bootstrap_means = []

    for _ in range(n_bootstrap):

        sampled = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrap_means.append(
            np.mean(sampled)
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    return (
        np.mean(values),
        np.percentile(
            bootstrap_means,
            2.5
        ),
        np.percentile(
            bootstrap_means,
            97.5
        )
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 100)
print("EXPERIMENT 31 — QRF–EWMA DISAGREEMENT VS BREACH PERSISTENCE")
print("=" * 100)

print("\nROOT:")
print(ROOT)

print("\nPANEL:")
print(PANEL_FILE)

print("\nEWMA:")
print(EWMA_FILE)

print("\nConfiguration:")
print(
    f"Train fraction:       {TRAIN_FRACTION:.2f}"
)
print(
    f"Block length:         {BLOCK_LENGTH}"
)
print(
    f"Within-ticker boots:  {WITHIN_TICKER_BOOTSTRAPS}"
)
print(
    f"Ticker boots:         {TICKER_BOOTSTRAPS}"
)
print(
    f"Random state:         {RANDOM_STATE}"
)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n" + "=" * 100)
print("LOADING DATA")
print("=" * 100)

if not PANEL_FILE.exists():
    raise FileNotFoundError(
        f"Panel file not found:\n{PANEL_FILE}"
    )

if not EWMA_FILE.exists():
    raise FileNotFoundError(
        f"EWMA file not found:\n{EWMA_FILE}"
    )

panel = pd.read_csv(
    PANEL_FILE
)

ewma = pd.read_csv(
    EWMA_FILE
)

print("\nPanel shape:")
print(panel.shape)

print("\nEWMA shape:")
print(ewma.shape)

print("\nPanel columns:")
print(list(panel.columns))

print("\nEWMA columns:")
print(list(ewma.columns))


# =============================================================================
# IDENTIFY COLUMNS
# =============================================================================

panel_ticker = find_column(
    panel,
    ["ticker"],
    "panel ticker column"
)

panel_date = find_column(
    panel,
    ["forecast_date"],
    "panel forecast date"
)

panel_qrf = find_column(
    panel,
    [
        "q05_forecast",
        "q05_ensemble",
        "VaR_5pct"
    ],
    "QRF 5% VaR forecast"
)

ewma_ticker = find_column(
    ewma,
    ["ticker"],
    "EWMA ticker column"
)

ewma_date = find_column(
    ewma,
    ["forecast_date"],
    "EWMA forecast date"
)

ewma_q05 = find_column(
    ewma,
    [
        "ewma_q05",
        "q05_ewma"
    ],
    "EWMA 5% VaR forecast"
)

print("\nIdentified columns:")
print(
    f"Panel ticker:     {panel_ticker}"
)
print(
    f"Panel date:       {panel_date}"
)
print(
    f"Panel QRF VaR:    {panel_qrf}"
)
print(
    f"EWMA ticker:      {ewma_ticker}"
)
print(
    f"EWMA date:        {ewma_date}"
)
print(
    f"EWMA VaR:         {ewma_q05}"
)


# =============================================================================
# STANDARDIZE DATA TYPES
# =============================================================================

panel[panel_date] = pd.to_datetime(
    panel[panel_date]
)

ewma[ewma_date] = pd.to_datetime(
    ewma[ewma_date]
)

panel[panel_ticker] = (
    panel[panel_ticker]
    .astype(str)
    .str.strip()
)

ewma[ewma_ticker] = (
    ewma[ewma_ticker]
    .astype(str)
    .str.strip()
)


# =============================================================================
# DUPLICATE CHECKS
# =============================================================================

print("\n" + "=" * 100)
print("KEY SANITY CHECKS")
print("=" * 100)

panel_duplicates = panel.duplicated(
    subset=[
        panel_ticker,
        panel_date
    ]
).sum()

ewma_duplicates = ewma.duplicated(
    subset=[
        ewma_ticker,
        ewma_date
    ]
).sum()

print(
    f"\nPanel duplicate ticker-date rows: {panel_duplicates}"
)

print(
    f"EWMA duplicate ticker-date rows:   {ewma_duplicates}"
)

if panel_duplicates > 0:
    raise ValueError(
        "Panel contains duplicate ticker-date rows."
    )

if ewma_duplicates > 0:
    raise ValueError(
        "EWMA contains duplicate ticker-date rows."
    )


# =============================================================================
# PREPARE MERGE TABLES
# =============================================================================

panel_merge = panel.copy()

panel_merge = panel_merge.rename(
    columns={
        panel_ticker: "ticker",
        panel_date: "forecast_date",
        panel_qrf: "qrf_q05"
    }
)

ewma_merge = ewma[
    [
        ewma_ticker,
        ewma_date,
        ewma_q05
    ]
].copy()

ewma_merge = ewma_merge.rename(
    columns={
        ewma_ticker: "ticker",
        ewma_date: "forecast_date",
        ewma_q05: "ewma_q05"
    }
)


# =============================================================================
# MERGE QRF + EWMA
# =============================================================================

print("\n" + "=" * 100)
print("MERGING QRF + EWMA")
print("=" * 100)

merged = panel_merge.merge(
    ewma_merge,
    on=[
        "ticker",
        "forecast_date"
    ],
    how="left",
    validate="one_to_one"
)

print("\nPanel rows:")
print(len(panel))

print("Merged rows:")
print(len(merged))

missing_ewma = merged[
    "ewma_q05"
].isna().sum()

print("Missing EWMA forecasts:")
print(missing_ewma)

complete_rows = merged[
    merged["qrf_q05"].notna()
    &
    merged["ewma_q05"].notna()
].shape[0]

print("Complete disagreement rows:")
print(complete_rows)

if len(merged) != len(panel):
    raise ValueError(
        "Merge changed the number of panel rows."
    )

if missing_ewma > 0:
    raise ValueError(
        "EWMA forecasts are missing for some panel rows."
    )


# =============================================================================
# COMPUTE PRE-SPECIFIED DISAGREEMENT SIGNAL
# =============================================================================

print("\n" + "=" * 100)
print("COMPUTING QRF–EWMA DISAGREEMENT")
print("=" * 100)

#
# CMD_t =
#
#       q05_QRF,t - q05_EWMA,t
#       -----------------------
#             |q05_EWMA,t|
#
# Larger positive values:
#   QRF is less negative / less conservative than EWMA.
#
# More negative values:
#   QRF is more negative / more conservative than EWMA.
#

denominator = (
    merged["ewma_q05"]
    .abs()
)

merged["qrf_ewma_disagreement"] = np.where(
    denominator > 1e-12,
    (
        merged["qrf_q05"]
        - merged["ewma_q05"]
    )
    / denominator,
    np.nan
)

# Baseline reliability signal:
# prior breach streak
merged["persistence_score"] = (
    merged["breach_streak_prior"]
    .astype(float)
)

print("\nDisagreement summary:")
print(
    merged[
        "qrf_ewma_disagreement"
    ].describe()
)

missing_disagreement = (
    merged["qrf_ewma_disagreement"]
    .isna()
    .sum()
)

infinite_disagreement = np.isinf(
    merged["qrf_ewma_disagreement"]
).sum()

print(
    "\nMissing disagreement:",
    missing_disagreement
)

print(
    "Infinite disagreement:",
    infinite_disagreement
)

if missing_disagreement > 0:
    raise ValueError(
        "Missing disagreement values detected."
    )

if infinite_disagreement > 0:
    raise ValueError(
        "Infinite disagreement values detected."
    )


# =============================================================================
# TICKER COVERAGE
# =============================================================================

print("\n" + "=" * 100)
print("TICKER COVERAGE")
print("=" * 100)

coverage = (
    merged
    .groupby("ticker")
    .agg(
        total_rows=(
            "ticker",
            "size"
        ),
        disagreement_rows=(
            "qrf_ewma_disagreement",
            lambda x: x.notna().sum()
        )
    )
    .reset_index()
)

coverage["coverage_rate"] = (
    coverage["disagreement_rows"]
    / coverage["total_rows"]
)

print(
    coverage.to_string(
        index=False
    )
)

print(
    "\nTickers with 100% disagreement coverage:",
    int(
        (
            coverage["coverage_rate"]
            == 1.0
        ).sum()
    )
)

print(
    "Tickers with incomplete disagreement coverage:",
    int(
        (
            coverage["coverage_rate"]
            < 1.0
        ).sum()
    )
)


# =============================================================================
# PRIMARY TARGET IDENTIFICATION
# =============================================================================

print("\n" + "=" * 100)
print("PRIMARY TARGET IDENTIFICATION")
print("=" * 100)

for horizon in HORIZONS:

    target_col = get_primary_target(
        merged,
        horizon
    )

    valid = (
        merged[target_col]
        .dropna()
    )

    print(
        f"\nPrimary {horizon}d target: {target_col}"
    )

    print(
        f"Valid: {len(valid):,}"
    )

    print(
        "Unique values:",
        sorted(
            valid.unique().tolist()
        )
    )

    print(
        f"Positive: {int(valid.sum()):,}"
    )

    print(
        f"Positive rate: {valid.mean():.4%}"
    )


tickers = sorted(
    merged["ticker"]
    .dropna()
    .unique()
)

print(
    f"\nTickers found: {len(tickers)}"
)


# =============================================================================
# STORAGE
# =============================================================================

summary_results = []
ticker_results = []


# =============================================================================
# HORIZON LOOP
# =============================================================================

for horizon in HORIZONS:

    target_col = get_primary_target(
        merged,
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
        f"TARGET  = {target_col}"
    )

    print(
        "=" * 100
    )

    ticker_prediction_tables = []

    eligible_tickers = 0
    total_test_observations = 0


    # =========================================================================
    # TICKER LOOP
    # =========================================================================

    for ticker in tickers:

        g = merged[
            merged["ticker"] == ticker
        ].copy()

        required_columns = [
            "forecast_date",
            target_col,
            "persistence_score",
            "qrf_ewma_disagreement"
        ]

        g = (
            g
            .dropna(
                subset=required_columns
            )
            .sort_values(
                "forecast_date"
            )
            .reset_index(
                drop=True
            )
        )

        # Minimum data check
        if len(g) < 100:
            continue

        split_idx = int(
            np.floor(
                len(g)
                * TRAIN_FRACTION
            )
        )

        if (
            split_idx <= 20
            or
            split_idx >= len(g) - 20
        ):
            continue

        train = g.iloc[
            :split_idx
        ].copy()

        test = g.iloc[
            split_idx:
        ].copy()

        # Both classes must exist in training and test
        if len(
            np.unique(
                train[target_col]
            )
        ) < 2:
            continue

        if len(
            np.unique(
                test[target_col]
            )
        ) < 2:
            continue


        # =====================================================================
        # MODEL 1 — BREACH PERSISTENCE
        # =====================================================================

        persistence_pred = fit_logistic(

            train_x=train[
                [
                    "persistence_score"
                ]
            ],

            train_y=train[
                target_col
            ],

            test_x=test[
                [
                    "persistence_score"
                ]
            ]
        )

        if persistence_pred is None:
            continue


        # =====================================================================
        # MODEL 2 — QRF-EWMA DISAGREEMENT
        # =====================================================================

        disagreement_pred = fit_logistic(

            train_x=train[
                [
                    "qrf_ewma_disagreement"
                ]
            ],

            train_y=train[
                target_col
            ],

            test_x=test[
                [
                    "qrf_ewma_disagreement"
                ]
            ]
        )

        if disagreement_pred is None:
            continue


        # =====================================================================
        # MODEL 3 — PERSISTENCE + DISAGREEMENT
        # =====================================================================

        combined_pred = fit_logistic(

            train_x=train[
                [
                    "persistence_score",
                    "qrf_ewma_disagreement"
                ]
            ],

            train_y=train[
                target_col
            ],

            test_x=test[
                [
                    "persistence_score",
                    "qrf_ewma_disagreement"
                ]
            ]
        )

        if combined_pred is None:
            continue


        # =====================================================================
        # TEST AUC
        # =====================================================================

        y_test = (
            test[target_col]
            .astype(int)
            .to_numpy()
        )

        auc_persistence = compute_auc(
            y_test,
            persistence_pred
        )

        auc_disagreement = compute_auc(
            y_test,
            disagreement_pred
        )

        auc_combined = compute_auc(
            y_test,
            combined_pred
        )

        if (
            np.isnan(auc_persistence)
            or
            np.isnan(auc_disagreement)
            or
            np.isnan(auc_combined)
        ):
            continue


        # =====================================================================
        # TICKER DELTAS
        # =====================================================================

        ticker_delta_disagreement = (
            auc_disagreement
            - auc_persistence
        )

        ticker_delta_combined = (
            auc_combined
            - auc_persistence
        )


        # =====================================================================
        # STORE TICKER RESULT
        # =====================================================================

        ticker_results.append(
            {
                "horizon": horizon,
                "ticker": ticker,

                "train_observations": len(train),
                "test_observations": len(test),

                "persistence_auc":
                    auc_persistence,

                "disagreement_auc":
                    auc_disagreement,

                "combined_auc":
                    auc_combined,

                "ticker_delta_disagreement":
                    ticker_delta_disagreement,

                "ticker_delta_combined":
                    ticker_delta_combined
            }
        )


        # =====================================================================
        # STORE TEST PREDICTIONS
        # =====================================================================

        temp = test[
            [
                "ticker",
                "forecast_date",
                target_col,
                "qrf_ewma_disagreement"
            ]
        ].copy()

        temp["target"] = (
            temp[target_col]
            .astype(int)
        )

        temp["persistence_pred"] = (
            persistence_pred
        )

        temp["disagreement_pred"] = (
            disagreement_pred
        )

        temp["combined_pred"] = (
            combined_pred
        )

        temp["horizon"] = horizon

        ticker_prediction_tables.append(
            temp[
                [
                    "ticker",
                    "forecast_date",
                    "target",
                    "persistence_pred",
                    "disagreement_pred",
                    "combined_pred",
                    "qrf_ewma_disagreement",
                    "horizon"
                ]
            ]
        )

        eligible_tickers += 1
        total_test_observations += len(test)


    # =========================================================================
    # NO RESULTS CHECK
    # =========================================================================

    if not ticker_prediction_tables:

        print(
            "\nNo eligible ticker predictions."
        )

        continue


    # =========================================================================
    # POOLED TEST SET
    # =========================================================================

    test_predictions = pd.concat(
        ticker_prediction_tables,
        ignore_index=True
    )

    y = (
        test_predictions["target"]
        .astype(int)
    )


    # =========================================================================
    # POOLED AUC
    # =========================================================================

    persistence_auc = roc_auc_score(
        y,
        test_predictions[
            "persistence_pred"
        ]
    )

    disagreement_auc = roc_auc_score(
        y,
        test_predictions[
            "disagreement_pred"
        ]
    )

    combined_auc = roc_auc_score(
        y,
        test_predictions[
            "combined_pred"
        ]
    )


    # =========================================================================
    # DELTAS
    # =========================================================================

    delta_disagreement = (
        disagreement_auc
        - persistence_auc
    )

    delta_combined = (
        combined_auc
        - persistence_auc
    )


    # =========================================================================
    # WITHIN-TICKER BLOCK BOOTSTRAP
    # =========================================================================

    bootstrap_input_disagreement = (
        test_predictions[
            [
                "ticker",
                "target",
                "persistence_pred",
                "disagreement_pred"
            ]
        ]
        .rename(
            columns={
                "persistence_pred":
                    "persistence_score",

                "disagreement_pred":
                    "qrf_ewma_disagreement"
            }
        )
    )

    (
        bootstrap_mean_disagreement,
        block_low_disagreement,
        block_high_disagreement
    ) = bootstrap_pooled_delta(

        ticker_test_data=
            bootstrap_input_disagreement,

        signal_column=
            "qrf_ewma_disagreement",

        n_bootstrap=
            WITHIN_TICKER_BOOTSTRAPS,

        block_length=
            BLOCK_LENGTH,

        random_state=
            RANDOM_STATE + horizon
    )


    # -------------------------------------------------------------------------
    # COMBINED MODEL BOOTSTRAP
    #
    # persistence_pred is still the baseline.
    # combined_pred is the signal model.
    # -------------------------------------------------------------------------

    bootstrap_input_combined = (
        test_predictions[
            [
                "ticker",
                "target",
                "persistence_pred",
                "combined_pred"
            ]
        ]
        .rename(
            columns={
                "persistence_pred":
                    "persistence_score",

                "combined_pred":
                    "qrf_ewma_disagreement"
            }
        )
    )

    (
        bootstrap_mean_combined,
        block_low_combined,
        block_high_combined
    ) = bootstrap_pooled_delta(

        ticker_test_data=
            bootstrap_input_combined,

        signal_column=
            "qrf_ewma_disagreement",

        n_bootstrap=
            WITHIN_TICKER_BOOTSTRAPS,

        block_length=
            BLOCK_LENGTH,

        random_state=
            RANDOM_STATE + 100 + horizon
    )


    # =========================================================================
    # TICKER-LEVEL RESULTS FOR THIS HORIZON
    # =========================================================================

    ticker_df_h = pd.DataFrame(
        ticker_results
    )

    ticker_df_h = ticker_df_h[
        ticker_df_h["horizon"] == horizon
    ].copy()


    # =========================================================================
    # CROSS-SECTIONAL BOOTSTRAP — STANDALONE DISAGREEMENT
    # =========================================================================

    (
        mean_ticker_disagreement,
        ticker_low_disagreement,
        ticker_high_disagreement
    ) = bootstrap_ticker_delta(

        ticker_summary=
            ticker_df_h,

        delta_column=
            "ticker_delta_disagreement",

        n_bootstrap=
            TICKER_BOOTSTRAPS,

        random_state=
            RANDOM_STATE + 200 + horizon
    )


    # =========================================================================
    # CROSS-SECTIONAL BOOTSTRAP — COMBINED MODEL
    # =========================================================================

    (
        mean_ticker_combined,
        ticker_low_combined,
        ticker_high_combined
    ) = bootstrap_ticker_delta(

        ticker_summary=
            ticker_df_h,

        delta_column=
            "ticker_delta_combined",

        n_bootstrap=
            TICKER_BOOTSTRAPS,

        random_state=
            RANDOM_STATE + 300 + horizon
    )


    # =========================================================================
    # BETTER TICKERS
    # =========================================================================

    better_disagreement = int(
        (
            ticker_df_h[
                "ticker_delta_disagreement"
            ] > 0
        ).sum()
    )

    better_combined = int(
        (
            ticker_df_h[
                "ticker_delta_combined"
            ] > 0
        ).sum()
    )

    total_tickers = len(
        ticker_df_h
    )


    # =========================================================================
    # PRINT RESULTS — STANDALONE
    # =========================================================================

    print(
        f"\nEligible tickers:       "
        f"{eligible_tickers}"
    )

    print(
        f"Test observations:      "
        f"{total_test_observations:,}"
    )

    print(
        "\n--- STANDALONE QRF-EWMA DISAGREEMENT ---"
    )

    print(
        f"Persistence AUC:        "
        f"{persistence_auc:.4f}"
    )

    print(
        f"QRF-EWMA disagreement:  "
        f"{disagreement_auc:.4f}"
    )

    print(
        f"Delta AUC:              "
        f"{delta_disagreement:+.4f}"
    )

    print(
        f"Block bootstrap mean:    "
        f"{bootstrap_mean_disagreement:+.4f}"
    )

    print(
        f"Block CI:               "
        f"[{block_low_disagreement:+.4f}, "
        f"{block_high_disagreement:+.4f}]"
    )

    print(
        f"Mean ticker Delta:      "
        f"{mean_ticker_disagreement:+.4f}"
    )

    print(
        f"Cross-sectional CI:     "
        f"[{ticker_low_disagreement:+.4f}, "
        f"{ticker_high_disagreement:+.4f}]"
    )

    print(
        f"Better tickers:         "
        f"{better_disagreement}/{total_tickers}"
    )

    print(
        "\nBlock CI entirely > 0:  ",
        block_low_disagreement > 0
    )

    print(
        "Ticker CI entirely > 0:",
        ticker_low_disagreement > 0
    )


    # =========================================================================
    # PRINT RESULTS — COMBINED
    # =========================================================================

    print(
        "\n--- PERSISTENCE + QRF-EWMA DISAGREEMENT ---"
    )

    print(
        f"Persistence AUC:        "
        f"{persistence_auc:.4f}"
    )

    print(
        f"Combined AUC:           "
        f"{combined_auc:.4f}"
    )

    print(
        f"Delta AUC:              "
        f"{delta_combined:+.4f}"
    )

    print(
        f"Block bootstrap mean:    "
        f"{bootstrap_mean_combined:+.4f}"
    )

    print(
        f"Block CI:               "
        f"[{block_low_combined:+.4f}, "
        f"{block_high_combined:+.4f}]"
    )

    print(
        f"Mean ticker Delta:      "
        f"{mean_ticker_combined:+.4f}"
    )

    print(
        f"Cross-sectional CI:     "
        f"[{ticker_low_combined:+.4f}, "
        f"{ticker_high_combined:+.4f}]"
    )

    print(
        f"Better tickers:         "
        f"{better_combined}/{total_tickers}"
    )

    print(
        "\nCombined block CI entirely > 0:",
        block_low_combined > 0
    )

    print(
        "Combined ticker CI entirely > 0:",
        ticker_low_combined > 0
    )


    # =========================================================================
    # SAVE HORIZON SUMMARY
    # =========================================================================

    summary_results.append(
        {
            "horizon":
                horizon,

            "eligible_tickers":
                eligible_tickers,

            "test_observations":
                total_test_observations,

            "persistence_auc":
                persistence_auc,

            "disagreement_auc":
                disagreement_auc,

            "delta_auc_disagreement":
                delta_disagreement,

            "block_bootstrap_mean_disagreement":
                bootstrap_mean_disagreement,

            "block_ci_low_disagreement":
                block_low_disagreement,

            "block_ci_high_disagreement":
                block_high_disagreement,

            "mean_ticker_delta_disagreement":
                mean_ticker_disagreement,

            "ticker_ci_low_disagreement":
                ticker_low_disagreement,

            "ticker_ci_high_disagreement":
                ticker_high_disagreement,

            "better_tickers_disagreement":
                better_disagreement,

            "combined_auc":
                combined_auc,

            "delta_auc_combined":
                delta_combined,

            "block_bootstrap_mean_combined":
                bootstrap_mean_combined,

            "block_ci_low_combined":
                block_low_combined,

            "block_ci_high_combined":
                block_high_combined,

            "mean_ticker_delta_combined":
                mean_ticker_combined,

            "ticker_ci_low_combined":
                ticker_low_combined,

            "ticker_ci_high_combined":
                ticker_high_combined,

            "better_tickers_combined":
                better_combined
        }
    )


# =============================================================================
# FINAL DATAFRAMES
# =============================================================================

results_df = pd.DataFrame(
    summary_results
)

ticker_results_df = pd.DataFrame(
    ticker_results
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FINAL EXPERIMENT 31 SUMMARY"
)

print(
    "=" * 100
)

if not results_df.empty:

    display_columns = [
        "horizon",
        "eligible_tickers",
        "test_observations",

        "persistence_auc",

        "disagreement_auc",
        "delta_auc_disagreement",

        "block_ci_low_disagreement",
        "block_ci_high_disagreement",

        "combined_auc",
        "delta_auc_combined",

        "block_ci_low_combined",
        "block_ci_high_combined"
    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# =============================================================================
# ROBUSTNESS CONCLUSION
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "ROBUSTNESS CONCLUSION"
)

print(
    "=" * 100
)

if not results_df.empty:

    standalone_robust = (
        results_df[
            "block_ci_low_disagreement"
        ] > 0
    ).any()

    combined_robust = (
        results_df[
            "block_ci_low_combined"
        ] > 0
    ).any()

    ticker_standalone_robust = (
        results_df[
            "ticker_ci_low_disagreement"
        ] > 0
    ).any()

    ticker_combined_robust = (
        results_df[
            "ticker_ci_low_combined"
        ] > 0
    ).any()


    print(
        "\nStandalone QRF-EWMA disagreement:"
    )

    print(
        "  Any horizon with within-ticker "
        "block CI entirely > 0:",
        standalone_robust
    )

    print(
        "  Any horizon with cross-sectional "
        "ticker CI entirely > 0:",
        ticker_standalone_robust
    )


    print(
        "\nPersistence + QRF-EWMA disagreement:"
    )

    print(
        "  Any horizon with within-ticker "
        "block CI entirely > 0:",
        combined_robust
    )

    print(
        "  Any horizon with cross-sectional "
        "ticker CI entirely > 0:",
        ticker_combined_robust
    )


    if (
        not standalone_robust
        and not combined_robust
    ):

        print(
            "\nConclusion:"
        )

        print(
            "  QRF-EWMA disagreement does not provide "
            "statistically robust incremental predictive "
            "value beyond breach persistence."
        )

    else:

        print(
            "\nConclusion:"
        )

        print(
            "  At least one specification/horizon shows "
            "a statistically positive incremental result."
        )

        print(
            "  Inspect the horizon-specific estimates before "
            "drawing conclusions."
        )


# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df.to_csv(
    OUTPUT_MAIN,
    index=False
)

ticker_results_df.to_csv(
    OUTPUT_TICKER,
    index=False
)


# =============================================================================
# SAVE DISAGREEMENT DATASET FOR TRACEABILITY
# =============================================================================

DISAGREEMENT_DATASET = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_qrf_ewma_disagreement.csv"
)

save_columns = [
    "ticker",
    "forecast_date",
    "realized_date",
    "qrf_q05",
    "ewma_q05",
    "qrf_ewma_disagreement"
]

existing_save_columns = [
    c for c in save_columns
    if c in merged.columns
]

merged[
    existing_save_columns
].to_csv(
    DISAGREEMENT_DATASET,
    index=False
)


# =============================================================================
# FINAL SAVE MESSAGE
# =============================================================================

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
    OUTPUT_TICKER
)

print(
    DISAGREEMENT_DATASET
)

print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 31 COMPLETE"
)

print(
    "=" * 100
)