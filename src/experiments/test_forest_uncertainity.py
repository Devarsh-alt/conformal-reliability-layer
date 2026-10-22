"""
EXPERIMENT 30
FOREST ENSEMBLE DISPERSION VS BREACH PERSISTENCE

Tests whether inter-tree Random Forest prediction dispersion adds
predictive information beyond breach persistence for future QRF
VaR deterioration.

Primary causal targets:
    deterioration_5d
    deterioration_10d
    deterioration_20d

Evaluation:
    - chronological 60/40 split within ticker
    - univariate logistic regression
    - pooled AUC
    - ticker-level AUC
    - Delta AUC
    - within-ticker moving-block bootstrap CI
    - cross-sectional ticker bootstrap CI
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PANEL_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

FOREST_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_forest_uncertainty.csv"
)

OUTPUT_SUMMARY = (
    ROOT
    / "data"
    / "monitoring"
    / "forest_uncertainty_test_results.csv"
)

OUTPUT_TICKER = (
    ROOT
    / "data"
    / "monitoring"
    / "forest_uncertainty_test_by_ticker.csv"
)

# Primary experiment settings
HORIZONS = [5, 10, 20]

TRAIN_FRACTION = 0.60

BLOCK_LENGTH = 20

N_BLOCK_BOOTSTRAPS = 1000

N_TICKER_BOOTSTRAPS = 2000

RANDOM_STATE = 42

MIN_TRAIN_ROWS = 100

BASELINE_SIGNAL = "breach_lag1"

FOREST_SIGNAL = "forest_uncertainty_norm"


# =============================================================================
# HELPER: FIND COLUMN
# =============================================================================

def find_column(df, candidates, description):
    """
    Find one column using exact case-insensitive matching.
    """

    column_map = {
        str(column).lower(): column
        for column in df.columns
    }

    matches = []

    for candidate in candidates:

        candidate_lower = candidate.lower()

        if candidate_lower in column_map:
            matches.append(
                column_map[candidate_lower]
            )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:

        raise ValueError(
            f"\nCould not find {description}.\n"
            f"Candidates: {candidates}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    raise ValueError(
        f"\nMultiple columns found for {description}: "
        f"{matches}\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# HELPER: PRIMARY TARGET
# =============================================================================

def get_primary_target(df, horizon):
    """
    Explicitly select the binary primary causal target.

    IMPORTANT:
    Do not search using generic 'deterioration' substring matching,
    because the dataset also contains continuous historical thresholds.
    """

    target = f"deterioration_{horizon}d"

    if target not in df.columns:

        raise ValueError(
            f"\nMissing primary target: {target}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    return target


# =============================================================================
# HELPER: AUC
# =============================================================================

def calculate_auc(y, predictions):
    """
    Safe ROC AUC.
    """

    y = np.asarray(y)

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if len(np.unique(y)) < 2:
        return np.nan

    try:
        return float(
            roc_auc_score(
                y,
                predictions,
            )
        )

    except ValueError:
        return np.nan


# =============================================================================
# HELPER: LOGISTIC REGRESSION
# =============================================================================

def fit_logistic(
    x_train,
    y_train,
    x_test,
):
    """
    Train:
        StandardScaler
        LogisticRegression

    Both scaling and model fitting use training data only.
    """

    x_train = np.asarray(
        x_train,
        dtype=float,
    ).reshape(-1, 1)

    x_test = np.asarray(
        x_test,
        dtype=float,
    ).reshape(-1, 1)

    y_train = np.asarray(
        y_train,
        dtype=int,
    )

    # Cannot train a classifier with one class.
    if len(np.unique(y_train)) < 2:
        return None

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(
        x_train
    )

    x_test_scaled = scaler.transform(
        x_test
    )

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    model.fit(
        x_train_scaled,
        y_train,
    )

    return model.predict_proba(
        x_test_scaled
    )[:, 1]


# =============================================================================
# HELPER: MOVING BLOCK BOOTSTRAP INDICES
# =============================================================================

def moving_block_indices(
    n,
    block_length,
    rng,
):
    """
    Generate n observations using moving-block bootstrap.
    """

    if n <= 1:
        return np.arange(n)

    block_length = min(
        block_length,
        n,
    )

    indices = []

    while len(indices) < n:

        start = rng.integers(
            0,
            n - block_length + 1,
        )

        block = np.arange(
            start,
            start + block_length,
        )

        indices.extend(
            block.tolist()
        )

    return np.asarray(
        indices[:n]
    )


# =============================================================================
# WITHIN-TICKER BLOCK BOOTSTRAP
# =============================================================================

def bootstrap_block_delta(
    ticker_objects,
    n_boot,
    block_length,
    random_state,
):
    """
    Bootstrap temporal blocks independently within each ticker.

    Returns:
        2.5th percentile
        97.5th percentile
    """

    if len(ticker_objects) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(
        random_state
    )

    bootstrap_deltas = []

    attempts = 0

    max_attempts = n_boot * 5

    while (
        len(bootstrap_deltas) < n_boot
        and attempts < max_attempts
    ):

        attempts += 1

        pooled_y = []

        pooled_baseline = []

        pooled_forest = []

        for item in ticker_objects:

            y = item["y"]

            baseline = item[
                "baseline_predictions"
            ]

            forest = item[
                "forest_predictions"
            ]

            indices = moving_block_indices(
                len(y),
                block_length,
                rng,
            )

            pooled_y.extend(
                y[indices]
                .tolist()
            )

            pooled_baseline.extend(
                baseline[indices]
                .tolist()
            )

            pooled_forest.extend(
                forest[indices]
                .tolist()
            )

        pooled_y = np.asarray(
            pooled_y,
            dtype=int,
        )

        pooled_baseline = np.asarray(
            pooled_baseline,
            dtype=float,
        )

        pooled_forest = np.asarray(
            pooled_forest,
            dtype=float,
        )

        if len(np.unique(pooled_y)) < 2:
            continue

        try:

            baseline_auc = roc_auc_score(
                pooled_y,
                pooled_baseline,
            )

            forest_auc = roc_auc_score(
                pooled_y,
                pooled_forest,
            )

            bootstrap_deltas.append(
                forest_auc - baseline_auc
            )

        except ValueError:

            continue

    if len(bootstrap_deltas) < max(
        100,
        n_boot // 10,
    ):
        return np.nan, np.nan

    bootstrap_deltas = np.asarray(
        bootstrap_deltas
    )

    lower = np.percentile(
        bootstrap_deltas,
        2.5,
    )

    upper = np.percentile(
        bootstrap_deltas,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# CROSS-SECTIONAL TICKER BOOTSTRAP
# =============================================================================

def bootstrap_ticker_delta(
    ticker_deltas,
    n_boot,
    random_state,
):
    """
    Bootstrap ticker-level Delta AUC values.

    Each bootstrap replicate samples tickers with replacement and
    calculates the mean ticker Delta AUC.
    """

    values = np.asarray(
        ticker_deltas,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(
        random_state
    )

    n = len(values)

    bootstrap_means = []

    for _ in range(n_boot):

        sample = rng.choice(
            values,
            size=n,
            replace=True,
        )

        bootstrap_means.append(
            np.mean(sample)
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    lower = np.percentile(
        bootstrap_means,
        2.5,
    )

    upper = np.percentile(
        bootstrap_means,
        97.5,
    )

    return (
        float(lower),
        float(upper),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print(
        "EXPERIMENT 30 — FOREST ENSEMBLE DISPERSION VS "
        "BREACH PERSISTENCE"
    )
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------

    print()
    print("ROOT:")
    print(ROOT)

    print()
    print("PANEL:")
    print(PANEL_FILE)

    print()
    print("FOREST DISPERSION:")
    print(FOREST_FILE)

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    print()
    print("Configuration:")
    print(
        f"Train fraction:       {TRAIN_FRACTION:.2f}"
    )

    print(
        f"Block length:         {BLOCK_LENGTH}"
    )

    print(
        f"Within-ticker boots:  {N_BLOCK_BOOTSTRAPS}"
    )

    print(
        f"Ticker boots:         {N_TICKER_BOOTSTRAPS}"
    )

    print(
        f"Random state:         {RANDOM_STATE}"
    )

    # -------------------------------------------------------------------------
    # Check files
    # -------------------------------------------------------------------------

    if not PANEL_FILE.exists():

        sys.exit(
            f"\nERROR: Panel file not found:\n"
            f"{PANEL_FILE}"
        )

    if not FOREST_FILE.exists():

        sys.exit(
            f"\nERROR: Forest file not found:\n"
            f"{FOREST_FILE}"
        )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    panel = pd.read_csv(
        PANEL_FILE
    )

    forest = pd.read_csv(
        FOREST_FILE
    )

    print()
    print("Loaded panel shape:")
    print(panel.shape)

    print()
    print("Loaded forest shape:")
    print(forest.shape)

    print()
    print("Panel columns:")
    print(list(panel.columns))

    print()
    print("Forest columns:")
    print(list(forest.columns))

    # -------------------------------------------------------------------------
    # Identify columns
    # -------------------------------------------------------------------------

    panel_ticker_col = find_column(
        panel,
        ["ticker"],
        "panel ticker column",
    )

    panel_date_col = find_column(
        panel,
        [
            "forecast_date",
            "date",
            "Date",
        ],
        "panel forecast date column",
    )

    forest_ticker_col = find_column(
        forest,
        ["ticker"],
        "forest ticker column",
    )

    forest_date_col = find_column(
        forest,
        [
            "forecast_date",
            "date",
            "Date",
        ],
        "forest forecast date column",
    )

    # -------------------------------------------------------------------------
    # Required signals
    # -------------------------------------------------------------------------

    if BASELINE_SIGNAL not in panel.columns:

        raise ValueError(
            f"\nMissing baseline signal: "
            f"{BASELINE_SIGNAL}"
        )

    if FOREST_SIGNAL not in forest.columns:

        raise ValueError(
            f"\nMissing forest signal: "
            f"{FOREST_SIGNAL}"
        )

    # -------------------------------------------------------------------------
    # Create merge keys
    # -------------------------------------------------------------------------

    panel["__ticker"] = (
        panel[panel_ticker_col]
        .astype(str)
    )

    forest["__ticker"] = (
        forest[forest_ticker_col]
        .astype(str)
    )

    panel["__forecast_date"] = pd.to_datetime(
        panel[panel_date_col],
        errors="coerce",
    )

    forest["__forecast_date"] = pd.to_datetime(
        forest[forest_date_col],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Duplicate checks
    # -------------------------------------------------------------------------

    panel_duplicates = panel.duplicated(
        [
            "__ticker",
            "__forecast_date",
        ]
    ).sum()

    forest_duplicates = forest.duplicated(
        [
            "__ticker",
            "__forecast_date",
        ]
    ).sum()

    print()
    print("=" * 100)
    print("KEY SANITY CHECKS")
    print("=" * 100)

    print(
        f"Panel duplicate ticker-date rows:  "
        f"{panel_duplicates}"
    )

    print(
        f"Forest duplicate ticker-date rows: "
        f"{forest_duplicates}"
    )

    if panel_duplicates > 0:

        raise ValueError(
            "Panel contains duplicate ticker-date rows."
        )

    if forest_duplicates > 0:

        raise ValueError(
            "Forest file contains duplicate ticker-date rows."
        )

    # -------------------------------------------------------------------------
    # Select forest signal
    # -------------------------------------------------------------------------

    forest_small = forest[
        [
            "__ticker",
            "__forecast_date",
            FOREST_SIGNAL,
        ]
    ].copy()

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------

    merged = panel.merge(
        forest_small,
        on=[
            "__ticker",
            "__forecast_date",
        ],
        how="inner",
        validate="one_to_one",
    )

    print()
    print("Merged rows:")
    print(
        f"{len(merged):,}"
    )

    print()
    print("Panel rows:")
    print(
        f"{len(panel):,}"
    )

    if len(merged) != len(panel):

        print()
        print(
            "WARNING: not every panel row contains forest dispersion."
        )

    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    merged[BASELINE_SIGNAL] = pd.to_numeric(
        merged[BASELINE_SIGNAL],
        errors="coerce",
    )

    merged[FOREST_SIGNAL] = pd.to_numeric(
        merged[FOREST_SIGNAL],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Forest signal sanity check
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("FOREST SIGNAL SANITY CHECK")
    print("=" * 100)

    print()
    print(
        merged[
            FOREST_SIGNAL
        ].describe()
    )

    missing_forest = (
        merged[FOREST_SIGNAL]
        .isna()
        .sum()
    )

    negative_forest = (
        merged[FOREST_SIGNAL] < 0
    ).sum()

    print()
    print(
        f"Missing forest signal: "
        f"{missing_forest:,}"
    )

    print(
        f"Negative forest signal: "
        f"{negative_forest:,}"
    )

    if negative_forest > 0:

        raise ValueError(
            "Forest dispersion cannot be negative."
        )

    # -------------------------------------------------------------------------
    # Identify targets explicitly
    # -------------------------------------------------------------------------

    target_columns = {}

    print()
    print("=" * 100)
    print("PRIMARY TARGET IDENTIFICATION")
    print("=" * 100)

    for horizon in HORIZONS:

        target = get_primary_target(
            merged,
            horizon,
        )

        target_columns[
            horizon
        ] = target

        merged[target] = (
            pd.to_numeric(
                merged[target],
                errors="coerce",
            )
        )

        valid = (
            merged[target]
            .notna()
        )

        unique_values = sorted(
            merged.loc[
                valid,
                target,
            ]
            .unique()
            .tolist()
        )

        print()
        print(
            f"Primary {horizon}d target: "
            f"{target}"
        )

        print(
            f"Valid: "
            f"{valid.sum():,}"
        )

        print(
            f"Unique values: "
            f"{unique_values}"
        )

        positives = int(
            merged.loc[
                valid,
                target,
            ].sum()
        )

        positive_rate = (
            positives
            / valid.sum()
            if valid.sum() > 0
            else np.nan
        )

        print(
            f"Positive: "
            f"{positives:,}"
        )

        print(
            f"Positive rate: "
            f"{positive_rate:.4%}"
        )

        if not set(
            unique_values
        ).issubset({0, 1}):

            raise ValueError(
                f"{target} is not binary 0/1."
            )

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    merged = (
        merged
        .sort_values(
            [
                "__ticker",
                "__forecast_date",
            ]
        )
        .reset_index(drop=True)
    )

    tickers = sorted(
        merged[
            "__ticker"
        ]
        .dropna()
        .unique()
    )

    print()
    print(
        f"Tickers found: {len(tickers)}"
    )

    # -------------------------------------------------------------------------
    # Results containers
    # -------------------------------------------------------------------------

    summary_results = []

    ticker_results = []

    # =========================================================================
    # HORIZON LOOP
    # =========================================================================

    for horizon in HORIZONS:

        target_col = target_columns[
            horizon
        ]

        print()
        print()
        print("=" * 100)
        print(
            f"HORIZON = {horizon} DAYS"
        )
        print(
            f"TARGET  = {target_col}"
        )
        print("=" * 100)

        pooled_y = []

        pooled_baseline = []

        pooled_forest = []

        block_bootstrap_objects = []

        ticker_deltas = []

        eligible_tickers = 0

        total_test_rows = 0

        # =====================================================================
        # TICKER LOOP
        # =====================================================================

        for ticker in tickers:

            ticker_df = merged[
                merged["__ticker"] == ticker
            ].copy()

            # -----------------------------------------------------------------
            # Complete cases
            # -----------------------------------------------------------------

            ticker_df = ticker_df.dropna(
                subset=[
                    BASELINE_SIGNAL,
                    FOREST_SIGNAL,
                    target_col,
                ]
            ).copy()

            if len(ticker_df) < (
                MIN_TRAIN_ROWS + 30
            ):
                continue

            if (
                ticker_df[target_col]
                .nunique()
                < 2
            ):
                continue

            # -----------------------------------------------------------------
            # Chronological split
            # -----------------------------------------------------------------

            n = len(ticker_df)

            split_index = int(
                np.floor(
                    n * TRAIN_FRACTION
                )
            )

            if split_index < MIN_TRAIN_ROWS:
                continue

            if split_index >= n:
                continue

            train = ticker_df.iloc[
                :split_index
            ].copy()

            test = ticker_df.iloc[
                split_index:
            ].copy()

            if len(test) < 30:
                continue

            if (
                train[target_col]
                .nunique()
                < 2
            ):
                continue

            if (
                test[target_col]
                .nunique()
                < 2
            ):
                continue

            # -----------------------------------------------------------------
            # Baseline model
            # -----------------------------------------------------------------

            baseline_predictions = fit_logistic(
                train[
                    BASELINE_SIGNAL
                ].values,
                train[
                    target_col
                ].values,
                test[
                    BASELINE_SIGNAL
                ].values,
            )

            # -----------------------------------------------------------------
            # Forest dispersion model
            # -----------------------------------------------------------------

            forest_predictions = fit_logistic(
                train[
                    FOREST_SIGNAL
                ].values,
                train[
                    target_col
                ].values,
                test[
                    FOREST_SIGNAL
                ].values,
            )

            if baseline_predictions is None:
                continue

            if forest_predictions is None:
                continue

            # -----------------------------------------------------------------
            # Test target
            # -----------------------------------------------------------------

            y_test = (
                test[target_col]
                .astype(int)
                .values
            )

            # -----------------------------------------------------------------
            # AUCs
            # -----------------------------------------------------------------

            baseline_auc = calculate_auc(
                y_test,
                baseline_predictions,
            )

            forest_auc = calculate_auc(
                y_test,
                forest_predictions,
            )

            if not np.isfinite(
                baseline_auc
            ):
                continue

            if not np.isfinite(
                forest_auc
            ):
                continue

            delta_auc = (
                forest_auc
                - baseline_auc
            )

            # -----------------------------------------------------------------
            # Store pooled data
            # -----------------------------------------------------------------

            pooled_y.extend(
                y_test.tolist()
            )

            pooled_baseline.extend(
                baseline_predictions.tolist()
            )

            pooled_forest.extend(
                forest_predictions.tolist()
            )

            # -----------------------------------------------------------------
            # Store bootstrap information
            # -----------------------------------------------------------------

            block_bootstrap_objects.append(
                {
                    "ticker": ticker,
                    "n_test": len(y_test),
                    "y": y_test,
                    "baseline_predictions":
                        baseline_predictions,
                    "forest_predictions":
                        forest_predictions,
                }
            )

            # -----------------------------------------------------------------
            # Store ticker statistics
            # -----------------------------------------------------------------

            ticker_deltas.append(
                delta_auc
            )

            ticker_results.append(
                {
                    "horizon": horizon,
                    "ticker": ticker,
                    "n_total": n,
                    "n_train": len(train),
                    "n_test": len(test),
                    "train_positive":
                        int(
                            train[target_col]
                            .sum()
                        ),
                    "test_positive":
                        int(
                            test[target_col]
                            .sum()
                        ),
                    "baseline_auc":
                        baseline_auc,
                    "forest_dispersion_auc":
                        forest_auc,
                    "delta_auc":
                        delta_auc,
                    "forest_better":
                        int(
                            delta_auc > 0
                        ),
                }
            )

            eligible_tickers += 1

            total_test_rows += len(
                test
            )

        # =====================================================================
        # POOLED AUC
        # =====================================================================

        pooled_y = np.asarray(
            pooled_y,
            dtype=int,
        )

        pooled_baseline = np.asarray(
            pooled_baseline,
            dtype=float,
        )

        pooled_forest = np.asarray(
            pooled_forest,
            dtype=float,
        )

        pooled_baseline_auc = calculate_auc(
            pooled_y,
            pooled_baseline,
        )

        pooled_forest_auc = calculate_auc(
            pooled_y,
            pooled_forest,
        )

        pooled_delta_auc = (
            pooled_forest_auc
            - pooled_baseline_auc
        )

        # =====================================================================
        # MEAN TICKER DELTA
        # =====================================================================

        ticker_deltas = np.asarray(
            ticker_deltas,
            dtype=float,
        )

        mean_ticker_delta = float(
            np.mean(
                ticker_deltas
            )
        )

        better_tickers = int(
            np.sum(
                ticker_deltas > 0
            )
        )

        # =====================================================================
        # WITHIN-TICKER BLOCK BOOTSTRAP
        # =====================================================================

        block_ci_low, block_ci_high = (
            bootstrap_block_delta(
                block_bootstrap_objects,
                n_boot=N_BLOCK_BOOTSTRAPS,
                block_length=BLOCK_LENGTH,
                random_state=(
                    RANDOM_STATE
                    + horizon
                ),
            )
        )

        # =====================================================================
        # CROSS-SECTIONAL TICKER BOOTSTRAP
        # =====================================================================

        ticker_ci_low, ticker_ci_high = (
            bootstrap_ticker_delta(
                ticker_deltas,
                n_boot=N_TICKER_BOOTSTRAPS,
                random_state=(
                    RANDOM_STATE
                    + 1000
                    + horizon
                ),
            )
        )

        # =====================================================================
        # CI FLAGS
        # =====================================================================

        block_ci_positive = (
            np.isfinite(block_ci_low)
            and block_ci_low > 0
        )

        ticker_ci_positive = (
            np.isfinite(ticker_ci_low)
            and ticker_ci_low > 0
        )

        # =====================================================================
        # PRINT
        # =====================================================================

        print()
        print(
            f"Eligible tickers:       "
            f"{eligible_tickers}"
        )

        print(
            f"Test observations:      "
            f"{total_test_rows:,}"
        )

        print()

        print(
            f"Persistence AUC:        "
            f"{pooled_baseline_auc:.4f}"
        )

        print(
            f"Forest dispersion AUC:  "
            f"{pooled_forest_auc:.4f}"
        )

        print(
            f"Delta AUC:              "
            f"{pooled_delta_auc:+.4f}"
        )

        print(
            f"Block CI:               "
            f"[{block_ci_low:+.4f}, "
            f"{block_ci_high:+.4f}]"
        )

        print(
            f"Mean ticker Delta:      "
            f"{mean_ticker_delta:+.4f}"
        )

        print(
            f"Cross-sectional CI:     "
            f"[{ticker_ci_low:+.4f}, "
            f"{ticker_ci_high:+.4f}]"
        )

        print(
            f"Better tickers:         "
            f"{better_tickers}/"
            f"{eligible_tickers}"
        )

        print()

        print(
            f"Block CI entirely > 0:  "
            f"{block_ci_positive}"
        )

        print(
            f"Ticker CI entirely > 0: "
            f"{ticker_ci_positive}"
        )

        # =====================================================================
        # STORE SUMMARY
        # =====================================================================

        summary_results.append(
            {
                "horizon": horizon,
                "baseline":
                    "breach_persistence",
                "signal":
                    "forest_ensemble_dispersion",
                "eligible_tickers":
                    eligible_tickers,
                "test_observations":
                    total_test_rows,
                "persistence_auc":
                    pooled_baseline_auc,
                "forest_dispersion_auc":
                    pooled_forest_auc,
                "delta_auc":
                    pooled_delta_auc,
                "block_ci_low":
                    block_ci_low,
                "block_ci_high":
                    block_ci_high,
                "mean_ticker_delta":
                    mean_ticker_delta,
                "ticker_ci_low":
                    ticker_ci_low,
                "ticker_ci_high":
                    ticker_ci_high,
                "better_tickers":
                    better_tickers,
                "block_ci_entirely_positive":
                    int(
                        block_ci_positive
                    ),
                "ticker_ci_entirely_positive":
                    int(
                        ticker_ci_positive
                    ),
            }
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    summary_df = pd.DataFrame(
        summary_results
    )

    ticker_df = pd.DataFrame(
        ticker_results
    )

    print()
    print()
    print("=" * 100)
    print("FINAL EXPERIMENT 30 SUMMARY")
    print("=" * 100)

    print()

    print(
        summary_df[
            [
                "horizon",
                "eligible_tickers",
                "test_observations",
                "persistence_auc",
                "forest_dispersion_auc",
                "delta_auc",
                "block_ci_low",
                "block_ci_high",
                "mean_ticker_delta",
                "ticker_ci_low",
                "ticker_ci_high",
                "better_tickers",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # =========================================================================
    # ROBUSTNESS CONCLUSION
    # =========================================================================

    print()
    print()
    print("=" * 100)
    print("ROBUSTNESS CONCLUSION")
    print("=" * 100)

    robust_block = summary_df[
        summary_df[
            "block_ci_entirely_positive"
        ] == 1
    ]

    robust_ticker = summary_df[
        summary_df[
            "ticker_ci_entirely_positive"
        ] == 1
    ]

    print()

    if len(robust_block) == 0:

        print(
            "Within-ticker block bootstrap:"
        )

        print(
            "  No horizon has a 95% CI entirely above zero."
        )

    else:

        print(
            "Within-ticker block bootstrap:"
        )

        for _, row in robust_block.iterrows():

            print(
                f"  {int(row['horizon'])}-day: "
                f"Delta={row['delta_auc']:+.4f}, "
                f"CI=["
                f"{row['block_ci_low']:+.4f}, "
                f"{row['block_ci_high']:+.4f}]"
            )

    print()

    if len(robust_ticker) == 0:

        print(
            "Cross-sectional ticker bootstrap:"
        )

        print(
            "  No horizon has a 95% CI entirely above zero."
        )

    else:

        print(
            "Cross-sectional ticker bootstrap:"
        )

        for _, row in robust_ticker.iterrows():

            print(
                f"  {int(row['horizon'])}-day: "
                f"Mean Delta={row['mean_ticker_delta']:+.4f}, "
                f"CI=["
                f"{row['ticker_ci_low']:+.4f}, "
                f"{row['ticker_ci_high']:+.4f}]"
            )

    # =========================================================================
    # SAVE
    # =========================================================================

    summary_df.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    ticker_df.to_csv(
        OUTPUT_TICKER,
        index=False,
    )

    print()
    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)

    print()
    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_TICKER
    )

    print()
    print("=" * 100)
    print("EXPERIMENT 30 COMPLETE")
    print("=" * 100)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()