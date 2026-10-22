"""
EXPERIMENT 29 — FAST FOREST UNCERTAINTY

Purpose
-------
Construct an ensemble-dispersion reliability signal without
re-running the expensive quantile_forest fitting procedure.

The official QRF q05 forecast is taken directly from:

    QRF_25ticker_oos_predictions.csv

A standard RandomForestRegressor is fitted using the same
frozen forest configuration:

    rolling window      = 1000
    min_samples_leaf    = 20
    max_features        = 1.0
    n_estimators        = 500
    random_state        = 42
    refit frequency     = every 20 observations

For each OOS block:

    1. Recover the exact 1000-observation training sample.
    2. Fit a standard random forest.
    3. Get predictions from all 500 trees.
    4. Compute their cross-sectional standard deviation.
    5. Normalize by the absolute SAVED QRF q05.

Main signal:

    forest_uncertainty_norm
        = SD(tree predictions) / |QRF q05|

IMPORTANT RESEARCH TERMINOLOGY
------------------------------
This is called:

    Inter-Tree Prediction Dispersion

We do NOT call it:
    - individual-tree q05 dispersion
    - epistemic uncertainty
    - quantile uncertainty

because individual sklearn regression trees produce conditional
mean predictions, not independent 5% quantile estimates.

The official VaR forecast remains the saved QRF q05.
"""


from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW = 1000

MIN_SAMPLES_LEAF = 20
MAX_FEATURES = 1.0
N_ESTIMATORS = 500

REFIT_EVERY = 20
ALPHA = 0.05

RANDOM_STATE = 42
N_JOBS = -1

TEST_START = pd.Timestamp("2023-01-01")


FEATURES = [
    "Log_Return",
    "return_lag1",
    "return_lag5",
    "return_lag10",
    "vol_10d",
    "vol_20d",
    "volume_ratio",
    "VIX",
]


TICKERS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CAT",
    "COIN",
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


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = (
    SCRIPT_DIR.parent.parent
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

MONITORING_DIR = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
)

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Existing frozen QRF predictions
QRF_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_oos_predictions.csv"
)


# Output
OUTPUT_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_forest_uncertainty.csv"
)


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_processed_data(
    df: pd.DataFrame,
    ticker: str
) -> None:

    required = [
        "Date"
    ] + FEATURES

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{ticker}: missing columns: {missing}"
        )


def prepare_processed_data(
    ticker: str
) -> pd.DataFrame:

    file_path = (
        PROCESSED_DIR
        / f"{ticker}.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(
            f"Missing processed file:\n"
            f"{file_path}"
        )

    df = pd.read_csv(
        file_path
    )

    validate_processed_data(
        df,
        ticker
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Keep exactly the model features used by QRF.
    # --------------------------------------------------------

    df_model = df[
        ["Date"] + FEATURES
    ].copy()

    for col in FEATURES:

        df_model[col] = pd.to_numeric(
            df_model[col],
            errors="coerce"
        )

    df_model = (
        df_model
        .dropna(
            subset=FEATURES
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Next-day target.
    #
    # X_t -> R_(t+1)
    # --------------------------------------------------------

    df_model["target"] = (
        df_model["Log_Return"]
        .shift(-1)
    )

    df_model["realized_date"] = (
        df_model["Date"]
        .shift(-1)
    )

    df_model = (
        df_model
        .dropna(
            subset=[
                "target",
                "realized_date"
            ]
        )
        .reset_index(drop=True)
    )

    return df_model


# ============================================================
# TREE PREDICTION MATRIX
# ============================================================

def get_tree_prediction_matrix(
    model: RandomForestRegressor,
    X: np.ndarray
) -> np.ndarray:

    """
    Evaluate all fitted trees over a block of observations.

    Returns:

        shape = (n_trees, n_observations)
    """

    estimators = model.estimators_

    if len(estimators) != N_ESTIMATORS:

        raise RuntimeError(
            f"Expected {N_ESTIMATORS} trees, "
            f"got {len(estimators)}."
        )

    predictions = []

    for estimator in estimators:

        pred = estimator.predict(
            X
        )

        pred = np.asarray(
            pred,
            dtype=float
        ).reshape(-1)

        if len(pred) != len(X):

            raise RuntimeError(
                "Individual tree prediction "
                "length mismatch."
            )

        predictions.append(
            pred
        )

    matrix = np.vstack(
        predictions
    )

    if matrix.shape != (
        N_ESTIMATORS,
        len(X)
    ):

        raise RuntimeError(
            f"Unexpected tree matrix shape: "
            f"{matrix.shape}"
        )

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print(
        "EXPERIMENT 29 — FAST INTER-TREE "
        "PREDICTION DISPERSION"
    )
    print("=" * 100)

    print()
    print(
        f"Project root:          {PROJECT_ROOT}"
    )

    print(
        f"Training window:       {WINDOW}"
    )

    print(
        f"Min samples leaf:      {MIN_SAMPLES_LEAF}"
    )

    print(
        f"Max features:          {MAX_FEATURES}"
    )

    print(
        f"Trees:                 {N_ESTIMATORS}"
    )

    print(
        f"Refit frequency:       {REFIT_EVERY}"
    )

    print(
        f"Random state:          {RANDOM_STATE}"
    )

    print()
    print("=" * 100)
    print("LOADING FROZEN QRF FORECASTS")
    print("=" * 100)

    if not QRF_FILE.exists():

        raise FileNotFoundError(
            f"Frozen QRF file not found:\n"
            f"{QRF_FILE}"
        )

    qrf = pd.read_csv(
        QRF_FILE
    )

    required_qrf = [
        "ticker",
        "forecast_date",
        "realized_date",
        "q05_forecast",
        "fit_number",
    ]

    missing_qrf = [
        col
        for col in required_qrf
        if col not in qrf.columns
    ]

    if missing_qrf:

        raise ValueError(
            "Frozen QRF file is missing columns: "
            f"{missing_qrf}"
        )

    qrf["forecast_date"] = pd.to_datetime(
        qrf["forecast_date"],
        errors="coerce"
    )

    qrf["realized_date"] = pd.to_datetime(
        qrf["realized_date"],
        errors="coerce"
    )

    qrf["q05_forecast"] = pd.to_numeric(
        qrf["q05_forecast"],
        errors="coerce"
    )

    qrf["fit_number"] = pd.to_numeric(
        qrf["fit_number"],
        errors="coerce"
    )

    qrf = (
        qrf
        .sort_values(
            [
                "ticker",
                "forecast_date"
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"QRF rows:             {len(qrf):,}"
    )

    print(
        f"QRF tickers:          "
        f"{qrf['ticker'].nunique()}"
    )

    print(
        f"QRF date range:       "
        f"{qrf['forecast_date'].min().date()} "
        f"to "
        f"{qrf['forecast_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Frozen QRF sign check.
    # --------------------------------------------------------

    median_q05 = (
        qrf["q05_forecast"]
        .median()
    )

    print()
    print(
        f"Median frozen QRF q05: "
        f"{median_q05:.8f}"
    )

    if median_q05 >= 0:

        raise RuntimeError(
            "Frozen QRF q05 appears to be positive. "
            "Expected a signed lower-tail quantile."
        )

    all_results = []

    # ========================================================
    # TICKER LOOP
    # ========================================================

    print()
    print("=" * 100)
    print("PROCESSING TICKERS")
    print("=" * 100)

    for ticker in TICKERS:

        print()
        print("-" * 100)
        print(
            f"PROCESSING {ticker}"
        )
        print("-" * 100)

        ticker_qrf = (
            qrf[
                qrf["ticker"] == ticker
            ]
            .copy()
            .sort_values("forecast_date")
            .reset_index(drop=True)
        )

        if len(ticker_qrf) == 0:

            print(
                f"WARNING: no QRF rows for {ticker}"
            )

            continue

        df_model = prepare_processed_data(
            ticker
        )

        print(
            f"Processed rows:       "
            f"{len(df_model)}"
        )

        print(
            f"QRF OOS rows:         "
            f"{len(ticker_qrf)}"
        )

        # ----------------------------------------------------
        # Verify the expected test start.
        # ----------------------------------------------------

        first_qrf_date = (
            ticker_qrf[
                "forecast_date"
            ].min()
        )

        print(
            f"First forecast:        "
            f"{first_qrf_date.date()}"
        )

        # ----------------------------------------------------
        # Group forecasts by frozen QRF fit_number.
        #
        # This reconstructs the exact 20-observation blocks
        # used by the existing QRF OOS predictions.
        # ----------------------------------------------------

        grouped = (
            ticker_qrf
            .groupby(
                "fit_number",
                sort=True
            )
        )

        ticker_results = []

        # ====================================================
        # FIT BLOCKS
        # ====================================================

        for fit_number, block in grouped:

            block = (
                block
                .sort_values("forecast_date")
                .reset_index(drop=True)
            )

            first_forecast_date = (
                block[
                    "forecast_date"
                ].iloc[0]
            )

            last_forecast_date = (
                block[
                    "forecast_date"
                ].iloc[-1]
            )

            # ------------------------------------------------
            # Recover training data.
            #
            # Training must end BEFORE the first forecast date.
            #
            # The last training row's target corresponds to the
            # first forecast day's realized return.
            # ------------------------------------------------

            training_candidates = (
                df_model[
                    df_model["Date"]
                    < first_forecast_date
                ]
                .copy()
            )

            if len(training_candidates) < WINDOW:

                raise RuntimeError(
                    f"{ticker} fit {fit_number}: "
                    f"only "
                    f"{len(training_candidates)} "
                    f"training rows available; "
                    f"need {WINDOW}."
                )

            train_slice = (
                training_candidates
                .tail(WINDOW)
                .copy()
            )

            # ------------------------------------------------
            # Confirm exact training size.
            # ------------------------------------------------

            if len(train_slice) != WINDOW:

                raise RuntimeError(
                    f"{ticker} fit {fit_number}: "
                    f"training size is "
                    f"{len(train_slice)}, "
                    f"expected {WINDOW}."
                )

            X_train = (
                train_slice[
                    FEATURES
                ]
                .to_numpy(
                    dtype=float
                )
            )

            y_train = (
                train_slice[
                    "target"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            # ------------------------------------------------
            # Forecast block.
            # ------------------------------------------------

            forecast_dates = (
                block[
                    "forecast_date"
                ]
                .to_numpy()
            )

            # Create lookup from dates to processed data.
            block_model = (
                df_model[
                    df_model["Date"].isin(
                        forecast_dates
                    )
                ]
                .copy()
            )

            block_model = (
                block_model
                .sort_values("Date")
                .reset_index(drop=True)
            )

            if len(block_model) != len(block):

                raise RuntimeError(
                    f"{ticker} fit {fit_number}: "
                    f"forecast date mismatch. "
                    f"QRF block has {len(block)} rows, "
                    f"processed data has "
                    f"{len(block_model)} rows."
                )

            # ------------------------------------------------
            # Ensure dates align exactly.
            # ------------------------------------------------

            if not np.array_equal(
                block_model["Date"].to_numpy(),
                block["forecast_date"].to_numpy()
            ):

                raise RuntimeError(
                    f"{ticker} fit {fit_number}: "
                    f"forecast-date alignment failed."
                )

            X_block = (
                block_model[
                    FEATURES
                ]
                .to_numpy(
                    dtype=float
                )
            )

            # =================================================
            # FIT STANDARD RF
            # =================================================

            print(
                f"  FIT {int(fit_number):3d} | "
                f"block={len(block):2d} | "
                f"forecast="
                f"{first_forecast_date.date()} "
                f"to "
                f"{last_forecast_date.date()}"
            )

            rf = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=MAX_FEATURES,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,

                # Explicit sklearn defaults used by the
                # standard regression forest.
                criterion="squared_error",
                max_depth=None,
                min_samples_split=2,
                min_weight_fraction_leaf=0.0,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                warm_start=False,
                ccp_alpha=0.0,
                max_samples=None,
            )

            with warnings.catch_warnings():

                warnings.simplefilter(
                    "ignore"
                )

                rf.fit(
                    X_train,
                    y_train
                )

            # ------------------------------------------------
            # Verify 500 trees.
            # ------------------------------------------------

            if len(
                rf.estimators_
            ) != N_ESTIMATORS:

                raise RuntimeError(
                    f"{ticker} fit {fit_number}: "
                    f"expected {N_ESTIMATORS} trees, "
                    f"got "
                    f"{len(rf.estimators_)}."
                )

            # =================================================
            # TREE PREDICTIONS
            # =================================================

            tree_matrix = (
                get_tree_prediction_matrix(
                    rf,
                    X_block
                )
            )

            # tree_matrix:
            #
            # rows    = trees
            # columns = dates

            tree_mean = np.mean(
                tree_matrix,
                axis=0
            )

            tree_median = np.median(
                tree_matrix,
                axis=0
            )

            tree_std = np.std(
                tree_matrix,
                axis=0,
                ddof=1
            )

            tree_p10 = np.percentile(
                tree_matrix,
                10,
                axis=0
            )

            tree_p90 = np.percentile(
                tree_matrix,
                90,
                axis=0
            )

            tree_iqr = (
                tree_p90
                - tree_p10
            )

            # ------------------------------------------------
            # Retrieve official frozen QRF q05.
            # ------------------------------------------------

            qrf_q05 = (
                block[
                    "q05_forecast"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            denominator = np.abs(
                qrf_q05
            )

            forest_uncertainty_norm = (
                tree_std
                / denominator
            )

            forest_uncertainty_norm[
                denominator <= 1e-12
            ] = np.nan

            # ------------------------------------------------
            # Additional diagnostic:
            # share of trees above zero.
            # Descriptive only.
            # ------------------------------------------------

            tree_positive_share = (
                np.mean(
                    tree_matrix > 0,
                    axis=0
                )
            )

            # =================================================
            # STORE BLOCK
            # =================================================

            for j in range(
                len(block)
            ):

                ticker_results.append(
                    {
                        "ticker":
                            ticker,

                        "forecast_date":
                            block.loc[
                                j,
                                "forecast_date"
                            ],

                        "realized_date":
                            block.loc[
                                j,
                                "realized_date"
                            ],

                        # Official QRF forecast
                        "q05_ensemble":
                            float(
                                qrf_q05[j]
                            ),

                        # Standard RF tree distribution
                        "tree_prediction_mean":
                            float(
                                tree_mean[j]
                            ),

                        "tree_prediction_median":
                            float(
                                tree_median[j]
                            ),

                        "tree_prediction_std":
                            float(
                                tree_std[j]
                            ),

                        "tree_prediction_p10":
                            float(
                                tree_p10[j]
                            ),

                        "tree_prediction_p90":
                            float(
                                tree_p90[j]
                            ),

                        # MAIN SIGNAL
                        "forest_uncertainty_norm":
                            float(
                                forest_uncertainty_norm[j]
                            ),

                        # Descriptive spread
                        "tree_p90_minus_p10":
                            float(
                                tree_iqr[j]
                            ),

                        "tree_positive_share":
                            float(
                                tree_positive_share[j]
                            ),

                        "n_trees_used":
                            int(
                                tree_matrix.shape[0]
                            ),

                        "fit_number":
                            int(
                                fit_number
                            ),
                    }
                )

        # ====================================================
        # TICKER RESULTS
        # ====================================================

        ticker_df = pd.DataFrame(
            ticker_results
        )

        print()
        print(
            f"{ticker}: generated "
            f"{len(ticker_df)} rows"
        )

        if len(ticker_df) > 0:

            print(
                f"{ticker}: uncertainty mean = "
                f"{ticker_df['forest_uncertainty_norm'].mean():.6f}"
            )

            print(
                f"{ticker}: uncertainty median = "
                f"{ticker_df['forest_uncertainty_norm'].median():.6f}"
            )

            print(
                f"{ticker}: uncertainty max = "
                f"{ticker_df['forest_uncertainty_norm'].max():.6f}"
            )

            print(
                f"{ticker}: trees min/max = "
                f"{ticker_df['n_trees_used'].min()} / "
                f"{ticker_df['n_trees_used'].max()}"
            )

        all_results.append(
            ticker_df
        )

    # ========================================================
    # COMBINE
    # ========================================================

    print()
    print("=" * 100)
    print("COMBINING RESULTS")
    print("=" * 100)

    if not all_results:

        raise RuntimeError(
            "No results generated."
        )

    out = pd.concat(
        all_results,
        ignore_index=True
    )

    out = (
        out
        .sort_values(
            [
                "ticker",
                "forecast_date"
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    print()
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)

    print()
    print(
        f"Total rows:             "
        f"{len(out):,}"
    )

    print(
        f"Tickers:                "
        f"{out['ticker'].nunique()}"
    )

    print()
    print(
        "Rows per ticker:"
    )

    print(
        out
        .groupby("ticker")
        .size()
        .to_string()
    )

    # --------------------------------------------------------
    # Expected rows
    # --------------------------------------------------------

    print()
    print(
        "Expected frozen QRF rows: "
        f"{len(qrf):,}"
    )

    print(
        "Generated forest rows:    "
        f"{len(out):,}"
    )

    if len(out) != len(qrf):

        raise RuntimeError(
            "Generated row count does not match "
            "the frozen QRF OOS predictions."
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicate_count = (
        out[
            [
                "ticker",
                "forecast_date"
            ]
        ]
        .duplicated()
        .sum()
    )

    print()
    print(
        f"Duplicate ticker-date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count > 0:

        raise RuntimeError(
            "Duplicate ticker-date rows detected."
        )

    # --------------------------------------------------------
    # Tree count
    # --------------------------------------------------------

    print()
    print(
        "Tree count distribution:"
    )

    print(
        out[
            "n_trees_used"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if (
        out[
            "n_trees_used"
        ] != N_ESTIMATORS
    ).any():

        raise RuntimeError(
            "Not every observation used "
            f"{N_ESTIMATORS} trees."
        )

    # --------------------------------------------------------
    # Main signal
    # --------------------------------------------------------

    print()
    print(
        "Forest uncertainty statistics:"
    )

    print(
        out[
            "forest_uncertainty_norm"
        ]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_signal = (
        out[
            "forest_uncertainty_norm"
        ]
        .isna()
        .sum()
    )

    print()
    print(
        f"Missing uncertainty rows: "
        f"{missing_signal}"
    )

    # --------------------------------------------------------
    # Negative uncertainty
    # --------------------------------------------------------

    negative_signal = (
        out[
            "forest_uncertainty_norm"
        ] < 0
    ).sum()

    print()
    print(
        f"Negative uncertainty rows: "
        f"{negative_signal}"
    )

    if negative_signal > 0:

        raise RuntimeError(
            "Uncertainty cannot be negative."
        )

    # ========================================================
    # ALIGNMENT CHECK WITH ORIGINAL QRF
    # ========================================================

    print()
    print("=" * 100)
    print("QRF ALIGNMENT CHECK")
    print("=" * 100)

    compare = out.merge(
        qrf[
            [
                "ticker",
                "forecast_date",
                "realized_date",
                "q05_forecast",
                "fit_number",
            ]
        ],
        on=[
            "ticker",
            "forecast_date"
        ],
        how="inner",
        suffixes=(
            "_forest",
            "_qrf"
        )
    )

    if len(compare) != len(out):

        raise RuntimeError(
            "Could not fully align forest output "
            "with frozen QRF predictions."
        )

    realized_date_diff = (
        (
            compare[
                "realized_date_forest"
            ]
            !=
            compare[
                "realized_date_qrf"
            ]
        )
        .sum()
    )

    print(
        f"Realized-date mismatches: "
        f"{realized_date_diff}"
    )

    fit_number_diff = (
        (
            compare[
                "fit_number_forest"
            ]
            !=
            compare[
                "fit_number_qrf"
            ]
        )
        .sum()
    )

    print(
        f"Fit-number mismatches:     "
        f"{fit_number_diff}"
    )

    q05_diff = np.max(
        np.abs(
            compare[
                "q05_ensemble"
            ]
            -
            compare[
                "q05_forecast"
            ]
        )
    )

    print(
        f"Max QRF q05 difference:     "
        f"{q05_diff:.12e}"
    )

    if realized_date_diff > 0:

        raise RuntimeError(
            "Realized dates do not match."
        )

    if fit_number_diff > 0:

        raise RuntimeError(
            "Fit numbers do not match."
        )

    if q05_diff > 1e-12:

        raise RuntimeError(
            "Official QRF q05 values were altered."
        )

    # ========================================================
    # AAPL CHECK
    # ========================================================

    print()
    print("=" * 100)
    print("AAPL CHECK")
    print("=" * 100)

    aapl = (
        out[
            out["ticker"] == "AAPL"
        ]
        .copy()
    )

    if len(aapl) > 0:

        print()
        print(
            aapl[
                [
                    "forecast_date",
                    "realized_date",
                    "q05_ensemble",
                    "tree_prediction_std",
                    "forest_uncertainty_norm",
                    "n_trees_used",
                    "fit_number",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        print()
        print(
            "Last 10:"
        )

        print(
            aapl[
                [
                    "forecast_date",
                    "realized_date",
                    "q05_ensemble",
                    "tree_prediction_std",
                    "forest_uncertainty_norm",
                    "n_trees_used",
                    "fit_number",
                ]
            ]
            .tail(10)
            .to_string(index=False)
        )

    # ========================================================
    # SAVE
    # ========================================================

    print()
    print("=" * 100)
    print("SAVING")
    print("=" * 100)

    out.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 100)
    print("EXPERIMENT 29 COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()