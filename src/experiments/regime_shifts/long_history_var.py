from __future__ import annotations

import io
import zipfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import t as student_t
from sklearn.ensemble import RandomForestRegressor

try:
    from arch import arch_model
except ImportError as exc:
    raise ImportError(
        "The 'arch' package is required.\n\n"
        "Install it with:\n"
        "pip install arch"
    ) from exc


warnings.filterwarnings("ignore")


# ============================================================
# RS01 — LONG-HISTORY VaR RECONSTRUCTION
# ============================================================
#
# Research purpose:
#
# Build a strictly causal long-history VaR forecast panel
# for:
#
#   1. Historical Simulation (HS)
#   2. EWMA
#   3. GARCH(1,1)-Student-t
#   4. Quantile Random Forest (QRF)
#
# This is ONLY RS01.
#
# Later experiments will use this panel to study:
#
#   - regime shifts
#   - VaR forecast deterioration
#   - reliability monitoring
#   - trust / warning scores
#   - event studies
#
#
# CAUSAL TIMING
# -------------
#
# Information available through date t
#                 |
#                 v
#          VaR forecast t+1
#                 |
#                 v
#       realized return at t+1
#
#
# The forecast is therefore evaluated against the next
# trading day's return.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"D:\reliability_layer"
)

RAW_ZIP_PATH = (
    ROOT
    / "data"
    / "raw.zip"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PANEL = (
    OUTPUT_DIR
    / "RS01_long_history_var_panel.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "RS01_long_history_var_summary.csv"
)

OUTPUT_COVERAGE = (
    OUTPUT_DIR
    / "RS01_long_history_var_coverage.csv"
)

OUTPUT_LOG = (
    OUTPUT_DIR
    / "RS01_long_history_var_log.txt"
)


# ============================================================
# FIXED 24-TICKER UNIVERSE
# ============================================================

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


# ============================================================
# MODEL CONFIGURATION
# ============================================================

ALPHA = 0.05

# Historical observations used by each model.
WINDOW = 1000

# Expensive models refitted every 20 forecast dates.
REFIT_EVERY = 20


# ============================================================
# EWMA
# ============================================================

EWMA_LAMBDA = 0.94


# ============================================================
# GARCH-T
# ============================================================

GARCH_P = 1
GARCH_Q = 1


# ============================================================
# QRF
# ============================================================

QRF_TREES = 500
QRF_MIN_LEAF = 20
QRF_MAX_FEATURES = 1.0
QRF_RANDOM_STATE = 42
QRF_N_JOBS = -1

QRF_FEATURES = [
    "log_return",
    "return_lag1",
    "return_lag5",
    "return_lag10",
    "vol_10d",
    "vol_20d",
    "volume_ratio",
    "vix",
]


# ============================================================
# LOGGING
# ============================================================

LOG_LINES: list[str] = []


def write_log(message: str) -> None:

    print(message)

    LOG_LINES.append(
        str(message)
    )


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_columns(
    df: pd.DataFrame
) -> pd.DataFrame:

    rename_map = {}

    for column in df.columns:

        cleaned = (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        rename_map[column] = cleaned

    return df.rename(
        columns=rename_map
    )


def find_date_column(columns):

    candidates = [
        "date",
        "datetime",
        "timestamp",
        "time",
    ]

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def find_close_column(columns):

    candidates = [
        "close",
        "adj_close",
        "adjusted_close",
        "adjclose",
    ]

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


def find_volume_column(columns):

    candidates = [
        "volume",
    ]

    for candidate in candidates:

        if candidate in columns:
            return candidate

    return None


# ============================================================
# READ CSV INSIDE ZIP
# ============================================================

def read_csv_from_zip(
    z: zipfile.ZipFile,
    member: str
) -> pd.DataFrame:

    raw_bytes = z.read(
        member
    )

    return pd.read_csv(
        io.BytesIO(
            raw_bytes
        )
    )


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_raw_data():

    write_log(
        "=" * 100
    )

    write_log(
        "RAW DATA LOADING"
    )

    write_log(
        "=" * 100
    )

    write_log(
        f"\nZIP:\n{RAW_ZIP_PATH}"
    )

    if not RAW_ZIP_PATH.exists():

        raise FileNotFoundError(
            "\nRaw ZIP not found.\n\n"
            f"Expected:\n{RAW_ZIP_PATH}\n"
        )

    stocks = {}
    vix_df = None

    with zipfile.ZipFile(
        RAW_ZIP_PATH,
        "r"
    ) as z:

        csv_members = [
            member
            for member in z.namelist()
            if (
                not member.endswith("/")
                and member.lower().endswith(".csv")
            )
        ]

        write_log(
            f"\nCSV files in ZIP: "
            f"{len(csv_members)}"
        )

        for member in csv_members:

            filename = Path(
                member
            ).name

            stem = (
                Path(filename)
                .stem
                .upper()
                .strip()
            )

            try:

                df = read_csv_from_zip(
                    z,
                    member
                )

            except Exception as exc:

                write_log(
                    f"\nSkipping unreadable file:"
                    f"\n{member}"
                    f"\nReason: {exc}"
                )

                continue

            df = normalize_columns(
                df
            )

            date_column = (
                find_date_column(
                    df.columns
                )
            )

            close_column = (
                find_close_column(
                    df.columns
                )
            )

            if (
                date_column is None
                or close_column is None
            ):

                continue

            # =================================================
            # VIX FILE
            # =================================================

            is_vix = (
                stem in {
                    "VIX",
                    "^VIX",
                    "VIX_INDEX",
                }
                or "VIX" in stem
            )

            if is_vix:

                temp = df[
                    [
                        date_column,
                        close_column,
                    ]
                ].copy()

                temp["date"] = pd.to_datetime(
                    temp[date_column],
                    errors="coerce"
                )

                temp["vix"] = pd.to_numeric(
                    temp[close_column],
                    errors="coerce"
                )

                temp = (
                    temp[
                        [
                            "date",
                            "vix",
                        ]
                    ]
                    .dropna()
                    .drop_duplicates(
                        subset=["date"]
                    )
                    .sort_values(
                        "date"
                    )
                    .reset_index(
                        drop=True
                    )
                )

                if len(temp) > 0:

                    vix_df = temp

                continue

            # =================================================
            # DETECT TICKER
            # =================================================

            ticker = None

            # Try explicit ticker column first.
            if "ticker" in df.columns:

                unique_tickers = (
                    df["ticker"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .unique()
                )

                for candidate in unique_tickers:

                    if candidate in TICKERS:

                        ticker = candidate
                        break

            # Otherwise use filename.
            if ticker is None:

                for candidate in TICKERS:

                    if (
                        stem == candidate
                        or stem.startswith(
                            candidate + "_"
                        )
                        or stem.endswith(
                            "_" + candidate
                        )
                        or stem.startswith(
                            candidate + "."
                        )
                        or stem.endswith(
                            "." + candidate
                        )
                    ):

                        ticker = candidate
                        break

            if ticker is None:

                continue

            # =================================================
            # STANDARDIZE STOCK DATA
            # =================================================

            volume_column = (
                find_volume_column(
                    df.columns
                )
            )

            selected_columns = [
                date_column,
                close_column,
            ]

            if volume_column is not None:

                selected_columns.append(
                    volume_column
                )

            temp = df[
                selected_columns
            ].copy()

            temp = temp.rename(
                columns={
                    date_column: "date",
                    close_column: "close",
                }
            )

            if volume_column is not None:

                temp = temp.rename(
                    columns={
                        volume_column: "volume"
                    }
                )

            else:

                temp["volume"] = np.nan

            temp["date"] = pd.to_datetime(
                temp["date"],
                errors="coerce"
            )

            temp["close"] = pd.to_numeric(
                temp["close"],
                errors="coerce"
            )

            temp["volume"] = pd.to_numeric(
                temp["volume"],
                errors="coerce"
            )

            temp = (
                temp[
                    [
                        "date",
                        "close",
                        "volume",
                    ]
                ]
                .dropna(
                    subset=[
                        "date",
                        "close",
                    ]
                )
                .drop_duplicates(
                    subset=["date"]
                )
                .sort_values(
                    "date"
                )
                .reset_index(
                    drop=True
                )
            )

            # Positive prices only.
            temp = temp[
                temp["close"] > 0
            ].copy()

            if len(temp) > 0:

                stocks[ticker] = temp

    # ========================================================
    # AUDIT DISCOVERY
    # ========================================================

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "RAW DATA AUDIT"
    )

    write_log(
        "=" * 100
    )

    write_log(
        f"\nRequired tickers: "
        f"{len(TICKERS)}"
    )

    write_log(
        f"Discovered tickers: "
        f"{len(stocks)}"
    )

    write_log(
        str(
            sorted(
                stocks.keys()
            )
        )
    )

    missing = [
        ticker
        for ticker in TICKERS
        if ticker not in stocks
    ]

    if missing:

        raise RuntimeError(
            "\nMissing required tickers:\n"
            + "\n".join(missing)
        )

    if vix_df is None:

        raise RuntimeError(
            "\nVIX was not found in raw.zip."
        )

    write_log(
        f"\nVIX rows: "
        f"{len(vix_df):,}"
    )

    write_log(
        f"VIX range: "
        f"{vix_df['date'].min().date()} "
        f"to "
        f"{vix_df['date'].max().date()}"
    )

    for ticker in TICKERS:

        temp = stocks[ticker]

        write_log(
            f"{ticker:6s} | "
            f"rows={len(temp):,} | "
            f"{temp['date'].min().date()} "
            f"to "
            f"{temp['date'].max().date()}"
        )

    return stocks, vix_df


# ============================================================
# BUILD HISTORICAL FEATURES
# ============================================================

def build_features(
    stock_df: pd.DataFrame,
    vix_df: pd.DataFrame
) -> pd.DataFrame:

    df = (
        stock_df
        .sort_values("date")
        .drop_duplicates(
            "date"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    # --------------------------------------------------------
    # Daily log return
    # --------------------------------------------------------

    df["log_return"] = (
        np.log(
            df["close"]
        ).diff()
    )

    # --------------------------------------------------------
    # Lagged returns
    # --------------------------------------------------------

    df["return_lag1"] = (
        df["log_return"]
        .shift(1)
    )

    df["return_lag5"] = (
        df["log_return"]
        .shift(5)
    )

    df["return_lag10"] = (
        df["log_return"]
        .shift(10)
    )

    # --------------------------------------------------------
    # Historical volatility
    #
    # These include information through date t.
    # The forecast is for t+1.
    # --------------------------------------------------------

    df["vol_10d"] = (
        df["log_return"]
        .rolling(
            10,
            min_periods=10
        )
        .std(
            ddof=0
        )
    )

    df["vol_20d"] = (
        df["log_return"]
        .rolling(
            20,
            min_periods=20
        )
        .std(
            ddof=0
        )
    )

    # --------------------------------------------------------
    # Volume ratio
    # --------------------------------------------------------

    volume_mean = (
        df["volume"]
        .rolling(
            20,
            min_periods=20
        )
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"]
        / volume_mean
    )

    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    df = df.merge(
        vix_df[
            [
                "date",
                "vix",
            ]
        ],
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Keep both the realized date and realized return.
    # This avoids the previous final-row indexing bug.
    # --------------------------------------------------------

    df["realized_date"] = (
        df["date"]
        .shift(-1)
    )

    df["next_return"] = (
        df["log_return"]
        .shift(-1)
    )

    return df


# ============================================================
# PINBALL LOSS
# ============================================================

def calculate_pinball_loss(
    actual: float,
    forecast: float,
    alpha: float = ALPHA,
):

    if (
        not np.isfinite(actual)
        or not np.isfinite(forecast)
    ):

        return np.nan

    error = (
        actual
        - forecast
    )

    if error >= 0:

        return (
            alpha
            * error
        )

    return (
        (alpha - 1.0)
        * error
    )


# ============================================================
# HISTORICAL SIMULATION
# ============================================================

def historical_simulation_var(
    historical_returns: np.ndarray
):

    returns = np.asarray(
        historical_returns,
        dtype=float
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) == 0:

        return np.nan

    return float(
        np.quantile(
            returns,
            ALPHA
        )
    )


# ============================================================
# EWMA
# ============================================================

def ewma_var(
    historical_returns: np.ndarray
):

    returns = np.asarray(
        historical_returns,
        dtype=float
    )

    returns = returns[
        np.isfinite(returns)
    ]

    if len(returns) < 2:

        return np.nan

    variance = float(
        np.var(
            returns,
            ddof=0
        )
    )

    # Recursively update through all information
    # available at the forecast date.
    for value in returns:

        variance = (
            EWMA_LAMBDA
            * variance
            + (
                1.0
                - EWMA_LAMBDA
            )
            * value
            * value
        )

    sigma = np.sqrt(
        max(
            variance,
            1e-16
        )
    )

    # Standard normal 5% quantile.
    z05 = -1.644853626951473

    return float(
        z05 * sigma
    )


# ============================================================
# GARCH-T BLOCK
# ============================================================

def fit_garch_block(
    returns_by_date: pd.Series,
    dates: pd.DatetimeIndex,
    positions: np.ndarray,
    ticker: str,
    fit_number: int,
):

    if len(positions) == 0:

        return []

    first_position = positions[0]

    first_forecast_date = (
        dates[
            first_position
        ]
    )

    # --------------------------------------------------------
    # Training information available through first forecast
    # date.
    # --------------------------------------------------------

    training_returns = (
        returns_by_date.loc[
            returns_by_date.index
            <= first_forecast_date
        ]
        .dropna()
    )

    if len(
        training_returns
    ) < WINDOW:

        return []

    training_window = (
        training_returns
        .iloc[-WINDOW:]
        .to_numpy(
            dtype=float
        )
    )

    # Scale returns by 100.
    scaled_returns = (
        training_window
        * 100.0
    )

    # --------------------------------------------------------
    # GARCH(1,1)-Student-t
    # --------------------------------------------------------

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=GARCH_P,
        q=GARCH_Q,
        dist="t",
        rescale=False,
    )

    try:

        fitted = model.fit(
            disp="off"
        )

    except Exception as exc:

        write_log(
            f"{ticker}: GARCH fit failure "
            f"at fit {fit_number}: "
            f"{exc}"
        )

        return []

    params = fitted.params

    mu = float(
        params.get(
            "mu",
            0.0
        )
    )

    omega = float(
        params.get(
            "omega",
            1e-8
        )
    )

    alpha1 = float(
        params.get(
            "alpha[1]",
            0.0
        )
    )

    beta1 = float(
        params.get(
            "beta[1]",
            0.0
        )
    )

    nu = float(
        params.get(
            "nu",
            8.0
        )
    )

    # --------------------------------------------------------
    # Last fitted conditional variance
    # --------------------------------------------------------

    conditional_volatility = np.asarray(
        fitted.conditional_volatility,
        dtype=float
    )

    sigma2_t = float(
        conditional_volatility[-1] ** 2
    )

    # --------------------------------------------------------
    # Compute one-step-ahead variance.
    # --------------------------------------------------------

    last_return_scaled = (
        training_window[-1]
        * 100.0
    )

    epsilon_t = (
        last_return_scaled
        - mu
    )

    current_variance = (
        omega
        + alpha1
        * epsilon_t
        * epsilon_t
        + beta1
        * sigma2_t
    )

    current_variance = max(
        current_variance,
        1e-12
    )

    # --------------------------------------------------------
    # Standardized Student-t 5% quantile.
    # --------------------------------------------------------

    raw_quantile = student_t.ppf(
        ALPHA,
        df=nu
    )

    if nu > 2:

        standardized_quantile = (
            raw_quantile
            * np.sqrt(
                (nu - 2.0)
                / nu
            )
        )

    else:

        standardized_quantile = (
            raw_quantile
        )

    rows = []

    # --------------------------------------------------------
    # Frozen parameter block
    # --------------------------------------------------------

    for j, position in enumerate(
        positions
    ):

        forecast_date = dates[
            position
        ]

        # ----------------------------------------------------
        # For later observations in the block, update the
        # conditional variance using the newly observed
        # previous return.
        # ----------------------------------------------------

        if j > 0:

            previous_position = (
                positions[j - 1]
            )

            previous_date = dates[
                previous_position
            ]

            previous_return = (
                returns_by_date.get(
                    previous_date,
                    np.nan
                )
            )

            if np.isfinite(
                previous_return
            ):

                previous_scaled = (
                    previous_return
                    * 100.0
                )

                previous_epsilon = (
                    previous_scaled
                    - mu
                )

                current_variance = (
                    omega
                    + alpha1
                    * previous_epsilon
                    * previous_epsilon
                    + beta1
                    * current_variance
                )

                current_variance = max(
                    current_variance,
                    1e-12
                )

        sigma = (
            np.sqrt(
                current_variance
            )
            / 100.0
        )

        garch_quantile = (
            mu / 100.0
            + standardized_quantile
            * sigma
        )

        rows.append(
            {
                "ticker":
                    ticker,

                "forecast_date":
                    forecast_date,

                "garch_var":
                    float(
                        garch_quantile
                    ),

                "garch_mu":
                    float(
                        mu / 100.0
                    ),

                "garch_sigma":
                    float(
                        sigma
                    ),

                "garch_df":
                    float(
                        nu
                    ),

                "garch_fit_number":
                    fit_number,
            }
        )

    return rows


# ============================================================
# QRF BLOCK
# ============================================================

def fit_qrf_block(
    feature_df: pd.DataFrame,
    positions: np.ndarray,
    ticker: str,
    fit_number: int,
):
    """
    Fit a Quantile Regression Forest for one refit block.

    Method:
    - Train a standard Random Forest on causal training observations.
    - Use forest leaves to construct Meinshausen-style observation weights.
    - Estimate the conditional 5% quantile directly from the weighted
      distribution of training returns.

    Important:
    We DO NOT take the 5th percentile of individual tree predictions.
    Tree predictions are leaf means and their cross-tree distribution is
    not the predictive distribution of returns.
    """

    # --------------------------------------------------------
    # Basic check
    # --------------------------------------------------------

    if len(positions) == 0:
        return []

    first_position = positions[0]

    first_forecast_date = feature_df.iloc[first_position]["date"]

    # --------------------------------------------------------
    # CAUSAL TRAINING SAMPLE
    #
    # Training feature date:
    #     s < first_forecast_date
    #
    # Training target:
    #     next_return = return at s+1
    #
    # Therefore no target after the first forecast date can
    # enter the training sample.
    # --------------------------------------------------------

    training = feature_df[
        feature_df["date"] < first_forecast_date
    ].copy()

    training = training.dropna(
        subset=QRF_FEATURES + ["next_return"]
    )

    # Need at least WINDOW valid training observations.
    if len(training) < WINDOW:
        return []

    # Use the most recent WINDOW observations only.
    training = (
        training
        .sort_values("date")
        .iloc[-WINDOW:]
        .copy()
    )

    # --------------------------------------------------------
    # TRAINING MATRICES
    # --------------------------------------------------------

    X_train = training[
        QRF_FEATURES
    ].to_numpy(dtype=float)

    y_train = training[
        "next_return"
    ].to_numpy(dtype=float)

    # Safety check
    if (
        not np.isfinite(X_train).all()
        or not np.isfinite(y_train).all()
    ):
        return []

    # --------------------------------------------------------
    # RANDOM FOREST
    #
    # The forest is NOT itself used by taking quantiles of
    # tree predictions.
    #
    # Instead, its leaf structure determines similarity
    # weights between the forecast observation and historical
    # training observations.
    # --------------------------------------------------------

    forest = RandomForestRegressor(
        n_estimators=QRF_TREES,
        min_samples_leaf=QRF_MIN_LEAF,
        max_features=QRF_MAX_FEATURES,
        random_state=QRF_RANDOM_STATE,
        n_jobs=-1,
    )

    try:
        forest.fit(
            X_train,
            y_train,
        )
    except Exception as exc:
        print(
            f"[QRF FIT ERROR] "
            f"{ticker} "
            f"fit={fit_number}: {exc}",
            flush=True,
        )
        return []

    # ========================================================
    # MEINSHAUSEN-STYLE QUANTILE REGRESSION FOREST
    # ========================================================

    # --------------------------------------------------------
    # 1. Find the terminal leaf occupied by every training
    #    observation in every tree.
    #
    # Shape:
    #     (n_train, n_trees)
    # --------------------------------------------------------

    leaves_train = forest.apply(X_train)

    n_train, n_trees = leaves_train.shape

    # --------------------------------------------------------
    # 2. For each training observation and tree, calculate
    #
    #        1 / number of training observations in its leaf
    #
    # This is the contribution that observation receives when
    # a test point falls into the same leaf.
    # --------------------------------------------------------

    inv_size = np.empty(
        (n_train, n_trees),
        dtype=float,
    )

    for t in range(n_trees):

        tree_leaves = leaves_train[:, t]

        # Leaf IDs can be used directly with bincount.
        counts = np.bincount(tree_leaves)

        inv_size[:, t] = (
            1.0
            / counts[tree_leaves]
        )

    # --------------------------------------------------------
    # 3. Collect ALL test observations for this refit block.
    #
    # This is important for speed:
    # forest.apply() is called once for the entire block rather
    # than once for every forecast observation.
    # --------------------------------------------------------

    X_test = (
        feature_df
        .iloc[positions][QRF_FEATURES]
        .to_numpy(dtype=float)
    )

    # Only retain finite feature rows.
    ok = np.isfinite(X_test).all(axis=1)

    if not ok.any():
        return []

    pos_ok = positions[ok]

    X_test_ok = X_test[ok]

    # --------------------------------------------------------
    # 4. Find terminal leaves for every forecast observation.
    #
    # Shape:
    #     (n_test, n_trees)
    # --------------------------------------------------------

    leaves_test = forest.apply(X_test_ok)

    # --------------------------------------------------------
    # 5. Construct QRF observation weights.
    #
    # match shape:
    #
    #     (n_test, n_train, n_trees)
    #
    # For each test observation, training observation and tree,
    # this is True when they occupy the same terminal leaf.
    # --------------------------------------------------------

    match = (
        leaves_train[None, :, :]
        ==
        leaves_test[:, None, :]
    )

    # Meinshausen-style weights:
    #
    # w_i(x)
    # =
    # 1/T * sum_t
    # [
    #   I(X_i in same leaf as x)
    #   /
    #   leaf_size_t(x)
    # ]
    #
    # Shape:
    #     (n_test, n_train)
    # --------------------------------------------------------

    W = (
        match
        * inv_size[None, :, :]
    ).sum(axis=2) / n_trees

    # --------------------------------------------------------
    # Numerical safety.
    #
    # In theory every row should sum to 1.
    # Normalize explicitly to protect against floating-point
    # accumulation error.
    # --------------------------------------------------------

    weight_sums = W.sum(
        axis=1,
        keepdims=True,
    )

    valid_weights = (
        np.isfinite(weight_sums[:, 0])
        & (weight_sums[:, 0] > 0)
    )

    if not valid_weights.all():

        pos_ok = pos_ok[valid_weights]
        W = W[valid_weights]

        if len(pos_ok) == 0:
            return []

        weight_sums = W.sum(
            axis=1,
            keepdims=True,
        )

    W = W / weight_sums

    # --------------------------------------------------------
    # 6. Weighted conditional quantile.
    #
    # Sort training returns from smallest to largest.
    # --------------------------------------------------------

    order = np.argsort(y_train)

    y_sorted = y_train[order]

    W_sorted = W[:, order]

    # Weighted empirical CDF for every forecast observation.
    cdf = np.cumsum(
        W_sorted,
        axis=1,
    )

    # Find first training return for which:
    #
    #     F(y | X=x) >= ALPHA
    #
    # ALPHA = 0.05 for 5% VaR.
    # --------------------------------------------------------

    quantile_indices = (
        cdf >= ALPHA
    ).argmax(axis=1)

    quantile_indices = np.minimum(
        quantile_indices,
        n_train - 1,
    )

    q05 = y_sorted[
        quantile_indices
    ]

    # --------------------------------------------------------
    # 7. Conditional weighted mean and standard deviation.
    #
    # These replace the old misleading tree-prediction mean
    # and standard deviation.
    # --------------------------------------------------------

    mean_w = W @ y_train

    second_moment = (
        W @ (y_train ** 2)
    )

    variance_w = np.maximum(
        second_moment
        - mean_w ** 2,
        0.0,
    )

    std_w = np.sqrt(
        variance_w
    )

    # --------------------------------------------------------
    # 8. Build output rows.
    # --------------------------------------------------------

    rows = []

    for k, position in enumerate(pos_ok):

        row = feature_df.iloc[position]

        rows.append(
            {
                "ticker": ticker,

                "forecast_date": row["date"],

                "qrf_var": float(
                    q05[k]
                ),

                # These names are retained so that the existing
                # RS01 panel/audit code continues to work.
                #
                # Their interpretation is now:
                #
                # qrf_tree_mean =
                # conditional weighted return mean
                #
                # qrf_tree_std =
                # conditional weighted return std
                "qrf_tree_mean": float(
                    mean_w[k]
                ),

                "qrf_tree_std": float(
                    std_w[k]
                ),

                "qrf_n_trees": int(
                    n_trees
                ),

                "qrf_fit_number": int(
                    fit_number
                ),
            }
        )

    return rows
    # --------------------------------------------------------
    # Random forest used as a quantile random forest:
    # empirical distribution of tree predictions.
    # --------------------------------------------------------

    forest = RandomForestRegressor(
        n_estimators=QRF_TREES,
        min_samples_leaf=QRF_MIN_LEAF,
        max_features=QRF_MAX_FEATURES,
        random_state=(
            QRF_RANDOM_STATE
            + fit_number
        ),
        n_jobs=QRF_N_JOBS,
    )

    try:

        forest.fit(
            X_train,
            y_train
        )

    except Exception as exc:

        write_log(
            f"{ticker}: QRF fit failure "
            f"at fit {fit_number}: "
            f"{exc}"
        )

        return []

    rows = []

    # --------------------------------------------------------
    # Frozen forest over block.
    # --------------------------------------------------------

    for position in positions:

        row = feature_df.iloc[
            position
        ]

        X_test = (
            row[
                QRF_FEATURES
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            X_test
        ).all():

            continue

        tree_predictions = np.asarray(
            [
                estimator.predict(
                    X_test.reshape(
                        1,
                        -1
                    )
                )[0]
                for estimator
                in forest.estimators_
            ],
            dtype=float
        )

        q05 = float(
            np.quantile(
                tree_predictions,
                ALPHA
            )
        )

        rows.append(
            {
                "ticker":
                    ticker,

                "forecast_date":
                    row["date"],

                "qrf_var":
                    q05,

                "qrf_tree_mean":
                    float(
                        tree_predictions.mean()
                    ),

                "qrf_tree_std":
                    float(
                        tree_predictions.std(
                            ddof=0
                        )
                    ),

                "qrf_n_trees":
                    int(
                        len(
                            forest.estimators_
                        )
                    ),

                "qrf_fit_number":
                    fit_number,
            }
        )

    return rows


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def process_ticker(
    ticker: str,
    stock_df: pd.DataFrame,
    vix_df: pd.DataFrame,
):

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        f"PROCESSING {ticker}"
    )

    write_log(
        "=" * 100
    )

    df = build_features(
        stock_df,
        vix_df
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Forecast rows require:
    #
    #   current return
    #   next return
    #   realized date
    # --------------------------------------------------------

    usable = df[
        df["log_return"].notna()
        & df["next_return"].notna()
        & df["realized_date"].notna()
    ].copy()

    usable = (
        usable
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(usable) < (
        WINDOW + 100
    ):

        raise RuntimeError(
            f"{ticker}: insufficient usable rows: "
            f"{len(usable)}"
        )

    # ========================================================
    # DATE / RETURN SERIES
    # ========================================================

    dates = pd.DatetimeIndex(
        usable["date"]
    )

    returns_by_date = (
        usable
        .set_index("date")[
            "log_return"
        ]
        .sort_index()
    )

    # ========================================================
    # VALID FORECAST POSITIONS
    # ========================================================
    #
    # The maximum possible position is the final row in
    # usable. Its next_return and realized_date are already
    # stored in that same row.
    #
    # DO NOT use dates[position + 1].
    # ========================================================

    possible_positions = np.arange(
        WINDOW - 1,
        len(usable)
    )

    if len(
        possible_positions
    ) == 0:

        raise RuntimeError(
            f"{ticker}: no forecast positions."
        )

    # ========================================================
    # BUILD REFIT BLOCKS
    # ========================================================

    blocks = [
        possible_positions[
            start:
            start + REFIT_EVERY
        ]
        for start in range(
            0,
            len(possible_positions),
            REFIT_EVERY
        )
    ]

    total_blocks = len(
        blocks
    )

    garch_rows = []
    qrf_rows = []

    # ========================================================
    # EXPENSIVE MODEL LOOP
    # ========================================================

    for block_number, positions in enumerate(
        blocks,
        start=1
    ):

        first_date = dates[
            positions[0]
        ]

        last_date = dates[
            positions[-1]
        ]

        print(
            f"{ticker}: "
            f"FIT {block_number:03d}/"
            f"{total_blocks:03d} | "
            f"{first_date.date()} "
            f"to "
            f"{last_date.date()}"
        )

        # ----------------------------------------------------
        # GARCH
        # ----------------------------------------------------

        garch_block = fit_garch_block(
            returns_by_date,
            dates,
            positions,
            ticker,
            block_number
        )

        garch_rows.extend(
            garch_block
        )

        # ----------------------------------------------------
        # QRF
        # ----------------------------------------------------

        qrf_block = fit_qrf_block(
            usable,
            positions,
            ticker,
            block_number
        )

        qrf_rows.extend(
            qrf_block
        )

    # ========================================================
    # LOOKUP DICTIONARIES
    # ========================================================

    garch_lookup = {
        (
            row["ticker"],
            row["forecast_date"]
        ): row
        for row in garch_rows
    }

    qrf_lookup = {
        (
            row["ticker"],
            row["forecast_date"]
        ): row
        for row in qrf_rows
    }

    # ========================================================
    # BUILD FINAL DAILY PANEL
    # ========================================================

    output_rows = []

    for position in possible_positions:

        # ----------------------------------------------------
        # IMPORTANT FIX
        #
        # realized_date comes directly from the row.
        # This is what fixes the previous IndexError.
        # ----------------------------------------------------

        forecast_date = (
            usable.iloc[
                position
            ]["date"]
        )

        realized_date = (
            usable.iloc[
                position
            ]["realized_date"]
        )

        actual_return = float(
            usable.iloc[
                position
            ]["next_return"]
        )

        # ----------------------------------------------------
        # Historical return window available at forecast date.
        # ----------------------------------------------------

        historical_returns = (
            returns_by_date
            .loc[:forecast_date]
            .dropna()
            .iloc[-WINDOW:]
            .to_numpy(
                dtype=float
            )
        )

        if len(
            historical_returns
        ) < WINDOW:

            continue

        # ====================================================
        # HS
        # ====================================================

        hs_forecast = (
            historical_simulation_var(
                historical_returns
            )
        )

        hs_breach = (
            int(
                actual_return
                < hs_forecast
            )
            if np.isfinite(
                hs_forecast
            )
            else np.nan
        )

        hs_loss = (
            calculate_pinball_loss(
                actual_return,
                hs_forecast
            )
        )

        # ====================================================
        # EWMA
        # ====================================================

        ewma_forecast = (
            ewma_var(
                historical_returns
            )
        )

        ewma_breach = (
            int(
                actual_return
                < ewma_forecast
            )
            if np.isfinite(
                ewma_forecast
            )
            else np.nan
        )

        ewma_loss = (
            calculate_pinball_loss(
                actual_return,
                ewma_forecast
            )
        )

        # ====================================================
        # GARCH
        # ====================================================

        garch = garch_lookup.get(
            (
                ticker,
                forecast_date
            )
        )

        if garch is not None:

            garch_forecast = (
                garch["garch_var"]
            )

            garch_mu = (
                garch["garch_mu"]
            )

            garch_sigma = (
                garch["garch_sigma"]
            )

            garch_df = (
                garch["garch_df"]
            )

            garch_fit_number = (
                garch["garch_fit_number"]
            )

            garch_breach = int(
                actual_return
                < garch_forecast
            )

            garch_loss = (
                calculate_pinball_loss(
                    actual_return,
                    garch_forecast
                )
            )

        else:

            garch_forecast = np.nan
            garch_mu = np.nan
            garch_sigma = np.nan
            garch_df = np.nan
            garch_fit_number = np.nan
            garch_breach = np.nan
            garch_loss = np.nan

        # ====================================================
        # QRF
        # ====================================================

        qrf = qrf_lookup.get(
            (
                ticker,
                forecast_date
            )
        )

        if qrf is not None:

            qrf_forecast = (
                qrf["qrf_var"]
            )

            qrf_tree_mean = (
                qrf["qrf_tree_mean"]
            )

            qrf_tree_std = (
                qrf["qrf_tree_std"]
            )

            qrf_n_trees = (
                qrf["qrf_n_trees"]
            )

            qrf_fit_number = (
                qrf["qrf_fit_number"]
            )

            qrf_breach = int(
                actual_return
                < qrf_forecast
            )

            qrf_loss = (
                calculate_pinball_loss(
                    actual_return,
                    qrf_forecast
                )
            )

        else:

            qrf_forecast = np.nan
            qrf_tree_mean = np.nan
            qrf_tree_std = np.nan
            qrf_n_trees = np.nan
            qrf_fit_number = np.nan
            qrf_breach = np.nan
            qrf_loss = np.nan

        # ====================================================
        # APPEND ROW
        # ====================================================

        output_rows.append(
            {
                # -------------------------
                # Timing
                # -------------------------

                "ticker":
                    ticker,

                "forecast_date":
                    forecast_date,

                "realized_date":
                    realized_date,

                "actual_return":
                    actual_return,

                # -------------------------
                # HS
                # -------------------------

                "hs_var":
                    hs_forecast,

                "hs_breach":
                    hs_breach,

                "hs_pinball_loss":
                    hs_loss,

                # -------------------------
                # EWMA
                # -------------------------

                "ewma_var":
                    ewma_forecast,

                "ewma_breach":
                    ewma_breach,

                "ewma_pinball_loss":
                    ewma_loss,

                # -------------------------
                # GARCH-t
                # -------------------------

                "garch_var":
                    garch_forecast,

                "garch_mu":
                    garch_mu,

                "garch_sigma":
                    garch_sigma,

                "garch_df":
                    garch_df,

                "garch_breach":
                    garch_breach,

                "garch_pinball_loss":
                    garch_loss,

                "garch_fit_number":
                    garch_fit_number,

                # -------------------------
                # QRF
                # -------------------------

                "qrf_var":
                    qrf_forecast,

                "qrf_tree_mean":
                    qrf_tree_mean,

                "qrf_tree_std":
                    qrf_tree_std,

                "qrf_n_trees":
                    qrf_n_trees,

                "qrf_breach":
                    qrf_breach,

                "qrf_pinball_loss":
                    qrf_loss,

                "qrf_fit_number":
                    qrf_fit_number,
            }
        )

    return pd.DataFrame(
        output_rows
    )


# ============================================================
# SUMMARY TABLE
# ============================================================

def build_summary(
    panel: pd.DataFrame
):

    model_definitions = {
        "HS": (
            "hs_var",
            "hs_breach",
            "hs_pinball_loss",
        ),

        "EWMA": (
            "ewma_var",
            "ewma_breach",
            "ewma_pinball_loss",
        ),

        "GARCH-t": (
            "garch_var",
            "garch_breach",
            "garch_pinball_loss",
        ),

        "QRF": (
            "qrf_var",
            "qrf_breach",
            "qrf_pinball_loss",
        ),
    }

    rows = []

    for model, (
        var_column,
        breach_column,
        loss_column,
    ) in model_definitions.items():

        valid = panel[
            panel[var_column]
            .notna()
        ].copy()

        rows.append(
            {
                "model":
                    model,

                "rows":
                    len(valid),

                "tickers":
                    valid[
                        "ticker"
                    ].nunique(),

                "forecast_start":
                    valid[
                        "forecast_date"
                    ].min(),

                "forecast_end":
                    valid[
                        "forecast_date"
                    ].max(),

                "breach_rate":
                    valid[
                        breach_column
                    ].mean(),

                "mean_pinball_loss":
                    valid[
                        loss_column
                    ].mean(),

                "median_pinball_loss":
                    valid[
                        loss_column
                    ].median(),

                "mean_abs_var":
                    valid[
                        var_column
                    ].abs().mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COVERAGE TABLE
# ============================================================

def build_coverage(
    panel: pd.DataFrame
):

    model_columns = {
        "HS":
            "hs_var",

        "EWMA":
            "ewma_var",

        "GARCH-t":
            "garch_var",

        "QRF":
            "qrf_var",
    }

    rows = []

    for ticker, group in (
        panel.groupby(
            "ticker"
        )
    ):

        for model, var_column in (
            model_columns.items()
        ):

            valid = (
                group[var_column]
                .notna()
            )

            rows.append(
                {
                    "ticker":
                        ticker,

                    "model":
                        model,

                    "rows":
                        int(
                            valid.sum()
                        ),

                    "first_forecast":
                        group.loc[
                            valid,
                            "forecast_date"
                        ].min(),

                    "last_forecast":
                        group.loc[
                            valid,
                            "forecast_date"
                        ].max(),

                    "coverage":
                        float(
                            valid.mean()
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks(
    panel: pd.DataFrame
):

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "SANITY CHECKS"
    )

    write_log(
        "=" * 100
    )

    # --------------------------------------------------------
    # Duplicate ticker-date rows
    # --------------------------------------------------------

    duplicate_count = (
        panel
        .duplicated(
            subset=[
                "ticker",
                "forecast_date",
            ]
        )
        .sum()
    )

    write_log(
        f"\nDuplicate ticker-date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:

        raise RuntimeError(
            "Duplicate ticker-date rows detected."
        )

    # --------------------------------------------------------
    # Forecast must precede realization.
    # --------------------------------------------------------

    timing_ok = (
        panel["realized_date"]
        > panel["forecast_date"]
    )

    write_log(
        "All realized dates after "
        "forecast dates: "
        f"{bool(timing_ok.all())}"
    )

    if not timing_ok.all():

        raise RuntimeError(
            "Forecast timing violation detected."
        )

    # --------------------------------------------------------
    # Actual return range
    # --------------------------------------------------------

    huge_returns = (
        panel["actual_return"]
        .abs()
        > 1.0
    ).sum()

    write_log(
        "Next-day returns with "
        "|return| > 100%: "
        f"{huge_returns}"
    )

    # --------------------------------------------------------
    # Model breach rates
    # --------------------------------------------------------

    breach_columns = {
        "HS":
            "hs_breach",

        "EWMA":
            "ewma_breach",

        "GARCH-t":
            "garch_breach",

        "QRF":
            "qrf_breach",
    }

    for model, column in (
        breach_columns.items()
    ):

        valid = panel[
            column
        ].notna()

        if valid.sum() == 0:

            write_log(
                f"{model:8s} breach rate: "
                "NO VALID OBSERVATIONS"
            )

            continue

        rate = (
            panel.loc[
                valid,
                column
            ]
            .mean()
        )

        write_log(
            f"{model:8s} breach rate: "
            f"{rate:.4%}"
        )

    # --------------------------------------------------------
    # QRF trees
    # --------------------------------------------------------

    qrf_tree_counts = (
        panel[
            "qrf_n_trees"
        ]
        .dropna()
    )

    if len(qrf_tree_counts) > 0:

        write_log(
            "QRF trees min/max: "
            f"{int(qrf_tree_counts.min())} / "
            f"{int(qrf_tree_counts.max())}"
        )

    # --------------------------------------------------------
    # Rows per ticker
    # --------------------------------------------------------

    write_log(
        "\nRows per ticker:"
    )

    rows_per_ticker = (
        panel
        .groupby(
            "ticker"
        )
        .size()
    )

    print(
        rows_per_ticker.to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    write_log(
        "=" * 100
    )

    write_log(
        "RS01 — LONG-HISTORY VaR RECONSTRUCTION"
    )

    write_log(
        "=" * 100
    )

    write_log(
        "\nResearch purpose:"
    )

    write_log(
        "Construct a strictly causal long-history "
        "VaR forecast panel for HS, EWMA, GARCH-t "
        "and QRF before studying regime shifts."
    )

    write_log(
        f"\nROOT:\n{ROOT}"
    )

    write_log(
        f"\nRAW ZIP:\n{RAW_ZIP_PATH}"
    )

    write_log(
        f"\nWindow: {WINDOW}"
    )

    write_log(
        f"Refit frequency: every "
        f"{REFIT_EVERY} forecasts"
    )

    write_log(
        f"QRF trees: {QRF_TREES}"
    )

    write_log(
        f"QRF minimum leaf: {QRF_MIN_LEAF}"
    )

    # ========================================================
    # LOAD RAW DATA
    # ========================================================

    stocks, vix_df = (
        load_raw_data()
    )

    # ========================================================
    # PROCESS ALL 24 TICKERS
    # ========================================================

    ticker_panels = []

    for ticker in TICKERS:

        result = process_ticker(
            ticker,
            stocks[ticker],
            vix_df
        )

        write_log(
            f"\n{ticker}: generated "
            f"{len(result):,} forecast rows"
        )

        ticker_panels.append(
            result
        )

    # ========================================================
    # COMBINE
    # ========================================================

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "COMBINING RESULTS"
    )

    write_log(
        "=" * 100
    )

    panel = pd.concat(
        ticker_panels,
        ignore_index=True
    )

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

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    write_log(
        f"\nTotal rows: "
        f"{len(panel):,}"
    )

    write_log(
        f"Tickers: "
        f"{panel['ticker'].nunique()}"
    )

    write_log(
        f"Forecast range: "
        f"{panel['forecast_date'].min().date()} "
        f"to "
        f"{panel['forecast_date'].max().date()}"
    )

    write_log(
        f"Realized range: "
        f"{panel['realized_date'].min().date()} "
        f"to "
        f"{panel['realized_date'].max().date()}"
    )

    # ========================================================
    # SANITY CHECKS
    # ========================================================

    run_sanity_checks(
        panel
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        panel
    )

    coverage = build_coverage(
        panel
    )

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "MODEL SUMMARY"
    )

    write_log(
        "=" * 100
    )

    print(
        summary.to_string(
            index=False,
            float_format="%.6f"
        )
    )

    # ========================================================
    # SAVE PANEL
    # ========================================================

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "SAVING OUTPUTS"
    )

    write_log(
        "=" * 100
    )

    panel.to_csv(
        OUTPUT_PANEL,
        index=False
    )

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False
    )

    coverage.to_csv(
        OUTPUT_COVERAGE,
        index=False
    )

    write_log(
        f"\nPanel:\n{OUTPUT_PANEL}"
    )

    write_log(
        f"\nSummary:\n{OUTPUT_SUMMARY}"
    )

    write_log(
        f"\nCoverage:\n{OUTPUT_COVERAGE}"
    )

    # ========================================================
    # SAVE LOG
    # ========================================================

    with open(
        OUTPUT_LOG,
        "w",
        encoding="utf-8"
    ) as log_file:

        log_file.write(
            "\n".join(
                LOG_LINES
            )
        )

    write_log(
        f"\nLog:\n{OUTPUT_LOG}"
    )

    write_log(
        "\n" + "=" * 100
    )

    write_log(
        "RS01 COMPLETE"
    )

    write_log(
        "=" * 100
    )


if __name__ == "__main__":

    main()