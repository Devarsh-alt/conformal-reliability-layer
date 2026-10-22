import os
import warnings
import numpy as np
import pandas as pd

from scipy.stats import chi2, norm
from quantile_forest import RandomForestQuantileRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\reliability_layer"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "monitoring"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Frozen configuration from AAPL validation
# ------------------------------------------------------------

ALPHA = 0.05

TRAIN_WINDOW = 1000

MIN_SAMPLES_LEAF = 20

MAX_FEATURES = 1.0

N_ESTIMATORS = 500

REFIT_EVERY = 20

TEST_START = pd.Timestamp(
    "2023-01-01"
)


# ------------------------------------------------------------
# Same features used in AAPL QRF
# ------------------------------------------------------------

FEATURES = [
    "Log_Return",
    "return_lag1",
    "return_lag5",
    "return_lag10",
    "vol_10d",
    "vol_20d",
    "volume_ratio",
    "VIX"
]


# ------------------------------------------------------------
# 25 ticker universe
# ------------------------------------------------------------

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
    "XOM"
]


# ============================================================
# OUTPUT FILES
# ============================================================

PREDICTION_FILE = os.path.join(
    OUTPUT_DIR,
    "QRF_25ticker_oos_predictions.csv"
)

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "QRF_25ticker_summary.csv"
)

POOLED_FILE = os.path.join(
    OUTPUT_DIR,
    "QRF_25ticker_pooled_oos.csv"
)


# ============================================================
# PINBALL LOSS
# ============================================================

def pinball_loss(
    actual,
    forecast,
    alpha=0.05
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )

    hit = (
        actual < forecast
    ).astype(float)

    return (
        actual - forecast
    ) * (
        alpha - hit
    )


# ============================================================
# KUPIEC TEST
# ============================================================

def kupiec_test(
    breaches,
    alpha=0.05
):
    """
    Kupiec unconditional coverage test.
    """

    hits = np.asarray(
        breaches,
        dtype=int
    )

    n = len(hits)

    if n == 0:
        return np.nan

    x = int(
        hits.sum()
    )

    if x == 0 or x == n:
        # Handle boundary cases
        p_hat = x / n

        if x == 0:
            log_l1 = n * np.log(
                max(1.0 - p_hat, 1e-15)
            )
        else:
            log_l1 = n * np.log(
                max(p_hat, 1e-15)
            )

        log_l0 = (
            x * np.log(alpha)
            +
            (n - x)
            * np.log(1.0 - alpha)
        )

    else:

        p_hat = x / n

        log_l1 = (
            x * np.log(p_hat)
            +
            (n - x)
            * np.log(1.0 - p_hat)
        )

        log_l0 = (
            x * np.log(alpha)
            +
            (n - x)
            * np.log(1.0 - alpha)
        )

    lr = -2.0 * (
        log_l0 - log_l1
    )

    return float(
        chi2.sf(
            lr,
            1
        )
    )


# ============================================================
# CHRISTOFFERSEN INDEPENDENCE TEST
# ============================================================

def christoffersen_independence(
    breaches
):

    hits = np.asarray(
        breaches,
        dtype=int
    )

    if len(hits) < 2:
        return np.nan

    # Transition counts
    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for i in range(
        1,
        len(hits)
    ):

        prev = hits[i - 1]
        curr = hits[i]

        if prev == 0 and curr == 0:
            n00 += 1

        elif prev == 0 and curr == 1:
            n01 += 1

        elif prev == 1 and curr == 0:
            n10 += 1

        elif prev == 1 and curr == 1:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return np.nan

    pi0 = n01 / n0
    pi1 = n11 / n1

    total_ones = n01 + n11
    total = (
        n00
        + n01
        + n10
        + n11
    )

    pi = total_ones / max(
        total,
        1
    )

    def safe_log(x):
        return np.log(
            max(x, 1e-15)
        )

    log_l_ind = (
        (n01 + n11)
        * safe_log(pi)
        +
        (n00 + n10)
        * safe_log(1.0 - pi)
    )

    log_l_markov = (
        n01 * safe_log(pi0)
        +
        n00 * safe_log(1.0 - pi0)
        +
        n11 * safe_log(pi1)
        +
        n10 * safe_log(1.0 - pi1)
    )

    lr = -2.0 * (
        log_l_ind
        -
        log_l_markov
    )

    return float(
        chi2.sf(
            lr,
            1
        )
    )


# ============================================================
# DQ TEST
# ============================================================

def dq_test(
    returns,
    quantile_forecasts,
    alpha=0.05,
    lags=5
):
    """
    Dynamic Quantile test.

    Hit_t = I(R_t < q_t)

    Centered hit:
        h_t = Hit_t - alpha

    Regress:
        h_t =
            beta_0
            + beta_1 h_{t-1}
            + ...
            + beta_k h_{t-k}
            + beta_q q_t
            + error

    DQ statistic:
        beta'X'X beta / [alpha(1-alpha)]

    H0:
        Correct dynamic quantile specification.
    """

    returns = np.asarray(
        returns,
        dtype=float
    )

    q = np.asarray(
        quantile_forecasts,
        dtype=float
    )

    if len(returns) != len(q):
        raise ValueError(
            "Returns and forecasts "
            "must have same length."
        )

    hits = (
        returns < q
    ).astype(float)

    centered = (
        hits - alpha
    )

    X_rows = []
    y_rows = []

    for t in range(
        lags,
        len(hits)
    ):

        row = [1.0]

        # Lagged violations
        for lag in range(
            1,
            lags + 1
        ):

            row.append(
                centered[
                    t - lag
                ]
            )

        # Current quantile forecast
        row.append(
            q[t]
        )

        X_rows.append(row)
        y_rows.append(
            centered[t]
        )

    X = np.asarray(
        X_rows,
        dtype=float
    )

    y = np.asarray(
        y_rows,
        dtype=float
    )

    n = len(y)

    if n <= X.shape[1]:

        return np.nan

    XtX = X.T @ X

    try:
        XtX_inv = np.linalg.inv(
            XtX
        )
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(
            XtX
        )

    beta = (
        XtX_inv
        @ X.T
        @ y
    )

    dq_stat = (
        beta.T
        @ XtX
        @ beta
        /
        (
            alpha
            *
            (1.0 - alpha)
        )
    )

    dq_stat = float(
        dq_stat
    )

    df = X.shape[1]

    return float(
        chi2.sf(
            dq_stat,
            df
        )
    )


# ============================================================
# MAX CONSECUTIVE BREACHES
# ============================================================

def max_consecutive_breaches(
    breaches
):

    hits = np.asarray(
        breaches,
        dtype=int
    )

    max_run = 0
    current = 0

    for hit in hits:

        if hit == 1:

            current += 1

            max_run = max(
                max_run,
                current
            )

        else:

            current = 0

    return int(
        max_run
    )


# ============================================================
# QRF PREDICTION HELPER
# ============================================================

def predict_quantile(
    model,
    X
):

    prediction = model.predict(
        X,
        quantiles=ALPHA
    )

    prediction = np.asarray(
        prediction
    )

    # quantile_forest can return
    # either (n,) or (n,1)
    if prediction.ndim == 2:

        prediction = prediction[:, 0]

    return prediction.astype(
        float
    )


# ============================================================
# PROCESS ONE TICKER
# ============================================================

def run_ticker(
    ticker
):

    file_path = os.path.join(
        DATA_DIR,
        f"{ticker}.csv"
    )

    print("\n" + "=" * 80)
    print(
        f"{ticker} — QRF FINAL FROZEN OOS"
    )
    print("=" * 80)

    if not os.path.exists(
        file_path
    ):

        print(
            f"SKIP: file not found: "
            f"{file_path}"
        )

        return None, None

    df = pd.read_csv(
        file_path
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required = (
        ["Date"]
        + FEATURES
    )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        print(
            f"SKIP: missing columns "
            f"{missing}"
        )

        return None, None

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

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
    # Create one-step-ahead target explicitly
    #
    # Row t:
    #   features from t
    #
    # Forecast:
    #   R(t+1)
    #
    # realized_date:
    #   t+1
    # --------------------------------------------------------

    df["target_return"] = (
        df["Log_Return"]
        .shift(-1)
    )

    df["realized_date"] = (
        df["Date"]
        .shift(-1)
    )

    # --------------------------------------------------------
    # Keep rows with all required information
    # --------------------------------------------------------

    model_df = df[
        [
            "Date",
            "realized_date",
            "target_return"
        ]
        + FEATURES
    ].copy()

    model_df = (
        model_df
        .dropna()
        .reset_index(drop=True)
    )

    if len(model_df) <= TRAIN_WINDOW:

        print(
            "SKIP: fewer rows than "
            f"train window ({len(model_df)})"
        )

        return None, None

    print(
        f"Usable observations: "
        f"{len(model_df)}"
    )

    print(
        f"Date range: "
        f"{model_df['Date'].min().date()} "
        f"to "
        f"{model_df['realized_date'].max().date()}"
    )

    # --------------------------------------------------------
    # Find first possible OOS forecast
    # --------------------------------------------------------

    eligible = model_df.index[
        model_df["Date"] >= TEST_START
    ]

    if len(eligible) == 0:

        print(
            "SKIP: no observations "
            "after test start."
        )

        return None, None

    first_test_idx = int(
        eligible[0]
    )

    # Need 1000 previous observations
    if first_test_idx < TRAIN_WINDOW:

        first_test_idx = TRAIN_WINDOW

    # --------------------------------------------------------
    # Test rows
    # --------------------------------------------------------

    test_indices = np.arange(
        first_test_idx,
        len(model_df)
    )

    if len(test_indices) == 0:

        print(
            "SKIP: no OOS observations."
        )

        return None, None

    test_start_date = model_df.loc[
        test_indices[0],
        "Date"
    ]

    print(
        f"Effective test start: "
        f"{test_start_date.date()}"
    )

    print(
        f"OOS forecast observations: "
        f"{len(test_indices)}"
    )

    # --------------------------------------------------------
    # Walk-forward refitting
    # --------------------------------------------------------

    prediction_records = []

    fit_number = 0

    block_start = 0

    while block_start < len(
        test_indices
    ):

        block_indices = test_indices[
            block_start:
            block_start + REFIT_EVERY
        ]

        first_idx = int(
            block_indices[0]
        )

        last_idx = int(
            block_indices[-1]
        )

        # Training ends immediately
        # before the first test row.
        train_end = first_idx

        train_start = (
            train_end
            - TRAIN_WINDOW
        )

        if train_start < 0:

            raise ValueError(
                f"{ticker}: insufficient "
                "training history."
            )

        train_slice = model_df.iloc[
            train_start:
            train_end
        ]

        test_slice = model_df.iloc[
            first_idx:
            last_idx + 1
        ]

        X_train = train_slice[
            FEATURES
        ].values

        y_train = train_slice[
            "target_return"
        ].values

        X_test = test_slice[
            FEATURES
        ].values

        # ----------------------------------------------------
        # Fit QRF
        # ----------------------------------------------------

        model = (
            RandomForestQuantileRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=MAX_FEATURES,
                n_jobs=-1,
                random_state=1000 + fit_number
            )
        )

        model.fit(
            X_train,
            y_train
        )

        q05 = predict_quantile(
            model,
            X_test
        )

        # ----------------------------------------------------
        # Store forecasts
        # ----------------------------------------------------

        for i in range(
            len(test_slice)
        ):

            row = test_slice.iloc[
                i
            ]

            actual = float(
                row["target_return"]
            )

            forecast = float(
                q05[i]
            )

            breach = int(
                actual < forecast
            )

            loss = float(
                pinball_loss(
                    np.array([actual]),
                    np.array([forecast]),
                    ALPHA
                )[0]
            )

            prediction_records.append(
                {
                    "ticker": ticker,
                    "forecast_date":
                        row["Date"],
                    "realized_date":
                        row["realized_date"],
                    "actual_return":
                        actual,
                    "q05_forecast":
                        forecast,
                    "VaR_5pct":
                        -forecast,
                    "breach":
                        breach,
                    "pinball_loss":
                        loss,
                    "fit_number":
                        fit_number + 1
                }
            )

        fit_number += 1

        print(
            f"Fit {fit_number:3d} | "
            f"train="
            f"{train_slice['Date'].min().date()}"
            f" to "
            f"{train_slice['Date'].max().date()}"
            f" | test="
            f"{test_slice['Date'].min().date()}"
            f" to "
            f"{test_slice['Date'].max().date()}"
            f" | n_test="
            f"{len(test_slice)}"
        )

        block_start += REFIT_EVERY

    # ========================================================
    # PREDICTION DATAFRAME
    # ========================================================

    predictions = pd.DataFrame(
        prediction_records
    )

    if predictions.empty:

        return None, None

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    predictions = (
        predictions
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # METRICS
    # ========================================================

    actual = predictions[
        "actual_return"
    ].values

    q05 = predictions[
        "q05_forecast"
    ].values

    breaches = predictions[
        "breach"
    ].values

    losses = predictions[
        "pinball_loss"
    ].values

    n = len(
        predictions
    )

    breach_count = int(
        breaches.sum()
    )

    breach_rate = (
        breach_count / n
    )

    mean_loss = float(
        losses.mean()
    )

    mean_var = float(
        predictions[
            "VaR_5pct"
        ].mean()
    )

    median_var = float(
        predictions[
            "VaR_5pct"
        ].median()
    )

    kupiec_p = kupiec_test(
        breaches,
        ALPHA
    )

    christoffersen_p = (
        christoffersen_independence(
            breaches
        )
    )

    dq_p = dq_test(
        actual,
        q05,
        ALPHA,
        lags=5
    )

    max_consecutive = (
        max_consecutive_breaches(
            breaches
        )
    )

    summary = {
        "ticker": ticker,
        "n": n,
        "oos_forecast_start":
            predictions[
                "forecast_date"
            ].min(),
        "oos_realized_start":
            predictions[
                "realized_date"
            ].min(),
        "oos_realized_end":
            predictions[
                "realized_date"
            ].max(),
        "breaches":
            breach_count,
        "breach_rate":
            breach_rate,
        "target_rate":
            ALPHA,
        "breach_rate_error":
            breach_rate - ALPHA,
        "mean_pinball_loss":
            mean_loss,
        "mean_VaR":
            mean_var,
        "median_VaR":
            median_var,
        "kupiec_p":
            kupiec_p,
        "christoffersen_independence_p":
            christoffersen_p,
        "DQ_p":
            dq_p,
        "max_consecutive_breaches":
            max_consecutive,
        "kupiec_pass":
            (
                kupiec_p >= 0.05
                if np.isfinite(
                    kupiec_p
                )
                else False
            ),
        "independence_pass":
            (
                christoffersen_p >= 0.05
                if np.isfinite(
                    christoffersen_p
                )
                else False
            ),
        "DQ_pass":
            (
                dq_p >= 0.05
                if np.isfinite(dq_p)
                else False
            )
    }

    summary_df = pd.DataFrame(
        [summary]
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "-" * 80)
    print(
        f"{ticker} RESULTS"
    )
    print("-" * 80)

    print(
        f"Forecast dates:       "
        f"{summary['oos_forecast_start'].date()} "
        f"to "
        f"{summary['oos_realized_end'].date()}"
    )

    print(
        f"Observations:         "
        f"{n}"
    )

    print(
        f"Breaches:             "
        f"{breach_count}"
    )

    print(
        f"Breach rate:          "
        f"{breach_rate:.6f}"
    )

    print(
        f"Mean pinball loss:    "
        f"{mean_loss:.8f}"
    )

    print(
        f"Kupiec p:             "
        f"{kupiec_p:.6f}"
    )

    print(
        f"Independence p:       "
        f"{christoffersen_p:.6f}"
    )

    print(
        f"DQ p:                 "
        f"{dq_p:.6f}"
    )

    print(
        f"Max consecutive:      "
        f"{max_consecutive}"
    )

    return (
        predictions,
        summary_df
    )


# ============================================================
# MAIN LOOP
# ============================================================

print("\n" + "=" * 80)
print("25-TICKER QUANTILE RANDOM FOREST — FINAL OOS")
print("=" * 80)

print("\nFrozen configuration:")
print(
    f"  Train window:       {TRAIN_WINDOW}"
)

print(
    f"  Min leaf:           {MIN_SAMPLES_LEAF}"
)

print(
    f"  Max features:       {MAX_FEATURES}"
)

print(
    f"  Trees:              {N_ESTIMATORS}"
)

print(
    f"  Refit frequency:    {REFIT_EVERY}"
)

print(
    f"  Quantile:           {ALPHA}"
)

print(
    f"  Test start:         {TEST_START.date()}"
)

print(
    f"  Number of tickers:  {len(TICKERS)}"
)


all_predictions = []
all_summaries = []


for ticker_idx, ticker in enumerate(
    TICKERS,
    start=1
):

    print(
        f"\n\n{'#' * 80}"
    )

    print(
        f"# TICKER {ticker_idx}/{len(TICKERS)}: "
        f"{ticker}"
    )

    print(
        f"{'#' * 80}"
    )

    try:

        predictions, summary = (
            run_ticker(ticker)
        )

        if predictions is not None:

            all_predictions.append(
                predictions
            )

        if summary is not None:

            all_summaries.append(
                summary
            )

    except Exception as e:

        print(
            f"\nERROR processing {ticker}: "
            f"{type(e).__name__}: {e}"
        )

        continue


# ============================================================
# COMBINE RESULTS
# ============================================================

if not all_predictions:

    raise RuntimeError(
        "No ticker completed successfully."
    )


predictions_all = pd.concat(
    all_predictions,
    ignore_index=True
)

summary_all = pd.concat(
    all_summaries,
    ignore_index=True
)


# ============================================================
# SAVE INDIVIDUAL OOS PREDICTIONS
# ============================================================

predictions_all.to_csv(
    PREDICTION_FILE,
    index=False
)

summary_all.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# POOLED DESCRIPTIVE RESULTS
# ============================================================

pooled = predictions_all.copy()

pooled_breach_rate = (
    pooled["breach"].mean()
)

pooled_pinball = (
    pooled["pinball_loss"].mean()
)

pooled_n = len(
    pooled
)

pooled_breaches = int(
    pooled["breach"].sum()
)


# Equal-weight ticker metrics
mean_ticker_loss = (
    summary_all[
        "mean_pinball_loss"
    ].mean()
)

median_ticker_loss = (
    summary_all[
        "mean_pinball_loss"
    ].median()
)

mean_breach_rate = (
    summary_all[
        "breach_rate"
    ].mean()
)

median_breach_rate = (
    summary_all[
        "breach_rate"
    ].median()
)


pooled_summary = pd.DataFrame(
    [
        {
            "type": "pooled",
            "ticker": "ALL",
            "n": pooled_n,
            "breaches":
                pooled_breaches,
            "breach_rate":
                pooled_breach_rate,
            "mean_pinball_loss":
                pooled_pinball,
            "mean_ticker_pinball":
                mean_ticker_loss,
            "median_ticker_pinball":
                median_ticker_loss,
            "mean_ticker_breach_rate":
                mean_breach_rate,
            "median_ticker_breach_rate":
                median_breach_rate
        }
    ]
)


pooled_summary.to_csv(
    POOLED_FILE,
    index=False
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL 25-TICKER QRF SUMMARY")
print("=" * 80)

display_cols = [
    "ticker",
    "n",
    "breaches",
    "breach_rate",
    "mean_pinball_loss",
    "kupiec_p",
    "christoffersen_independence_p",
    "DQ_p",
    "max_consecutive_breaches",
    "kupiec_pass",
    "independence_pass",
    "DQ_pass"
]


print(
    summary_all[
        display_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# PASS COUNTS
# ============================================================

print("\n" + "=" * 80)
print("CALIBRATION PASS COUNTS")
print("=" * 80)

print(
    f"Tickers completed:          "
    f"{len(summary_all)}"
)

print(
    f"Kupiec pass:                "
    f"{summary_all['kupiec_pass'].sum()}"
    f"/{len(summary_all)}"
)

print(
    f"Independence pass:          "
    f"{summary_all['independence_pass'].sum()}"
    f"/{len(summary_all)}"
)

print(
    f"DQ pass:                    "
    f"{summary_all['DQ_pass'].sum()}"
    f"/{len(summary_all)}"
)

print(
    f"All 3 tests pass:           "
    f"{(
        summary_all[
            [
                'kupiec_pass',
                'independence_pass',
                'DQ_pass'
            ]
        ].all(axis=1)
    ).sum()}"
    f"/{len(summary_all)}"
)


# ============================================================
# BREACH-RATE DISPERSION
# ============================================================

print("\n" + "=" * 80)
print("BREACH RATE DISTRIBUTION")
print("=" * 80)

print(
    f"Mean ticker breach rate:    "
    f"{mean_breach_rate:.6f}"
)

print(
    f"Median ticker breach rate:  "
    f"{median_breach_rate:.6f}"
)

print(
    f"Min ticker breach rate:     "
    f"{summary_all['breach_rate'].min():.6f}"
)

print(
    f"Max ticker breach rate:     "
    f"{summary_all['breach_rate'].max():.6f}"
)


# ============================================================
# PINBALL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("PINBALL LOSS DISTRIBUTION")
print("=" * 80)

print(
    f"Pooled mean loss:            "
    f"{pooled_pinball:.8f}"
)

print(
    f"Mean ticker loss:            "
    f"{mean_ticker_loss:.8f}"
)

print(
    f"Median ticker loss:          "
    f"{median_ticker_loss:.8f}"
)

print(
    f"Best ticker loss:            "
    f"{summary_all['mean_pinball_loss'].min():.8f}"
)

print(
    f"Worst ticker loss:           "
    f"{summary_all['mean_pinball_loss'].max():.8f}"
)


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"\nOOS predictions:\n"
    f"{PREDICTION_FILE}"
)

print(
    f"\nTicker summary:\n"
    f"{SUMMARY_FILE}"
)

print(
    f"\nPooled summary:\n"
    f"{POOLED_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("25-TICKER QRF BASELINE COMPLETE")
print("=" * 80)