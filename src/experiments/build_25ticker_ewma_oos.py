# ============================================================
# 27_build_25ticker_ewma_oos.py
#
# Purpose:
#   Build 25-ticker EWMA 5% VaR forecasts aligned to the
#   existing frozen QRF out-of-sample forecasts.
#
# EWMA:
#   lambda = 0.94
#
# Forecast:
#   sigma_t^2 = lambda * sigma_(t-1)^2
#               + (1-lambda) * R_t^2
#
#   q05_t = Phi^(-1)(0.05) * sigma_t
#
# IMPORTANT:
#   The EWMA forecast dated t uses information available through
#   return date t and forecasts the next trading day's return.
#
#   This makes it directly comparable to the QRF forecast made
#   on the same forecast_date.
#
# Inputs:
#   data/monitoring/QRF_25ticker_oos_predictions.csv
#   data/processed/<ticker>.csv
#
# Output:
#   data/monitoring/EWMA_25ticker_oos_predictions.csv
#
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# CONFIGURATION
# ============================================================

LAMBDA = 0.94
ALPHA = 0.05

# Use an initial 250-observation variance for the EWMA
# recursion, consistent with the project's standard VaR
# estimation setup.
INITIAL_WINDOW = 250

MIN_OBSERVATIONS = 300

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

ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)

MONITORING_DIR = (
    ROOT
    / "data"
    / "monitoring"
)

QRF_FILE = (
    MONITORING_DIR
    / "QRF_25ticker_oos_predictions.csv"
)

OUTPUT_FILE = (
    MONITORING_DIR
    / "EWMA_25ticker_oos_predictions.csv"
)


# ============================================================
# HELPER
# ============================================================

def find_column(df, candidates):
    """
    Exact case-insensitive column match.
    """

    mapping = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in mapping:
            return mapping[key]

    raise ValueError(
        f"\nCould not find any of {candidates}"
        f"\nAvailable columns:"
        f"\n{df.columns.tolist()}"
    )


def load_processed(ticker):
    """
    Load processed return data for one ticker.
    """

    path = (
        PROCESSED_DIR
        / f"{ticker}.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"\nProcessed file not found:"
            f"\n{path}"
        )

    df = pd.read_csv(path)

    date_col = find_column(
        df,
        ["Date", "date"],
    )

    return_col = find_column(
        df,
        [
            "Log_Return",
            "log_return",
        ],
    )

    out = df[
        [
            date_col,
            return_col,
        ]
    ].copy()

    out.columns = [
        "date",
        "return",
    ]

    out["date"] = pd.to_datetime(
        out["date"],
        errors="coerce",
    )

    out["return"] = pd.to_numeric(
        out["return"],
        errors="coerce",
    )

    out = (
        out
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out


def compute_ewma_series(
    returns,
    dates,
    lam=0.94,
    initial_window=250,
):
    """
    Compute the EWMA conditional volatility series.

    For date t:

        sigma_t^2 =
            lambda * sigma_(t-1)^2
            + (1-lambda) * R_t^2

    The resulting sigma_t is the volatility estimate
    available after observing return R_t.

    Therefore the associated q05_t forecasts R_(t+1).
    """

    returns = np.asarray(
        returns,
        dtype=float,
    )

    dates = pd.to_datetime(
        dates
    )

    n = len(returns)

    sigma2 = np.full(
        n,
        np.nan,
        dtype=float,
    )

    if n < initial_window:
        return pd.DataFrame(
            {
                "date": dates,
                "ewma_sigma": sigma2,
                "ewma_q05": np.nan,
            }
        )

    # --------------------------------------------------------
    # Initial variance
    # --------------------------------------------------------

    initial_var = np.var(
        returns[
            :initial_window
        ],
        ddof=1,
    )

    sigma2[
        initial_window - 1
    ] = initial_var

    # --------------------------------------------------------
    # EWMA recursion
    # --------------------------------------------------------

    for i in range(
        initial_window,
        n,
    ):

        sigma2[i] = (
            lam
            * sigma2[i - 1]
            +
            (1.0 - lam)
            * returns[i] ** 2
        )

    sigma = np.sqrt(
        sigma2
    )

    q05 = (
        norm.ppf(ALPHA)
        * sigma
    )

    out = pd.DataFrame(
        {
            "date": dates,
            "ewma_sigma": sigma,
            "ewma_q05": q05,
        }
    )

    return out


# ============================================================
# START
# ============================================================

print("=" * 100)
print("BUILD 25-TICKER EWMA OOS VaR")
print("=" * 100)

print(
    f"\nLambda: "
    f"{LAMBDA}"
)

print(
    f"Alpha: "
    f"{ALPHA}"
)

print(
    f"Initial variance window: "
    f"{INITIAL_WINDOW}"
)


# ============================================================
# LOAD QRF OOS
# ============================================================

print("\n" + "=" * 100)
print("LOADING QRF OOS DATA")
print("=" * 100)

if not QRF_FILE.exists():

    raise FileNotFoundError(
        f"\nQRF file not found:\n{QRF_FILE}"
    )

qrf = pd.read_csv(
    QRF_FILE
)

qrf["forecast_date"] = pd.to_datetime(
    qrf["forecast_date"],
    errors="coerce",
)

qrf["realized_date"] = pd.to_datetime(
    qrf["realized_date"],
    errors="coerce",
)

print(
    f"\nQRF rows: "
    f"{len(qrf):,}"
)

print(
    f"QRF tickers: "
    f"{qrf['ticker'].nunique()}"
)


# ============================================================
# BUILD EWMA
# ============================================================

all_results = []

summary_rows = []


for ticker in TICKERS:

    print("\n" + "-" * 90)
    print(
        f"PROCESSING {ticker}"
    )
    print("-" * 90)

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
            "WARNING: no QRF observations."
        )

        continue

    processed = load_processed(
        ticker
    )

    print(
        f"Processed rows: "
        f"{len(processed)}"
    )

    print(
        f"Processed dates: "
        f"{processed['date'].min().date()}"
        f" to "
        f"{processed['date'].max().date()}"
    )

    if len(processed) < MIN_OBSERVATIONS:

        print(
            "WARNING: insufficient observations."
        )

        continue

    # --------------------------------------------------------
    # Compute full EWMA series
    # --------------------------------------------------------

    ewma = compute_ewma_series(
        returns=processed["return"],
        dates=processed["date"],
        lam=LAMBDA,
        initial_window=INITIAL_WINDOW,
    )

    # --------------------------------------------------------
    # Keep only EWMA values at QRF forecast dates
    # --------------------------------------------------------

    merged = ticker_qrf[
        [
            "ticker",
            "forecast_date",
            "realized_date",
            "actual_return",
        ]
    ].merge(
        ewma,
        left_on="forecast_date",
        right_on="date",
        how="left",
    )

    merged = merged.drop(
        columns=["date"]
    )

    # --------------------------------------------------------
    # Verify forecast / realized ordering
    # --------------------------------------------------------

    merged["calendar_gap_days"] = (
        merged["realized_date"]
        - merged["forecast_date"]
    ).dt.days

    # --------------------------------------------------------
    # EWMA breach against next-day realized return
    # --------------------------------------------------------

    merged["ewma_breach"] = np.where(
        merged["actual_return"]
        <
        merged["ewma_q05"],
        1,
        0,
    )

    # --------------------------------------------------------
    # Retain valid forecasts only
    # --------------------------------------------------------

    valid = merged[
        [
            "ewma_sigma",
            "ewma_q05",
        ]
    ].notna().all(axis=1)

    merged_valid = merged[
        valid
    ].copy()

    print(
        f"QRF rows: "
        f"{len(ticker_qrf)}"
    )

    print(
        f"EWMA valid rows: "
        f"{len(merged_valid)}"
    )

    if len(merged_valid) > 0:

        print(
            f"EWMA breach rate: "
            f"{merged_valid['ewma_breach'].mean():.6f}"
        )

        print(
            f"EWMA q05 mean: "
            f"{merged_valid['ewma_q05'].mean():.6f}"
        )

        print(
            f"EWMA sigma mean: "
            f"{merged_valid['ewma_sigma'].mean():.6f}"
        )

        print(
            f"Forecast-realized gap:"
            f" {merged_valid['calendar_gap_days'].median():.0f}"
            f" calendar day median"
        )

    all_results.append(
        merged_valid[
            [
                "ticker",
                "forecast_date",
                "realized_date",
                "actual_return",
                "ewma_sigma",
                "ewma_q05",
                "ewma_breach",
            ]
        ]
    )

    summary_rows.append(
        {
            "ticker":
                ticker,

            "qrf_rows":
                len(ticker_qrf),

            "ewma_valid_rows":
                len(merged_valid),

            "ewma_breach_rate":
                (
                    merged_valid[
                        "ewma_breach"
                    ].mean()
                    if len(merged_valid) > 0
                    else np.nan
                ),

            "mean_ewma_q05":
                (
                    merged_valid[
                        "ewma_q05"
                    ].mean()
                    if len(merged_valid) > 0
                    else np.nan
                ),

            "mean_ewma_sigma":
                (
                    merged_valid[
                        "ewma_sigma"
                    ].mean()
                    if len(merged_valid) > 0
                    else np.nan
                ),
        }
    )


# ============================================================
# CONCATENATE
# ============================================================

if len(all_results) == 0:

    raise RuntimeError(
        "No EWMA results generated."
    )

ewma_all = pd.concat(
    all_results,
    ignore_index=True,
)

summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SORT
# ============================================================

ewma_all = (
    ewma_all
    .sort_values(
        [
            "ticker",
            "realized_date",
        ]
    )
    .reset_index(drop=True)
)


summary = summary.sort_values(
    "ticker"
).reset_index(drop=True)


# ============================================================
# SAVE
# ============================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ewma_all.to_csv(
    OUTPUT_FILE,
    index=False,
)


SUMMARY_FILE = (
    MONITORING_DIR
    / "EWMA_25ticker_summary.csv"
)

summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FINAL EWMA SUMMARY")
print("=" * 100)

print(
    f"\nRows: "
    f"{len(ewma_all):,}"
)

print(
    f"Tickers: "
    f"{ewma_all['ticker'].nunique()}"
)

print(
    f"Forecast dates:"
    f" {ewma_all['forecast_date'].min().date()}"
    f" to "
    f"{ewma_all['forecast_date'].max().date()}"
)

print(
    f"Realized dates:"
    f" {ewma_all['realized_date'].min().date()}"
    f" to "
    f"{ewma_all['realized_date'].max().date()}"
)


print(
    "\nTicker summary:"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# OVERALL CHECKS
# ============================================================

print("\n" + "=" * 100)
print("OVERALL SANITY CHECKS")
print("=" * 100)

print(
    f"\nOverall EWMA breach rate:"
    f" {ewma_all['ewma_breach'].mean():.6f}"
)

print(
    f"Mean EWMA q05:"
    f" {ewma_all['ewma_q05'].mean():.6f}"
)

print(
    f"Mean EWMA sigma:"
    f" {ewma_all['ewma_sigma'].mean():.6f}"
)

print(
    "\nForecast -> realized calendar gap:"
)

print(
    ewma_all[
        "realized_date"
    ]
    .sub(
        ewma_all[
            "forecast_date"
        ]
    )
    .dt.days
    .describe()
    .to_string()
)


# ============================================================
# AAPL CROSS-CHECK
# ============================================================

aapl = ewma_all[
    ewma_all["ticker"] == "AAPL"
].copy()


print("\n" + "=" * 100)
print("AAPL CROSS-CHECK")
print("=" * 100)

if len(aapl) > 0:

    print(
        "\nLast 10 AAPL EWMA forecasts:"
    )

    print(
        aapl[
            [
                "forecast_date",
                "realized_date",
                "actual_return",
                "ewma_sigma",
                "ewma_q05",
                "ewma_breach",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


# ============================================================
# SAVE PATHS
# ============================================================

print("\n" + "=" * 100)
print("SAVED")
print("=" * 100)

print(
    OUTPUT_FILE
)

print(
    SUMMARY_FILE
)

print("\n" + "=" * 100)
print(
    "25-TICKER EWMA OOS BUILD COMPLETE"
)
print("=" * 100)