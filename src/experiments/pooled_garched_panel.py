from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import t as student_t

warnings.filterwarnings("ignore")

# =============================================================================
# POOLED GARCH(1,1)-STUDENT-t OOS PANEL
# =============================================================================
# Build a 25-ticker GARCH(1,1)-Student-t VaR panel aligned to the QRF forecast
# dates.
#
# Rolling protocol:
#   - 1000-return estimation window
#   - refit every 20 forecast observations
#   - strict historical cutoff: return_date < first_forecast_date
#   - parameters frozen within each 20-observation forecast block
#   - conditional variance recursively updated with newly observed returns
#
# Outputs:
#   data/monitoring/GARCH_25ticker_oos_predictions.csv
#   data/monitoring/GARCH_25ticker_summary.csv
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
MONITORING_DIR = ROOT / "data" / "monitoring"

QRF_PATH = MONITORING_DIR / "QRF_25ticker_oos_predictions.csv"
HS_PANEL_PATH = MONITORING_DIR / "pooled_hs_reliability_panel.csv"

OUTPUT_PATH = MONITORING_DIR / "GARCH_25ticker_oos_predictions.csv"
SUMMARY_PATH = MONITORING_DIR / "GARCH_25ticker_summary.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

ALPHA = 0.05
GARCH_WINDOW = 1000
REFIT_EVERY = 20
TICKER_MIN_HISTORY = 1000

EXPECTED_TICKERS = [
    "AAPL", "AMZN", "BA", "BAC", "CAT",
    "COIN", "CVX", "DIS", "GME", "GOOGL",
    "GS", "JNJ", "JPM", "KO", "MCD",
    "MSFT", "NFLX", "NVDA", "PFE", "PG",
    "TSLA", "UNH", "V", "WMT", "XOM",
]


# =============================================================================
# HELPERS
# =============================================================================

def pinball_loss(actual: float, forecast: float, alpha: float = ALPHA) -> float:
    actual = float(actual)
    forecast = float(forecast)

    if actual < forecast:
        return (alpha - 1.0) * (actual - forecast)

    return alpha * (actual - forecast)


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    description: str,
) -> str:
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not identify {description}.\n"
        f"Candidates: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def standardized_student_t_quantile(alpha: float, nu: float) -> float:
    """
    Quantile of arch's unit-variance standardized Student-t innovation.
    """
    nu = float(nu)

    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError(
            f"Invalid Student-t degrees of freedom: {nu}"
        )

    return float(
        student_t.ppf(alpha, nu)
        * np.sqrt((nu - 2.0) / nu)
    )


def fit_garch(returns_percent: np.ndarray):
    """
    Fit GARCH(1,1)-Student-t.
    """
    returns_percent = np.asarray(
        returns_percent,
        dtype=float,
    )

    returns_percent = returns_percent[
        np.isfinite(returns_percent)
    ]

    if len(returns_percent) < TICKER_MIN_HISTORY:
        raise ValueError(
            f"Insufficient GARCH history: "
            f"{len(returns_percent)} observations"
        )

    model = arch_model(
        returns_percent,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )

    return model.fit(
        disp="off",
        update_freq=0,
    )


def extract_parameters(result):
    """
    Extract fitted GARCH parameters and the final in-sample
    conditional variance.

    IMPORTANT:
    arch can return conditional_volatility as a NumPy ndarray.
    Therefore np.asarray() is used instead of .to_numpy().
    """
    params = result.params

    mu = float(
        params.get(
            "mu",
            0.0,
        )
    )

    omega = float(
        params["omega"]
    )

    alpha = float(
        params["alpha[1]"]
    )

    beta = float(
        params["beta[1]"]
    )

    nu = float(
        params["nu"]
    )

    # FIX 1:
    # conditional_volatility may already be a NumPy ndarray.
    conditional_vol = np.asarray(
        result.conditional_volatility,
        dtype=float,
    )

    conditional_vol = conditional_vol[
        np.isfinite(conditional_vol)
    ]

    if len(conditional_vol) == 0:
        raise ValueError(
            "No finite conditional volatility values in GARCH fit."
        )

    last_sigma2 = float(
        conditional_vol[-1] ** 2
    )

    return (
        mu,
        omega,
        alpha,
        beta,
        nu,
        last_sigma2,
    )


def generate_block_forecasts(
    result,
    block_actual_returns: np.ndarray,
):
    """
    Generate sequential one-step-ahead forecasts with parameters
    frozen for the entire block.

    Each newly observed realized return updates the conditional
    variance recursion.
    """

    (
        mu,
        omega,
        alpha,
        beta,
        nu,
        sigma2_last,
    ) = extract_parameters(
        result
    )

    std_t_q = standardized_student_t_quantile(
        ALPHA,
        nu,
    )

    conditional_sigma2 = float(
        sigma2_last
    )

    # FIX 2:
    # result.model.y can also already be a NumPy ndarray.
    train_returns = np.asarray(
        result.model.y,
        dtype=float,
    )

    train_returns = train_returns[
        np.isfinite(train_returns)
    ]

    if len(train_returns) == 0:
        raise ValueError(
            "No finite training returns available in GARCH result."
        )

    last_train_return = float(
        train_returns[-1]
    )

    last_residual = (
        last_train_return - mu
    )

    outputs = []

    for actual_return in np.asarray(
        block_actual_returns,
        dtype=float,
    ):

        # -------------------------------------------------------------
        # One-step-ahead conditional variance
        # -------------------------------------------------------------

        conditional_sigma2 = (
            omega
            + alpha * (
                last_residual ** 2
            )
            + beta * conditional_sigma2
        )

        conditional_sigma2 = max(
            float(conditional_sigma2),
            1e-12,
        )

        sigma_percent = float(
            np.sqrt(
                conditional_sigma2
            )
        )

        q05_percent = (
            mu
            + sigma_percent * std_t_q
        )

        # GARCH fitting uses percentage-scaled returns.
        q05_decimal = (
            q05_percent / 100.0
        )

        sigma_decimal = (
            sigma_percent / 100.0
        )

        mu_decimal = (
            mu / 100.0
        )

        actual_decimal = float(
            actual_return
        )

        breach = int(
            actual_decimal < q05_decimal
        )

        loss = pinball_loss(
            actual_decimal,
            q05_decimal,
            ALPHA,
        )

        outputs.append(
            {
                "actual_return":
                    actual_decimal,

                "garch_mu":
                    mu_decimal,

                "garch_sigma":
                    sigma_decimal,

                "garch_q05":
                    q05_decimal,

                "garch_df":
                    nu,

                "garch_breach":
                    breach,

                "garch_pinball_loss":
                    loss,
            }
        )

        # -------------------------------------------------------------
        # Update recursion using newly observed return
        # -------------------------------------------------------------

        realized_percent = (
            actual_decimal * 100.0
        )

        last_residual = (
            realized_percent - mu
        )

    return outputs


# =============================================================================
# LOAD DATA
# =============================================================================

print(
    "=" * 100,
    flush=True,
)

print(
    "POOLED GARCH(1,1)-STUDENT-t OOS PANEL",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    f"Project root: {ROOT}",
    flush=True,
)

if not QRF_PATH.exists():
    raise FileNotFoundError(
        f"QRF file not found:\n{QRF_PATH}"
    )

if not HS_PANEL_PATH.exists():
    raise FileNotFoundError(
        f"HS panel not found:\n{HS_PANEL_PATH}"
    )

qrf = pd.read_csv(
    QRF_PATH
)

hs = pd.read_csv(
    HS_PANEL_PATH
)

qrf["ticker"] = (
    qrf["ticker"]
    .astype(str)
    .str.strip()
)

hs["ticker"] = (
    hs["ticker"]
    .astype(str)
    .str.strip()
)

qrf["forecast_date"] = pd.to_datetime(
    qrf["forecast_date"]
)

if "realized_date" in qrf.columns:
    qrf["realized_date"] = pd.to_datetime(
        qrf["realized_date"],
        errors="coerce",
    )

print(
    f"\nQRF rows: {len(qrf):,}",
    flush=True,
)

print(
    f"QRF tickers: {qrf['ticker'].nunique()}",
    flush=True,
)

print(
    f"HS panel rows: {len(hs):,}",
    flush=True,
)


# =============================================================================
# IDENTIFY HS DATE / RETURN COLUMNS
# =============================================================================

hs_realized_date_col = find_column(
    hs,
    [
        "realized_date",
        "outcome_date",
        "date",
        "Date",
    ],
    "HS realized-date column",
)

hs_return_col = find_column(
    hs,
    [
        "actual_return",
        "Log_Return",
        "log_return",
        "return",
        "Return",
    ],
    "HS return column",
)

print(
    f"\nHS realized-date column: "
    f"{hs_realized_date_col}",
    flush=True,
)

print(
    f"HS return column: "
    f"{hs_return_col}",
    flush=True,
)

hs[hs_realized_date_col] = pd.to_datetime(
    hs[hs_realized_date_col],
    errors="coerce",
)

hs[hs_return_col] = pd.to_numeric(
    hs[hs_return_col],
    errors="coerce",
)

hs = hs[
    [
        "ticker",
        hs_realized_date_col,
        hs_return_col,
    ]
].copy()

hs = hs.rename(
    columns={
        hs_realized_date_col:
            "return_date",

        hs_return_col:
            "actual_return",
    }
)

hs = hs.dropna(
    subset=[
        "ticker",
        "return_date",
        "actual_return",
    ]
)

hs = (
    hs
    .sort_values(
        [
            "ticker",
            "return_date",
        ]
    )
    .drop_duplicates(
        [
            "ticker",
            "return_date",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)

qrf = (
    qrf
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# TICKER CHECK
# =============================================================================

qrf_tickers = sorted(
    qrf["ticker"].unique()
)

hs_tickers = sorted(
    hs["ticker"].unique()
)

print(
    "\nQRF tickers:",
    qrf_tickers,
    flush=True,
)

print(
    "\nHS tickers:",
    hs_tickers,
    flush=True,
)

missing_hs = sorted(
    set(qrf_tickers)
    -
    set(hs_tickers)
)

if missing_hs:
    raise ValueError(
        "Missing HS history for tickers:\n"
        f"{missing_hs}"
    )

missing_expected = sorted(
    set(EXPECTED_TICKERS)
    -
    set(qrf_tickers)
)

extra_tickers = sorted(
    set(qrf_tickers)
    -
    set(EXPECTED_TICKERS)
)

if missing_expected:
    print(
        "\nWARNING: Missing expected "
        f"QRF tickers: {missing_expected}",
        flush=True,
    )

if extra_tickers:
    print(
        "\nWARNING: Extra QRF tickers: "
        f"{extra_tickers}",
        flush=True,
    )


# =============================================================================
# GARCH PROCESSING
# =============================================================================

all_results = []
summary_rows = []

for ticker_number, ticker in enumerate(
    qrf_tickers,
    start=1,
):

    print(
        "\n" + "-" * 100,
        flush=True,
    )

    print(
        f"[{ticker_number}/{len(qrf_tickers)}] "
        f"PROCESSING {ticker}",
        flush=True,
    )

    print(
        "-" * 100,
        flush=True,
    )

    qrf_ticker = (
        qrf[
            qrf["ticker"] == ticker
        ]
        .sort_values(
            "forecast_date"
        )
        .reset_index(
            drop=True
        )
    )

    history = (
        hs[
            hs["ticker"] == ticker
        ]
        .sort_values(
            "return_date"
        )
        .reset_index(
            drop=True
        )
    )

    if len(history) < GARCH_WINDOW:
        print(
            f"{ticker}: insufficient history "
            f"({len(history)} rows)",
            flush=True,
        )
        continue

    ticker_outputs = []

    fit_count = 0
    convergence_failures = 0

    # =================================================================
    # BLOCKED REFIT LOOP
    # =================================================================

    for start in range(
        0,
        len(qrf_ticker),
        REFIT_EVERY,
    ):

        block = qrf_ticker.iloc[
            start:
            start + REFIT_EVERY
        ].copy()

        if block.empty:
            continue

        first_forecast_date = (
            block[
                "forecast_date"
            ].iloc[0]
        )

        # -------------------------------------------------------------
        # STRICT CAUSAL HISTORICAL CUTOFF
        # -------------------------------------------------------------

        available_history = history[
            history["return_date"]
            <
            first_forecast_date
        ]

        if len(available_history) < GARCH_WINDOW:
            print(
                f"{ticker}: skipping block "
                f"{start} — only "
                f"{len(available_history)} "
                f"historical observations",
                flush=True,
            )
            continue

        train = (
            available_history
            .tail(GARCH_WINDOW)[
                "actual_return"
            ]
            .to_numpy(
                dtype=float
            )
        )

        train = train[
            np.isfinite(train)
        ]

        if len(train) < GARCH_WINDOW:
            print(
                f"{ticker}: skipping block "
                f"{start} — only "
                f"{len(train)} finite "
                f"training observations",
                flush=True,
            )
            continue

        train_percent = (
            train * 100.0
        )

        # -------------------------------------------------------------
        # FIT GARCH
        # -------------------------------------------------------------

        try:

            result = fit_garch(
                train_percent
            )

            fit_count += 1

        except Exception as exc:

            convergence_failures += 1

            print(
                f"{ticker}: GARCH fit failed "
                f"at block {start}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            continue

        # -------------------------------------------------------------
        # ACTUAL RETURNS FOR FORECAST BLOCK
        # -------------------------------------------------------------

        if "actual_return" not in block.columns:
            raise ValueError(
                "QRF panel does not contain actual_return."
            )

        actual_block = (
            block[
                "actual_return"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.all(
            np.isfinite(actual_block)
        ):
            raise ValueError(
                f"{ticker}: non-finite "
                f"actual_return values "
                f"in forecast block."
            )

        # -------------------------------------------------------------
        # GENERATE RECURSIVE FORECASTS
        # -------------------------------------------------------------

        forecasts = (
            generate_block_forecasts(
                result,
                actual_block,
            )
        )

        if len(forecasts) != len(block):
            raise RuntimeError(
                f"{ticker}: forecast/output "
                f"length mismatch at block {start}."
            )

        # -------------------------------------------------------------
        # ATTACH DATES
        # -------------------------------------------------------------

        for (
            (_, qrf_row),
            forecast,
        ) in zip(
            block.iterrows(),
            forecasts,
        ):

            if (
                "realized_date"
                in qrf_row.index
            ):
                realized_date = (
                    qrf_row[
                        "realized_date"
                    ]
                )
            else:
                realized_date = pd.NaT

            ticker_outputs.append(
                {
                    "ticker":
                        ticker,

                    "forecast_date":
                        qrf_row[
                            "forecast_date"
                        ],

                    "realized_date":
                        realized_date,

                    "actual_return":
                        forecast[
                            "actual_return"
                        ],

                    "garch_mu":
                        forecast[
                            "garch_mu"
                        ],

                    "garch_sigma":
                        forecast[
                            "garch_sigma"
                        ],

                    "garch_q05":
                        forecast[
                            "garch_q05"
                        ],

                    "garch_df":
                        forecast[
                            "garch_df"
                        ],

                    "garch_breach":
                        forecast[
                            "garch_breach"
                        ],

                    "garch_pinball_loss":
                        forecast[
                            "garch_pinball_loss"
                        ],

                    "fit_number":
                        fit_count,
                }
            )

        print(
            f"  FIT {fit_count:>3} | "
            f"block={len(block):>2} | "
            f"forecast="
            f"{block['forecast_date'].iloc[0].date()} "
            f"to "
            f"{block['forecast_date'].iloc[-1].date()} | "
            f"train_n={len(train)}",
            flush=True,
        )

    # =================================================================
    # TICKER SUMMARY
    # =================================================================

    ticker_df = pd.DataFrame(
        ticker_outputs
    )

    if ticker_df.empty:
        print(
            f"{ticker}: no GARCH output.",
            flush=True,
        )
        continue

    all_results.append(
        ticker_df
    )

    clean_df = (
        ticker_df
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    n_obs = len(
        clean_df
    )

    breaches = int(
        clean_df[
            "garch_breach"
        ].sum()
    )

    breach_rate = (
        breaches / n_obs
        if n_obs
        else np.nan
    )

    mean_pinball = (
        clean_df[
            "garch_pinball_loss"
        ]
        .mean()
    )

    mean_sigma = (
        clean_df[
            "garch_sigma"
        ]
        .mean()
    )

    mean_df_value = (
        clean_df[
            "garch_df"
        ]
        .mean()
    )

    summary_rows.append(
        {
            "ticker":
                ticker,

            "n":
                n_obs,

            "breaches":
                breaches,

            "breach_rate":
                breach_rate,

            "mean_pinball_loss":
                mean_pinball,

            "mean_sigma":
                mean_sigma,

            "mean_student_t_df":
                mean_df_value,

            "fits":
                fit_count,

            "fit_failures":
                convergence_failures,
        }
    )

    print(
        f"\n{ticker} RESULTS",
        flush=True,
    )

    print(
        f"  Observations: "
        f"{n_obs}",
        flush=True,
    )

    print(
        f"  Breaches: "
        f"{breaches}",
        flush=True,
    )

    print(
        f"  Breach rate: "
        f"{breach_rate:.6f}",
        flush=True,
    )

    print(
        f"  Mean pinball loss: "
        f"{mean_pinball:.8f}",
        flush=True,
    )

    print(
        f"  Mean sigma: "
        f"{mean_sigma:.6f}",
        flush=True,
    )

    print(
        f"  Mean Student-t df: "
        f"{mean_df_value:.4f}",
        flush=True,
    )

    print(
        f"  Fits: "
        f"{fit_count}",
        flush=True,
    )

    print(
        f"  Fit failures: "
        f"{convergence_failures}",
        flush=True,
    )


# =============================================================================
# COMBINE
# =============================================================================

if not all_results:
    raise RuntimeError(
        "No GARCH predictions were generated."
    )

predictions = pd.concat(
    all_results,
    ignore_index=True,
)

predictions = (
    predictions
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(
        drop=True
    )
)

summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# ALIGNMENT CHECK
# =============================================================================

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "ALIGNMENT CHECKS",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

qrf_keys = (
    qrf[
        [
            "ticker",
            "forecast_date",
        ]
    ]
    .drop_duplicates()
)

garch_keys = (
    predictions[
        [
            "ticker",
            "forecast_date",
        ]
    ]
    .drop_duplicates()
)

merged_keys = qrf_keys.merge(
    garch_keys,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="outer",
    indicator=True,
)

missing_predictions = (
    merged_keys[
        merged_keys[
            "_merge"
        ]
        ==
        "left_only"
    ]
)

extra_predictions = (
    merged_keys[
        merged_keys[
            "_merge"
        ]
        ==
        "right_only"
    ]
)

duplicate_keys = int(
    predictions.duplicated(
        [
            "ticker",
            "forecast_date",
        ]
    ).sum()
)

print(
    f"QRF expected rows: "
    f"{len(qrf_keys):,}",
    flush=True,
)

print(
    f"GARCH generated rows: "
    f"{len(garch_keys):,}",
    flush=True,
)

print(
    f"Missing QRF forecast rows: "
    f"{len(missing_predictions):,}",
    flush=True,
)

print(
    f"Extra GARCH rows: "
    f"{len(extra_predictions):,}",
    flush=True,
)

print(
    f"Duplicate ticker-date rows: "
    f"{duplicate_keys}",
    flush=True,
)


# =============================================================================
# ACTUAL-RETURN ALIGNMENT CHECK
# =============================================================================

qrf_actual_check = (
    qrf[
        [
            "ticker",
            "forecast_date",
            "actual_return",
        ]
    ]
    .rename(
        columns={
            "actual_return":
                "qrf_actual_return"
        }
    )
)

actual_merged = predictions.merge(
    qrf_actual_check,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="left",
)

actual_diff = (
    actual_merged[
        "actual_return"
    ]
    -
    actual_merged[
        "qrf_actual_return"
    ]
).abs()

max_actual_diff = (
    float(actual_diff.max())
    if len(actual_diff)
    else np.nan
)

print(
    "Maximum actual-return "
    f"mismatch vs QRF: "
    f"{max_actual_diff:.12g}",
    flush=True,
)


# =============================================================================
# ROW COUNT SANITY
# =============================================================================

rows_per_ticker = (
    predictions
    .groupby(
        "ticker"
    )
    .size()
    .sort_index()
)

print(
    "\nRows per ticker:",
    flush=True,
)

print(
    rows_per_ticker.to_string(),
    flush=True,
)


# =============================================================================
# POOLED SUMMARY
# =============================================================================

pooled_breach_rate = (
    predictions[
        "garch_breach"
    ].mean()
)

pooled_pinball = (
    predictions[
        "garch_pinball_loss"
    ].mean()
)

pooled_sigma = (
    predictions[
        "garch_sigma"
    ].mean()
)

pooled_df = (
    predictions[
        "garch_df"
    ].mean()
)

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "POOLED GARCH SUMMARY",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    f"Total rows: "
    f"{len(predictions):,}",
    flush=True,
)

print(
    f"Tickers: "
    f"{predictions['ticker'].nunique()}",
    flush=True,
)

print(
    f"Overall breach rate: "
    f"{pooled_breach_rate:.6f}",
    flush=True,
)

print(
    f"Mean pinball loss: "
    f"{pooled_pinball:.8f}",
    flush=True,
)

print(
    f"Mean sigma: "
    f"{pooled_sigma:.6f}",
    flush=True,
)

print(
    f"Mean Student-t df: "
    f"{pooled_df:.4f}",
    flush=True,
)


# =============================================================================
# AAPL CROSS-CHECK
# =============================================================================

aapl = (
    predictions[
        predictions[
            "ticker"
        ]
        ==
        "AAPL"
    ]
    .sort_values(
        "forecast_date"
    )
)

if not aapl.empty:

    print(
        "\n" + "=" * 100,
        flush=True,
    )

    print(
        "AAPL LAST 10 GARCH FORECASTS",
        flush=True,
    )

    print(
        "=" * 100,
        flush=True,
    )

    print(
        aapl[
            [
                "forecast_date",
                "realized_date",
                "actual_return",
                "garch_sigma",
                "garch_q05",
                "garch_df",
                "garch_breach",
                "fit_number",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        ),
        flush=True,
    )


# =============================================================================
# SAVE
# =============================================================================

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

predictions.to_csv(
    OUTPUT_PATH,
    index=False,
)

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "SAVED",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)

print(
    OUTPUT_PATH,
    flush=True,
)

print(
    SUMMARY_PATH,
    flush=True,
)


# =============================================================================
# FINAL CHECK
# =============================================================================

if len(missing_predictions) > 0:

    print(
        "\nWARNING: Some QRF forecast dates "
        "do not have GARCH predictions.",
        flush=True,
    )

if len(extra_predictions) > 0:

    print(
        "\nWARNING: GARCH generated extra "
        "forecast dates not present in QRF.",
        flush=True,
    )

if (
    not np.isnan(max_actual_diff)
    and max_actual_diff > 1e-12
):

    raise RuntimeError(
        "Actual-return alignment failed. "
        f"Maximum mismatch = {max_actual_diff}"
    )

if duplicate_keys > 0:

    raise RuntimeError(
        "Duplicate GARCH ticker-date rows detected."
    )

print(
    "\n" + "=" * 100,
    flush=True,
)

print(
    "25-TICKER GARCH OOS BUILD COMPLETE",
    flush=True,
)

print(
    "=" * 100,
    flush=True,
)