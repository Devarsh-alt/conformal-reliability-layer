from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

OUTPUT_DIR = ROOT / "data" / "monitoring"

OUTPUT = (
    OUTPUT_DIR
    / "pooled_hs_reliability_panel.csv"
)

ALPHA = 0.05

HS_WINDOW = 250

VOL_WINDOWS = [20, 60, 120]

KS_CURRENT_WINDOW = 20
KS_REFERENCE_WINDOW = 120

BOOTSTRAPS = 20

RNG_SEED = 42

FORWARD_HORIZONS = [5, 10, 20]


# ============================================================
# Helper: pinball loss
# ============================================================

def pinball_loss(y, q, alpha):
    """
    Lower-tail quantile loss.

    y = realized return
    q = predicted VaR quantile
    """

    error = y - q

    return np.where(
        error < 0,
        (1 - alpha) * (-error),
        alpha * error
    )


# ============================================================
# Helper: EWMA breach rate
# ============================================================

def compute_ewma(values, decay=0.94):
    """
    EWMA of the breach indicator.
    """

    values = np.asarray(values, dtype=float)

    result = np.full(
        len(values),
        np.nan
    )

    if len(values) == 0:
        return result

    result[0] = values[0]

    for i in range(1, len(values)):

        result[i] = (
            decay * result[i - 1]
            + (1 - decay) * values[i]
        )

    return result


# ============================================================
# Helper: CUSUM on violation process
# ============================================================

def compute_cusum(breaches, alpha):
    """
    One-sided CUSUM of excess violations.

    S_t = max(0, S_{t-1} + I_t - alpha)

    High values mean sustained excess violation activity.
    """

    breaches = np.asarray(
        breaches,
        dtype=float
    )

    result = np.zeros(
        len(breaches)
    )

    for i in range(1, len(breaches)):

        result[i] = max(
            0.0,
            result[i - 1]
            + breaches[i]
            - alpha
        )

    return result


# ============================================================
# Helper: rolling historical VaR
# ============================================================

def compute_hs_var(returns, window, alpha):

    result = np.full(
        len(returns),
        np.nan
    )

    values = np.asarray(
        returns,
        dtype=float
    )

    for i in range(window, len(values)):

        history = values[
            i - window:i
        ]

        result[i] = np.quantile(
            history,
            alpha
        )

    return result


# ============================================================
# Helper: bootstrap HS VaR disagreement
# ============================================================

def compute_bootstrap_disagreement(
    returns,
    window,
    alpha,
    n_bootstrap,
    rng,
):
    """
    At each date:

    1. Take the historical HS estimation window.
    2. Bootstrap-resample that window.
    3. Recalculate the alpha quantile.
    4. Measure the standard deviation of the
       bootstrap VaR estimates.

    This is directly tied to the HS VaR model.
    """

    values = np.asarray(
        returns,
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan
    )

    for i in range(window, len(values)):

        history = values[
            i - window:i
        ]

        # Draw bootstrap samples.
        indices = rng.integers(
            low=0,
            high=window,
            size=(
                n_bootstrap,
                window
            )
        )

        samples = history[
            indices
        ]

        bootstrap_vars = np.quantile(
            samples,
            alpha,
            axis=1
        )

        result[i] = np.std(
            bootstrap_vars,
            ddof=1
        )

    return result


# ============================================================
# Helper: KS drift
# ============================================================

def compute_ks_drift(
    returns,
    current_window,
    reference_window,
):
    """
    Compare:

        most recent current_window returns

    against

        immediately preceding reference_window returns.

    Both windows are strictly available at the monitoring date.
    """

    values = np.asarray(
        returns,
        dtype=float
    )

    minimum_required = (
        current_window
        + reference_window
    )

    result = np.full(
        len(values),
        np.nan
    )

    for i in range(
        minimum_required,
        len(values)
    ):

        current = values[
            i - current_window:i
        ]

        reference = values[
            i - current_window
            - reference_window:
            i - current_window
        ]

        result[i] = ks_2samp(
            current,
            reference,
            alternative="two-sided",
            mode="auto"
        ).statistic

    return result


# ============================================================
# Discover tickers
# ============================================================

files = sorted(
    PROCESSED_DIR.glob("*.csv")
)

tickers = [
    f.stem
    for f in files
]

print("=" * 80)
print("POOLED HS RELIABILITY PANEL")
print("=" * 80)

print(
    "Number of return series:",
    len(tickers)
)

print(
    "Tickers:",
    tickers
)


# ============================================================
# Process every ticker
# ============================================================

all_panels = []


for ticker_index, ticker in enumerate(tickers):

    print("\n" + "-" * 80)
    print(
        f"[{ticker_index + 1}/{len(tickers)}] "
        f"Processing {ticker}"
    )
    print("-" * 80)

    path = (
        PROCESSED_DIR
        / f"{ticker}.csv"
    )

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = (
        df.sort_values("Date")
        .reset_index(drop=True)
    )

    dates = df["Date"]
    returns = (
        df["Log_Return"]
        .astype(float)
        .to_numpy()
    )

    n = len(df)

    # --------------------------------------------------------
    # HS VaR
    # --------------------------------------------------------

    hs_var = compute_hs_var(
        returns,
        HS_WINDOW,
        ALPHA
    )

    # --------------------------------------------------------
    # Current breaches
    # --------------------------------------------------------

    breach = np.where(
        np.isnan(hs_var),
        np.nan,
        (returns < hs_var).astype(float)
    )

    # --------------------------------------------------------
    # EWMA breach activity
    # --------------------------------------------------------

    valid_breach = np.nan_to_num(
        breach,
        nan=0.0
    )

    ewma_breach = compute_ewma(
        valid_breach,
        decay=0.94
    )

    # --------------------------------------------------------
    # CUSUM
    # --------------------------------------------------------

    cusum = compute_cusum(
        valid_breach,
        ALPHA
    )

    # --------------------------------------------------------
    # Realized volatility
    # --------------------------------------------------------

    vol_20 = (
        pd.Series(returns)
        .rolling(20)
        .std()
        .to_numpy()
    )

    vol_60 = (
        pd.Series(returns)
        .rolling(60)
        .std()
        .to_numpy()
    )

    vol_120 = (
        pd.Series(returns)
        .rolling(120)
        .std()
        .to_numpy()
    )

    # --------------------------------------------------------
    # Rolling breach rates
    # --------------------------------------------------------

    breach_series = pd.Series(
        valid_breach
    )

    breach_rate_20 = (
        breach_series
        .rolling(20)
        .mean()
        .to_numpy()
    )

    breach_rate_60 = (
        breach_series
        .rolling(60)
        .mean()
        .to_numpy()
    )

    breach_rate_120 = (
        breach_series
        .rolling(120)
        .mean()
        .to_numpy()
    )

    # --------------------------------------------------------
    # Bootstrap VaR disagreement
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RNG_SEED
        + ticker_index * 1000
    )

    bootstrap_disagreement = (
        compute_bootstrap_disagreement(
            returns,
            HS_WINDOW,
            ALPHA,
            BOOTSTRAPS,
            rng
        )
    )

    # --------------------------------------------------------
    # Normalize bootstrap disagreement
    # --------------------------------------------------------

    bootstrap_disagreement_pct = (
        bootstrap_disagreement
        / np.maximum(
            np.abs(hs_var),
            1e-8
        )
    )

    # --------------------------------------------------------
    # KS distribution drift
    # --------------------------------------------------------

    ks_stat = compute_ks_drift(
        returns,
        KS_CURRENT_WINDOW,
        KS_REFERENCE_WINDOW
    )

    # --------------------------------------------------------
    # Current VaR severity
    # --------------------------------------------------------

    var_abs = np.abs(
        hs_var
    )

    # --------------------------------------------------------
    # VaR movement
    # --------------------------------------------------------

    var_change = np.full(
        n,
        np.nan
    )

    valid_var = ~np.isnan(
        hs_var
    )

    valid_indices = np.where(
        valid_var
    )[0]

    if len(valid_indices) > 1:

        for j in range(
            1,
            len(valid_indices)
        ):

            current_index = (
                valid_indices[j]
            )

            previous_index = (
                valid_indices[j - 1]
            )

            var_change[
                current_index
            ] = abs(
                hs_var[current_index]
                - hs_var[previous_index]
            )

    # --------------------------------------------------------
    # Current breach severity
    # --------------------------------------------------------

    breach_severity = np.maximum(
        hs_var - returns,
        0
    )

    breach_severity[
        np.isnan(hs_var)
    ] = np.nan

    # --------------------------------------------------------
    # Pinball loss
    # --------------------------------------------------------

    current_pinball = np.full(
        n,
        np.nan
    )

    valid = ~np.isnan(
        hs_var
    )

    current_pinball[
        valid
    ] = pinball_loss(
        returns[valid],
        hs_var[valid],
        ALPHA
    )

    # --------------------------------------------------------
    # Build dataframe
    # --------------------------------------------------------

    panel = pd.DataFrame({
        "date": dates,
        "ticker": ticker,

        "return": returns,

        "HS_VaR": hs_var,
        "HS_breach": breach,

        "var_abs": var_abs,
        "var_change": var_change,

        "breach_severity": breach_severity,
        "pinball_loss": current_pinball,

        # Persistence
        "breach_rate_20d":
            breach_rate_20,
        "breach_rate_60d":
            breach_rate_60,
        "breach_rate_120d":
            breach_rate_120,

        # EWMA / CUSUM
        "ewma_breach_rate":
            ewma_breach,
        "cusum_breach":
            cusum,

        # Volatility
        "realized_vol_20d":
            vol_20,
        "realized_vol_60d":
            vol_60,
        "realized_vol_120d":
            vol_120,

        # Direct reliability signals
        "ks_stat":
            ks_stat,

        "bootstrap_var_disagreement":
            bootstrap_disagreement,

        "bootstrap_var_disagreement_pct":
            bootstrap_disagreement_pct,
    })

    # ========================================================
    # Future targets
    # ========================================================

    for horizon in FORWARD_HORIZONS:

        future_breach_rate = np.full(
            n,
            np.nan
        )

        future_pinball = np.full(
            n,
            np.nan
        )

        future_severity = np.full(
            n,
            np.nan
        )

        future_normalized_severity = np.full(
            n,
            np.nan
        )

        for i in range(n):

            start = i + 1
            end = i + 1 + horizon

            if end > n:
                continue

            future_breaches = (
                breach[start:end]
            )

            future_losses = (
                current_pinball[
                    start:end
                ]
            )

            future_severities = (
                breach_severity[
                    start:end
                ]
            )

            future_vars = np.abs(
                hs_var[start:end]
            )

            if (
                np.isnan(
                    future_breaches
                ).any()
                or np.isnan(
                    future_losses
                ).any()
                or np.isnan(
                    future_severities
                ).any()
                or np.isnan(
                    future_vars
                ).any()
            ):
                continue

            future_breach_rate[i] = (
                np.mean(
                    future_breaches
                )
            )

            future_pinball[i] = (
                np.mean(
                    future_losses
                )
            )

            future_severity[i] = (
                np.mean(
                    future_severities
                )
            )

            future_normalized_severity[i] = (
                np.mean(
                    future_severities
                    / np.maximum(
                        future_vars,
                        1e-8
                    )
                )
            )

        panel[
            f"future_breach_rate_{horizon}d"
        ] = future_breach_rate

        panel[
            f"future_pinball_{horizon}d"
        ] = future_pinball

        panel[
            f"future_severity_{horizon}d"
        ] = future_severity

        panel[
            f"future_normalized_severity_{horizon}d"
        ] = future_normalized_severity

    # --------------------------------------------------------
    # Drop incomplete rows
    # --------------------------------------------------------

    panel = panel.dropna().reset_index(
        drop=True
    )

    print(
        "Usable rows:",
        len(panel)
    )

    print(
        "Date range:",
        panel["date"].min().date(),
        "to",
        panel["date"].max().date()
    )

    print(
        "HS breach rate:",
        panel["HS_breach"].mean()
    )

    all_panels.append(
        panel
    )


# ============================================================
# Pool all tickers
# ============================================================

pooled = pd.concat(
    all_panels,
    ignore_index=True
)

pooled = (
    pooled
    .sort_values(
        ["ticker", "date"]
    )
    .reset_index(drop=True)
)


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

pooled.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 80)
print("POOLED PANEL SUMMARY")
print("=" * 80)

print(
    "Rows:",
    len(pooled)
)

print(
    "Tickers:",
    pooled["ticker"].nunique()
)

print(
    "Ticker list:",
    sorted(
        pooled["ticker"].unique()
    )
)

print(
    "Date range:",
    pooled["date"].min().date(),
    "to",
    pooled["date"].max().date()
)


# ============================================================
# Per-ticker statistics
# ============================================================

ticker_summary = (
    pooled
    .groupby("ticker")
    .agg(
        n=("return", "size"),
        breach_rate=(
            "HS_breach",
            "mean"
        ),
        mean_var=(
            "var_abs",
            "mean"
        ),
        mean_bootstrap_disagreement=(
            "bootstrap_var_disagreement_pct",
            "mean"
        ),
        mean_ks=(
            "ks_stat",
            "mean"
        ),
        mean_vol_20=(
            "realized_vol_20d",
            "mean"
        ),
    )
    .sort_values(
        "breach_rate"
    )
)

print("\n" + "=" * 80)
print("PER-TICKER SUMMARY")
print("=" * 80)

print(
    ticker_summary.to_string()
)


# ============================================================
# Pooled signal correlations
# ============================================================

signals = [
    "breach_rate_20d",
    "breach_rate_60d",
    "breach_rate_120d",
    "ewma_breach_rate",
    "cusum_breach",
    "realized_vol_20d",
    "realized_vol_60d",
    "realized_vol_120d",
    "ks_stat",
    "bootstrap_var_disagreement_pct",
    "var_abs",
    "var_change",
]

outcomes = []

for horizon in FORWARD_HORIZONS:

    outcomes.extend([
        f"future_breach_rate_{horizon}d",
        f"future_pinball_{horizon}d",
        f"future_normalized_severity_{horizon}d",
    ])


correlation_rows = []

for signal in signals:

    for outcome in outcomes:

        pair = pooled[
            [signal, outcome]
        ].dropna()

        if len(pair) < 20:
            continue

        corr = pair.corr().iloc[
            0,
            1
        ]

        correlation_rows.append({
            "signal": signal,
            "outcome": outcome,
            "pearson_corr": corr,
            "n": len(pair),
        })


correlations = pd.DataFrame(
    correlation_rows
)

print("\n" + "=" * 80)
print("TOP SIGNAL → FUTURE OUTCOME CORRELATIONS")
print("=" * 80)

print(
    correlations
    .sort_values(
        "pearson_corr",
        ascending=False
    )
    .head(40)
    .to_string(index=False)
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    "Saved to:",
    OUTPUT
)