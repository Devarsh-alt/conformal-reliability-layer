from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from arch import arch_model


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TICKER = "AAPL"
ALPHA = 0.05
WINDOW = 250
LAMBDA = 0.94

INPUT = ROOT / "data" / "processed" / f"{TICKER}.csv"
OUTPUT_DIR = ROOT / "data" / "var"
OUTPUT = OUTPUT_DIR / f"{TICKER}_var_5pct.csv"


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])

df = (
    df.sort_values("Date")
    .reset_index(drop=True)
)

returns = df["Log_Return"].astype(float)


print("=" * 80)
print(f"VaR MODEL CONSTRUCTION — {TICKER}")
print("=" * 80)
print(f"Observations: {len(df)}")
print(f"Window: {WINDOW}")
print(f"VaR level: {ALPHA:.0%}")


# ============================================================
# Containers
# ============================================================

dates = []
realized_returns = []

hs_var = []
ewma_var = []
garch_var = []


# ============================================================
# Rolling VaR estimation
# ============================================================

for i in range(WINDOW, len(df)):

    # --------------------------------------------------------
    # Only use information available BEFORE day i
    # --------------------------------------------------------

    history = returns.iloc[i - WINDOW:i].dropna()

    current_return = returns.iloc[i]
    current_date = df["Date"].iloc[i]

    # ========================================================
    # 1. Historical Simulation VaR
    # ========================================================

    # Empirical 5% lower-tail quantile.
    hs = history.quantile(ALPHA)

    # ========================================================
    # 2. EWMA VaR
    # ========================================================

    squared_returns = history.to_numpy() ** 2

    # Initialize variance using the oldest observation.
    ewma_variance = squared_returns[0]

    for r2 in squared_returns[1:]:
        ewma_variance = (
            LAMBDA * ewma_variance
            + (1 - LAMBDA) * r2
        )

    ewma_sigma = np.sqrt(ewma_variance)

    # Lower 5% normal quantile.
    # norm.ppf(0.05) ≈ -1.645
    #
    # Therefore VaR is a NEGATIVE return threshold.
    ewma = norm.ppf(ALPHA) * ewma_sigma

    # ========================================================
    # 3. GARCH(1,1) VaR
    # ========================================================

    # Scale returns by 100 for numerical stability.
    scaled_history = history * 100

    try:

        model = arch_model(
            scaled_history,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="t",
            rescale=False,
        )

        fitted = model.fit(
            disp="off",
            show_warning=False,
        )

        forecast = fitted.forecast(horizon=1)

        mean_forecast = forecast.mean.iloc[-1, 0]
        variance_forecast = forecast.variance.iloc[-1, 0]

        sigma_forecast = np.sqrt(variance_forecast)

        # Degrees of freedom of Student-t distribution.
        nu = fitted.params.get("nu", 8.0)

        # Lower-tail Student-t quantile.
        q = t.ppf(ALPHA, df=nu)

        # Convert back from percentage units.
        garch = (
            mean_forecast
            + sigma_forecast * q
        ) / 100.0

    except Exception:

        garch = np.nan

    # ========================================================
    # Store
    # ========================================================

    dates.append(current_date)
    realized_returns.append(current_return)

    hs_var.append(hs)
    ewma_var.append(ewma)
    garch_var.append(garch)


# ============================================================
# Create output dataframe
# ============================================================

result = pd.DataFrame({
    "date": dates,
    "return": realized_returns,
    "HS_VaR": hs_var,
    "EWMA_VaR": ewma_var,
    "GARCH_VaR": garch_var,
})


# ============================================================
# Remove rows where GARCH failed
# ============================================================

# We don't want failed GARCH estimations to silently
# contaminate the evaluation.
result = result.dropna(
    subset=["HS_VaR", "EWMA_VaR", "GARCH_VaR"]
).reset_index(drop=True)


# ============================================================
# Breach indicators
# ============================================================

# VaR represents the lower-tail return threshold.
#
# Example:
#
# VaR = -0.03
# Return = -0.04
#
# Since -0.04 < -0.03, the loss breached VaR.

result["HS_breach"] = (
    result["return"] < result["HS_VaR"]
).astype(int)

result["EWMA_breach"] = (
    result["return"] < result["EWMA_VaR"]
).astype(int)

result["GARCH_breach"] = (
    result["return"] < result["GARCH_VaR"]
).astype(int)


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

result.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

print("Output:", OUTPUT)
print("Rows:", len(result))

print("\nMissing VaR values:")
print(
    result[
        ["HS_VaR", "EWMA_VaR", "GARCH_VaR"]
    ].isna().sum()
)

print("\nBreach counts:")
print(
    result[
        ["HS_breach", "EWMA_breach", "GARCH_breach"]
    ].sum()
)

print("\nBreach rates:")
print(
    result[
        ["HS_breach", "EWMA_breach", "GARCH_breach"]
    ].mean()
)

print("\nAverage VaR:")
print(
    result[
        ["HS_VaR", "EWMA_VaR", "GARCH_VaR"]
    ].mean()
)

print("\nLast 10 observations:")
print(
    result.tail(10).to_string(index=False)
)