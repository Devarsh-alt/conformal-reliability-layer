import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from quantile_forest import RandomForestQuantileRegressor


# ============================================================
# CONFIG
# ============================================================

TICKER = "AAPL"
ALPHA = 0.05

# First sanity-check specification.
# We will tune these later.
TRAIN_WINDOW = 1000
REFIT_EVERY = 20

N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 10
MAX_FEATURES = "sqrt"
RANDOM_STATE = 42

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    f"{TICKER}.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "monitoring"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    f"{TICKER}_QRF_5pct_baseline.csv"
)


# ============================================================
# METRICS
# ============================================================

def pinball_loss(y, q, alpha=0.05):
    """
    Quantile / pinball loss.

    q = predicted alpha quantile
    y = realized return
    """
    error = y - q
    return np.where(
        error >= 0,
        alpha * error,
        (alpha - 1.0) * error
    )


def kupiec_uc(breaches, alpha=0.05):
    """
    Kupiec unconditional coverage test.

    Null hypothesis:
        observed breach probability = alpha
    """

    breaches = np.asarray(breaches, dtype=int)
    n = len(breaches)
    x = breaches.sum()

    if n == 0 or x == 0 or x == n:
        return np.nan, np.nan

    p_hat = x / n

    log_l_null = (
        (n - x) * np.log(1 - alpha)
        + x * np.log(alpha)
    )

    log_l_alt = (
        (n - x) * np.log(1 - p_hat)
        + x * np.log(p_hat)
    )

    lr = -2.0 * (log_l_null - log_l_alt)
    p_value = 1.0 - chi2.cdf(lr, df=1)

    return lr, p_value


def christoffersen_independence(breaches):
    """
    Christoffersen independence test.

    Tests whether breaches are independent over time.
    """

    breaches = np.asarray(breaches, dtype=int)

    if len(breaches) < 2:
        return np.nan, np.nan

    n00 = n01 = n10 = n11 = 0

    for prev, curr in zip(breaches[:-1], breaches[1:]):
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

    total = n0 + n1

    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    pi0 = n01 / n0
    pi1 = n11 / n1
    pi = (n01 + n11) / total

    eps = 1e-12

    pi0 = np.clip(pi0, eps, 1 - eps)
    pi1 = np.clip(pi1, eps, 1 - eps)
    pi = np.clip(pi, eps, 1 - eps)

    log_l_ind = (
        n00 * np.log(1 - pi)
        + n01 * np.log(pi)
        + n10 * np.log(1 - pi)
        + n11 * np.log(pi)
    )

    log_l_dep = (
        n00 * np.log(1 - pi0)
        + n01 * np.log(pi0)
        + n10 * np.log(1 - pi1)
        + n11 * np.log(pi1)
    )

    lr = -2.0 * (log_l_ind - log_l_dep)
    p_value = 1.0 - chi2.cdf(lr, df=1)

    return lr, p_value


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("QUANTILE RANDOM FOREST — AAPL DIRECT 5% VaR")
print("=" * 80)

print(f"Data path: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print(f"Raw rows: {len(df)}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")


# ============================================================
# FEATURES
# ============================================================

# These are already available in our processed dataset.
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

missing_features = [c for c in FEATURES if c not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

# Next-day return is the target.
df["target"] = df["Log_Return"].shift(-1)

model_df = df[
    ["Date"] + FEATURES + ["target"]
].copy()

model_df = model_df.replace([np.inf, -np.inf], np.nan)
model_df = model_df.dropna().reset_index(drop=True)

print(f"Usable rows after feature/target construction: {len(model_df)}")
print()


# ============================================================
# WALK-FORWARD QRF
# ============================================================

predictions = []

first_test_idx = TRAIN_WINDOW

print("=" * 80)
print("WALK-FORWARD FORECASTING")
print("=" * 80)

fit_number = 0

for start_idx in range(
    first_test_idx,
    len(model_df),
    REFIT_EVERY
):

    end_idx = min(
        start_idx + REFIT_EVERY,
        len(model_df)
    )

    train_start = start_idx - TRAIN_WINDOW
    train_end = start_idx

    train_df = model_df.iloc[train_start:train_end]
    test_df = model_df.iloc[start_idx:end_idx]

    X_train = train_df[FEATURES]
    y_train = train_df["target"]

    X_test = test_df[FEATURES]
    y_test = test_df["target"]

    model = RandomForestQuantileRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    q_pred = model.predict(
        X_test,
        quantiles=[ALPHA]
    )

    # quantile_forest returns one column for one quantile
    q_pred = np.asarray(q_pred).reshape(-1)

    block = pd.DataFrame({
        "Date": test_df["Date"].values,
        "actual_return": y_test.values,
        "q05_forecast": q_pred,
    })

    block["VaR_5pct"] = -block["q05_forecast"]

    block["breach"] = (
        block["actual_return"] < block["q05_forecast"]
    ).astype(int)

    block["pinball_loss"] = pinball_loss(
        block["actual_return"].values,
        block["q05_forecast"].values,
        ALPHA
    )

    block["fit_number"] = fit_number

    predictions.append(block)

    fit_number += 1

    print(
        f"Fit {fit_number:3d} | "
        f"train={train_df['Date'].iloc[0].date()} "
        f"to {train_df['Date'].iloc[-1].date()} | "
        f"test={test_df['Date'].iloc[0].date()} "
        f"to {test_df['Date'].iloc[-1].date()} | "
        f"n_test={len(test_df)}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

pred = pd.concat(
    predictions,
    ignore_index=True
)

pred = pred.sort_values("Date").reset_index(drop=True)


# ============================================================
# SUMMARY
# ============================================================

breach_rate = pred["breach"].mean()
mean_pinball = pred["pinball_loss"].mean()

kupiec_lr, kupiec_p = kupiec_uc(
    pred["breach"].values,
    ALPHA
)

ind_lr, ind_p = christoffersen_independence(
    pred["breach"].values
)

max_consecutive = 0
current = 0

for b in pred["breach"]:
    if b == 1:
        current += 1
        max_consecutive = max(max_consecutive, current)
    else:
        current = 0


print()
print("=" * 80)
print("QRF OUT-OF-SAMPLE RESULTS")
print("=" * 80)

print(f"Test observations:       {len(pred)}")
print(
    f"Test date range:         "
    f"{pred['Date'].min().date()} "
    f"to {pred['Date'].max().date()}"
)

print(f"Observed breach rate:    {breach_rate:.6f}")
print(f"Target breach rate:      {ALPHA:.6f}")
print(f"Mean pinball loss:       {mean_pinball:.8f}")

print()
print("Kupiec unconditional coverage:")
print(f"  LR statistic:           {kupiec_lr:.6f}")
print(f"  p-value:                {kupiec_p:.6f}")

print()
print("Christoffersen independence:")
print(f"  LR statistic:           {ind_lr:.6f}")
print(f"  p-value:                {ind_p:.6f}")

print()
print(f"Maximum consecutive breaches: {max_consecutive}")

print()
print("First 10 forecasts:")
print(
    pred[
        [
            "Date",
            "actual_return",
            "q05_forecast",
            "VaR_5pct",
            "breach",
            "pinball_loss",
        ]
    ].head(10).to_string(index=False)
)

print()
print("Last 10 forecasts:")
print(
    pred[
        [
            "Date",
            "actual_return",
            "q05_forecast",
            "VaR_5pct",
            "breach",
            "pinball_loss",
        ]
    ].tail(10).to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

pred.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)
print(OUTPUT_PATH)
print()