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

# ------------------------------------------------------------
# FROZEN FROM VALIDATION
# ------------------------------------------------------------

TRAIN_WINDOW = 1000
REFIT_EVERY = 20

N_ESTIMATORS = 500
MIN_SAMPLES_LEAF = 20
MAX_FEATURES = 1.0

RANDOM_STATE = 42

# ------------------------------------------------------------
# FINAL TEST PERIOD
# ------------------------------------------------------------

TEST_START_YEAR = 2023

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    f"{TICKER}.csv"
)

TRADITIONAL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "var",
    f"{TICKER}_var_5pct.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "monitoring",
    f"{TICKER}_QRF_final_oos.csv"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "data",
    "monitoring",
    f"{TICKER}_final_oos_model_comparison.csv"
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# PINBALL LOSS
# ============================================================

def pinball_loss(y, q, alpha=0.05):

    error = y - q

    return np.where(
        error >= 0,
        alpha * error,
        (alpha - 1.0) * error
    )


# ============================================================
# KUPIEC TEST
# ============================================================

def kupiec_uc(breaches, alpha=0.05):

    breaches = np.asarray(
        breaches,
        dtype=int
    )

    n = len(breaches)
    x = int(breaches.sum())

    if n == 0:
        return np.nan, np.nan

    if x == 0 or x == n:
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

    lr = -2.0 * (
        log_l_null - log_l_alt
    )

    p_value = 1.0 - chi2.cdf(
        lr,
        df=1
    )

    return lr, p_value


# ============================================================
# CHRISTOFFERSEN INDEPENDENCE
# ============================================================

def christoffersen_independence(breaches):

    breaches = np.asarray(
        breaches,
        dtype=int
    )

    if len(breaches) < 2:
        return np.nan, np.nan

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for prev, curr in zip(
        breaches[:-1],
        breaches[1:]
    ):

        if prev == 0 and curr == 0:
            n00 += 1

        elif prev == 0 and curr == 1:
            n01 += 1

        elif prev == 1 and curr == 0:
            n10 += 1

        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    total = n0 + n1

    pi0 = n01 / n0
    pi1 = n11 / n1
    pi = (n01 + n11) / total

    eps = 1e-12

    pi0 = np.clip(
        pi0,
        eps,
        1 - eps
    )

    pi1 = np.clip(
        pi1,
        eps,
        1 - eps
    )

    pi = np.clip(
        pi,
        eps,
        1 - eps
    )

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

    lr = -2.0 * (
        log_l_ind - log_l_dep
    )

    p_value = 1.0 - chi2.cdf(
        lr,
        df=1
    )

    return lr, p_value


# ============================================================
# MAX CONSECUTIVE BREACHES
# ============================================================

def max_consecutive_breaches(breaches):

    best = 0
    current = 0

    for breach in breaches:

        if breach == 1:

            current += 1

            best = max(
                best,
                current
            )

        else:

            current = 0

    return best


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("FINAL OUT-OF-SAMPLE QRF — AAPL")
print("=" * 80)

print()
print("Frozen configuration:")
print(f"  Train window:       {TRAIN_WINDOW}")
print(f"  Min leaf:           {MIN_SAMPLES_LEAF}")
print(f"  Max features:       {MAX_FEATURES}")
print(f"  Trees:              {N_ESTIMATORS}")
print(f"  Refit frequency:    {REFIT_EVERY}")
print(f"  Test start year:    {TEST_START_YEAR}")


df = pd.read_csv(
    DATA_PATH
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

# One-step-ahead target:
# X_t -> R_(t+1)

df["target"] = (
    df["Log_Return"]
    .shift(-1)
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    c
    for c in FEATURES
    if c not in df.columns
]

if missing_features:

    raise ValueError(
        f"Missing features: {missing_features}"
    )


# ============================================================
# BUILD MODEL DATASET
# ============================================================

model_df = df[
    ["Date"] + FEATURES + ["target"]
].copy()

model_df = model_df.replace(
    [np.inf, -np.inf],
    np.nan
)

model_df = (
    model_df
    .dropna()
    .sort_values("Date")
    .reset_index(drop=True)
)


print()
print(
    f"Usable rows: {len(model_df)}"
)


# ============================================================
# FINAL TEST MASK
# ============================================================

test_mask = (
    model_df["Date"].dt.year
    >= TEST_START_YEAR
)

test_indices = np.where(
    test_mask.values
)[0]

if len(test_indices) == 0:

    raise ValueError(
        "No final test observations found."
    )


test_start_idx = test_indices[0]

test_end_idx = (
    test_indices[-1] + 1
)


print()
print("=" * 80)
print("FINAL TEST PERIOD")
print("=" * 80)

print(
    f"Dates: "
    f"{model_df.iloc[test_start_idx]['Date'].date()} "
    f"to "
    f"{model_df.iloc[test_end_idx - 1]['Date'].date()}"
)

print(
    f"Rows: "
    f"{test_end_idx - test_start_idx}"
)


# ============================================================
# WALK-FORWARD FINAL TEST
# ============================================================

predictions = []

fit_number = 0

for block_start in range(
    test_start_idx,
    test_end_idx,
    REFIT_EVERY
):

    block_end = min(
        block_start + REFIT_EVERY,
        test_end_idx
    )

    # --------------------------------------------------------
    # TRAINING WINDOW
    # --------------------------------------------------------

    train_end = block_start

    train_start = (
        train_end - TRAIN_WINDOW
    )

    if train_start < 0:

        raise ValueError(
            f"Not enough history for "
            f"block starting at {block_start}"
        )

    train_df = model_df.iloc[
        train_start:train_end
    ]

    test_df = model_df.iloc[
        block_start:block_end
    ]


    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "target"
    ]

    X_test = test_df[
        FEATURES
    ]


    # --------------------------------------------------------
    # FIT FROZEN QRF
    # --------------------------------------------------------

    model = (
        RandomForestQuantileRegressor(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # PREDICT 5% QUANTILE
    # --------------------------------------------------------

    q_pred = model.predict(
        X_test,
        quantiles=[ALPHA]
    )

    q_pred = np.asarray(
        q_pred
    ).reshape(-1)


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    block = pd.DataFrame({

        "Date":
            test_df["Date"].values,

        "actual_return":
            test_df["target"].values,

        "q05_forecast":
            q_pred,

    })

    block["VaR_5pct"] = (
        -block["q05_forecast"]
    )

    block["breach"] = (
        block["actual_return"]
        < block["q05_forecast"]
    ).astype(int)

    block["pinball_loss"] = (
        pinball_loss(
            block["actual_return"].values,
            block["q05_forecast"].values,
            ALPHA
        )
    )

    block["fit_number"] = fit_number

    predictions.append(block)

    print(
        f"Fit {fit_number + 1:3d} | "
        f"train={train_df['Date'].iloc[0].date()} "
        f"to {train_df['Date'].iloc[-1].date()} | "
        f"test={test_df['Date'].iloc[0].date()} "
        f"to {test_df['Date'].iloc[-1].date()}"
    )

    fit_number += 1


# ============================================================
# COMBINE
# ============================================================

qrf = pd.concat(
    predictions,
    ignore_index=True
)

qrf = (
    qrf
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# QRF METRICS
# ============================================================

qrf_breaches = (
    qrf["breach"].to_numpy()
)

qrf_loss = (
    qrf["pinball_loss"].to_numpy()
)

qrf_kupiec_lr, qrf_kupiec_p = (
    kupiec_uc(
        qrf_breaches,
        ALPHA
    )
)

qrf_ind_lr, qrf_ind_p = (
    christoffersen_independence(
        qrf_breaches
    )
)


# ============================================================
# LOAD TRADITIONAL MODELS
# ============================================================

print()
print("=" * 80)
print("LOADING TRADITIONAL VAR")
print("=" * 80)

traditional = pd.read_csv(
    TRADITIONAL_PATH
)

print(
    f"Traditional rows: "
    f"{len(traditional)}"
)

# Date
if "date" in traditional.columns:

    traditional["Date"] = pd.to_datetime(
        traditional["date"]
    )

elif "Date" in traditional.columns:

    traditional["Date"] = pd.to_datetime(
        traditional["Date"]
    )

else:

    raise ValueError(
        "No Date/date column in traditional VaR file."
    )


required_traditional = [
    "HS_VaR",
    "EWMA_VaR",
    "GARCH_VaR",
]

missing_traditional = [
    c
    for c in required_traditional
    if c not in traditional.columns
]

if missing_traditional:

    raise ValueError(
        "Missing traditional VaR columns:\n"
        f"{missing_traditional}\n\n"
        f"Available:\n"
        f"{traditional.columns.tolist()}"
    )


# ------------------------------------------------------------
# Traditional models contain negative quantile values.
# Convert them into positive VaR magnitudes.
# ------------------------------------------------------------

for model_name in [
    "HS",
    "EWMA",
    "GARCH"
]:

    raw_col = f"{model_name}_VaR"

    traditional[raw_col] = pd.to_numeric(
        traditional[raw_col],
        errors="coerce"
    )


    median_value = (
        traditional[raw_col]
        .dropna()
        .median()
    )

    if median_value < 0:

        traditional[
            f"{model_name}_VaR_mag"
        ] = -traditional[raw_col]

    else:

        traditional[
            f"{model_name}_VaR_mag"
        ] = traditional[raw_col]


# ============================================================
# MERGE TRADITIONAL WITH FINAL QRF
# ============================================================

comparison = pd.merge(
    qrf[
        [
            "Date",
            "actual_return"
        ]
    ],
    traditional[
        [
            "Date",
            "HS_VaR_mag",
            "EWMA_VaR_mag",
            "GARCH_VaR_mag"
        ]
    ],
    on="Date",
    how="inner"
)


# Only compare the exact QRF test period
comparison = comparison[
    comparison["Date"]
    >= qrf["Date"].min()
]

comparison = comparison[
    comparison["Date"]
    <= qrf["Date"].max()
]

comparison = (
    comparison
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_var(
    y,
    positive_var,
    alpha=0.05
):

    positive_var = np.asarray(
        positive_var,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    q = -positive_var

    breaches = (
        y < q
    ).astype(int)

    losses = pinball_loss(
        y,
        q,
        alpha
    )

    kupiec_lr, kupiec_p = (
        kupiec_uc(
            breaches,
            alpha
        )
    )

    ind_lr, ind_p = (
        christoffersen_independence(
            breaches
        )
    )

    excess = np.where(
        breaches == 1,
        q - y,
        0.0
    )

    breach_excess = (
        excess[breaches == 1].mean()
        if breaches.sum() > 0
        else np.nan
    )

    return {

        "n": len(y),

        "breaches": int(
            breaches.sum()
        ),

        "breach_rate":
            breaches.mean(),

        "mean_pinball_loss":
            losses.mean(),

        "mean_VaR":
            positive_var.mean(),

        "median_VaR":
            np.median(positive_var),

        "kupiec_p":
            kupiec_p,

        "christoffersen_independence_p":
            ind_p,

        "mean_excess_loss":
            breach_excess,

        "max_consecutive_breaches":
            max_consecutive_breaches(
                breaches
            )
    }


# ============================================================
# BUILD FINAL COMPARISON
# ============================================================

final_results = []


# QRF
qrf_common = qrf[
    qrf["Date"].isin(
        comparison["Date"]
    )
].sort_values("Date")

qrf_metrics = evaluate_var(
    qrf_common["actual_return"].values,
    qrf_common["VaR_5pct"].values,
    ALPHA
)

qrf_metrics["model"] = "QRF"

final_results.append(
    qrf_metrics
)


# Traditional
for model_name in [
    "HS",
    "EWMA",
    "GARCH"
]:

    metrics = evaluate_var(

        comparison["actual_return"].values,

        comparison[
            f"{model_name}_VaR_mag"
        ].values,

        ALPHA
    )

    metrics["model"] = model_name

    final_results.append(
        metrics
    )


final_results_df = pd.DataFrame(
    final_results
)


final_results_df = (
    final_results_df[
        [
            "model",
            "n",
            "breaches",
            "breach_rate",
            "mean_pinball_loss",
            "mean_VaR",
            "median_VaR",
            "kupiec_p",
            "christoffersen_independence_p",
            "mean_excess_loss",
            "max_consecutive_breaches"
        ]
    ]
    .sort_values(
        "mean_pinball_loss"
    )
    .reset_index(drop=True)
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 80)
print("FINAL 2023–2026 OOS RESULTS")
print("=" * 80)

print()

print(
    f"QRF test dates: "
    f"{qrf['Date'].min().date()} "
    f"to "
    f"{qrf['Date'].max().date()}"
)

print()

print(
    final_results_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# QRF SUMMARY
# ============================================================

print()
print("=" * 80)
print("QRF FINAL OOS SUMMARY")
print("=" * 80)

print(
    f"Breach rate:       "
    f"{qrf_metrics['breach_rate']:.6f}"
)

print(
    f"Pinball loss:      "
    f"{qrf_metrics['mean_pinball_loss']:.8f}"
)

print(
    f"Kupiec p-value:    "
    f"{qrf_metrics['kupiec_p']:.6f}"
)

print(
    f"Independence p:    "
    f"{qrf_metrics['christoffersen_independence_p']:.6f}"
)

print(
    f"Max consecutive:   "
    f"{qrf_metrics['max_consecutive_breaches']}"
)


# ============================================================
# SAVE QRF FORECASTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

qrf.to_csv(
    OUTPUT_PATH,
    index=False
)

final_results_df.to_csv(
    COMPARISON_PATH,
    index=False
)


print()
print("=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"QRF forecasts:\n"
    f"{OUTPUT_PATH}"
)

print()

print(
    f"Model comparison:\n"
    f"{COMPARISON_PATH}"
)

print()
print("=" * 80)
print("FINAL OOS TEST COMPLETE")
print("=" * 80)