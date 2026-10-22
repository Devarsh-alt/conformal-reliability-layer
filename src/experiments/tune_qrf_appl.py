import os
import warnings
import itertools

import numpy as np
import pandas as pd
from scipy.stats import chi2
from quantile_forest import RandomForestQuantileRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

TICKER = "AAPL"
ALPHA = 0.05

# ------------------------------------------------------------
# DEVELOPMENT / VALIDATION SPLIT
# ------------------------------------------------------------
#
# We use 2022 as the validation period.
# 2023 onward remains untouched for the final test.
#
VALIDATION_YEAR = 2022

# ------------------------------------------------------------
# WALK-FORWARD SETTINGS
# ------------------------------------------------------------

REFIT_EVERY = 20

# ------------------------------------------------------------
# QRF TUNING GRID
# ------------------------------------------------------------

WINDOWS = [
    750,
    1000,
    1500,
]

MIN_SAMPLES_LEAF_VALUES = [
    2,
    5,
    10,
    20,
]

MAX_FEATURES_VALUES = [
    "sqrt",
    1.0,
]

# Keep tree count fixed during hyperparameter tuning.
# We can increase this for the final selected model.
N_ESTIMATORS = 500

RANDOM_STATE = 42

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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "monitoring"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    f"{TICKER}_QRF_tuning_results.csv"
)

SELECTION_PATH = os.path.join(
    OUTPUT_DIR,
    f"{TICKER}_QRF_selected_config.txt"
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
    """
    Quantile / pinball loss.

    y = realized next-day return
    q = predicted alpha quantile
    """

    y = np.asarray(y)
    q = np.asarray(q)

    error = y - q

    return np.where(
        error >= 0,
        alpha * error,
        (alpha - 1.0) * error
    )


# ============================================================
# KUPIEC
# ============================================================

def kupiec_uc(breaches, alpha=0.05):
    """
    Kupiec unconditional coverage test.
    """

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
    """
    Christoffersen independence test.
    """

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
print("QRF HYPERPARAMETER TUNING — AAPL")
print("=" * 80)

print()
print(f"Data: {DATA_PATH}")

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        f"Data file not found:\n{DATA_PATH}"
    )


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

print(
    f"Raw rows: {len(df)}"
)

print(
    f"Date range: "
    f"{df['Date'].min().date()} "
    f"to "
    f"{df['Date'].max().date()}"
)


# ============================================================
# TARGET
# ============================================================

#
# Row t:
#
# Features = information known at t
# Target   = R_(t+1)
#
# Therefore:
#
# q_t = Q_0.05(R_(t+1) | X_t)
#

df["target"] = (
    df["Log_Return"]
    .shift(-1)
)


# ============================================================
# VALIDATE FEATURES
# ============================================================

missing_features = [
    col
    for col in FEATURES
    if col not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features:\n"
        f"{missing_features}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

model_df = df[
    [
        "Date"
    ]
    + FEATURES
    + [
        "target"
    ]
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


print(
    f"Usable rows: {len(model_df)}"
)


# ============================================================
# VALIDATION PERIOD
# ============================================================

validation_mask = (
    model_df["Date"].dt.year
    == VALIDATION_YEAR
)

validation_indices = np.where(
    validation_mask.values
)[0]

if len(validation_indices) == 0:

    raise ValueError(
        f"No observations found for "
        f"validation year {VALIDATION_YEAR}."
    )


validation_start_idx = (
    validation_indices[0]
)

validation_end_idx = (
    validation_indices[-1] + 1
)

print()
print("=" * 80)
print("VALIDATION PERIOD")
print("=" * 80)

print(
    f"Validation year: "
    f"{VALIDATION_YEAR}"
)

print(
    f"Validation rows: "
    f"{len(validation_indices)}"
)

print(
    f"Validation dates: "
    f"{model_df.iloc[validation_start_idx]['Date'].date()} "
    f"to "
    f"{model_df.iloc[validation_end_idx - 1]['Date'].date()}"
)


# ============================================================
# CHECK THAT ALL WINDOWS FIT
# ============================================================

for window in WINDOWS:

    if validation_start_idx < window:

        raise ValueError(
            f"Window {window} is too large. "
            "Not enough historical observations "
            "before the validation period."
        )


# ============================================================
# BUILD CONFIG GRID
# ============================================================

config_grid = list(
    itertools.product(
        WINDOWS,
        MIN_SAMPLES_LEAF_VALUES,
        MAX_FEATURES_VALUES
    )
)

print()
print(
    f"Total configurations: "
    f"{len(config_grid)}"
)

print(
    f"Trees per fit: "
    f"{N_ESTIMATORS}"
)

print(
    f"Refit frequency: "
    f"every {REFIT_EVERY} observations"
)


# ============================================================
# VALIDATION BLOCKS
# ============================================================

validation_blocks = []

for block_start in range(
    validation_start_idx,
    validation_end_idx,
    REFIT_EVERY
):

    block_end = min(
        block_start + REFIT_EVERY,
        validation_end_idx
    )

    validation_blocks.append(
        (
            block_start,
            block_end
        )
    )


print(
    f"Validation blocks: "
    f"{len(validation_blocks)}"
)


# ============================================================
# TUNING
# ============================================================

all_results = []


for config_number, (
    window,
    min_leaf,
    max_features
) in enumerate(
    config_grid,
    start=1
):

    print()
    print("=" * 80)

    print(
        f"CONFIGURATION "
        f"{config_number}/{len(config_grid)}"
    )

    print("=" * 80)

    print(
        f"Window:          {window}"
    )

    print(
        f"Min samples leaf:{min_leaf}"
    )

    print(
        f"Max features:    {max_features}"
    )

    print(
        f"Trees:            {N_ESTIMATORS}"
    )


    block_predictions = []


    # --------------------------------------------------------
    # WALK FORWARD THROUGH 2022
    # --------------------------------------------------------

    for block_number, (
        block_start,
        block_end
    ) in enumerate(
        validation_blocks,
        start=1
    ):

        # ----------------------------------------------------
        # TRAINING WINDOW
        # ----------------------------------------------------

        train_end = block_start

        train_start = (
            train_end - window
        )

        if train_start < 0:

            continue


        train_df = model_df.iloc[
            train_start:train_end
        ]

        validation_df = model_df.iloc[
            block_start:block_end
        ]


        X_train = train_df[
            FEATURES
        ]

        y_train = train_df[
            "target"
        ]

        X_validation = validation_df[
            FEATURES
        ]

        y_validation = validation_df[
            "target"
        ]


        # ----------------------------------------------------
        # FIT QRF
        # ----------------------------------------------------

        model = (
            RandomForestQuantileRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=min_leaf,
                max_features=max_features,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )

        model.fit(
            X_train,
            y_train
        )


        # ----------------------------------------------------
        # PREDICT 5% QUANTILE
        # ----------------------------------------------------

        q_pred = model.predict(
            X_validation,
            quantiles=[ALPHA]
        )

        q_pred = np.asarray(
            q_pred
        ).reshape(-1)


        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        block_result = pd.DataFrame({

            "Date":
                validation_df["Date"].values,

            "actual_return":
                y_validation.values,

            "q05_forecast":
                q_pred,

        })

        block_predictions.append(
            block_result
        )


        print(
            f"  Block "
            f"{block_number:02d}/"
            f"{len(validation_blocks)} | "
            f"{validation_df['Date'].iloc[0].date()} "
            f"to "
            f"{validation_df['Date'].iloc[-1].date()}"
        )


    # --------------------------------------------------------
    # COMBINE CONFIGURATION RESULTS
    # --------------------------------------------------------

    if not block_predictions:

        print(
            "No validation predictions. "
            "Skipping."
        )

        continue


    config_predictions = pd.concat(
        block_predictions,
        ignore_index=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    y = (
        config_predictions[
            "actual_return"
        ].to_numpy()
    )

    q = (
        config_predictions[
            "q05_forecast"
        ].to_numpy()
    )


    losses = pinball_loss(
        y,
        q,
        ALPHA
    )


    breaches = (
        y < q
    ).astype(int)


    breach_rate = (
        breaches.mean()
    )


    mean_loss = (
        losses.mean()
    )


    median_loss = (
        np.median(losses)
    )


    mean_var = (
        (-q).mean()
    )


    median_var = (
        np.median(-q)
    )


    kupiec_lr, kupiec_p = (
        kupiec_uc(
            breaches,
            ALPHA
        )
    )


    ind_lr, ind_p = (
        christoffersen_independence(
            breaches
        )
    )


    max_consecutive = (
        max_consecutive_breaches(
            breaches
        )
    )


    calibration_distance = abs(
        breach_rate - ALPHA
    )


    all_results.append({

        "window":
            window,

        "min_samples_leaf":
            min_leaf,

        "max_features":
            str(max_features),

        "n_estimators":
            N_ESTIMATORS,

        "n_validation":
            len(config_predictions),

        "validation_breach_rate":
            breach_rate,

        "validation_breach_distance":
            calibration_distance,

        "mean_pinball_loss":
            mean_loss,

        "median_pinball_loss":
            median_loss,

        "mean_VaR":
            mean_var,

        "median_VaR":
            median_var,

        "kupiec_lr":
            kupiec_lr,

        "kupiec_p":
            kupiec_p,

        "christoffersen_lr":
            ind_lr,

        "christoffersen_independence_p":
            ind_p,

        "max_consecutive_breaches":
            max_consecutive,

    })


    print()
    print(
        "  Validation pinball loss: "
        f"{mean_loss:.8f}"
    )

    print(
        "  Validation breach rate:  "
        f"{breach_rate:.6f}"
    )

    print(
        "  Kupiec p-value:           "
        f"{kupiec_p:.6f}"
    )

    print(
        "  Independence p-value:    "
        f"{ind_p:.6f}"
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ------------------------------------------------------------
# PRIMARY RANKING
# ------------------------------------------------------------

results_df = (
    results_df
    .sort_values(
        [
            "mean_pinball_loss",
            "validation_breach_distance"
        ],
        ascending=[
            True,
            True
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# DISPLAY RANKING
# ============================================================

print()
print("=" * 80)
print("QRF VALIDATION RANKING")
print("=" * 80)

print()
print(
    "Primary criterion: "
    "validation pinball loss"
)

print(
    "Tie-breaker: "
    "distance of breach rate from 5%"
)

display_columns = [

    "window",

    "min_samples_leaf",

    "max_features",

    "n_estimators",

    "n_validation",

    "validation_breach_rate",

    "mean_pinball_loss",

    "kupiec_p",

    "christoffersen_independence_p",

    "max_consecutive_breaches"
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# BEST CONFIGURATION
# ============================================================

best = results_df.iloc[0]


print()
print("=" * 80)
print("SELECTED QRF CONFIGURATION")
print("=" * 80)

print(
    f"Window:                    "
    f"{int(best['window'])}"
)

print(
    f"Min samples leaf:          "
    f"{int(best['min_samples_leaf'])}"
)

print(
    f"Max features:              "
    f"{best['max_features']}"
)

print(
    f"Trees during tuning:       "
    f"{int(best['n_estimators'])}"
)

print(
    f"Validation pinball loss:   "
    f"{best['mean_pinball_loss']:.8f}"
)

print(
    f"Validation breach rate:    "
    f"{best['validation_breach_rate']:.6f}"
)

print(
    f"Kupiec p-value:            "
    f"{best['kupiec_p']:.6f}"
)

print(
    f"Independence p-value:      "
    f"{best['christoffersen_independence_p']:.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


results_df.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SAVE SELECTED CONFIG
# ============================================================

with open(
    SELECTION_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "SELECTED QRF CONFIGURATION\n"
    )

    f.write(
        "===========================\n"
    )

    f.write(
        f"Ticker: {TICKER}\n"
    )

    f.write(
        f"Validation year: "
        f"{VALIDATION_YEAR}\n"
    )

    f.write(
        f"Window: "
        f"{int(best['window'])}\n"
    )

    f.write(
        f"Min samples leaf: "
        f"{int(best['min_samples_leaf'])}\n"
    )

    f.write(
        f"Max features: "
        f"{best['max_features']}\n"
    )

    f.write(
        f"N estimators: "
        f"{int(best['n_estimators'])}\n"
    )

    f.write(
        f"Validation pinball loss: "
        f"{best['mean_pinball_loss']:.10f}\n"
    )

    f.write(
        f"Validation breach rate: "
        f"{best['validation_breach_rate']:.10f}\n"
    )

    f.write(
        f"Kupiec p-value: "
        f"{best['kupiec_p']:.10f}\n"
    )

    f.write(
        f"Christoffersen independence p-value: "
        f"{best['christoffersen_independence_p']:.10f}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"Results:\n{RESULTS_PATH}"
)

print()

print(
    f"Selected configuration:\n"
    f"{SELECTION_PATH}"
)

print()
print("=" * 80)
print("TUNING COMPLETE")
print("=" * 80)