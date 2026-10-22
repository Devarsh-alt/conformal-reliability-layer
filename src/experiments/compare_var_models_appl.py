import os
import numpy as np
import pandas as pd
from scipy.stats import chi2


# ============================================================
# CONFIG
# ============================================================

TICKER = "AAPL"
ALPHA = 0.05

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

QRF_PATH = os.path.join(
    BASE_DIR,
    "data",
    "monitoring",
    f"{TICKER}_QRF_5pct_baseline.csv"
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
    f"{TICKER}_var_model_comparison.csv"
)


# ============================================================
# METRICS
# ============================================================

def pinball_loss(y, q, alpha=0.05):
    """
    Pinball loss for quantile alpha.

    y = realized return
    q = predicted alpha quantile
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

    H0:
        observed breach probability = alpha
    """

    breaches = np.asarray(breaches, dtype=int)

    n = len(breaches)
    x = breaches.sum()

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

    lr = -2.0 * (log_l_null - log_l_alt)

    p_value = 1.0 - chi2.cdf(
        lr,
        df=1
    )

    return lr, p_value


def christoffersen_independence(breaches):
    """
    Christoffersen independence test.

    H0:
        VaR breaches are independent over time.
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

        elif prev == 1 and curr == 1:
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

    # Independence model
    log_l_ind = (
        n00 * np.log(1 - pi)
        + n01 * np.log(pi)
        + n10 * np.log(1 - pi)
        + n11 * np.log(pi)
    )

    # Markov dependence model
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


def max_consecutive_breaches(breaches):
    """
    Maximum number of consecutive VaR breaches.
    """

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
# COLUMN DETECTION
# ============================================================

def find_date_column(df):
    """
    Automatically find date column.
    """

    candidates = [
        "Date",
        "date",
        "DATE"
    ]

    for col in candidates:

        if col in df.columns:
            return col

    raise ValueError(
        "Could not find date column.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


def find_var_column(
    df,
    model_name
):
    """
    Find VaR column using multiple possible naming schemes.
    """

    model_lower = model_name.lower()

    # First try exact common names
    exact_candidates = {

        "HS": [
            "HS_VaR",
            "HS_5pct",
            "HS",
            "hs_var",
            "hs_5pct"
        ],

        "EWMA": [
            "EWMA_VaR",
            "EWMA_5pct",
            "EWMA",
            "ewma_var",
            "ewma_5pct"
        ],

        "GARCH": [
            "GARCH_VaR",
            "GARCH_5pct",
            "GARCH",
            "garch_var",
            "garch_5pct"
        ]
    }

    for col in exact_candidates[model_name]:

        if col in df.columns:
            return col

    # Then do flexible matching
    for col in df.columns:

        col_lower = str(col).lower()

        if model_lower in col_lower and "var" in col_lower:
            return col

    # Some files may use quantile naming
    for col in df.columns:

        col_lower = str(col).lower()

        if model_lower in col_lower and (
            "q05" in col_lower
            or "05" in col_lower
            or "quantile" in col_lower
        ):
            return col

    raise ValueError(
        f"Could not find {model_name} VaR column.\n"
        f"Available columns:\n{df.columns.tolist()}"
    )


# ============================================================
# LOAD QRF
# ============================================================

print("=" * 80)
print("AAPL — FAIR VaR MODEL COMPARISON")
print("=" * 80)

print()
print("Loading QRF...")

if not os.path.exists(QRF_PATH):
    raise FileNotFoundError(
        f"QRF file not found:\n{QRF_PATH}"
    )

qrf = pd.read_csv(
    QRF_PATH
)

qrf_date_col = find_date_column(
    qrf
)

qrf[qrf_date_col] = pd.to_datetime(
    qrf[qrf_date_col]
)

qrf = qrf.rename(
    columns={
        qrf_date_col: "Date"
    }
)

required_qrf = [
    "Date",
    "actual_return",
    "q05_forecast",
    "VaR_5pct"
]

missing_qrf = [
    col
    for col in required_qrf
    if col not in qrf.columns
]

if missing_qrf:
    raise ValueError(
        "Missing QRF columns:\n"
        f"{missing_qrf}\n\n"
        "Available columns:\n"
        f"{qrf.columns.tolist()}"
    )

qrf = qrf[
    [
        "Date",
        "actual_return",
        "q05_forecast",
        "VaR_5pct"
    ]
].copy()

qrf = qrf.rename(
    columns={
        "actual_return": "Return",
        "q05_forecast": "QRF_quantile",
        "VaR_5pct": "QRF_VaR"
    }
)

print(f"QRF rows: {len(qrf)}")


# ============================================================
# LOAD TRADITIONAL VAR
# ============================================================

print()
print("Loading traditional VaR models...")

if not os.path.exists(TRADITIONAL_PATH):
    raise FileNotFoundError(
        f"Traditional VaR file not found:\n"
        f"{TRADITIONAL_PATH}"
    )

traditional = pd.read_csv(
    TRADITIONAL_PATH
)

print(
    f"Traditional rows: "
    f"{len(traditional)}"
)

print()
print("Traditional columns:")
print(traditional.columns.tolist())


# ============================================================
# STANDARDIZE DATE
# ============================================================

traditional_date_col = find_date_column(
    traditional
)

traditional[traditional_date_col] = pd.to_datetime(
    traditional[traditional_date_col]
)

traditional = traditional.rename(
    columns={
        traditional_date_col: "Date"
    }
)


# ============================================================
# FIND TRADITIONAL VAR COLUMNS
# ============================================================

hs_col = find_var_column(
    traditional,
    "HS"
)

ewma_col = find_var_column(
    traditional,
    "EWMA"
)

garch_col = find_var_column(
    traditional,
    "GARCH"
)

print()
print("Detected VaR columns:")

print(f"HS:    {hs_col}")
print(f"EWMA:  {ewma_col}")
print(f"GARCH: {garch_col}")


# ============================================================
# KEEP ONLY REQUIRED COLUMNS
# ============================================================

traditional = traditional[
    [
        "Date",
        hs_col,
        ewma_col,
        garch_col
    ]
].copy()

traditional = traditional.rename(
    columns={
        hs_col: "HS_raw",
        ewma_col: "EWMA_raw",
        garch_col: "GARCH_raw"
    }
)


# ============================================================
# NORMALIZE SIGN CONVENTION
# ============================================================

print()
print("=" * 80)
print("SIGN CHECK")
print("=" * 80)

for model in [
    "HS",
    "EWMA",
    "GARCH"
]:

    raw_col = f"{model}_raw"

    median_value = (
        traditional[raw_col]
        .dropna()
        .median()
    )

    print(
        f"{model}: median raw value = "
        f"{median_value:.8f}"
    )

    # If the stored quantity is a negative quantile,
    # convert it to positive VaR magnitude.
    if median_value < 0:

        traditional[f"{model}_VaR"] = (
            -traditional[raw_col]
        )

        print(
            f"  -> interpreted as negative "
            f"quantile; multiplied by -1"
        )

    else:

        traditional[f"{model}_VaR"] = (
            traditional[raw_col]
        )

        print(
            f"  -> interpreted as positive "
            f"VaR magnitude"
        )


# ============================================================
# MERGE ON COMMON DATES
# ============================================================

print()
print("=" * 80)
print("MERGING COMMON SAMPLE")
print("=" * 80)

df = pd.merge(
    qrf[
        [
            "Date",
            "Return",
            "QRF_quantile",
            "QRF_VaR"
        ]
    ],
    traditional[
        [
            "Date",
            "HS_VaR",
            "EWMA_VaR",
            "GARCH_VaR"
        ]
    ],
    on="Date",
    how="inner"
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna(
    subset=[
        "Date",
        "Return",
        "QRF_quantile",
        "QRF_VaR",
        "HS_VaR",
        "EWMA_VaR",
        "GARCH_VaR"
    ]
)

df = (
    df
    .sort_values("Date")
    .drop_duplicates("Date")
    .reset_index(drop=True)
)

print(
    f"Common-sample rows: "
    f"{len(df)}"
)

print(
    f"Common date range: "
    f"{df['Date'].min().date()} "
    f"to "
    f"{df['Date'].max().date()}"
)


# ============================================================
# VERIFY RETURN / QRF CONSISTENCY
# ============================================================

print()
print("=" * 80)
print("QRF CONSISTENCY CHECK")
print("=" * 80)

# The QRF quantile should be negative and VaR should be positive.

qrf_sign_mismatch = (
    (df["QRF_quantile"] >= 0)
    |
    (df["QRF_VaR"] <= 0)
).mean()

print(
    "Fraction of QRF rows with suspicious "
    f"sign convention: {qrf_sign_mismatch:.6f}"
)

if qrf_sign_mismatch > 0:

    print(
        "WARNING: Some QRF rows have an "
        "unexpected sign convention."
    )


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

print()
print("=" * 80)
print("EVALUATING MODELS")
print("=" * 80)

models = {

    "QRF": "QRF_VaR",

    "HS": "HS_VaR",

    "EWMA": "EWMA_VaR",

    "GARCH": "GARCH_VaR"
}

results = []


for model_name, var_col in models.items():

    print()
    print(f"Evaluating {model_name}...")

    y = df["Return"].to_numpy()

    positive_var = (
        df[var_col]
        .to_numpy()
    )

    # Positive VaR magnitude
    # becomes negative return quantile.
    q = -positive_var

    # --------------------------------------------------------
    # BREACHES
    # --------------------------------------------------------

    breaches = (
        y < q
    ).astype(int)

    breach_rate = breaches.mean()

    # --------------------------------------------------------
    # PINBALL LOSS
    # --------------------------------------------------------

    losses = pinball_loss(
        y,
        q,
        ALPHA
    )

    mean_loss = losses.mean()

    median_loss = np.median(
        losses
    )

    # --------------------------------------------------------
    # COVERAGE
    # --------------------------------------------------------

    kupiec_lr, kupiec_p = kupiec_uc(
        breaches,
        ALPHA
    )

    # --------------------------------------------------------
    # INDEPENDENCE
    # --------------------------------------------------------

    ind_lr, ind_p = (
        christoffersen_independence(
            breaches
        )
    )

    # --------------------------------------------------------
    # BREACH SEVERITY
    # --------------------------------------------------------

    excess_losses = np.where(
        breaches == 1,
        q - y,
        0.0
    )

    mean_excess = (
        excess_losses[breaches == 1].mean()
        if breaches.sum() > 0
        else np.nan
    )

    # --------------------------------------------------------
    # CONSECUTIVE BREACHES
    # --------------------------------------------------------

    max_consecutive = (
        max_consecutive_breaches(
            breaches
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results.append({

        "model": model_name,

        "n": len(df),

        "breaches": int(
            breaches.sum()
        ),

        "breach_rate": breach_rate,

        "target_rate": ALPHA,

        "mean_pinball_loss": mean_loss,

        "median_pinball_loss": median_loss,

        "mean_VaR": positive_var.mean(),

        "median_VaR": np.median(
            positive_var
        ),

        "kupiec_lr": kupiec_lr,

        "kupiec_p": kupiec_p,

        "christoffersen_lr": ind_lr,

        "christoffersen_independence_p":
            ind_p,

        "mean_excess_loss":
            mean_excess,

        "max_consecutive_breaches":
            max_consecutive
    })


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = (
    results_df
    .sort_values(
        "mean_pinball_loss",
        ascending=True
    )
    .reset_index(drop=True)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 80)
print("FINAL MODEL RANKING")
print("=" * 80)

print()
print(
    "Lower pinball loss = better tail-quantile forecast."
)

print()

display_cols = [
    "model",
    "n",
    "breaches",
    "breach_rate",
    "mean_pinball_loss",
    "mean_VaR",
    "kupiec_p",
    "christoffersen_independence_p",
    "mean_excess_loss",
    "max_consecutive_breaches"
]

print(
    results_df[
        display_cols
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.8f}"
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print()
print("=" * 80)
print("BEST MODEL BY PINBALL LOSS")
print("=" * 80)

print(
    f"Model:              "
    f"{best_model['model']}"
)

print(
    f"Pinball loss:       "
    f"{best_model['mean_pinball_loss']:.8f}"
)

print(
    f"Breach rate:        "
    f"{best_model['breach_rate']:.6f}"
)

print(
    f"Kupiec p-value:     "
    f"{best_model['kupiec_p']:.6f}"
)

print(
    f"Independence p:     "
    f"{best_model['christoffersen_independence_p']:.6f}"
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_PATH)