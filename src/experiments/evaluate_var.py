from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TICKER = "AAPL"
ALPHA = 0.05
ROLLING_WINDOW = 250

INPUT = ROOT / "data" / "var" / f"{TICKER}_var_5pct.csv"
OUTPUT_DIR = ROOT / "data" / "var"
OUTPUT = OUTPUT_DIR / f"{TICKER}_var_evaluation.csv"


# ============================================================
# Helper functions
# ============================================================

def safe_log_likelihood(x, n, p):
    """
    Bernoulli log-likelihood:
        x successes out of n observations
        success probability p
    """
    p = np.clip(p, 1e-12, 1 - 1e-12)

    x = int(x)
    n = int(n)

    return (
        x * np.log(p)
        + (n - x) * np.log(1 - p)
    )


def kupiec_test(breaches, alpha):
    """
    Kupiec unconditional coverage test.

    H0:
        Actual breach probability = alpha

    LR_uc ~ Chi-square(1)
    """

    breaches = np.asarray(breaches, dtype=int)

    n = len(breaches)
    x = int(breaches.sum())

    p_hat = x / n

    log_likelihood_null = safe_log_likelihood(
        x,
        n,
        alpha
    )

    log_likelihood_alt = safe_log_likelihood(
        x,
        n,
        p_hat
    )

    lr_uc = -2 * (
        log_likelihood_null
        - log_likelihood_alt
    )

    p_value = 1 - chi2.cdf(lr_uc, df=1)

    return {
        "n": n,
        "breaches": x,
        "breach_rate": p_hat,
        "LR_uc": lr_uc,
        "pvalue_uc": p_value,
    }


def christoffersen_test(breaches, alpha):
    """
    Christoffersen independence and conditional coverage tests.

    Independence:
        Are breaches independent over time?

    Conditional coverage:
        Correct unconditional coverage + independence.
    """

    breaches = np.asarray(breaches, dtype=int)

    # --------------------------------------------------------
    # Transition counts
    # --------------------------------------------------------

    previous = breaches[:-1]
    current = breaches[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    # --------------------------------------------------------
    # Transition probabilities
    # --------------------------------------------------------

    n0 = n00 + n01
    n1 = n10 + n11

    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0

    total_transitions = n0 + n1

    pi = (
        (n01 + n11) / total_transitions
        if total_transitions > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Log-likelihood under independence
    # --------------------------------------------------------

    ll_independent = (
        safe_log_likelihood(
            n01 + n11,
            total_transitions,
            pi
        )
    )

    # --------------------------------------------------------
    # Log-likelihood under first-order dependence
    # --------------------------------------------------------

    ll_dependent = 0.0

    if n00 + n01 > 0:
        ll_dependent += safe_log_likelihood(
            n01,
            n00 + n01,
            pi0
        )

    if n10 + n11 > 0:
        ll_dependent += safe_log_likelihood(
            n11,
            n10 + n11,
            pi1
        )

    # --------------------------------------------------------
    # Independence LR
    # --------------------------------------------------------

    lr_ind = max(
        0.0,
        -2 * (
            ll_independent
            - ll_dependent
        )
    )

    pvalue_ind = 1 - chi2.cdf(
        lr_ind,
        df=1
    )

    # --------------------------------------------------------
    # Unconditional coverage
    # --------------------------------------------------------

    uc = kupiec_test(
        breaches,
        alpha
    )

    # --------------------------------------------------------
    # Conditional coverage
    # --------------------------------------------------------

    lr_cc = uc["LR_uc"] + lr_ind

    pvalue_cc = 1 - chi2.cdf(
        lr_cc,
        df=2
    )

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "pi0": pi0,
        "pi1": pi1,
        "LR_ind": lr_ind,
        "pvalue_ind": pvalue_ind,
        "LR_cc": lr_cc,
        "pvalue_cc": pvalue_cc,
    }


def max_consecutive_ones(values):
    """
    Maximum number of consecutive breaches.
    """

    values = np.asarray(values, dtype=int)

    max_run = 0
    current_run = 0

    for value in values:

        if value == 1:
            current_run += 1
            max_run = max(
                max_run,
                current_run
            )
        else:
            current_run = 0

    return max_run


def breach_severity(df, var_column, breach_column):
    """
    Measures how far realized returns went beyond VaR.

    Since VaR and return are negative:
        return - VaR < 0  -> no excess loss
        VaR - return > 0  -> breach severity
    """

    breaches = df[breach_column] == 1

    if breaches.sum() == 0:
        return {
            "breach_count": 0,
            "mean_excess": np.nan,
            "median_excess": np.nan,
            "max_excess": np.nan,
        }

    excess = (
        df.loc[breaches, var_column]
        - df.loc[breaches, "return"]
    )

    return {
        "breach_count": int(breaches.sum()),
        "mean_excess": excess.mean(),
        "median_excess": excess.median(),
        "max_excess": excess.max(),
    }


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df.sort_values("date")
    .reset_index(drop=True)
)


print("=" * 80)
print(f"VaR STATISTICAL EVALUATION — {TICKER}")
print("=" * 80)

print("Rows:", len(df))
print("VaR level:", f"{ALPHA:.0%}")
print("Rolling window:", ROLLING_WINDOW)


# ============================================================
# Models
# ============================================================

models = {
    "Historical Simulation": {
        "var": "HS_VaR",
        "breach": "HS_breach",
    },
    "EWMA": {
        "var": "EWMA_VaR",
        "breach": "EWMA_breach",
    },
    "GARCH(1,1)": {
        "var": "GARCH_VaR",
        "breach": "GARCH_breach",
    },
}


# ============================================================
# Statistical evaluation
# ============================================================

results = []

for model_name, columns in models.items():

    var_column = columns["var"]
    breach_column = columns["breach"]

    breaches = df[breach_column].to_numpy()

    # --------------------------------------------------------
    # Kupiec
    # --------------------------------------------------------

    kupiec = kupiec_test(
        breaches,
        ALPHA
    )

    # --------------------------------------------------------
    # Christoffersen
    # --------------------------------------------------------

    christ = christoffersen_test(
        breaches,
        ALPHA
    )

    # --------------------------------------------------------
    # Rolling breach rate
    # --------------------------------------------------------

    rolling_rate = (
        df[breach_column]
        .rolling(ROLLING_WINDOW)
        .mean()
    )

    valid_rolling = rolling_rate.dropna()

    # --------------------------------------------------------
    # Breach severity
    # --------------------------------------------------------

    severity = breach_severity(
        df,
        var_column,
        breach_column
    )

    # --------------------------------------------------------
    # VaR statistics
    # --------------------------------------------------------

    avg_var = df[var_column].mean()
    median_var = df[var_column].median()

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    row = {
        "model": model_name,

        "n": kupiec["n"],
        "breaches": kupiec["breaches"],
        "breach_rate": kupiec["breach_rate"],

        "LR_uc": kupiec["LR_uc"],
        "pvalue_uc": kupiec["pvalue_uc"],

        "n00": christ["n00"],
        "n01": christ["n01"],
        "n10": christ["n10"],
        "n11": christ["n11"],

        "pi0": christ["pi0"],
        "pi1": christ["pi1"],

        "LR_ind": christ["LR_ind"],
        "pvalue_ind": christ["pvalue_ind"],

        "LR_cc": christ["LR_cc"],
        "pvalue_cc": christ["pvalue_cc"],

        "rolling_rate_mean": valid_rolling.mean(),
        "rolling_rate_std": valid_rolling.std(),
        "rolling_rate_min": valid_rolling.min(),
        "rolling_rate_max": valid_rolling.max(),

        "max_consecutive_breaches":
            max_consecutive_ones(breaches),

        "mean_excess":
            severity["mean_excess"],

        "median_excess":
            severity["median_excess"],

        "max_excess":
            severity["max_excess"],

        "average_var":
            avg_var,

        "median_var":
            median_var,
    }

    results.append(row)


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Pass / fail at 5% significance
# ============================================================

results_df["Kupiec_pass"] = (
    results_df["pvalue_uc"] >= 0.05
)

results_df["Christoffersen_independence_pass"] = (
    results_df["pvalue_ind"] >= 0.05
)

results_df["Christoffersen_CC_pass"] = (
    results_df["pvalue_cc"] >= 0.05
)


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# Print results
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\n" + "=" * 80)
print("MAIN RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "model",
            "breaches",
            "breach_rate",
            "pvalue_uc",
            "pvalue_ind",
            "pvalue_cc",
            "max_consecutive_breaches",
            "mean_excess",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("ROLLING BREACH-RATE RESULTS")
print("=" * 80)

print(
    results_df[
        [
            "model",
            "rolling_rate_mean",
            "rolling_rate_std",
            "rolling_rate_min",
            "rolling_rate_max",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 80)
print("PASS / FAIL AT 5% SIGNIFICANCE")
print("=" * 80)

for _, row in results_df.iterrows():

    print(f"\n{row['model']}")

    print(
        "  Kupiec:",
        "PASS" if row["Kupiec_pass"] else "FAIL",
        f"(p={row['pvalue_uc']:.4f})"
    )

    print(
        "  Christoffersen independence:",
        "PASS"
        if row["Christoffersen_independence_pass"]
        else "FAIL",
        f"(p={row['pvalue_ind']:.4f})"
    )

    print(
        "  Christoffersen conditional coverage:",
        "PASS"
        if row["Christoffersen_CC_pass"]
        else "FAIL",
        f"(p={row['pvalue_cc']:.4f})"
    )


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("Saved to:", OUTPUT)