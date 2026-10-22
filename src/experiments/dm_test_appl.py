import os
import numpy as np
import pandas as pd
from scipy.stats import norm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"D:\reliability_layer"

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "monitoring",
    "AAPL_final_aligned_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "monitoring",
    "AAPL_dm_results.csv"
)

ALPHA = 0.05


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DIEBOLD-MARIANO TEST — AAPL FINAL OOS")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nInput columns:")
print(df.columns.tolist())

required_cols = [
    "realized_date",
    "actual_return",
    "QRF_q05",
    "HS_q05",
    "EWMA_q05",
    "GARCH_q05"
]

missing = [
    c for c in required_cols
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing required columns: {missing}"
    )

df["realized_date"] = pd.to_datetime(
    df["realized_date"],
    errors="coerce"
)

for col in required_cols[1:]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = (
    df[
        required_cols
    ]
    .dropna()
    .sort_values("realized_date")
    .reset_index(drop=True)
)


print(
    f"\nRows: {len(df)}"
)

print(
    f"Dates: "
    f"{df['realized_date'].min().date()} "
    f"to "
    f"{df['realized_date'].max().date()}"
)


# ============================================================
# VERIFY RETURN / FORECAST SANITY
# ============================================================

print("\n" + "=" * 80)
print("DATA SANITY CHECK")
print("=" * 80)

print(
    f"Actual return mean: "
    f"{df['actual_return'].mean():.8f}"
)

print(
    f"Actual return std:  "
    f"{df['actual_return'].std():.8f}"
)

for model in [
    "QRF",
    "HS",
    "EWMA",
    "GARCH"
]:

    col = f"{model}_q05"

    print(
        f"{model:6s} q05 range: "
        f"{df[col].min():.6f} "
        f"to "
        f"{df[col].max():.6f}"
    )


# ============================================================
# PINBALL LOSS
# ============================================================

def pinball_loss(
    actual,
    forecast,
    alpha=0.05
):
    """
    Quantile / pinball loss:

        rho_alpha(y, q)
        = (y-q) * (alpha - I[y<q])

    Lower is better.
    """

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

    loss = (
        actual - forecast
    ) * (
        alpha - hit
    )

    return loss


# ============================================================
# CALCULATE MODEL LOSSES
# ============================================================

models = {
    "QRF": "QRF_q05",
    "HS": "HS_q05",
    "EWMA": "EWMA_q05",
    "GARCH": "GARCH_q05"
}

actual = df["actual_return"].values

for model, col in models.items():

    df[f"{model}_loss"] = pinball_loss(
        actual,
        df[col].values,
        ALPHA
    )


# ============================================================
# LOSS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("MEAN PINBALL LOSS")
print("=" * 80)

loss_summary = []

for model in models:

    mean_loss = df[
        f"{model}_loss"
    ].mean()

    loss_summary.append(
        {
            "model": model,
            "mean_pinball_loss": mean_loss
        }
    )

loss_summary = pd.DataFrame(
    loss_summary
)

print(
    loss_summary.to_string(
        index=False
    )
)


# ============================================================
# NEWey-WEST HAC VARIANCE
# ============================================================

def newey_west_variance(
    series,
    max_lag=None
):
    """
    HAC estimate of the variance of the sample mean.

    Bartlett kernel.
    """

    x = np.asarray(
        series,
        dtype=float
    )

    x = x[np.isfinite(x)]

    n = len(x)

    if n < 2:
        raise ValueError(
            "Not enough observations."
        )

    x_centered = (
        x - x.mean()
    )

    # --------------------------------------------------------
    # Automatic bandwidth
    # --------------------------------------------------------

    if max_lag is None:

        max_lag = int(
            np.floor(
                4
                *
                (n / 100.0)
                ** (2.0 / 9.0)
            )
        )

    max_lag = min(
        max_lag,
        n - 1
    )

    # --------------------------------------------------------
    # Gamma_0
    # --------------------------------------------------------

    gamma_0 = np.mean(
        x_centered ** 2
    )

    long_run_variance = gamma_0

    # --------------------------------------------------------
    # Autocovariances
    # --------------------------------------------------------

    for lag in range(
        1,
        max_lag + 1
    ):

        gamma_lag = np.mean(
            x_centered[lag:]
            *
            x_centered[:-lag]
        )

        # Bartlett weight
        weight = (
            1.0
            -
            lag / (max_lag + 1.0)
        )

        long_run_variance += (
            2.0
            *
            weight
            *
            gamma_lag
        )

    # Variance of sample mean
    variance_mean = (
        long_run_variance
        /
        n
    )

    return (
        variance_mean,
        max_lag
    )


# ============================================================
# DIEBOLD-MARIANO TEST
# ============================================================

def dm_test(
    loss_a,
    loss_b,
    alternative="two-sided"
):
    """
    Diebold-Mariano test using HAC/Newey-West
    variance.

    Define:

        d_t = L_A,t - L_B,t

    Null:
        E[d_t] = 0

    Interpretation:

        mean_d < 0
            A has lower loss than B.

        mean_d > 0
            B has lower loss than A.

    Two-sided test is used by default.
    """

    loss_a = np.asarray(
        loss_a,
        dtype=float
    )

    loss_b = np.asarray(
        loss_b,
        dtype=float
    )

    valid = (
        np.isfinite(loss_a)
        &
        np.isfinite(loss_b)
    )

    loss_a = loss_a[valid]
    loss_b = loss_b[valid]

    if len(loss_a) != len(loss_b):
        raise ValueError(
            "Loss arrays have different lengths."
        )

    n = len(loss_a)

    if n < 20:
        raise ValueError(
            "Too few observations for DM test."
        )

    # --------------------------------------------------------
    # Loss differential
    # --------------------------------------------------------

    d = (
        loss_a
        -
        loss_b
    )

    mean_d = d.mean()

    # --------------------------------------------------------
    # HAC variance
    # --------------------------------------------------------

    variance_mean, nw_lag = (
        newey_west_variance(d)
    )

    if variance_mean <= 0:
        raise ValueError(
            "Non-positive HAC variance."
        )

    standard_error = np.sqrt(
        variance_mean
    )

    dm_stat = (
        mean_d
        /
        standard_error
    )

    # --------------------------------------------------------
    # P-value
    # --------------------------------------------------------

    if alternative == "two-sided":

        p_value = (
            2.0
            *
            norm.sf(
                abs(dm_stat)
            )
        )

    elif alternative == "less":

        # H1: model A better than B
        # mean differential < 0

        p_value = norm.cdf(
            dm_stat
        )

    elif alternative == "greater":

        # H1: model A worse than B
        # mean differential > 0

        p_value = norm.sf(
            dm_stat
        )

    else:

        raise ValueError(
            "Invalid alternative."
        )

    return {
        "n": n,
        "mean_loss_diff": mean_d,
        "std_error": standard_error,
        "DM_stat": dm_stat,
        "p_value": p_value,
        "NW_lag": nw_lag
    }


# ============================================================
# MODEL COMPARISONS
# ============================================================

comparisons = [
    ("QRF", "EWMA"),
    ("QRF", "HS"),
    ("QRF", "GARCH")
]

results = []


for model_a, model_b in comparisons:

    loss_a = df[
        f"{model_a}_loss"
    ].values

    loss_b = df[
        f"{model_b}_loss"
    ].values

    result = dm_test(
        loss_a,
        loss_b,
        alternative="two-sided"
    )

    mean_a = loss_a.mean()
    mean_b = loss_b.mean()

    # --------------------------------------------------------
    # Percentage improvement of A over B
    # --------------------------------------------------------

    if mean_b != 0:

        pct_improvement = (
            (
                mean_b
                -
                mean_a
            )
            /
            mean_b
            * 100.0
        )

    else:

        pct_improvement = np.nan

    # --------------------------------------------------------
    # Which model has lower loss?
    # --------------------------------------------------------

    if mean_a < mean_b:

        lower_loss_model = model_a

    elif mean_b < mean_a:

        lower_loss_model = model_b

    else:

        lower_loss_model = "TIE"

    # --------------------------------------------------------
    # Statistical significance
    # --------------------------------------------------------

    if result["p_value"] < 0.05:

        if result["mean_loss_diff"] < 0:

            interpretation = (
                f"{model_a} significantly "
                f"outperforms {model_b}"
            )

        else:

            interpretation = (
                f"{model_b} significantly "
                f"outperforms {model_a}"
            )

    else:

        interpretation = (
            "No statistically significant "
            "difference in predictive accuracy"
        )

    results.append(
        {
            "model_A": model_a,
            "model_B": model_b,
            "n": result["n"],
            "mean_loss_A": mean_a,
            "mean_loss_B": mean_b,
            "mean_loss_diff_A_minus_B":
                result["mean_loss_diff"],
            "std_error":
                result["std_error"],
            "pct_improvement_A_vs_B":
                pct_improvement,
            "lower_loss_model":
                lower_loss_model,
            "DM_stat":
                result["DM_stat"],
            "p_value":
                result["p_value"],
            "NW_lag":
                result["NW_lag"],
            "significant_5pct":
                result["p_value"] < 0.05,
            "interpretation":
                interpretation
        }
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 80)
print("DIEBOLD-MARIANO TEST RESULTS")
print("=" * 80)

display_cols = [
    "model_A",
    "model_B",
    "n",
    "mean_loss_A",
    "mean_loss_B",
    "mean_loss_diff_A_minus_B",
    "pct_improvement_A_vs_B",
    "DM_stat",
    "p_value",
    "NW_lag",
    "significant_5pct"
]

print(
    results_df[
        display_cols
    ].to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATIONS
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)

for _, row in results_df.iterrows():

    print(
        f"\n{row['model_A']} vs "
        f"{row['model_B']}:"
    )

    print(
        f"  Mean loss difference: "
        f"{row['mean_loss_diff_A_minus_B']:.8f}"
    )

    print(
        f"  DM statistic: "
        f"{row['DM_stat']:.6f}"
    )

    print(
        f"  p-value: "
        f"{row['p_value']:.6f}"
    )

    print(
        f"  Percentage improvement: "
        f"{row['pct_improvement_A_vs_B']:.3f}%"
    )

    if row["p_value"] < 0.05:

        print(
            f"  RESULT: "
            f"{row['interpretation']}"
        )

    else:

        print(
            "  RESULT: "
            "Difference is not statistically significant "
            "at the 5% level."
        )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)

print("\n" + "=" * 80)
print("DM TEST COMPLETE")
print("=" * 80)