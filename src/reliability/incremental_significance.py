from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "fixed_ablation"
    / "oos_predictions.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "fixed_ablation"
    / "incremental_significance.csv"
)

BOOTSTRAPS = 5000
BLOCK_SIZE = 20
RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT_PATH)

df["prediction_date"] = pd.to_datetime(
    df["prediction_date"]
)

print(f"Rows loaded: {len(df):,}")

required = {
    "architecture",
    "prediction_date",
    "miscovered",
    "predicted_miscoverage",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# CREATE ONE DAILY BRIER SCORE PER ARCHITECTURE
#
# This prevents the 24 assets on the same day from being
# treated as 24 independent time observations.
# ============================================================

architectures = sorted(
    df["architecture"].unique()
)

daily = (
    df.assign(
        squared_error=(
            df["miscovered"]
            - df["predicted_miscoverage"]
        ) ** 2
    )
    .groupby(
        ["prediction_date", "architecture"],
        as_index=False,
    )["squared_error"]
    .mean()
    .rename(
        columns={
            "squared_error": "daily_brier"
        }
    )
)

pivot = daily.pivot(
    index="prediction_date",
    columns="architecture",
    values="daily_brier",
).sort_index()

print(
    f"Number of dates: {len(pivot):,}"
)

print(
    "\nArchitectures:"
)

print(
    list(pivot.columns)
)


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)


def block_bootstrap_mean(
    differences,
    block_size=20,
    n_bootstrap=5000,
):
    """
    Moving/non-overlapping block bootstrap over the
    chronological daily loss-difference sequence.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)

    if n == 0:
        raise ValueError("No finite observations.")

    # Construct sequential blocks.
    blocks = [
        differences[i:i + block_size]
        for i in range(
            0,
            n,
            block_size,
        )
    ]

    # Drop incomplete final block.
    blocks = [
        b
        for b in blocks
        if len(b) == block_size
    ]

    if len(blocks) < 5:
        raise ValueError(
            "Too few complete blocks for bootstrap."
        )

    block_means = np.array(
        [b.mean() for b in blocks]
    )

    n_blocks = len(block_means)

    bootstrap_means = np.empty(
        n_bootstrap
    )

    for i in range(n_bootstrap):

        sample = rng.integers(
            low=0,
            high=n_blocks,
            size=n_blocks,
        )

        bootstrap_means[i] = (
            block_means[sample].mean()
        )

    observed = differences.mean()

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
    )

    return (
        observed,
        ci_low,
        ci_high,
        bootstrap_means,
    )


# ============================================================
# COMPARISONS
#
# Positive difference means challenger has LOWER Brier than
# reference:
#
#     reference_brier - challenger_brier
#
# Therefore:
#
#     positive = improvement
# ============================================================

comparisons = [
    ("constant", "D"),
    ("D", "D+O"),
    ("D", "D+M"),
    ("D", "D+W"),
    ("D", "D+O+M+W"),
]


results = []


for reference, challenger in comparisons:

    pair = pivot[
        [reference, challenger]
    ].dropna()

    improvement = (
        pair[reference]
        - pair[challenger]
    )

    observed, ci_low, ci_high, bootstrap = (
        block_bootstrap_mean(
            improvement.values,
            block_size=BLOCK_SIZE,
            n_bootstrap=BOOTSTRAPS,
        )
    )

    # Probability that the bootstrap improvement is > 0.
    prob_positive = np.mean(
        bootstrap > 0
    )

    results.append(
        {
            "reference": reference,
            "challenger": challenger,
            "n_dates": len(pair),
            "mean_brier_improvement": observed,
            "ci_2.5": ci_low,
            "ci_97.5": ci_high,
            "bootstrap_probability_improvement_gt_0":
                prob_positive,
        }
    )

    print("\n" + "=" * 75)
    print(
        f"{reference} -> {challenger}"
    )
    print("=" * 75)

    print(
        f"Mean Brier improvement : {observed:.8f}"
    )

    print(
        f"95% bootstrap CI       : "
        f"[{ci_low:.8f}, {ci_high:.8f}]"
    )

    print(
        f"P(improvement > 0)     : "
        f"{prob_positive:.4f}"
    )


# ============================================================
# SAVE
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n")
print("=" * 75)
print("SUMMARY")
print("=" * 75)

print(
    results_df.to_string(index=False)
)

print("\nSaved:")
print(OUTPUT_PATH)