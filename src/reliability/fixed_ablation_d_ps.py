from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "perturbation_sensitivity"
    / "candidate_dataset_with_ps.csv"
)

OUT_DIR = (
    ROOT
    / "data"
    / "reliability"
    / "perturbation_sensitivity"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

TARGET = "miscovered"

ARCHITECTURES = {
    "constant": [],
    "D": ["disagreement_pct"],
    "PS": ["ps_pct"],
    "D+PS": [
        "disagreement_pct",
        "ps_pct",
    ],
}

BOOTSTRAPS = 5000
BLOCK_SIZE = 20
RANDOM_STATE = 42


# ============================================================
# MODEL
# ============================================================

def build_model():

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def evaluate(
    y_true,
    probabilities,
):

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    result = {
        "brier":
            brier_score_loss(
                y_true,
                probabilities,
            ),
        "logloss":
            log_loss(
                y_true,
                probabilities,
                labels=[0, 1],
            ),
    }

    if len(
        np.unique(y_true)
    ) == 2:

        result["auc"] = (
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        result["pr_auc"] = (
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:

        result["auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("FIXED D vs PS vs D+PS")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH
)

df["prediction_date"] = (
    pd.to_datetime(
        df["prediction_date"]
    )
)

df["outcome_date"] = (
    pd.to_datetime(
        df["outcome_date"]
    )
)

required = {
    "ticker",
    "prediction_date",
    "outcome_date",
    TARGET,
    "disagreement_pct",
    "ps_pct",
}

missing = (
    required
    - set(df.columns)
)

if missing:

    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

print(
    f"Rows: {len(df):,}"
)


# ============================================================
# OUT-OF-SAMPLE PREDICTIONS
# ============================================================

results = []
predictions = []

for test_year in TEST_YEARS:

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year + 1}-01-01"
    )

    # --------------------------------------------------------
    # STRICT TEMPORAL TRAINING
    # --------------------------------------------------------

    train = df[
        (df["outcome_date"] < test_start)
        &
        (df["prediction_date"] < test_start)
    ].copy()

    test = df[
        (df["prediction_date"] >= test_start)
        &
        (df["prediction_date"] < test_end)
    ].copy()

    print("\n" + "=" * 80)
    print(
        f"TEST YEAR: {test_year}"
    )
    print("=" * 80)

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Test : {len(test):,}"
    )

    train_prevalence = (
        train[TARGET].mean()
    )

    # --------------------------------------------------------
    # CONSTANT
    # --------------------------------------------------------

    baseline_prob = np.full(
        len(test),
        train_prevalence,
    )

    metrics = evaluate(
        test[TARGET],
        baseline_prob,
    )

    results.append(
        {
            "test_year": test_year,
            "architecture": "constant",
            "n_train": len(train),
            "n_test": len(test),
            **metrics,
        }
    )

    pred = test[
        [
            "ticker",
            "prediction_date",
            "outcome_date",
            TARGET,
        ]
    ].copy()

    pred[
        "architecture"
    ] = "constant"

    pred[
        "predicted_miscoverage"
    ] = baseline_prob

    predictions.append(
        pred
    )

    # --------------------------------------------------------
    # SIGNAL ARCHITECTURES
    # --------------------------------------------------------

    for name, features in (
        ARCHITECTURES.items()
    ):

        if name == "constant":
            continue

        train_model = (
            train
            .dropna(
                subset=features + [TARGET]
            )
            .copy()
        )

        test_model = (
            test
            .dropna(
                subset=features + [TARGET]
            )
            .copy()
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  train usable = "
            f"{len(train_model):,}"
        )

        print(
            f"  test usable  = "
            f"{len(test_model):,}"
        )

        model = build_model()

        model.fit(
            train_model[features],
            train_model[TARGET].astype(int),
        )

        probability = (
            model.predict_proba(
                test_model[features]
            )[:, 1]
        )

        metrics = evaluate(
            test_model[TARGET],
            probability,
        )

        print(
            f"  Brier   = "
            f"{metrics['brier']:.6f}"
        )

        print(
            f"  LogLoss = "
            f"{metrics['logloss']:.6f}"
        )

        print(
            f"  AUC     = "
            f"{metrics['auc']:.6f}"
        )

        print(
            f"  PR-AUC  = "
            f"{metrics['pr_auc']:.6f}"
        )

        results.append(
            {
                "test_year": test_year,
                "architecture": name,
                "n_train": len(train_model),
                "n_test": len(test_model),
                **metrics,
            }
        )

        pred = test_model[
            [
                "ticker",
                "prediction_date",
                "outcome_date",
                TARGET,
            ]
        ].copy()

        pred[
            "architecture"
        ] = name

        pred[
            "predicted_miscoverage"
        ] = probability

        predictions.append(
            pred
        )


# ============================================================
# DATAFRAMES
# ============================================================

results = pd.DataFrame(
    results
)

predictions = pd.concat(
    predictions,
    ignore_index=True,
)


# ============================================================
# YEARLY BRIER IMPROVEMENT
# ============================================================

baseline = (
    results[
        results["architecture"]
        == "constant"
    ]
    .set_index("test_year")[
        "brier"
    ]
)

results[
    "brier_improvement"
] = np.nan

for i, row in results.iterrows():

    if row["architecture"] == "constant":
        continue

    results.loc[
        i,
        "brier_improvement"
    ] = (
        baseline.loc[
            row["test_year"]
        ]
        - row["brier"]
    )


# ============================================================
# DAILY BRIER LOSS
# ============================================================

predictions["squared_error"] = (
    predictions[TARGET]
    - predictions[
        "predicted_miscoverage"
    ]
) ** 2

daily = (
    predictions
    .groupby(
        [
            "prediction_date",
            "architecture",
        ]
    )[
        "squared_error"
    ]
    .mean()
    .reset_index()
)

pivot = daily.pivot(
    index="prediction_date",
    columns="architecture",
    values="squared_error",
).sort_index()


# ============================================================
# BLOCK BOOTSTRAP
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)


def block_bootstrap(
    differences,
    block_size=20,
    n_bootstrap=5000,
):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    blocks = [
        differences[i:i + block_size]
        for i in range(
            0,
            len(differences),
            block_size,
        )
    ]

    blocks = [
        block
        for block in blocks
        if len(block)
        == block_size
    ]

    block_means = np.array(
        [
            block.mean()
            for block in blocks
        ]
    )

    n_blocks = len(
        block_means
    )

    boot = np.empty(
        n_bootstrap
    )

    for i in range(
        n_bootstrap
    ):

        sample = rng.integers(
            0,
            n_blocks,
            size=n_blocks,
        )

        boot[i] = (
            block_means[
                sample
            ].mean()
        )

    observed = (
        differences.mean()
    )

    low, high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    return (
        observed,
        low,
        high,
        np.mean(boot > 0),
    )


# ============================================================
# SIGNIFICANCE COMPARISONS
# ============================================================

comparisons = [
    ("constant", "D"),
    ("constant", "PS"),
    ("D", "D+PS"),
]

significance = []

for reference, challenger in comparisons:

    pair = pivot[
        [reference, challenger]
    ].dropna()

    # Positive means challenger has
    # LOWER Brier loss.
    improvement = (
        pair[reference]
        - pair[challenger]
    )

    observed, low, high, probability = (
        block_bootstrap(
            improvement.values,
            block_size=BLOCK_SIZE,
            n_bootstrap=BOOTSTRAPS,
        )
    )

    significance.append(
        {
            "reference": reference,
            "challenger": challenger,
            "n_dates": len(pair),
            "mean_brier_improvement":
                observed,
            "ci_2.5": low,
            "ci_97.5": high,
            "probability_improvement_gt_0":
                probability,
        }
    )

significance = pd.DataFrame(
    significance
)


# ============================================================
# PRINT
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

print("\n")
print("=" * 100)
print("YEARLY RESULTS")
print("=" * 100)

print(
    results[
        [
            "test_year",
            "architecture",
            "n_test",
            "brier",
            "brier_improvement",
            "logloss",
            "auc",
            "pr_auc",
        ]
    ]
    .sort_values(
        [
            "test_year",
            "architecture",
        ]
    )
    .to_string(
        index=False
    )
)

print("\n")
print("=" * 100)
print("BLOCK BOOTSTRAP INCREMENTAL TEST")
print("=" * 100)

print(
    significance.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# POOLED
# ============================================================

pooled = []

for architecture in (
    predictions[
        "architecture"
    ].unique()
):

    subset = predictions[
        predictions[
            "architecture"
        ] == architecture
    ]

    metrics = evaluate(
        subset[TARGET],
        subset[
            "predicted_miscoverage"
        ],
    )

    pooled.append(
        {
            "architecture":
                architecture,
            "n":
                len(subset),
            **metrics,
        }
    )

pooled = pd.DataFrame(
    pooled
)

print("\n")
print("=" * 100)
print("POOLED OOS RESULTS")
print("=" * 100)

print(
    pooled.sort_values(
        "brier"
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUT_DIR
    / "d_ps_yearly_results.csv",
    index=False,
)

significance.to_csv(
    OUT_DIR
    / "d_ps_significance.csv",
    index=False,
)

pooled.to_csv(
    OUT_DIR
    / "d_ps_pooled_results.csv",
    index=False,
)

predictions.to_csv(
    OUT_DIR
    / "d_ps_oos_predictions.csv",
    index=False,
)

print("\nSaved:")
print(
    OUT_DIR
    / "d_ps_yearly_results.csv"
)

print(
    OUT_DIR
    / "d_ps_significance.csv"
)

print(
    OUT_DIR
    / "d_ps_pooled_results.csv"
)