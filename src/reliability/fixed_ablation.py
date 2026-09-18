from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

DATA_PATH = ROOT / "data" / "reliability" / "final_candidate_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "reliability" / "fixed_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_YEARS = [2021, 2022, 2023, 2024, 2025, 2026]

TARGET = "miscovered"

SIGNALS = {
    "D": ["disagreement_pct"],
    "O": ["novelty_pct"],
    "M": ["ewma_miscoverage_pct"],
    "W": ["interval_width_pct"],
}

# Fixed architectures.
# These are NOT selected using test performance.
ARCHITECTURES = {
    "constant": [],
    "D": SIGNALS["D"],
    "D+O": SIGNALS["D"] + SIGNALS["O"],
    "D+M": SIGNALS["D"] + SIGNALS["M"],
    "D+W": SIGNALS["D"] + SIGNALS["W"],
    "D+O+M+W": (
        SIGNALS["D"]
        + SIGNALS["O"]
        + SIGNALS["M"]
        + SIGNALS["W"]
    ),
}


# ============================================================
# HELPERS
# ============================================================

def build_model():
    """
    Same fixed meta-model for every architecture.

    Standardization is fit ONLY on the training fold.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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


def evaluate_predictions(y_true, y_prob):
    """
    Evaluate probabilistic miscoverage predictions.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    result = {
        "brier": brier_score_loss(y_true, y_prob),
        "logloss": log_loss(
            y_true,
            np.clip(y_prob, 1e-6, 1 - 1e-6),
            labels=[0, 1],
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["auc"] = roc_auc_score(y_true, y_prob)
        result["pr_auc"] = average_precision_score(y_true, y_prob)
    else:
        result["auc"] = np.nan
        result["pr_auc"] = np.nan

    return result


# ============================================================
# LOAD DATA
# ============================================================

print(f"Loading: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

df["prediction_date"] = pd.to_datetime(df["prediction_date"])
df["outcome_date"] = pd.to_datetime(df["outcome_date"])

df = df.sort_values(
    ["prediction_date", "ticker"]
).reset_index(drop=True)

required_columns = {
    "ticker",
    "prediction_date",
    "outcome_date",
    TARGET,
    "disagreement_pct",
    "novelty_pct",
    "ewma_miscoverage_pct",
    "interval_width_pct",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

print(f"Rows: {len(df):,}")
print(f"Tickers: {df['ticker'].nunique()}")
print(
    f"Date range: "
    f"{df['prediction_date'].min().date()} -> "
    f"{df['prediction_date'].max().date()}"
)


# ============================================================
# FIXED WALK-FORWARD EVALUATION
# ============================================================

all_results = []
all_predictions = []

for test_year in TEST_YEARS:

    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Training labels must actually be observable before the
    # beginning of the test period.
    #
    # Therefore:
    #
    #   outcome_date < test_start
    #
    # is required for training.
    # --------------------------------------------------------

    train = df[
        (df["outcome_date"] < test_start)
        & (df["prediction_date"] < test_start)
    ].copy()

    test = df[
        (df["prediction_date"] >= test_start)
        & (df["prediction_date"] < test_end)
    ].copy()

    print("\n" + "=" * 70)
    print(f"OUTER TEST YEAR: {test_year}")
    print("=" * 70)

    print(f"Train rows: {len(train):,}")
    print(f"Test rows : {len(test):,}")

    if len(train) == 0 or len(test) == 0:
        print("Skipping: insufficient train/test rows.")
        continue

    train_prevalence = train[TARGET].mean()
    test_prevalence = test[TARGET].mean()

    print(f"Train prevalence: {train_prevalence:.6f}")
    print(f"Test prevalence : {test_prevalence:.6f}")

    # --------------------------------------------------------
    # CONSTANT BASELINE
    # --------------------------------------------------------

    baseline_prob = np.full(
        len(test),
        train_prevalence,
        dtype=float,
    )

    baseline_metrics = evaluate_predictions(
        test[TARGET],
        baseline_prob,
    )

    baseline_record = {
        "test_year": test_year,
        "architecture": "constant",
        "features": "",
        "n_train": len(train),
        "n_test": len(test),
        "train_prevalence": train_prevalence,
        "test_prevalence": test_prevalence,
        **baseline_metrics,
    }

    all_results.append(baseline_record)

    baseline_predictions = test[
        ["ticker", "prediction_date", "outcome_date", TARGET]
    ].copy()

    baseline_predictions["test_year"] = test_year
    baseline_predictions["architecture"] = "constant"
    baseline_predictions["predicted_miscoverage"] = baseline_prob

    all_predictions.append(baseline_predictions)

    # --------------------------------------------------------
    # FIXED SIGNAL ARCHITECTURES
    # --------------------------------------------------------

    for architecture_name, features in ARCHITECTURES.items():

        if architecture_name == "constant":
            continue

        print(
            f"\nEvaluating {architecture_name}: "
            f"{features}"
        )

        # Keep only complete rows for THIS architecture.
        #
        # This means the usable sample can differ slightly
        # between architectures because some signals have
        # different causal warm-up periods.
        train_model = train.dropna(
            subset=features + [TARGET]
        ).copy()

        test_model = test.dropna(
            subset=features + [TARGET]
        ).copy()

        print(
            f"  train usable: {len(train_model):,} | "
            f"test usable: {len(test_model):,}"
        )

        if len(train_model) == 0 or len(test_model) == 0:
            print("  Skipping: no usable rows.")
            continue

        X_train = train_model[features]
        y_train = train_model[TARGET].astype(int)

        X_test = test_model[features]
        y_test = test_model[TARGET].astype(int)

        # ----------------------------------------------------
        # FIXED MODEL
        # ----------------------------------------------------

        model = build_model()

        model.fit(X_train, y_train)

        test_prob = model.predict_proba(X_test)[:, 1]

        metrics = evaluate_predictions(
            y_test,
            test_prob,
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        record = {
            "test_year": test_year,
            "architecture": architecture_name,
            "features": ",".join(features),
            "n_train": len(train_model),
            "n_test": len(test_model),
            "train_prevalence": y_train.mean(),
            "test_prevalence": y_test.mean(),
            **metrics,
        }

        all_results.append(record)

        # ----------------------------------------------------
        # STORE PREDICTIONS
        # ----------------------------------------------------

        pred = test_model[
            [
                "ticker",
                "prediction_date",
                "outcome_date",
                TARGET,
            ]
        ].copy()

        pred["test_year"] = test_year
        pred["architecture"] = architecture_name
        pred["predicted_miscoverage"] = test_prob

        all_predictions.append(pred)

        print(
            f"  Brier   : {metrics['brier']:.6f}"
        )
        print(
            f"  LogLoss : {metrics['logloss']:.6f}"
        )
        print(
            f"  AUC     : {metrics['auc']:.6f}"
        )
        print(
            f"  PR-AUC  : {metrics['pr_auc']:.6f}"
        )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(all_results)

predictions = pd.concat(
    all_predictions,
    ignore_index=True,
)


# ============================================================
# ADD DELTAS VS CONSTANT BASELINE
# ============================================================

baseline_lookup = (
    results[results["architecture"] == "constant"]
    .set_index("test_year")
)

results["brier_vs_constant"] = np.nan
results["brier_improvement_pct"] = np.nan

for idx, row in results.iterrows():

    if row["architecture"] == "constant":
        continue

    year = row["test_year"]

    if year in baseline_lookup.index:

        baseline_brier = baseline_lookup.loc[
            year, "brier"
        ]

        results.loc[
            idx,
            "brier_vs_constant"
        ] = row["brier"] - baseline_brier

        results.loc[
            idx,
            "brier_improvement_pct"
        ] = (
            (baseline_brier - row["brier"])
            / baseline_brier
            * 100
        )


# ============================================================
# MACRO YEAR SUMMARY
# ============================================================

summary = (
    results[
        results["architecture"] != "constant"
    ]
    .groupby("architecture")
    .agg(
        mean_brier=("brier", "mean"),
        median_brier=("brier", "median"),
        mean_logloss=("logloss", "mean"),
        mean_auc=("auc", "mean"),
        median_auc=("auc", "median"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_brier_improvement_pct=(
            "brier_improvement_pct",
            "mean",
        ),
        positive_brier_years=(
            "brier_improvement_pct",
            lambda x: (x > 0).sum(),
        ),
        total_years=(
            "test_year",
            "count",
        ),
    )
    .reset_index()
)

summary["positive_year_fraction"] = (
    summary["positive_brier_years"]
    / summary["total_years"]
)


# ============================================================
# POOLED OUT-OF-SAMPLE RESULTS
# ============================================================

pooled_rows = []

for architecture in predictions["architecture"].unique():

    subset = predictions[
        predictions["architecture"] == architecture
    ].copy()

    metrics = evaluate_predictions(
        subset[TARGET],
        subset["predicted_miscoverage"],
    )

    pooled_rows.append(
        {
            "architecture": architecture,
            "n_predictions": len(subset),
            **metrics,
        }
    )

pooled = pd.DataFrame(pooled_rows)


# ============================================================
# PRINT
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n")
print("=" * 100)
print("YEAR-BY-YEAR RESULTS")
print("=" * 100)

print(
    results[
        [
            "test_year",
            "architecture",
            "n_test",
            "brier",
            "brier_improvement_pct",
            "logloss",
            "auc",
            "pr_auc",
        ]
    ]
    .sort_values(
        ["test_year", "architecture"]
    )
    .to_string(index=False)
)

print("\n")
print("=" * 100)
print("MACRO ARCHITECTURE SUMMARY")
print("=" * 100)

print(
    summary[
        [
            "architecture",
            "mean_brier",
            "median_brier",
            "mean_logloss",
            "mean_auc",
            "median_auc",
            "mean_pr_auc",
            "mean_brier_improvement_pct",
            "positive_brier_years",
            "total_years",
            "positive_year_fraction",
        ]
    ]
    .sort_values("mean_brier")
    .to_string(index=False)
)

print("\n")
print("=" * 100)
print("POOLED OUT-OF-SAMPLE RESULTS")
print("=" * 100)

print(
    pooled
    .sort_values("brier")
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

results_path = OUTPUT_DIR / "yearly_results.csv"
summary_path = OUTPUT_DIR / "architecture_summary.csv"
pooled_path = OUTPUT_DIR / "pooled_results.csv"
predictions_path = OUTPUT_DIR / "oos_predictions.csv"

results.to_csv(results_path, index=False)
summary.to_csv(summary_path, index=False)
pooled.to_csv(pooled_path, index=False)
predictions.to_csv(predictions_path, index=False)

print("\nSaved:")
print(f"  {results_path}")
print(f"  {summary_path}")
print(f"  {pooled_path}")
print(f"  {predictions_path}")