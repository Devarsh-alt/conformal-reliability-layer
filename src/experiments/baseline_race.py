from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TICKER = "AAPL"
INPUT = (
    ROOT
    / "data"
    / "monitoring"
    / f"{TICKER}_HS_monitoring_panel.csv"
)

OUTPUT_DIR = ROOT / "data" / "monitoring"
OUTPUT = OUTPUT_DIR / f"{TICKER}_baseline_race.csv"

ALPHA = 0.05

# Forecast horizons to test
HORIZONS = [5, 10, 20]

# Train / validation / test split
TRAIN_FRAC = 0.60
VALID_FRAC = 0.20

# Embargo to prevent overlapping future labels across splits
EMBARGO = 20


# ============================================================
# Load panel
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = (
    df.sort_values("date")
    .reset_index(drop=True)
)

print("=" * 80)
print("BASELINE RACE — AAPL HS VaR")
print("=" * 80)

print("Rows:", len(df))
print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)


# ============================================================
# Feature sets
# ============================================================

# These are ALL observable at prediction time.

feature_sets = {

    # --------------------------------------------------------
    # Persistence / recent failure behaviour
    # --------------------------------------------------------
    "breach_rate": [
        "breach_rate_20d"
    ],

    # --------------------------------------------------------
    # Volatility-only baseline
    # --------------------------------------------------------
    "volatility": [
        "realized_vol_20d"
    ],

    # --------------------------------------------------------
    # VaR magnitude baseline
    # --------------------------------------------------------
    "var_level": [
        "var_abs"
    ],

    # --------------------------------------------------------
    # Severity baseline
    # --------------------------------------------------------
    "severity": [
        "severity_20d"
    ],

    # --------------------------------------------------------
    # Combined simple monitor
    # --------------------------------------------------------
    "breach_rate_plus_volatility": [
        "breach_rate_20d",
        "realized_vol_20d"
    ],

    # --------------------------------------------------------
    # Full simple observable set
    # --------------------------------------------------------
    "simple_all": [
        "breach_rate_20d",
        "realized_vol_20d",
        "severity_20d",
        "pinball_20d",
        "var_abs",
        "var_change",
    ],
}


# ============================================================
# Chronological split helper
# ============================================================

def make_splits(data, train_frac, valid_frac, embargo):

    n = len(data)

    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train = data.iloc[:train_end].copy()

    valid_start = train_end + embargo
    test_start = valid_end + embargo

    valid = data.iloc[valid_start:valid_end].copy()
    test = data.iloc[test_start:].copy()

    return train, valid, test


# ============================================================
# Main experiment
# ============================================================

results = []


for horizon in HORIZONS:

    print("\n" + "=" * 80)
    print(f"HORIZON = {horizon} DAYS")
    print("=" * 80)

    target_column = f"future_pinball_{horizon}d"

    # --------------------------------------------------------
    # Create binary deterioration target
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # We do NOT choose the deterioration threshold using
    # the entire dataset.
    #
    # The threshold is estimated using TRAINING DATA ONLY.
    #
    # Target = 1 means future pinball loss is unusually high.
    #
    # 75th percentile = "high future risk" regime.
    #

    train, valid, test = make_splits(
        df,
        TRAIN_FRAC,
        VALID_FRAC,
        EMBARGO
    )

    threshold = train[target_column].quantile(0.75)

    train["target"] = (
        train[target_column] > threshold
    ).astype(int)

    valid["target"] = (
        valid[target_column] > threshold
    ).astype(int)

    test["target"] = (
        test[target_column] > threshold
    ).astype(int)

    print("Training rows:", len(train))
    print("Validation rows:", len(valid))
    print("Test rows:", len(test))

    print(
        "Deterioration threshold:",
        threshold
    )

    print(
        "Train deterioration rate:",
        train["target"].mean()
    )

    print(
        "Validation deterioration rate:",
        valid["target"].mean()
    )

    print(
        "Test deterioration rate:",
        test["target"].mean()
    )

    # --------------------------------------------------------
    # Baseline models
    # --------------------------------------------------------

    for feature_set_name, features in feature_sets.items():

        # ----------------------------------------------------
        # Select data
        # ----------------------------------------------------

        X_train = train[features]
        y_train = train["target"]

        X_valid = valid[features]
        y_valid = valid["target"]

        X_test = test[features]
        y_test = test["target"]

        # ----------------------------------------------------
        # Logistic regression
        # ----------------------------------------------------

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        train_prob = model.predict_proba(X_train)[:, 1]
        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        # ----------------------------------------------------
        # AUC
        # ----------------------------------------------------

        train_auc = roc_auc_score(
            y_train,
            train_prob
        )

        valid_auc = roc_auc_score(
            y_valid,
            valid_prob
        )

        test_auc = roc_auc_score(
            y_test,
            test_prob
        )

        # ----------------------------------------------------
        # Simple correlation check
        # ----------------------------------------------------

        # Mean feature score
        feature_score = X_test.mean(axis=1)

        spearman_corr, spearman_p = spearmanr(
            feature_score,
            test[target_column]
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        results.append({
            "horizon": horizon,
            "feature_set": feature_set_name,
            "n_features": len(features),

            "features": ",".join(features),

            "train_auc": train_auc,
            "validation_auc": valid_auc,
            "test_auc": test_auc,

            "test_spearman":
                spearman_corr,

            "test_spearman_pvalue":
                spearman_p,

            "threshold":
                threshold,

            "train_positive_rate":
                y_train.mean(),

            "validation_positive_rate":
                y_valid.mean(),

            "test_positive_rate":
                y_test.mean(),
        })

        print(
            f"{feature_set_name:30s} "
            f"Train AUC={train_auc:.3f}  "
            f"Val AUC={valid_auc:.3f}  "
            f"Test AUC={test_auc:.3f}"
        )


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# Incremental value calculations
# ============================================================

for horizon in HORIZONS:

    mask = (
        results_df["horizon"] == horizon
    )

    volatility_auc = results_df.loc[
        mask
        & (
            results_df["feature_set"]
            == "volatility"
        ),
        "test_auc"
    ].iloc[0]

    breach_auc = results_df.loc[
        mask
        & (
            results_df["feature_set"]
            == "breach_rate"
        ),
        "test_auc"
    ].iloc[0]

    combined_auc = results_df.loc[
        mask
        & (
            results_df["feature_set"]
            == "breach_rate_plus_volatility"
        ),
        "test_auc"
    ].iloc[0]

    results_df.loc[
        mask,
        "delta_breach_vs_volatility"
    ] = (
        breach_auc - volatility_auc
    )

    results_df.loc[
        mask,
        "delta_combined_vs_volatility"
    ] = (
        combined_auc - volatility_auc
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
# Print final comparison
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST-SET COMPARISON")
print("=" * 80)

display_columns = [
    "horizon",
    "feature_set",
    "test_auc",
    "test_spearman",
    "test_spearman_pvalue",
]

print(
    results_df[
        display_columns
    ]
    .sort_values(
        ["horizon", "test_auc"],
        ascending=[True, False]
    )
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("INCREMENTAL VALUE OVER VOLATILITY")
print("=" * 80)

incremental_columns = [
    "horizon",
    "feature_set",
    "test_auc",
    "delta_breach_vs_volatility",
    "delta_combined_vs_volatility",
]

print(
    results_df[
        incremental_columns
    ]
    .drop_duplicates()
    .to_string(index=False)
)


# ============================================================
# Best model per horizon
# ============================================================

print("\n" + "=" * 80)
print("BEST TEST MODEL PER HORIZON")
print("=" * 80)

for horizon in HORIZONS:

    subset = results_df[
        results_df["horizon"] == horizon
    ]

    best = subset.loc[
        subset["test_auc"].idxmax()
    ]

    print(
        f"{horizon:2d}d: "
        f"{best['feature_set']} "
        f"AUC={best['test_auc']:.3f}"
    )


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print("Saved to:", OUTPUT)