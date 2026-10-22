from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss


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
OUTPUT = (
    OUTPUT_DIR
    / f"{TICKER}_learned_monitor_results.csv"
)

ALPHA = 0.05

HORIZONS = [5, 10, 20]

TRAIN_FRAC = 0.60
VALID_FRAC = 0.20

EMBARGO = 20


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(df["date"])

df = (
    df.sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# Chronological split
# ============================================================

def make_splits(data):

    n = len(data)

    train_end = int(
        n * TRAIN_FRAC
    )

    valid_end = int(
        n * (TRAIN_FRAC + VALID_FRAC)
    )

    train = data.iloc[:train_end].copy()

    valid = data.iloc[
        train_end + EMBARGO:
        valid_end
    ].copy()

    test = data.iloc[
        valid_end + EMBARGO:
    ].copy()

    return train, valid, test


train_base, valid_base, test_base = make_splits(df)


print("=" * 80)
print("LEARNED MONITOR — AAPL HS VaR")
print("=" * 80)

print("Total rows:", len(df))

print(
    "Train:",
    len(train_base)
)

print(
    "Validation:",
    len(valid_base)
)

print(
    "Test:",
    len(test_base)
)


# ============================================================
# Observable feature sets
# ============================================================

# ------------------------------------------------------------
# The strongest simple benchmark
# ------------------------------------------------------------

breach_baseline = [
    "breach_rate_20d"
]


# ------------------------------------------------------------
# Volatility baseline
# ------------------------------------------------------------

volatility_baseline = [
    "realized_vol_20d"
]


# ------------------------------------------------------------
# Expanded observable features
# ------------------------------------------------------------

expanded_features = [
    # Recent violations
    "breach_rate_20d",
    "breach_rate_60d",
    "breach_rate_120d",

    # Realized volatility
    "realized_vol_20d",
    "realized_vol_60d",
    "realized_vol_120d",

    # Violation severity
    "severity_20d",
    "severity_60d",
    "severity_120d",

    # Quantile forecast loss
    "pinball_20d",
    "pinball_60d",
    "pinball_120d",

    # VaR behaviour
    "var_abs",
    "var_change",

    # Current observation
    "breach_severity",
    "pinball_loss",
]


# ============================================================
# Models
# ============================================================

models = {

    "logistic": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                C=1.0,
                random_state=42
            )
        ),
    ]),

    "gradient_boosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.03,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42
    ),
}


# ============================================================
# Results container
# ============================================================

results = []


# ============================================================
# Run experiments
# ============================================================

for horizon in HORIZONS:

    print("\n" + "=" * 80)
    print(f"HORIZON = {horizon} DAYS")
    print("=" * 80)

    target_column = (
        f"future_pinball_{horizon}d"
    )

    # --------------------------------------------------------
    # Create target threshold using TRAIN ONLY
    # --------------------------------------------------------

    threshold = train_base[
        target_column
    ].quantile(0.75)

    train = train_base.copy()
    valid = valid_base.copy()
    test = test_base.copy()

    train["target"] = (
        train[target_column] > threshold
    ).astype(int)

    valid["target"] = (
        valid[target_column] > threshold
    ).astype(int)

    test["target"] = (
        test[target_column] > threshold
    ).astype(int)

    print(
        "Threshold:",
        threshold
    )

    print(
        "Train positive rate:",
        train["target"].mean()
    )

    print(
        "Validation positive rate:",
        valid["target"].mean()
    )

    print(
        "Test positive rate:",
        test["target"].mean()
    )

    # ========================================================
    # 1. Raw breach-rate baseline
    # ========================================================

    baseline_score = test[
        "breach_rate_20d"
    ].to_numpy()

    baseline_auc = roc_auc_score(
        test["target"],
        baseline_score
    )

    baseline_brier = brier_score_loss(
        test["target"],
        baseline_score
    )

    results.append({
        "horizon": horizon,
        "model": "breach_rate_20d",
        "test_auc": baseline_auc,
        "test_brier": baseline_brier,
        "delta_auc_vs_breach_rate": 0.0,
    })

    print(
        f"{'breach_rate_20d':25s} "
        f"AUC={baseline_auc:.4f} "
        f"Brier={baseline_brier:.6f}"
    )

    # ========================================================
    # 2. Volatility baseline
    # ========================================================

    volatility_score = test[
        "realized_vol_20d"
    ].to_numpy()

    volatility_auc = roc_auc_score(
        test["target"],
        volatility_score
    )

    # Normalize volatility for Brier score.
    vol_min = train[
        "realized_vol_20d"
    ].min()

    vol_max = train[
        "realized_vol_20d"
    ].max()

    volatility_prob = (
        volatility_score - vol_min
    ) / (
        vol_max - vol_min + 1e-12
    )

    volatility_prob = np.clip(
        volatility_prob,
        0.0,
        1.0
    )

    volatility_brier = brier_score_loss(
        test["target"],
        volatility_prob
    )

    results.append({
        "horizon": horizon,
        "model": "volatility_20d",
        "test_auc": volatility_auc,
        "test_brier": volatility_brier,
        "delta_auc_vs_breach_rate":
            volatility_auc - baseline_auc,
    })

    print(
        f"{'volatility_20d':25s} "
        f"AUC={volatility_auc:.4f} "
        f"Brier={volatility_brier:.6f}"
    )

    # ========================================================
    # 3. Logistic regression
    # ========================================================

    features = expanded_features

    X_train = train[features]
    y_train = train["target"]

    X_valid = valid[features]
    y_valid = valid["target"]

    X_test = test[features]
    y_test = test["target"]

    model = models["logistic"]

    model.fit(
        X_train,
        y_train
    )

    train_prob = model.predict_proba(
        X_train
    )[:, 1]

    valid_prob = model.predict_proba(
        X_valid
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    logistic_auc = roc_auc_score(
        y_test,
        test_prob
    )

    logistic_brier = brier_score_loss(
        y_test,
        test_prob
    )

    results.append({
        "horizon": horizon,
        "model": "logistic_expanded",
        "test_auc": logistic_auc,
        "test_brier": logistic_brier,
        "delta_auc_vs_breach_rate":
            logistic_auc - baseline_auc,
    })

    print(
        f"{'logistic_expanded':25s} "
        f"AUC={logistic_auc:.4f} "
        f"Brier={logistic_brier:.6f} "
        f"Delta={logistic_auc - baseline_auc:+.4f}"
    )

    # ========================================================
    # 4. Gradient boosting
    # ========================================================

    model = models[
        "gradient_boosting"
    ]

    model.fit(
        X_train,
        y_train
    )

    train_prob = model.predict_proba(
        X_train
    )[:, 1]

    valid_prob = model.predict_proba(
        X_valid
    )[:, 1]

    test_prob = model.predict_proba(
        X_test
    )[:, 1]

    boosting_auc = roc_auc_score(
        y_test,
        test_prob
    )

    boosting_brier = brier_score_loss(
        y_test,
        test_prob
    )

    results.append({
        "horizon": horizon,
        "model": "gradient_boosting_expanded",
        "test_auc": boosting_auc,
        "test_brier": boosting_brier,
        "delta_auc_vs_breach_rate":
            boosting_auc - baseline_auc,
    })

    print(
        f"{'gradient_boosting':25s} "
        f"AUC={boosting_auc:.4f} "
        f"Brier={boosting_brier:.6f} "
        f"Delta={boosting_auc - baseline_auc:+.4f}"
    )

    # ========================================================
    # Feature importance
    # ========================================================

    if hasattr(
        model,
        "feature_importances_"
    ):

        importance = pd.DataFrame({
            "feature": features,
            "importance":
                model.feature_importances_
        })

        importance = (
            importance
            .sort_values(
                "importance",
                ascending=False
            )
        )

        print("\nTop gradient-boosting features:")

        print(
            importance
            .head(10)
            .to_string(index=False)
        )


# ============================================================
# Results dataframe
# ============================================================

results_df = pd.DataFrame(results)


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
# Final table
# ============================================================

print("\n" + "=" * 80)
print("FINAL TEST RESULTS")
print("=" * 80)

print(
    results_df
    .sort_values(
        ["horizon", "test_auc"],
        ascending=[
            True,
            False
        ]
    )
    .to_string(index=False)
)


# ============================================================
# Go / no-go summary
# ============================================================

print("\n" + "=" * 80)
print("GO / NO-GO SUMMARY")
print("=" * 80)

for horizon in HORIZONS:

    subset = results_df[
        results_df["horizon"] == horizon
    ]

    baseline = subset[
        subset["model"]
        == "breach_rate_20d"
    ]["test_auc"].iloc[0]

    learned = subset[
        subset["model"]
        == "gradient_boosting_expanded"
    ]["test_auc"].iloc[0]

    delta = learned - baseline

    print(
        f"{horizon:2d}d: "
        f"baseline={baseline:.4f}, "
        f"learned={learned:.4f}, "
        f"delta={delta:+.4f}"
    )


print("\n" + "=" * 80)
print("OUTPUT")
print("=" * 80)

print(
    "Saved to:",
    OUTPUT
)