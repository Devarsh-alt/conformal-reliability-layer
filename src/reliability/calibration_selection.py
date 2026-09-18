from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FINAL_DATA_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "final_candidate_dataset.csv"
)

PS_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "perturbation_sensitivity"
    / "robustness"
    / "ps_robustness_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "reliability"
    / "calibration_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "d_ps_oos_predictions.csv"
)

YEARLY_METRICS_PATH = (
    OUTPUT_DIR
    / "d_ps_yearly_metrics.csv"
)

POOLED_METRICS_PATH = (
    OUTPUT_DIR
    / "d_ps_pooled_metrics.csv"
)

CALIBRATION_DECILES_PATH = (
    OUTPUT_DIR
    / "d_ps_calibration_deciles.csv"
)

SELECTIVE_PATH = (
    OUTPUT_DIR
    / "d_ps_selective_prediction.csv"
)

RISK_CONCENTRATION_PATH = (
    OUTPUT_DIR
    / "d_ps_risk_concentration.csv"
)

CALIBRATION_YEARLY_PATH = (
    OUTPUT_DIR
    / "d_ps_calibration_yearly.csv"
)

CALIBRATION_PLOT_PATH = (
    OUTPUT_DIR
    / "d_ps_calibration_plot.png"
)

SELECTION_PLOT_PATH = (
    OUTPUT_DIR
    / "d_ps_selection_curve.png"
)


# ============================================================
# CONFIG
# ============================================================

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]

# Frozen primary PS configuration
PS_COL_RAW = "w60_s0.50_n30"
PS_COL = (
    PS_COL_RAW
    + "_pct"
)

D_COL = "disagreement_pct"

TARGET_COL = "miscovered"

SELECTION_FRACTIONS = [
    1.00,
    0.90,
    0.75,
    0.50,
    0.25,
    0.10,
]

CALIBRATION_BINS = 10


# ============================================================
# HELPERS
# ============================================================

def build_meta_model():
    """
    Same reliability meta-model used in the
    fixed D vs PS vs D+PS experiment.
    """

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
                    random_state=42,
                ),
            ),
        ]
    )


def safe_clip_probabilities(p):
    """
    Prevent log-loss infinities.
    """

    return np.clip(
        np.asarray(p, dtype=float),
        1e-6,
        1 - 1e-6,
    )


def calibration_slope_intercept(
    y,
    p,
):
    """
    Descriptive calibration regression:

        logit(p_observed_model)
            ->
        actual binary outcome

    Ideal:
        intercept = 0
        slope = 1

    This is diagnostic only and is NOT used to
    recalibrate the test predictions.
    """

    y = np.asarray(
        y,
        dtype=int,
    )

    p = safe_clip_probabilities(p)

    logit_p = np.log(
        p / (1 - p)
    ).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        max_iter=5000,
    )

    model.fit(
        logit_p,
        y,
    )

    intercept = float(
        model.intercept_[0]
    )

    slope = float(
        model.coef_[0][0]
    )

    return (
        intercept,
        slope,
    )


def expected_calibration_error(
    y,
    p,
    n_bins=10,
):
    """
    Equal-frequency calibration bins.

    ECE = weighted mean absolute difference between
    predicted probability and observed frequency.
    """

    data = pd.DataFrame(
        {
            "y": np.asarray(
                y,
                dtype=int,
            ),
            "p": np.asarray(
                p,
                dtype=float,
            ),
        }
    )

    data = data.sort_values(
        "p"
    ).reset_index(
        drop=True
    )

    data["bin"] = (
        pd.qcut(
            data.index,
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    grouped = (
        data
        .groupby("bin")
        .agg(
            n=("y", "size"),
            mean_pred=("p", "mean"),
            observed_rate=("y", "mean"),
        )
    )

    grouped["abs_gap"] = (
        (
            grouped["mean_pred"]
            - grouped["observed_rate"]
        )
        .abs()
    )

    ece = (
        (
            grouped["n"]
            / len(data)
        )
        * grouped["abs_gap"]
    ).sum()

    return float(ece)


def make_calibration_deciles(
    y,
    p,
    bins=10,
):
    """
    Return calibration table.
    """

    data = pd.DataFrame(
        {
            "y": np.asarray(
                y,
                dtype=int,
            ),
            "predicted_probability": np.asarray(
                p,
                dtype=float,
            ),
        }
    )

    data = data.sort_values(
        "predicted_probability"
    ).reset_index(
        drop=True
    )

    data["decile"] = (
        pd.qcut(
            data.index,
            q=bins,
            labels=False,
            duplicates="drop",
        )
        + 1
    )

    result = (
        data
        .groupby("decile")
        .agg(
            n=("y", "size"),
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_miscoverage=(
                "y",
                "mean",
            ),
            min_predicted_probability=(
                "predicted_probability",
                "min",
            ),
            max_predicted_probability=(
                "predicted_probability",
                "max",
            ),
        )
        .reset_index()
    )

    result["calibration_gap"] = (
        result[
            "observed_miscoverage"
        ]
        -
        result[
            "mean_predicted_probability"
        ]
    )

    result["absolute_gap"] = (
        result["calibration_gap"]
        .abs()
    )

    return result


def selective_prediction_table(
    y,
    p,
    fractions,
):
    """
    Retain the lowest predicted-risk observations.

    This answers:
        "How much can we reduce realized miscoverage
         if we abstain from the highest-risk predictions?"
    """

    data = pd.DataFrame(
        {
            "y": np.asarray(
                y,
                dtype=int,
            ),
            "p": np.asarray(
                p,
                dtype=float,
            ),
        }
    )

    data = data.sort_values(
        "p",
        ascending=True,
    ).reset_index(
        drop=True
    )

    baseline_rate = (
        data["y"]
        .mean()
    )

    rows = []

    for fraction in fractions:

        n_retain = max(
            1,
            int(
                np.floor(
                    len(data)
                    * fraction
                )
            ),
        )

        retained = data.iloc[
            :n_retain
        ]

        retained_rate = (
            retained["y"]
            .mean()
        )

        relative_reduction = (
            1
            -
            retained_rate
            / baseline_rate
        )

        rows.append(
            {
                "retained_fraction": fraction,
                "discarded_fraction": (
                    1 - fraction
                ),
                "n_retained": len(retained),
                "threshold_max_risk_retained": (
                    retained["p"].max()
                ),
                "mean_predicted_risk_retained": (
                    retained["p"].mean()
                ),
                "realized_miscoverage_retained": (
                    retained_rate
                ),
                "baseline_miscoverage": (
                    baseline_rate
                ),
                "relative_miscoverage_reduction": (
                    relative_reduction
                ),
                "coverage_retained": (
                    1 - retained_rate
                ),
                "miscovered_count_retained": (
                    int(
                        retained["y"].sum()
                    )
                ),
                "miscovered_count_total": (
                    int(
                        data["y"].sum()
                    )
                ),
                "fraction_of_failures_retained": (
                    retained["y"].sum()
                    /
                    data["y"].sum()
                    if data["y"].sum() > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def risk_concentration_table(
    y,
    p,
):
    """
    Concentration of all failures inside the
    highest-risk tail.

    Example:
        top 10% risk captures X% of all failures.
    """

    data = pd.DataFrame(
        {
            "y": np.asarray(
                y,
                dtype=int,
            ),
            "p": np.asarray(
                p,
                dtype=float,
            ),
        }
    )

    data = data.sort_values(
        "p",
        ascending=False,
    ).reset_index(
        drop=True
    )

    total_failures = int(
        data["y"].sum()
    )

    fractions = [
        0.01,
        0.05,
        0.10,
        0.20,
        0.25,
        0.50,
    ]

    rows = []

    for fraction in fractions:

        n = max(
            1,
            int(
                np.ceil(
                    len(data)
                    * fraction
                )
            ),
        )

        high_risk = data.iloc[
            :n
        ]

        failures = int(
            high_risk["y"].sum()
        )

        rows.append(
            {
                "top_risk_fraction": fraction,
                "n_predictions": n,
                "mean_predicted_risk": (
                    high_risk["p"].mean()
                ),
                "realized_miscoverage": (
                    high_risk["y"].mean()
                ),
                "failures_captured": failures,
                "total_failures": total_failures,
                "fraction_of_all_failures_captured": (
                    failures
                    / total_failures
                    if total_failures > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("D + PS CALIBRATION / SELECTION ANALYSIS")
print("=" * 80)

df = pd.read_csv(
    FINAL_DATA_PATH
)

ps = pd.read_csv(
    PS_PATH
)

# ------------------------------------------------------------
# Dates
# ------------------------------------------------------------

df["prediction_date"] = pd.to_datetime(
    df["prediction_date"],
    errors="coerce",
)

df["outcome_date"] = pd.to_datetime(
    df["outcome_date"],
    errors="coerce",
)

ps["prediction_date"] = pd.to_datetime(
    ps["prediction_date"],
    errors="coerce",
)

# ------------------------------------------------------------
# Remove invalid rows
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        "prediction_date",
        "outcome_date",
    ]
)

ps = ps.dropna(
    subset=[
        "prediction_date",
    ]
)

# ------------------------------------------------------------
# Keep exactly the frozen PS configuration
# ------------------------------------------------------------

required_ps_cols = [
    "ticker",
    "prediction_date",
    PS_COL,
]

missing = [
    c
    for c in required_ps_cols
    if c not in ps.columns
]

if missing:

    raise RuntimeError(
        "Missing PS columns: "
        + str(missing)
    )

ps = ps[
    required_ps_cols
].drop_duplicates(
    subset=[
        "ticker",
        "prediction_date",
    ]
)

# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------

df = df.merge(
    ps,
    on=[
        "ticker",
        "prediction_date",
    ],
    how="inner",
    validate="one_to_one",
)

# ------------------------------------------------------------
# Required reliability columns
# ------------------------------------------------------------

required_cols = [
    D_COL,
    PS_COL,
    TARGET_COL,
]

missing = [
    c
    for c in required_cols
    if c not in df.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns: "
        + str(missing)
    )

# ------------------------------------------------------------
# Clean
# ------------------------------------------------------------

df = df.dropna(
    subset=[
        D_COL,
        PS_COL,
        TARGET_COL,
    ]
).copy()

df[TARGET_COL] = (
    df[TARGET_COL]
    .astype(int)
)

df = df.sort_values(
    [
        "prediction_date",
        "ticker",
    ]
).reset_index(
    drop=True
)

print(
    f"Usable rows: {len(df):,}"
)

print(
    f"PS configuration: {PS_COL_RAW}"
)


# ============================================================
# OUT-OF-SAMPLE PREDICTIONS
# ============================================================

print("\n")
print("=" * 80)
print("GENERATING STRICT OOS PREDICTIONS")
print("=" * 80)

oos_predictions = []

for test_year in TEST_YEARS:

    test_start = pd.Timestamp(
        f"{test_year}-01-01"
    )

    test_end = pd.Timestamp(
        f"{test_year + 1}-01-01"
    )

    # --------------------------------------------------------
    # Strict training chronology
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

    train = train.dropna(
        subset=[
            D_COL,
            PS_COL,
            TARGET_COL,
        ]
    )

    test = test.dropna(
        subset=[
            D_COL,
            PS_COL,
            TARGET_COL,
        ]
    )

    print(
        f"\n{test_year}"
    )

    print(
        f"Train: {len(train):,}"
    )

    print(
        f"Test : {len(test):,}"
    )

    if train.empty or test.empty:

        print(
            "Skipping year."
        )

        continue

    if (
        train[TARGET_COL]
        .nunique()
        < 2
    ):

        print(
            "Training target has <2 classes."
        )

        continue

    # --------------------------------------------------------
    # Frozen D + PS reliability model
    # --------------------------------------------------------

    model = build_meta_model()

    model.fit(
        train[
            [
                D_COL,
                PS_COL,
            ]
        ],
        train[
            TARGET_COL
        ],
    )

    probabilities = (
        model
        .predict_proba(
            test[
                [
                    D_COL,
                    PS_COL,
                ]
            ]
        )[:, 1]
    )

    probabilities = safe_clip_probabilities(
        probabilities
    )

    result = test[
        [
            "ticker",
            "prediction_date",
            "outcome_date",
            TARGET_COL,
            D_COL,
            PS_COL,
        ]
    ].copy()

    result["test_year"] = test_year

    result[
        "predicted_miscoverage_probability"
    ] = probabilities

    oos_predictions.append(
        result
    )

    # --------------------------------------------------------
    # Quick yearly metrics
    # --------------------------------------------------------

    y = test[
        TARGET_COL
    ].astype(int)

    brier = brier_score_loss(
        y,
        probabilities,
    )

    ll = log_loss(
        y,
        probabilities,
    )

    auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    print(
        f"Brier   = {brier:.6f}"
    )

    print(
        f"LogLoss = {ll:.6f}"
    )

    print(
        f"AUC     = {auc:.6f}"
    )

    print(
        f"PR-AUC  = {pr_auc:.6f}"
    )


# ============================================================
# COMBINE PREDICTIONS
# ============================================================

if not oos_predictions:

    raise RuntimeError(
        "No OOS predictions generated."
    )

oos = pd.concat(
    oos_predictions,
    ignore_index=True,
)

oos = oos.sort_values(
    [
        "prediction_date",
        "ticker",
    ]
).reset_index(
    drop=True
)

print("\n")
print(
    f"Total OOS predictions: "
    f"{len(oos):,}"
)


# ============================================================
# SAVE PREDICTION-LEVEL OUTPUT
# ============================================================

oos.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


# ============================================================
# POOLED METRICS
# ============================================================

print("\n")
print("=" * 80)
print("POOLED OOS CALIBRATION METRICS")
print("=" * 80)

y = oos[
    TARGET_COL
].astype(int)

p = safe_clip_probabilities(
    oos[
        "predicted_miscoverage_probability"
    ]
)

overall_brier = brier_score_loss(
    y,
    p,
)

overall_logloss = log_loss(
    y,
    p,
)

overall_auc = roc_auc_score(
    y,
    p,
)

overall_pr_auc = (
    average_precision_score(
        y,
        p,
    )
)

ece = expected_calibration_error(
    y,
    p,
    n_bins=CALIBRATION_BINS,
)

cal_intercept, cal_slope = (
    calibration_slope_intercept(
        y,
        p,
    )
)

actual_rate = y.mean()
predicted_mean = p.mean()

pooled_metrics = pd.DataFrame(
    [
        {
            "n": len(oos),
            "actual_miscoverage": actual_rate,
            "mean_predicted_probability": predicted_mean,
            "brier": overall_brier,
            "logloss": overall_logloss,
            "auc": overall_auc,
            "pr_auc": overall_pr_auc,
            "ece": ece,
            "calibration_intercept": cal_intercept,
            "calibration_slope": cal_slope,
        }
    ]
)

print(
    pooled_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

pooled_metrics.to_csv(
    POOLED_METRICS_PATH,
    index=False,
)


# ============================================================
# YEARLY METRICS + CALIBRATION
# ============================================================

print("\n")
print("=" * 80)
print("YEARLY CALIBRATION")
print("=" * 80)

yearly_rows = []

for year in TEST_YEARS:

    subset = oos[
        oos["test_year"] == year
    ].copy()

    if subset.empty:
        continue

    yy = subset[
        TARGET_COL
    ].astype(int)

    pp = safe_clip_probabilities(
        subset[
            "predicted_miscoverage_probability"
        ]
    )

    year_brier = (
        brier_score_loss(
            yy,
            pp,
        )
    )

    year_ll = (
        log_loss(
            yy,
            pp,
        )
    )

    year_auc = (
        roc_auc_score(
            yy,
            pp,
        )
    )

    year_pr_auc = (
        average_precision_score(
            yy,
            pp,
        )
    )

    year_ece = (
        expected_calibration_error(
            yy,
            pp,
            n_bins=CALIBRATION_BINS,
        )
    )

    intercept, slope = (
        calibration_slope_intercept(
            yy,
            pp,
        )
    )

    row = {
        "test_year": year,
        "n": len(subset),
        "actual_miscoverage": yy.mean(),
        "mean_predicted_probability": pp.mean(),
        "brier": year_brier,
        "logloss": year_ll,
        "auc": year_auc,
        "pr_auc": year_pr_auc,
        "ece": year_ece,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }

    yearly_rows.append(
        row
    )

yearly_metrics = pd.DataFrame(
    yearly_rows
)

print(
    yearly_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

yearly_metrics.to_csv(
    CALIBRATION_YEARLY_PATH,
    index=False,
)


# ============================================================
# CALIBRATION DECILES
# ============================================================

print("\n")
print("=" * 80)
print("CALIBRATION DECILES")
print("=" * 80)

calibration_deciles = (
    make_calibration_deciles(
        y,
        p,
        bins=CALIBRATION_BINS,
    )
)

print(
    calibration_deciles.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

calibration_deciles.to_csv(
    CALIBRATION_DECILES_PATH,
    index=False,
)


# ============================================================
# SELECTIVE PREDICTION
# ============================================================

print("\n")
print("=" * 80)
print("SELECTIVE PREDICTION")
print("=" * 80)

selection = (
    selective_prediction_table(
        y,
        p,
        SELECTION_FRACTIONS,
    )
)

print(
    selection.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

selection.to_csv(
    SELECTIVE_PATH,
    index=False,
)


# ============================================================
# RISK CONCENTRATION
# ============================================================

print("\n")
print("=" * 80)
print("HIGH-RISK FAILURE CONCENTRATION")
print("=" * 80)

concentration = (
    risk_concentration_table(
        y,
        p,
    )
)

print(
    concentration.to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)

concentration.to_csv(
    RISK_CONCENTRATION_PATH,
    index=False,
)


# ============================================================
# CALIBRATION PLOT
# ============================================================

print("\n")
print(
    "Creating calibration plot..."
)

plt.figure(
    figsize=(7, 7)
)

plt.plot(
    calibration_deciles[
        "mean_predicted_probability"
    ],
    calibration_deciles[
        "observed_miscoverage"
    ],
    marker="o",
    label="D + PS",
)

min_value = min(
    calibration_deciles[
        "mean_predicted_probability"
    ].min(),
    calibration_deciles[
        "observed_miscoverage"
    ].min(),
)

max_value = max(
    calibration_deciles[
        "mean_predicted_probability"
    ].max(),
    calibration_deciles[
        "observed_miscoverage"
    ].max(),
)

plt.plot(
    [
        min_value,
        max_value,
    ],
    [
        min_value,
        max_value,
    ],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel(
    "Mean predicted probability of miscoverage"
)

plt.ylabel(
    "Observed miscoverage rate"
)

plt.title(
    "Calibration of D + PS Reliability Layer"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CALIBRATION_PLOT_PATH,
    dpi=200,
)

plt.close()


# ============================================================
# SELECTION CURVE
# ============================================================

print(
    "Creating selection curve..."
)

selection_plot = selection.sort_values(
    "retained_fraction"
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    selection_plot[
        "retained_fraction"
    ] * 100,
    selection_plot[
        "realized_miscoverage_retained"
    ],
    marker="o",
)

plt.axhline(
    actual_rate,
    linestyle="--",
    label="No selection",
)

plt.xlabel(
    "Predictions retained (%)"
)

plt.ylabel(
    "Realized miscoverage rate"
)

plt.title(
    "Selective Prediction: Miscoverage vs Coverage of Prediction Set"
)

plt.legend()

plt.gca().invert_xaxis()

plt.tight_layout()

plt.savefig(
    SELECTION_PLOT_PATH,
    dpi=200,
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("KEY RESULTS")
print("=" * 80)

print(
    f"Overall actual miscoverage : "
    f"{actual_rate:.6f}"
)

print(
    f"Mean predicted probability : "
    f"{predicted_mean:.6f}"
)

print(
    f"Brier                     : "
    f"{overall_brier:.6f}"
)

print(
    f"LogLoss                   : "
    f"{overall_logloss:.6f}"
)

print(
    f"AUC                       : "
    f"{overall_auc:.6f}"
)

print(
    f"PR-AUC                    : "
    f"{overall_pr_auc:.6f}"
)

print(
    f"ECE                       : "
    f"{ece:.6f}"
)

print(
    f"Calibration intercept     : "
    f"{cal_intercept:.6f}"
)

print(
    f"Calibration slope         : "
    f"{cal_slope:.6f}"
)

print("\n")
print(
    "SELECTION SUMMARY"
)

for fraction in [
    0.90,
    0.75,
    0.50,
    0.25,
    0.10,
]:

    row = selection[
        selection[
            "retained_fraction"
        ] == fraction
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print(
        f"Retain {fraction * 100:5.1f}% | "
        f"miscoverage="
        f"{row['realized_miscoverage_retained']:.6f} | "
        f"reduction="
        f"{row['relative_miscoverage_reduction'] * 100:.2f}% | "
        f"failures retained="
        f"{row['fraction_of_failures_retained'] * 100:.2f}%"
    )


# ============================================================
# FILE OUTPUTS
# ============================================================

print("\n")
print("=" * 80)
print("SAVED OUTPUTS")
print("=" * 80)

print(
    PREDICTIONS_PATH
)

print(
    POOLED_METRICS_PATH
)

print(
    CALIBRATION_YEARLY_PATH
)

print(
    CALIBRATION_DECILES_PATH
)

print(
    SELECTIVE_PATH
)

print(
    RISK_CONCENTRATION_PATH
)

print(
    CALIBRATION_PLOT_PATH
)

print(
    SELECTION_PLOT_PATH
)

print("\nDONE.")