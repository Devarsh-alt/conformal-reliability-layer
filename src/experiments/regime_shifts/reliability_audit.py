from __future__ import annotations

"""
RS05 — ROBUST CALIBRATION + EMPIRICAL TRUST-BAND AUDIT
=======================================================

Purpose
-------
Audit the already-generated, strictly OOS RS05 probability forecasts for:

    T1 = future deterioration_10d
    T2 = future coverage_failure_10d

Primary specifications:
    S4 = persistence + market + own volatility
    S6 = S4 + VaR/model state + cross-model disagreement

Models:
    EWMA, GARCH-t, HS, QRF

Algorithm:
    Logistic Regression only

IMPORTANT
---------
This script does NOT train or recalibrate RS05.

It reads:
    RS05_oos_predictions_H10_calibration.csv.gz

which is the combined T1 + T2 prediction file.

The script is deliberately conservative about rare-event calibration:

1. Calibration-in-the-large (CITL) is always reported.
2. Brier score and log loss are always reported.
3. Calibration slope is estimated with a bounded profile-likelihood method.
   If the finite maximum-likelihood slope is not identifiable because of
   separation / a boundary solution, the slope is reported as NOT ESTIMABLE
   rather than printing an absurd coefficient.
4. For T2, empirical reliability bins are the primary calibration diagnostic.
5. Date-block bootstrap confidence intervals are used for CITL, Brier,
   log loss, and RED-vs-GREEN trust-band separation. Dates are the sampling
   unit because all tickers share dates and the H=10 labels overlap.
6. Trust-band thresholds are NOT learned from evaluation outcomes. They are
   the already-frozen RS05 thresholds:
       thr05, thr10, thr20
7. GREEN/AMBER/RED monotonicity is treated as a descriptive stratification
   result, not by itself as a significance test.
8. No future target/outcome columns are used to define any threshold/bin.

Outputs
-------
The script writes separate *_ROBUST files so that the previous audit results
are preserved.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit
from scipy.stats import spearmanr


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\reliability_layer")

RS05_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS05"
)

INPUT_PATH = (
    RS05_DIR
    / "RS05_oos_predictions_H10_calibration.csv.gz"
)

HEADLINE_START = pd.Timestamp("2020-01-01")

MODELS = ["EWMA", "GARCH-t", "HS", "QRF"]
SPECS = ["S4", "S6"]
TARGETS = ["T1", "T2"]

# Date-block bootstrap settings.
BOOT_REPS = 2000
BLOCK_LENGTH = 10
RANDOM_SEED = 42

# Quantile reliability bins.
N_RELIABILITY_BINS = 10

# Numerical settings.
EPS = 1e-8
MAX_CALIBRATION_SLOPE = 20.0
SLOPE_BOUND_TOL = 0.05


# =============================================================================
# OUTPUT PATHS
# =============================================================================

CALIBRATION_OUT = (
    RS05_DIR
    / "RS05_robust_calibration_summary_H10_T1_T2.csv"
)

RELIABILITY_OUT = (
    RS05_DIR
    / "RS05_robust_reliability_bins_H10_T1_T2.csv"
)

TRUST_BANDS_OUT = (
    RS05_DIR
    / "RS05_robust_trust_bands_H10_T1_T2.csv"
)

TRUST_DATE_OUT = (
    RS05_DIR
    / "RS05_robust_trust_bands_datelevel_H10_T1_T2.csv"
)

TRUST_SEPARATION_OUT = (
    RS05_DIR
    / "RS05_robust_trust_band_separation_H10_T1_T2.csv"
)

TRUST_BOOT_OUT = (
    RS05_DIR
    / "RS05_robust_trust_band_bootstrap_H10_T1_T2.csv"
)

YEARLY_OUT = (
    RS05_DIR
    / "RS05_robust_yearly_calibration_H10_T1_T2.csv"
)

PERIOD_OUT = (
    RS05_DIR
    / "RS05_robust_period_calibration_H10_T1_T2.csv"
)

GROUP_COUNTS_OUT = (
    RS05_DIR
    / "RS05_robust_group_audit_H10_T1_T2.csv"
)


# =============================================================================
# DEFINITIONS
# =============================================================================

PERIOD_WINDOWS = {
    "COVID_transition_stress":
        (pd.Timestamp("2020-02-24"), pd.Timestamp("2020-07-17")),
    "2022_rate_tightening":
        (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023_banking_stress":
        (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-31")),
    "2025_stress":
        (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
}


# =============================================================================
# BASIC NUMERICAL HELPERS
# =============================================================================

def clip_probability(p: np.ndarray) -> np.ndarray:
    return np.clip(
        np.asarray(p, dtype=float),
        EPS,
        1.0 - EPS,
    )


def logit(p: np.ndarray) -> np.ndarray:
    p = clip_probability(p)
    return np.log(p / (1.0 - p))


def safe_logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_probability(p)
    y = np.asarray(y, dtype=float)
    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y) * np.log(1.0 - p)
        )
    )


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def calibration_in_large(y: np.ndarray, p: np.ndarray) -> float:
    """
    CITL = logit(observed prevalence) - logit(mean predicted probability)

    CITL = 0 means pooled prediction level matches pooled event frequency.
    """
    y_bar = float(np.mean(y))
    p_bar = float(np.mean(p))

    y_bar = float(np.clip(y_bar, EPS, 1.0 - EPS))
    p_bar = float(np.clip(p_bar, EPS, 1.0 - EPS))

    return float(logit(np.array([y_bar]))[0] - logit(np.array([p_bar]))[0])


# =============================================================================
# STABLE CALIBRATION-SLOPE ESTIMATION
# =============================================================================

def _intercept_for_slope(
    slope: float,
    z: np.ndarray,
    prevalence: float,
) -> float:
    """
    For a fixed calibration slope b, find the intercept a such that:

        mean(sigmoid(a + b*z)) = prevalence

    The root is unique because the left side is monotone in a.
    """

    target = float(
        np.clip(
            prevalence,
            EPS,
            1.0 - EPS,
        )
    )

    def f(a: float) -> float:
        q = expit(
            np.clip(
                a + slope * z,
                -60.0,
                60.0,
            )
        )
        return float(np.mean(q) - target)

    # Very wide finite interval. A root exists for 0 < target < 1.
    return float(
        brentq(
            f,
            -60.0,
            60.0,
            xtol=1e-10,
            rtol=1e-10,
            maxiter=100,
        )
    )


def _profile_loglik(
    slope: float,
    z: np.ndarray,
    y: np.ndarray,
    prevalence: float,
) -> tuple[float, float]:
    """
    Profile the logistic calibration likelihood over the slope.

    Returns:
        log_likelihood, intercept
    """

    intercept = _intercept_for_slope(
        slope,
        z,
        prevalence,
    )

    eta = np.clip(
        intercept + slope * z,
        -60.0,
        60.0,
    )

    # Stable Bernoulli log-likelihood:
    # y*eta - log(1+exp(eta))
    ll = np.sum(
        y * eta
        - np.logaddexp(0.0, eta)
    )

    return float(ll), float(intercept)


def calibration_slope_mle(
    y: np.ndarray,
    p: np.ndarray,
) -> dict:
    """
    Estimate the conventional logistic calibration slope:

        logit(P(Y=1)) = alpha + beta * logit(p)

    The likelihood is optimized with a finite bound.

    A result is marked NOT_ESTIMABLE when:
        - there is no outcome variation,
        - predicted probabilities have no variation, or
        - the optimum sits essentially on the slope boundary.

    We intentionally do NOT return gigantic coefficients caused by numerical
    divergence / separation.
    """

    y = np.asarray(y, dtype=float)
    z = logit(p)

    if len(y) < 20:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "estimable": False,
            "reason": "too_few_rows",
        }

    if np.unique(y).size < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "estimable": False,
            "reason": "only_one_outcome_class",
        }

    if np.ptp(z) < 1e-12:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "estimable": False,
            "reason": "no_probability_variation",
        }

    prevalence = float(np.mean(y))

    def objective(beta: float) -> float:
        ll, _ = _profile_loglik(
            beta,
            z,
            y,
            prevalence,
        )
        return -ll

    result = minimize_scalar(
        objective,
        bounds=(
            -MAX_CALIBRATION_SLOPE,
            MAX_CALIBRATION_SLOPE,
        ),
        method="bounded",
        options={
            "xatol": 1e-6,
            "maxiter": 500,
        },
    )

    beta = float(result.x)

    try:
        ll, alpha = _profile_loglik(
            beta,
            z,
            y,
            prevalence,
        )
    except Exception:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "estimable": False,
            "reason": "optimization_failure",
        }

    near_boundary = (
        abs(beta)
        >= MAX_CALIBRATION_SLOPE - SLOPE_BOUND_TOL
    )

    if (
        not result.success
        or near_boundary
        or not np.isfinite(beta)
        or not np.isfinite(alpha)
    ):
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "estimable": False,
            "reason": "boundary_or_separation",
        }

    return {
        "slope": beta,
        "intercept": alpha,
        "estimable": True,
        "reason": "finite_mle",
    }


# =============================================================================
# WILSON INTERVAL
# =============================================================================

def wilson_interval(
    successes: int,
    n: int,
    z_value: float = 1.959963984540054,
) -> tuple[float, float]:
    """
    Wilson 95% interval for a binomial proportion.
    """

    if n <= 0:
        return np.nan, np.nan

    p_hat = successes / n
    z2 = z_value ** 2

    center = (
        p_hat
        + z2 / (2.0 * n)
    ) / (
        1.0
        + z2 / n
    )

    half = (
        z_value
        * math.sqrt(
            (
                p_hat * (1.0 - p_hat)
                / n
            )
            + z2 / (4.0 * n * n)
        )
        / (
            1.0
            + z2 / n
        )
    )

    return (
        max(0.0, center - half),
        min(1.0, center + half),
    )


# =============================================================================
# DATE-BLOCK BOOTSTRAP
# =============================================================================

def bootstrap_date_indices(
    dates: np.ndarray,
    rng: np.random.Generator,
    block_length: int,
) -> np.ndarray:
    """
    Moving-block bootstrap over DATE indices.

    Output contains approximately n_dates indices.
    """

    n = len(dates)

    if n <= 1:
        return np.arange(n)

    L = int(
        min(
            max(1, block_length),
            n,
        )
    )

    n_blocks = int(
        math.ceil(n / L)
    )

    starts = rng.integers(
        0,
        n - L + 1,
        size=n_blocks,
    )

    pieces = []

    for s in starts:
        pieces.append(
            np.arange(
                s,
                s + L,
            )
        )

    out = np.concatenate(
        pieces
    )

    return out[:n]


def bootstrap_metric_ci(
    y: np.ndarray,
    p: np.ndarray,
    dates: np.ndarray,
    metric_fn,
    reps: int,
    block_length: int,
    seed: int,
) -> tuple[float, float]:
    """
    Date-block bootstrap 95% percentile CI.
    """

    unique_dates = np.sort(
        np.unique(dates)
    )

    date_to_idx = {
        d: i
        for i, d in enumerate(unique_dates)
    }

    row_date_idx = np.array(
        [
            date_to_idx[d]
            for d in dates
        ],
        dtype=int,
    )

    rows_by_date = [
        np.flatnonzero(
            row_date_idx == i
        )
        for i in range(len(unique_dates))
    ]

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(reps):

        selected_dates = bootstrap_date_indices(
            unique_dates,
            rng,
            block_length,
        )

        row_parts = [
            rows_by_date[i]
            for i in selected_dates
        ]

        if not row_parts:
            continue

        idx = np.concatenate(
            row_parts
        )

        value = metric_fn(
            y[idx],
            p[idx],
        )

        if np.isfinite(value):
            values.append(
                float(value)
            )

    if len(values) < max(20, reps // 4):
        return np.nan, np.nan

    values = np.asarray(
        values,
        dtype=float,
    )

    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


# =============================================================================
# BASIC GROUP METRICS
# =============================================================================

def group_metrics(
    g: pd.DataFrame,
) -> dict:
    y = g["y"].to_numpy(
        dtype=float
    )

    p = clip_probability(
        g["p_cal"].to_numpy(
            dtype=float
        )
    )

    dates = g["forecast_date"].to_numpy()

    slope_info = calibration_slope_mle(
        y,
        p,
    )

    date_level = (
        g.groupby(
            "forecast_date",
            as_index=False,
        )
        .agg(
            observed=("y", "mean"),
            predicted=("p_cal", "mean"),
        )
    )

    if (
        date_level["observed"].nunique() > 1
        and date_level["predicted"].nunique() > 1
    ):
        date_rho = float(
            spearmanr(
                date_level["predicted"],
                date_level["observed"],
            ).statistic
        )
    else:
        date_rho = np.nan

    # Fully deterministic group-specific seed; do NOT use Python hash(),
    # because its hash randomization changes between processes.
    group_key = (
        str(g["target"].iloc[0])
        + "|"
        + str(g["model"].iloc[0])
        + "|"
        + str(g["spec"].iloc[0])
    )
    stable_seed = 0
    for ch in group_key:
        stable_seed = (stable_seed * 131 + ord(ch)) % 1000000007

    metric_boot_seed = RANDOM_SEED + stable_seed

    citl = calibration_in_large(
        y,
        p,
    )

    brier = brier_score(
        y,
        p,
    )

    ll = safe_logloss(
        y,
        p,
    )

    citl_lo, citl_hi = bootstrap_metric_ci(
        y,
        p,
        dates,
        calibration_in_large,
        BOOT_REPS,
        BLOCK_LENGTH,
        metric_boot_seed + 1,
    )

    brier_lo, brier_hi = bootstrap_metric_ci(
        y,
        p,
        dates,
        brier_score,
        BOOT_REPS,
        BLOCK_LENGTH,
        metric_boot_seed + 2,
    )

    ll_lo, ll_hi = bootstrap_metric_ci(
        y,
        p,
        dates,
        safe_logloss,
        BOOT_REPS,
        BLOCK_LENGTH,
        metric_boot_seed + 3,
    )

    return {
        "n_rows": int(len(g)),
        "n_dates": int(
            g["forecast_date"].nunique()
        ),
        "n_tickers": int(
            g["ticker"].nunique()
        ),
        "prevalence": float(
            np.mean(y)
        ),
        "mean_predicted": float(
            np.mean(p)
        ),
        "observed_minus_predicted":
            float(
                np.mean(y)
                - np.mean(p)
            ),
        "observed_predicted_ratio":
            float(
                np.mean(y)
                / max(np.mean(p), EPS)
            ),
        "brier": brier,
        "brier_ci_low": brier_lo,
        "brier_ci_high": brier_hi,
        "logloss": ll,
        "logloss_ci_low": ll_lo,
        "logloss_ci_high": ll_hi,
        "calibration_in_the_large": citl,
        "citl_ci_low": citl_lo,
        "citl_ci_high": citl_hi,
        "calibration_slope":
            slope_info["slope"],
        "calibration_intercept":
            slope_info["intercept"],
        "calibration_slope_estimable":
            bool(slope_info["estimable"]),
        "calibration_slope_status":
            slope_info["reason"],
        "date_spearman": date_rho,
    }


# =============================================================================
# RELIABILITY BINS
# =============================================================================

def reliability_bins(
    g: pd.DataFrame,
    n_bins: int = N_RELIABILITY_BINS,
) -> pd.DataFrame:
    """
    Create prediction-only quantile bins.

    Bin membership is determined entirely from p_cal, never from y.
    """

    x = g[
        [
            "target",
            "model",
            "spec",
            "forecast_date",
            "y",
            "p_cal",
        ]
    ].copy()

    p = clip_probability(
        x["p_cal"].to_numpy(
            dtype=float
        )
    )

    # Quantile edges based ONLY on predicted probabilities.
    quantiles = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    edges = np.quantile(
        p,
        quantiles,
    )

    edges = np.unique(
        edges
    )

    if len(edges) < 2:
        x["reliability_bin"] = 1
    else:
        x["reliability_bin"] = pd.cut(
            x["p_cal"],
            bins=edges,
            labels=False,
            include_lowest=True,
            duplicates="drop",
        ) + 1

    rows = []

    for b, sub in x.groupby(
        "reliability_bin",
        sort=True,
        observed=False,
    ):

        y = sub["y"].to_numpy(
            dtype=int
        )

        n = len(sub)
        successes = int(
            y.sum()
        )

        lo, hi = wilson_interval(
            successes,
            n,
        )

        rows.append(
            {
                "target":
                    sub["target"].iloc[0],
                "model":
                    sub["model"].iloc[0],
                "spec":
                    sub["spec"].iloc[0],
                "reliability_bin":
                    int(b),
                "n_rows":
                    n,
                "n_dates":
                    int(
                        sub[
                            "forecast_date"
                        ].nunique()
                    ),
                "mean_predicted_probability":
                    float(
                        sub["p_cal"].mean()
                    ),
                "realized_rate":
                    float(
                        sub["y"].mean()
                    ),
                "calibration_gap":
                    float(
                        sub["y"].mean()
                        - sub["p_cal"].mean()
                    ),
                "wilson_95_low":
                    lo,
                "wilson_95_high":
                    hi,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EMPIRICAL TRUST BANDS
# =============================================================================

def build_trust_bands(
    g: pd.DataFrame,
) -> pd.DataFrame:

    g = g.copy()

    # Frozen thresholds from RS05.
    g["trust_band"] = np.select(
        [
            g["p_cal"] >= g["thr10"],
            g["p_cal"] >= g["thr20"],
        ],
        [
            "RED",
            "AMBER",
        ],
        default="GREEN",
    )

    g["critical_5pct_alert"] = (
        g["p_cal"] >= g["thr05"]
    )

    threshold_violations = int(
        (
            g["thr05"] < g["thr10"]
        ).sum()
        +
        (
            g["thr10"] < g["thr20"]
        ).sum()
    )

    rows = []

    for band in [
        "GREEN",
        "AMBER",
        "RED",
    ]:

        sub = g[
            g["trust_band"] == band
        ]

        if len(sub) == 0:

            rows.append(
                {
                    "target":
                        g["target"].iloc[0],
                    "model":
                        g["model"].iloc[0],
                    "spec":
                        g["spec"].iloc[0],
                    "trust_band": band,
                    "n_rows": 0,
                    "n_dates": 0,
                    "mean_predicted_probability": np.nan,
                    "realized_rate": np.nan,
                    "calibration_gap": np.nan,
                    "critical_5pct_share": np.nan,
                    "wilson_95_low": np.nan,
                    "wilson_95_high": np.nan,
                    "threshold_order_violations":
                        threshold_violations,
                }
            )

            continue

        y = sub["y"].to_numpy(
            dtype=int
        )

        lo, hi = wilson_interval(
            int(y.sum()),
            int(len(y)),
        )

        rows.append(
            {
                "target":
                    g["target"].iloc[0],
                "model":
                    g["model"].iloc[0],
                "spec":
                    g["spec"].iloc[0],
                "trust_band": band,
                "n_rows":
                    int(len(sub)),
                "n_dates":
                    int(
                        sub[
                            "forecast_date"
                        ].nunique()
                    ),
                "mean_predicted_probability":
                    float(
                        sub["p_cal"].mean()
                    ),
                "realized_rate":
                    float(
                        sub["y"].mean()
                    ),
                "calibration_gap":
                    float(
                        sub["y"].mean()
                        - sub["p_cal"].mean()
                    ),
                "critical_5pct_share":
                    float(
                        sub[
                            "critical_5pct_alert"
                        ].mean()
                    ),
                "wilson_95_low":
                    lo,
                "wilson_95_high":
                    hi,
                "threshold_order_violations":
                    threshold_violations,
            }
        )

    return pd.DataFrame(
        rows
    )


def trust_date_level(
    g: pd.DataFrame,
) -> pd.DataFrame:

    g = g.copy()

    g["trust_band"] = np.select(
        [
            g["p_cal"] >= g["thr10"],
            g["p_cal"] >= g["thr20"],
        ],
        [
            "RED",
            "AMBER",
        ],
        default="GREEN",
    )

    out = (
        g.groupby(
            [
                "target",
                "model",
                "spec",
                "forecast_date",
                "trust_band",
            ],
            as_index=False,
        )
        .agg(
            n_rows=("y", "size"),
            realized_fraction=("y", "mean"),
            mean_probability=("p_cal", "mean"),
        )
    )

    return out


def trust_separation(
    trust: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for (
        target,
        model,
        spec,
    ), sub in trust.groupby(
        [
            "target",
            "model",
            "spec",
        ],
        sort=True,
    ):

        pivot = (
            sub.set_index("trust_band")
            .reindex(
                [
                    "GREEN",
                    "AMBER",
                    "RED",
                ]
            )
        )

        green = pivot.loc[
            "GREEN",
            "realized_rate",
        ]

        amber = pivot.loc[
            "AMBER",
            "realized_rate",
        ]

        red = pivot.loc[
            "RED",
            "realized_rate",
        ]

        monotone = (
            (
                pd.isna(green)
                or pd.isna(amber)
                or green <= amber
            )
            and
            (
                pd.isna(amber)
                or pd.isna(red)
                or amber <= red
            )
        )

        red_green = (
            np.nan
            if pd.isna(green)
            or pd.isna(red)
            else red - green
        )

        rows.append(
            {
                "target": target,
                "model": model,
                "spec": spec,
                "green_rate": green,
                "amber_rate": amber,
                "red_rate": red,
                "monotone_non_decreasing":
                    bool(monotone),
                "red_minus_green":
                    red_green,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# DATE-BLOCK BOOTSTRAP FOR TRUST-BAND RED-GREEN DIFFERENCE
# =============================================================================

def bootstrap_red_green_difference(
    g: pd.DataFrame,
    reps: int,
    block_length: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """
    Bootstrap the RED - GREEN realized-rate difference using dates as blocks.

    Band assignment is fixed before resampling because thresholds are frozen
    by RS05. Thus the bootstrap assesses uncertainty in the realized separation,
    not threshold selection.
    """

    x = g.copy()

    x["trust_band"] = np.select(
        [
            x["p_cal"] >= x["thr10"],
            x["p_cal"] >= x["thr20"],
        ],
        [
            "RED",
            "AMBER",
        ],
        default="GREEN",
    )

    red = x[
        x["trust_band"] == "RED"
    ]

    green = x[
        x["trust_band"] == "GREEN"
    ]

    if red.empty or green.empty:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    dates = np.sort(
        x["forecast_date"].unique()
    )

    date_to_index = {
        d: i
        for i, d in enumerate(dates)
    }

    row_date_idx = np.array(
        [
            date_to_index[d]
            for d in x[
                "forecast_date"
            ].to_numpy()
        ]
    )

    band_values = x[
        "trust_band"
    ].to_numpy()

    y = x[
        "y"
    ].to_numpy(
        dtype=float
    )

    rows_by_date_green = [
        np.flatnonzero(
            (
                row_date_idx == i
            )
            &
            (
                band_values == "GREEN"
            )
        )
        for i in range(len(dates))
    ]

    rows_by_date_red = [
        np.flatnonzero(
            (
                row_date_idx == i
            )
            &
            (
                band_values == "RED"
            )
        )
        for i in range(len(dates))
    ]

    rng = np.random.default_rng(
        seed
    )

    values = []

    for _ in range(reps):

        selected_dates = bootstrap_date_indices(
            dates,
            rng,
            block_length,
        )

        red_parts = []
        green_parts = []

        for i in selected_dates:
            if rows_by_date_red[i].size:
                red_parts.append(
                    rows_by_date_red[i]
                )

            if rows_by_date_green[i].size:
                green_parts.append(
                    rows_by_date_green[i]
                )

        if not red_parts or not green_parts:
            continue

        red_idx = np.concatenate(
            red_parts
        )

        green_idx = np.concatenate(
            green_parts
        )

        red_rate = float(
            np.mean(
                y[red_idx]
            )
        )

        green_rate = float(
            np.mean(
                y[green_idx]
            )
        )

        values.append(
            red_rate - green_rate
        )

    if len(values) < max(
        20,
        reps // 4,
    ):
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    point = float(
        red["y"].mean()
        - green["y"].mean()
    )

    lo = float(
        np.quantile(
            values,
            0.025,
        )
    )

    hi = float(
        np.quantile(
            values,
            0.975,
        )
    )

    p_gt_zero = float(
        np.mean(
            values > 0.0
        )
    )

    return (
        point,
        lo,
        hi,
        p_gt_zero,
    )


# =============================================================================
# YEARLY CALIBRATION
# =============================================================================

def yearly_calibration(
    g: pd.DataFrame,
) -> pd.DataFrame:

    x = g.copy()

    x["year"] = (
        x["forecast_date"]
        .dt.year
    )

    rows = []

    for year, sub in x.groupby(
        "year",
        sort=True,
    ):

        y = sub["y"].to_numpy(
            dtype=float
        )

        p = clip_probability(
            sub["p_cal"].to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "target":
                    sub["target"].iloc[0],
                "model":
                    sub["model"].iloc[0],
                "spec":
                    sub["spec"].iloc[0],
                "year":
                    int(year),
                "n_rows":
                    int(len(sub)),
                "n_dates":
                    int(
                        sub["forecast_date"]
                        .nunique()
                    ),
                "prevalence":
                    float(y.mean()),
                "mean_predicted":
                    float(p.mean()),
                "citl":
                    calibration_in_large(
                        y,
                        p,
                    ),
                "brier":
                    brier_score(
                        y,
                        p,
                    ),
                "logloss":
                    safe_logloss(
                        y,
                        p,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# PERIOD CALIBRATION
# =============================================================================

def period_calibration(
    g: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for period_name, (
        start,
        end,
    ) in PERIOD_WINDOWS.items():

        sub = g[
            (g["forecast_date"] >= start)
            &
            (g["forecast_date"] <= end)
        ]

        if sub.empty:
            continue

        y = sub["y"].to_numpy(
            dtype=float
        )

        p = clip_probability(
            sub["p_cal"].to_numpy(
                dtype=float
            )
        )

        rows.append(
            {
                "target":
                    g["target"].iloc[0],
                "model":
                    g["model"].iloc[0],
                "spec":
                    g["spec"].iloc[0],
                "period":
                    period_name,
                "start":
                    start,
                "end":
                    end,
                "n_rows":
                    int(len(sub)),
                "n_dates":
                    int(
                        sub[
                            "forecast_date"
                        ].nunique()
                    ),
                "prevalence":
                    float(y.mean()),
                "mean_predicted":
                    float(p.mean()),
                "citl":
                    calibration_in_large(
                        y,
                        p,
                    ),
                "brier":
                    brier_score(
                        y,
                        p,
                    ),
                "logloss":
                    safe_logloss(
                        y,
                        p,
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# GROUP INTEGRITY AUDIT
# =============================================================================

def group_integrity_audit(
    g: pd.DataFrame,
) -> dict:

    key_duplicates = int(
        g.duplicated(
            subset=[
                "forecast_date",
                "ticker",
            ]
        ).sum()
    )

    y_values = set(
        pd.to_numeric(
            g["y"],
            errors="coerce",
        ).dropna().astype(int)
    )

    valid_binary = y_values.issubset(
        {0, 1}
    )

    finite_probability = bool(
        np.isfinite(
            g["p_cal"].to_numpy(
                dtype=float
            )
        ).all()
    )

    finite_thresholds = bool(
        np.isfinite(
            g[
                [
                    "thr05",
                    "thr10",
                    "thr20",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    threshold_order_violations = int(
        (
            g["thr05"] < g["thr10"]
        ).sum()
        +
        (
            g["thr10"] < g["thr20"]
        ).sum()
    )

    return {
        "target":
            g["target"].iloc[0],
        "model":
            g["model"].iloc[0],
        "spec":
            g["spec"].iloc[0],
        "n_rows":
            int(len(g)),
        "n_dates":
            int(
                g["forecast_date"]
                .nunique()
            ),
        "n_tickers":
            int(
                g["ticker"]
                .nunique()
            ),
        "duplicate_date_ticker_rows":
            key_duplicates,
        "binary_target_ok":
            valid_binary,
        "finite_probabilities_ok":
            finite_probability,
        "finite_thresholds_ok":
            finite_thresholds,
        "threshold_order_violations":
            threshold_order_violations,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 110)
    print(
        "RS05 ROBUST CALIBRATION + EMPIRICAL TRUST-BAND AUDIT — H10 OOS"
    )
    print("=" * 110)
    print(
        f"Input:\n{INPUT_PATH}"
    )
    print()
    print(
        f"Bootstrap reps: {BOOT_REPS}"
    )
    print(
        f"Bootstrap block length: {BLOCK_LENGTH} dates"
    )
    print(
        f"Headline start: {HEADLINE_START.date()}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_PATH}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        raw = pd.read_csv(
            INPUT_PATH,
            compression="gzip",
            low_memory=False,
        )

    print()
    print(
        f"Raw rows loaded: {len(raw):,}"
    )

    required = [
        "target",
        "model",
        "algo",
        "spec",
        "forecast_date",
        "ticker",
        "y",
        "p_cal",
        "thr05",
        "thr10",
        "thr20",
    ]

    missing = [
        c
        for c in required
        if c not in raw.columns
    ]

    if missing:
        raise RuntimeError(
            "Required columns missing:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    # -------------------------------------------------------------------------
    # Normalize
    # -------------------------------------------------------------------------

    raw["forecast_date"] = pd.to_datetime(
        raw["forecast_date"],
        errors="coerce",
    )

    raw["target"] = (
        raw["target"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    raw["model"] = (
        raw["model"]
        .astype(str)
        .str.strip()
    )

    raw["algo"] = (
        raw["algo"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    raw["spec"] = (
        raw["spec"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    raw["ticker"] = (
        raw["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    raw["y"] = pd.to_numeric(
        raw["y"],
        errors="coerce",
    )

    raw["p_cal"] = pd.to_numeric(
        raw["p_cal"],
        errors="coerce",
    )

    for c in [
        "thr05",
        "thr10",
        "thr20",
    ]:
        raw[c] = pd.to_numeric(
            raw[c],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Filter exactly to intended calibration comparison.
    # -------------------------------------------------------------------------

    df = raw[
        (raw["forecast_date"] >= HEADLINE_START)
        &
        raw["target"].isin(TARGETS)
        &
        raw["model"].isin(MODELS)
        &
        raw["spec"].isin(SPECS)
        &
        (raw["algo"] == "logit")
    ].dropna(
        subset=[
            "forecast_date",
            "ticker",
            "target",
            "model",
            "spec",
            "y",
            "p_cal",
            "thr05",
            "thr10",
            "thr20",
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No rows remain after filtering."
        )

    # Validate target encoding.
    non_binary = sorted(
        set(
            df["y"].astype(int).unique()
        )
        -
        {0, 1}
    )

    if non_binary:
        raise RuntimeError(
            f"Non-binary target values found: {non_binary}"
        )

    df["y"] = df["y"].astype(int)

    # Validate probabilities.
    if (
        (~np.isfinite(
            df["p_cal"].to_numpy(
                dtype=float
            )
        )).any()
    ):
        raise RuntimeError(
            "Non-finite p_cal values remain after filtering."
        )

    # This should not happen for the RS05 predictions, but validate explicitly.
    if (
        (
            (df["p_cal"] < 0.0)
            | (df["p_cal"] > 1.0)
        ).any()
    ):
        raise RuntimeError(
            "p_cal contains values outside [0, 1]."
        )

    print()
    print(
        f"Filtered rows: {len(df):,}"
    )
    print(
        f"Filtered date range: "
        f"{df['forecast_date'].min().date()} "
        f"to "
        f"{df['forecast_date'].max().date()}"
    )

    # -------------------------------------------------------------------------
    # Group integrity.
    # -------------------------------------------------------------------------

    integrity_rows = []

    for (
        target,
        model,
        spec,
    ), g in df.groupby(
        [
            "target",
            "model",
            "spec",
        ],
        sort=True,
        observed=False,
    ):

        integrity_rows.append(
            group_integrity_audit(g)
        )

    integrity = pd.DataFrame(
        integrity_rows
    )

    integrity.to_csv(
        GROUP_COUNTS_OUT,
        index=False,
    )

    expected_groups = 16

    group_count = len(
        integrity
    )

    print()
    print(
        f"Calibration groups found: {group_count}"
    )

    if group_count != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} groups but found {group_count}."
        )

    if (
        integrity[
            "duplicate_date_ticker_rows"
        ]
        > 0
    ).any():
        raise RuntimeError(
            "Duplicate ticker + forecast_date rows detected."
        )

    if (
        ~integrity[
            "binary_target_ok"
        ]
    ).any():
        raise RuntimeError(
            "Binary target validation failed."
        )

    if (
        ~integrity[
            "finite_probabilities_ok"
        ]
    ).any():
        raise RuntimeError(
            "Non-finite probabilities detected."
        )

    if (
        ~integrity[
            "finite_thresholds_ok"
        ]
    ).any():
        raise RuntimeError(
            "Non-finite thresholds detected."
        )

    if (
        integrity[
            "threshold_order_violations"
        ]
        > 0
    ).any():
        raise RuntimeError(
            "Threshold ordering violated."
        )

    print(
        "Group integrity: PASS"
    )

    # -------------------------------------------------------------------------
    # Calibration + reliability bins + trust bands.
    # -------------------------------------------------------------------------

    calibration_rows = []
    reliability_frames = []
    trust_frames = []
    date_frames = []
    separation_rows = []
    trust_boot_rows = []
    yearly_frames = []
    period_frames = []

    for (
        target,
        model,
        spec,
    ), g in df.groupby(
        [
            "target",
            "model",
            "spec",
        ],
        sort=True,
        observed=False,
    ):

        print(
            f"\n[{target} | {model} | {spec}] "
            f"rows={len(g):,}, "
            f"dates={g['forecast_date'].nunique():,}"
        )

        # -------------------------------------------------------------
        # Calibration metrics.
        # -------------------------------------------------------------

        metrics = group_metrics(
            g
        )

        metrics.update(
            {
                "target": target,
                "model": model,
                "spec": spec,
            }
        )

        calibration_rows.append(
            metrics
        )

        # -------------------------------------------------------------
        # Reliability bins.
        # -------------------------------------------------------------

        rb = reliability_bins(
            g
        )

        reliability_frames.append(
            rb
        )

        # -------------------------------------------------------------
        # Trust bands.
        # -------------------------------------------------------------

        tb = build_trust_bands(
            g
        )

        trust_frames.append(
            tb
        )

        # -------------------------------------------------------------
        # Date-level trust data.
        # -------------------------------------------------------------

        tbd = trust_date_level(
            g
        )

        date_frames.append(
            tbd
        )

        # -------------------------------------------------------------
        # Separation.
        # -------------------------------------------------------------

        sep = trust_separation(
            tb
        )

        separation_rows.extend(
            sep.to_dict(
                orient="records"
            )
        )

        # -------------------------------------------------------------
        # Date-block bootstrap for RED-GREEN separation.
        # -------------------------------------------------------------

        boot = bootstrap_red_green_difference(
            g,
            BOOT_REPS,
            BLOCK_LENGTH,
            RANDOM_SEED
            + 1000
            + len(trust_boot_rows),
        )

        if len(boot) == 4:
            point, lo, hi, p_gt_zero = boot
        else:
            point, lo, hi = boot
            p_gt_zero = np.nan

        trust_boot_rows.append(
            {
                "target": target,
                "model": model,
                "spec": spec,
                "red_minus_green":
                    point,
                "bootstrap_95_low":
                    lo,
                "bootstrap_95_high":
                    hi,
                "bootstrap_probability_red_gt_green":
                    p_gt_zero,
                "bootstrap_reps":
                    BOOT_REPS,
                "block_length":
                    BLOCK_LENGTH,
            }
        )

        # -------------------------------------------------------------
        # Yearly.
        # -------------------------------------------------------------

        yearly_frames.append(
            yearly_calibration(
                g
            )
        )

        # -------------------------------------------------------------
        # Event periods.
        # -------------------------------------------------------------

        period_frames.append(
            period_calibration(
                g
            )
        )

        # -------------------------------------------------------------
        # Console summary.
        # -------------------------------------------------------------

        print(
            f"    prevalence={metrics['prevalence']:.6f}"
        )

        print(
            f"    mean_p={metrics['mean_predicted']:.6f}"
        )

        print(
            f"    CITL={metrics['calibration_in_the_large']:.6f} "
            f"[{metrics['citl_ci_low']:.6f}, "
            f"{metrics['citl_ci_high']:.6f}]"
        )

        print(
            f"    Brier={metrics['brier']:.6f} "
            f"[{metrics['brier_ci_low']:.6f}, "
            f"{metrics['brier_ci_high']:.6f}]"
        )

        print(
            f"    LogLoss={metrics['logloss']:.6f} "
            f"[{metrics['logloss_ci_low']:.6f}, "
            f"{metrics['logloss_ci_high']:.6f}]"
        )

        if metrics[
            "calibration_slope_estimable"
        ]:
            print(
                f"    Calibration slope="
                f"{metrics['calibration_slope']:.6f} "
                f"(finite MLE)"
            )
        else:
            print(
                "    Calibration slope=NOT ESTIMABLE "
                f"({metrics['calibration_slope_status']})"
            )

    # -------------------------------------------------------------------------
    # Assemble.
    # -------------------------------------------------------------------------

    calibration = pd.DataFrame(
        calibration_rows
    ).sort_values(
        [
            "target",
            "model",
            "spec",
        ]
    )

    reliability = pd.concat(
        reliability_frames,
        ignore_index=True,
    )

    trust_bands = pd.concat(
        trust_frames,
        ignore_index=True,
    )

    trust_date = pd.concat(
        date_frames,
        ignore_index=True,
    )

    separation = pd.DataFrame(
        separation_rows
    ).sort_values(
        [
            "target",
            "model",
            "spec",
        ]
    )

    trust_boot = pd.DataFrame(
        trust_boot_rows
    ).sort_values(
        [
            "target",
            "model",
            "spec",
        ]
    )

    yearly = pd.concat(
        yearly_frames,
        ignore_index=True,
    ).sort_values(
        [
            "target",
            "model",
            "spec",
            "year",
        ]
    )

    periods = pd.concat(
        period_frames,
        ignore_index=True,
    ).sort_values(
        [
            "target",
            "model",
            "spec",
            "period",
        ]
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    calibration.to_csv(
        CALIBRATION_OUT,
        index=False,
    )

    reliability.to_csv(
        RELIABILITY_OUT,
        index=False,
    )

    trust_bands.to_csv(
        TRUST_BANDS_OUT,
        index=False,
    )

    trust_date.to_csv(
        TRUST_DATE_OUT,
        index=False,
    )

    separation.to_csv(
        TRUST_SEPARATION_OUT,
        index=False,
    )

    trust_boot.to_csv(
        TRUST_BOOT_OUT,
        index=False,
    )

    yearly.to_csv(
        YEARLY_OUT,
        index=False,
    )

    periods.to_csv(
        PERIOD_OUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    monotone_count = int(
        separation[
            "monotone_non_decreasing"
        ]
        .fillna(False)
        .sum()
    )

    total_groups = len(
        separation
    )

    threshold_violations = int(
        (
            integrity[
                "threshold_order_violations"
            ]
            > 0
        ).sum()
    )

    print()
    print("=" * 110)
    print(
        "ROBUST CALIBRATION SUMMARY — 2020+"
    )
    print("=" * 110)

    display_cols = [
        "target",
        "model",
        "spec",
        "n_rows",
        "n_dates",
        "prevalence",
        "mean_predicted",
        "observed_minus_predicted",
        "observed_predicted_ratio",
        "brier",
        "logloss",
        "calibration_in_the_large",
        "calibration_slope",
        "calibration_slope_estimable",
        "date_spearman",
    ]

    print(
        calibration[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print(
        "TRUST-BAND SEPARATION — 2020+"
    )
    print("=" * 110)

    print(
        separation.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print(
        "TRUST-BAND DATE-BLOCK BOOTSTRAP"
    )
    print("=" * 110)

    print(
        trust_boot.to_string(
            index=False
        )
    )

    print()
    print("=" * 110)
    print(
        "SANITY CHECKS"
    )
    print("=" * 110)

    print(
        f"16 calibration groups: "
        f"{'PASS' if group_count == 16 else 'FAIL'}"
    )

    print(
        f"Group integrity: "
        f"{'PASS' if group_count == 16 else 'FAIL'}"
    )

    print(
        f"Threshold ordering: "
        f"{'PASS' if threshold_violations == 0 else 'FAIL'}"
    )

    print(
        f"Trust-band monotonicity: "
        f"{monotone_count}/{total_groups}"
    )

    slope_failures = int(
        (
            ~calibration[
                "calibration_slope_estimable"
            ]
        ).sum()
    )

    print(
        "Calibration slope finite-MLE status: "
        f"{len(calibration) - slope_failures}/"
        f"{len(calibration)} estimable"
    )

    print()
    print("=" * 110)
    print(
        "OUTPUTS"
    )
    print("=" * 110)

    for path in [
        CALIBRATION_OUT,
        RELIABILITY_OUT,
        TRUST_BANDS_OUT,
        TRUST_DATE_OUT,
        TRUST_SEPARATION_OUT,
        TRUST_BOOT_OUT,
        YEARLY_OUT,
        PERIOD_OUT,
        GROUP_COUNTS_OUT,
    ]:
        print(path)

    print()
    print("=" * 110)
    print(
        "RS05 ROBUST CALIBRATION + TRUST-BAND AUDIT COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
