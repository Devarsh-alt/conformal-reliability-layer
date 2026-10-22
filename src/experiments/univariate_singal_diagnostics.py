"""
24_univariate_signal_diagnostics.py

Experiment 3:
Univariate diagnostic of QRF reliability signals.

Question:
    Can information available at forecast origin t predict whether
    QRF VaR performance will deteriorate over the next h days?

Primary targets:
    deterioration_5d
    deterioration_10d
    deterioration_20d

Evaluation:
    - strictly out-of-sample train/test split
    - univariate logistic regression
    - ROC-AUC
    - PR-AUC
    - Spearman rank correlation
    - ticker-level AUC
    - block-bootstrap confidence intervals for pooled AUC

IMPORTANT
---------
No target information is used as an input feature.
All predictor columns are causal monitoring features from the panel.

The model is deliberately simple:
    one signal -> logistic regression

This experiment identifies whether an individual signal contains
predictive information before we build nested multivariate models.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

HORIZONS = [5, 10, 20]

# Train/test split within the available QRF OOS period.
# 60% train, 40% test, independently within each ticker.
TRAIN_FRACTION = 0.60

MIN_TRAIN = 150
MIN_TEST = 50

# Block bootstrap length.
# 20 is deliberately close to our longest deterioration horizon.
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 12345

# Minimum number of positive and negative observations required
# for an AUC to be meaningful.
MIN_CLASS_COUNT = 10

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel_with_targets.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "univariate_signal_diagnostics.csv"
)

TICKER_OUTPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "univariate_signal_by_ticker.csv"
)


# ============================================================
# SIGNAL DEFINITIONS
# ============================================================

SIGNALS = {
    "breach_lag1": "Breach Persistence",
    "breach_streak_prior": "Breach Streak",
    "breach_rate_5d": "5-Day Breach Rate",
    "breach_rate_20d": "20-Day Breach Rate",
    "breach_rate_60d": "60-Day Breach Rate",
    "ewma_breach_rate": "EWMA Breach Rate",
    "pinball_mean_5d_prior": "Recent 5-Day Pinball Loss",
    "pinball_mean_20d_prior": "Recent 20-Day Pinball Loss",
    "pinball_mean_60d_prior": "Recent 60-Day Pinball Loss",
    "pinball_acceleration": "Pinball Loss Acceleration",
    "realized_vol_20d": "20-Day Realized Volatility",
    "realized_vol_60d": "60-Day Realized Volatility",
    "vol_ratio_20_60": "Volatility Ratio",
    "ks_drift_60_vs_250": "KS Distribution Drift",
    "q05_change_1d": "VaR Quantile Change",
    "VaR_change_1d": "VaR Magnitude Change",
    "VaR_ratio_to_prior": "VaR Ratio to Prior",
    "VaR_ratio_to_20d_mean": "VaR Ratio to 20-Day Mean",
}


# ============================================================
# HELPERS
# ============================================================

def safe_auc(y_true, score):
    """
    Return ROC-AUC or NaN if both classes are not present.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score
        )
    )


def safe_prauc(y_true, score):
    """
    Return PR-AUC or NaN if both classes are not present.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        average_precision_score(
            y_true,
            score
        )
    )


def safe_spearman(x, y):

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if len(np.unique(x)) < 2:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        spearmanr(x, y).statistic
    )


def fit_univariate_logistic(
    x_train,
    y_train,
    x_test,
    y_test
):
    """
    Fit a one-signal logistic regression.

    Standardization is fitted ONLY on training observations.
    """

    x_train = np.asarray(
        x_train,
        float
    ).reshape(-1, 1)

    x_test = np.asarray(
        x_test,
        float
    ).reshape(-1, 1)

    y_train = np.asarray(
        y_train,
        int
    )

    y_test = np.asarray(
        y_test,
        int
    )

    if len(np.unique(y_train)) < 2:
        return {
            "auc": np.nan,
            "prauc": np.nan,
            "coef": np.nan,
            "intercept": np.nan,
        }

    scaler = StandardScaler()

    x_train_s = scaler.fit_transform(
        x_train
    )

    x_test_s = scaler.transform(
        x_test
    )

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(
        x_train_s,
        y_train
    )

    probability = model.predict_proba(
        x_test_s
    )[:, 1]

    return {
        "auc": safe_auc(
            y_test,
            probability
        ),
        "prauc": safe_prauc(
            y_test,
            probability
        ),
        "coef": float(
            model.coef_[0, 0]
        ),
        "intercept": float(
            model.intercept_[0]
        ),
    }


def block_bootstrap_auc(
    y,
    score,
    block_length=20,
    reps=1000,
    seed=12345,
):
    """
    Moving block bootstrap for AUC.

    The purpose is to preserve local serial dependence rather than
    independently resampling individual days.

    Returns:
        point estimate
        lower 95% CI
        upper 95% CI
    """

    y = np.asarray(y, int)
    score = np.asarray(score, float)

    valid = (
        np.isfinite(score)
        &
        np.isfinite(y)
    )

    y = y[valid]
    score = score[valid]

    n = len(y)

    if n < 2 * block_length:
        return (
            np.nan,
            np.nan,
            np.nan
        )

    point = safe_auc(
        y,
        score
    )

    if np.isnan(point):
        return (
            np.nan,
            np.nan,
            np.nan
        )

    rng = np.random.default_rng(
        seed
    )

    starts = np.arange(
        0,
        n - block_length + 1
    )

    boot_stats = []

    target_n = n

    for _ in range(reps):

        sampled_indices = []

        while len(sampled_indices) < target_n:

            start = rng.choice(
                starts
            )

            block = np.arange(
                start,
                min(
                    start + block_length,
                    n
                )
            )

            sampled_indices.extend(
                block.tolist()
            )

        sampled_indices = (
            sampled_indices[:target_n]
        )

        y_b = y[
            sampled_indices
        ]

        score_b = score[
            sampled_indices
        ]

        if len(np.unique(y_b)) < 2:
            continue

        auc_b = safe_auc(
            y_b,
            score_b
        )

        if np.isfinite(auc_b):
            boot_stats.append(
                auc_b
            )

    if len(boot_stats) < 100:
        return (
            point,
            np.nan,
            np.nan
        )

    boot_stats = np.asarray(
        boot_stats
    )

    lower = np.quantile(
        boot_stats,
        0.025
    )

    upper = np.quantile(
        boot_stats,
        0.975
    )

    return (
        point,
        float(lower),
        float(upper)
    )


def chronological_split(
    g,
    train_fraction=TRAIN_FRACTION
):
    """
    Chronological split within a ticker.

    Earlier observations -> train
    Later observations  -> test
    """

    g = (
        g
        .sort_values("forecast_date")
        .reset_index(drop=True)
        .copy()
    )

    n = len(g)

    split_idx = int(
        np.floor(
            n * train_fraction
        )
    )

    train = g.iloc[
        :split_idx
    ].copy()

    test = g.iloc[
        split_idx:
    ].copy()

    return train, test


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("UNIVARIATE QRF RELIABILITY-SIGNAL DIAGNOSTICS")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required = [
        "ticker",
        "forecast_date",
    ]

    for h in HORIZONS:
        required.append(
            f"deterioration_{h}d"
        )

    for signal in SIGNALS:
        required.append(
            signal
        )

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns:\n{missing}"
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce"
    )

    print(
        f"\nLoaded: {INPUT_FILE}"
    )

    print(
        f"Rows:    {len(df):,}"
    )

    print(
        f"Tickers: {df['ticker'].nunique()}"
    )

    # ========================================================
    # RUN SIGNALS
    # ========================================================

    pooled_results = []
    ticker_results = []

    for h in HORIZONS:

        target = (
            f"deterioration_{h}d"
        )

        print(
            "\n" + "=" * 90
        )

        print(
            f"HORIZON = {h} DAYS"
        )

        print(
            "=" * 90
        )

        for signal, signal_name in SIGNALS.items():

            pooled_y = []
            pooled_probability = []

            ticker_aucs = []

            n_train_total = 0
            n_test_total = 0

            # ------------------------------------------------
            # Fit independently within each ticker
            # ------------------------------------------------

            for ticker, g in df.groupby(
                "ticker",
                sort=True
            ):

                subset = g[
                    [
                        "forecast_date",
                        signal,
                        target,
                    ]
                ].copy()

                subset[signal] = pd.to_numeric(
                    subset[signal],
                    errors="coerce"
                )

                subset[target] = pd.to_numeric(
                    subset[target],
                    errors="coerce"
                )

                subset = (
                    subset
                    .dropna()
                    .sort_values("forecast_date")
                    .reset_index(drop=True)
                )

                if len(subset) < (
                    MIN_TRAIN + MIN_TEST
                ):
                    continue

                train, test = chronological_split(
                    subset
                )

                if len(train) < MIN_TRAIN:
                    continue

                if len(test) < MIN_TEST:
                    continue

                y_train = train[target].astype(int)
                y_test = test[target].astype(int)

                if (
                    y_train.sum() < MIN_CLASS_COUNT
                    or
                    (len(y_train) - y_train.sum())
                    < MIN_CLASS_COUNT
                ):
                    continue

                if (
                    y_test.sum() < MIN_CLASS_COUNT
                    or
                    (len(y_test) - y_test.sum())
                    < MIN_CLASS_COUNT
                ):
                    continue

                result = fit_univariate_logistic(
                    train[signal].values,
                    y_train.values,
                    test[signal].values,
                    y_test.values
                )

                test_probability = None

                # Refit here only so we can retain test probabilities
                xtr = train[signal].values.reshape(-1, 1)
                xte = test[signal].values.reshape(-1, 1)

                scaler = StandardScaler()

                xtr_s = scaler.fit_transform(
                    xtr
                )

                xte_s = scaler.transform(
                    xte
                )

                model = LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver="lbfgs"
                )

                model.fit(
                    xtr_s,
                    y_train.values
                )

                test_probability = (
                    model
                    .predict_proba(xte_s)[:, 1]
                )

                pooled_y.extend(
                    y_test.values.tolist()
                )

                pooled_probability.extend(
                    test_probability.tolist()
                )

                n_train_total += len(train)
                n_test_total += len(test)

                auc_ticker = safe_auc(
                    y_test.values,
                    test_probability
                )

                ticker_aucs.append(
                    auc_ticker
                )

                ticker_results.append({
                    "horizon": h,
                    "ticker": ticker,
                    "signal": signal,
                    "signal_name": signal_name,
                    "n_train": len(train),
                    "n_test": len(test),
                    "test_positive_rate": y_test.mean(),
                    "test_auc": auc_ticker,
                    "test_prauc": safe_prauc(
                        y_test.values,
                        test_probability
                    ),
                    "test_spearman": safe_spearman(
                        test[signal].values,
                        y_test.values
                    ),
                })

            # ------------------------------------------------
            # Pooled result
            # ------------------------------------------------

            pooled_y = np.asarray(
                pooled_y,
                int
            )

            pooled_probability = np.asarray(
                pooled_probability,
                float
            )

            if len(pooled_y) < 2:
                print(
                    f"{signal_name:<32} insufficient data"
                )
                continue

            pooled_auc = safe_auc(
                pooled_y,
                pooled_probability
            )

            pooled_prauc = safe_prauc(
                pooled_y,
                pooled_probability
            )

            # Spearman here measures monotonic relation between
            # the original signal and the binary target.
            # It is diagnostic, not the primary predictive metric.

            pooled_signal_for_corr = []

            # Need to reconstruct pooled held-out signals
            for ticker, g in df.groupby(
                "ticker",
                sort=True
            ):

                subset = g[
                    [
                        "forecast_date",
                        signal,
                        target,
                    ]
                ].copy()

                subset[signal] = pd.to_numeric(
                    subset[signal],
                    errors="coerce"
                )

                subset[target] = pd.to_numeric(
                    subset[target],
                    errors="coerce"
                )

                subset = (
                    subset
                    .dropna()
                    .sort_values("forecast_date")
                    .reset_index(drop=True)
                )

                if len(subset) < (
                    MIN_TRAIN + MIN_TEST
                ):
                    continue

                train, test = chronological_split(
                    subset
                )

                if (
                    len(train) < MIN_TRAIN
                    or
                    len(test) < MIN_TEST
                ):
                    continue

                y_train = train[target].astype(int)
                y_test = test[target].astype(int)

                if (
                    y_train.sum() < MIN_CLASS_COUNT
                    or
                    (len(y_train) - y_train.sum())
                    < MIN_CLASS_COUNT
                ):
                    continue

                if (
                    y_test.sum() < MIN_CLASS_COUNT
                    or
                    (len(y_test) - y_test.sum())
                    < MIN_CLASS_COUNT
                ):
                    continue

                pooled_signal_for_corr.extend(
                    [
                        (
                            float(x),
                            int(y)
                        )
                        for x, y
                        in zip(
                            test[signal],
                            y_test
                        )
                    ]
                )

            if pooled_signal_for_corr:

                signal_values = np.array(
                    [
                        x[0]
                        for x in pooled_signal_for_corr
                    ]
                )

                target_values = np.array(
                    [
                        x[1]
                        for x in pooled_signal_for_corr
                    ]
                )

                spearman = safe_spearman(
                    signal_values,
                    target_values
                )

            else:
                spearman = np.nan

            # ------------------------------------------------
            # Block bootstrap
            # ------------------------------------------------

            auc_point, auc_lower, auc_upper = (
                block_bootstrap_auc(
                    pooled_y,
                    pooled_probability,
                    block_length=BOOTSTRAP_BLOCK,
                    reps=BOOTSTRAP_REPS,
                    seed=(
                        BOOTSTRAP_SEED
                        + h
                        + len(pooled_results)
                    )
                )
            )

            mean_ticker_auc = (
                np.nanmean(ticker_aucs)
                if ticker_aucs
                else np.nan
            )

            median_ticker_auc = (
                np.nanmedian(ticker_aucs)
                if ticker_aucs
                else np.nan
            )

            print(
                f"{signal_name:<32} "
                f"AUC={pooled_auc:.4f} "
                f"CI=[{auc_lower:.4f}, {auc_upper:.4f}] "
                f"PR-AUC={pooled_prauc:.4f} "
                f"ticker-mean={mean_ticker_auc:.4f}"
            )

            pooled_results.append({
                "horizon": h,
                "signal": signal,
                "signal_name": signal_name,
                "n_train_total": n_train_total,
                "n_test_total": n_test_total,
                "pooled_test_positive_rate": (
                    pooled_y.mean()
                ),
                "pooled_auc": pooled_auc,
                "auc_ci_lower": auc_lower,
                "auc_ci_upper": auc_upper,
                "pooled_prauc": pooled_prauc,
                "pooled_spearman": spearman,
                "mean_ticker_auc": mean_ticker_auc,
                "median_ticker_auc": median_ticker_auc,
                "n_tickers": len(ticker_aucs),
            })

    # ========================================================
    # RESULT TABLE
    # ========================================================

    pooled = pd.DataFrame(
        pooled_results
    )

    ticker = pd.DataFrame(
        ticker_results
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "SUMMARY — POOLED OUT-OF-SAMPLE AUC"
    )

    print(
        "=" * 90
    )

    for h in HORIZONS:

        sub = (
            pooled[
                pooled["horizon"] == h
            ]
            .sort_values(
                "pooled_auc",
                ascending=False
            )
        )

        print(
            f"\n{h}-DAY"
        )

        print(
            sub[
                [
                    "signal_name",
                    "pooled_auc",
                    "auc_ci_lower",
                    "auc_ci_upper",
                    "pooled_prauc",
                    "mean_ticker_auc",
                    "n_tickers",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pooled.to_csv(
        OUTPUT_FILE,
        index=False
    )

    ticker.to_csv(
        TICKER_OUTPUT_FILE,
        index=False
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "SAVED"
    )

    print(
        "=" * 90
    )

    print(
        OUTPUT_FILE
    )

    print(
        TICKER_OUTPUT_FILE
    )

    print(
        "\n" + "=" * 90
    )

    print(
        "UNIVARIATE SIGNAL DIAGNOSTICS COMPLETE"
    )

    print(
        "=" * 90
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print(
            "\nFATAL ERROR:",
            exc,
            file=sys.stderr
        )

        raise