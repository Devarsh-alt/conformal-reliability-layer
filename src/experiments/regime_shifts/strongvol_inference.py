
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm


BASE = Path(r"D:\reliability_layer")

PRED_PATH = (
    BASE
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS05"
    / "RS05_oos_predictions_H10_main.csv.gz"
)

OUT_DIR = (
    BASE
    / "data"
    / "results"
    / "RS05_strongvol_inference"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADLINE_START = pd.Timestamp("2020-01-01")

COMPARISONS = [
    ("S6L", "S4L", "primary_S6L_minus_S4L"),
    ("S6", "S4", "original_S6_minus_S4"),
    ("S4L", "S4", "longvol_S4L_minus_S4"),
]


def safe_logloss(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_date_series(new_spec, old_spec, sub):
    new = sub[sub["spec"] == new_spec][
        ["forecast_date", "ticker", "y", "p_cal"]
    ].copy()

    old = sub[sub["spec"] == old_spec][
        ["forecast_date", "ticker", "y", "p_cal"]
    ].copy()

    new = new.rename(
        columns={
            "y": "y_new",
            "p_cal": "p_new",
        }
    )

    old = old.rename(
        columns={
            "y": "y_old",
            "p_cal": "p_old",
        }
    )

    pair = new.merge(
        old,
        on=["forecast_date", "ticker"],
        how="inner",
        validate="one_to_one",
    )

    if pair.empty:
        raise RuntimeError(
            f"No paired observations for {new_spec} vs {old_spec}."
        )

    if not np.array_equal(
        pair["y_new"].to_numpy(),
        pair["y_old"].to_numpy(),
    ):
        raise AssertionError(
            "Targets differ between the two specifications."
        )

    y = pair["y_new"].to_numpy(dtype=int)
    p_new = pair["p_new"].to_numpy(dtype=float)
    p_old = pair["p_old"].to_numpy(dtype=float)

    pair["brier_new"] = (y - p_new) ** 2
    pair["brier_old"] = (y - p_old) ** 2

    pair["logloss_new"] = safe_logloss(y, p_new)
    pair["logloss_old"] = safe_logloss(y, p_old)

    pair["brier_improvement"] = (
        pair["brier_old"] - pair["brier_new"]
    )
    pair["logloss_improvement"] = (
        pair["logloss_old"] - pair["logloss_new"]
    )

    date_df = (
        pair.groupby("forecast_date", as_index=False)
        .agg(
            brier_improvement=("brier_improvement", "mean"),
            logloss_improvement=("logloss_improvement", "mean"),
        )
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    return date_df


def hac_mean(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 10:
        return {
            "estimate": np.nan,
            "se": np.nan,
            "z": np.nan,
            "p": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }

    X = np.ones((len(x), 1))

    fit = sm.OLS(x, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag},
    )

    est = float(fit.params[0])
    se = float(fit.bse[0])
    z = est / se if se > 0 else np.nan
    p = 2.0 * norm.sf(abs(z)) if np.isfinite(z) else np.nan

    return {
        "estimate": est,
        "se": se,
        "z": z,
        "p": p,
        "ci_lo": est - 1.96 * se,
        "ci_hi": est + 1.96 * se,
    }


def block_bootstrap_ci(x, block_len, reps=1000, seed=42):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n == 0:
        return np.nan, np.nan, False

    rng = np.random.default_rng(seed)

    blocks = [
        np.arange(start, min(start + block_len, n))
        for start in range(0, n, block_len)
    ]

    boot = np.empty(reps, dtype=float)

    for b in range(reps):
        chosen = []
        while len(chosen) < n:
            block = blocks[
                rng.integers(0, len(blocks))
            ]
            chosen.extend(block.tolist())

        idx = np.asarray(chosen[:n], dtype=int)
        boot[b] = np.mean(x[idx])

    lo = float(np.quantile(boot, 0.025))
    hi = float(np.quantile(boot, 0.975))

    return lo, hi, bool(lo > 0)


def summarize_comparison(new_spec, old_spec, sub):
    dates = paired_date_series(new_spec, old_spec, sub)

    rows = []

    for metric in [
        "brier_improvement",
        "logloss_improvement",
    ]:
        x = dates[metric].to_numpy(dtype=float)

        row = {
            "metric": metric,
            "n_dates": int(len(x)),
            "mean_improvement": float(np.mean(x)),
            "median_improvement": float(np.median(x)),
            "positive_date_share": float(np.mean(x > 0)),
        }

        for lag in [9, 20, 40, 60]:
            h = hac_mean(x, lag)

            row[f"hac{lag}_estimate"] = h["estimate"]
            row[f"hac{lag}_se"] = h["se"]
            row[f"hac{lag}_p"] = h["p"]
            row[f"hac{lag}_ci_lo"] = h["ci_lo"]
            row[f"hac{lag}_ci_hi"] = h["ci_hi"]

        for block_len in [20, 40, 60]:
            lo, hi, robust = block_bootstrap_ci(
                x,
                block_len=block_len,
                reps=1000,
                seed=100 + block_len,
            )

            row[f"bootstrap{block_len}_ci_lo"] = lo
            row[f"bootstrap{block_len}_ci_hi"] = hi
            row[
                f"bootstrap{block_len}_robust_positive"
            ] = robust

        rows.append(row)

    return (
        pd.DataFrame(rows),
        dates,
    )


def fold_signs(new_spec, old_spec, sub, date_df):
    fold_map = (
        sub[sub["spec"] == new_spec][
            ["forecast_date", "fold"]
        ]
        .drop_duplicates()
    )

    d = date_df.merge(
        fold_map,
        on="forecast_date",
        how="left",
        validate="one_to_one",
    )

    rows = []

    for fold, g in d.groupby("fold", sort=True):
        rows.append(
            {
                "fold": fold,
                "n_dates": int(len(g)),
                "brier_delta": float(
                    g["brier_improvement"].mean()
                ),
                "logloss_delta": float(
                    g["logloss_improvement"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------

if not PRED_PATH.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PRED_PATH}"
    )

P = pd.read_csv(
    PRED_PATH,
    compression="gzip",
    low_memory=False,
)

P["forecast_date"] = pd.to_datetime(
    P["forecast_date"],
    errors="coerce",
)

P = P[
    P["forecast_date"] >= HEADLINE_START
].copy()

required = {
    "forecast_date",
    "ticker",
    "model",
    "target",
    "spec",
    "algo",
    "fold",
    "y",
    "p_cal",
}

missing = required - set(P.columns)

if missing:
    raise KeyError(
        f"Missing required columns: {sorted(missing)}"
    )

P = P[
    (P["target"] == "T1")
    & (P["algo"] == "logit")
    & (
        P["spec"].isin(
            ["S4", "S4L", "S6", "S6L"]
        )
    )
].copy()

print("=" * 110)
print("RS05 STRONG-VOLATILITY INFERENCE")
print("=" * 110)
print(f"Input: {PRED_PATH}")
print(f"Rows loaded for analysis: {len(P):,}")
print(
    f"Models: {sorted(P['model'].unique().tolist())}"
)
print(
    f"Date range: {P['forecast_date'].min().date()} "
    f"to {P['forecast_date'].max().date()}"
)


# ---------------------------------------------------------------------
# MODEL-SPECIFIC RESULTS
# ---------------------------------------------------------------------

all_summary = []
all_folds = []
all_dates = []

for model in sorted(P["model"].unique()):
    sub = P[P["model"] == model].copy()

    for new_spec, old_spec, tag in COMPARISONS:
        needed = {new_spec, old_spec}

        if not needed.issubset(
            set(sub["spec"].unique())
        ):
            continue

        summary, date_df = summarize_comparison(
            new_spec,
            old_spec,
            sub,
        )

        summary.insert(0, "model", model)
        summary.insert(1, "comparison", tag)
        summary.insert(2, "new_spec", new_spec)
        summary.insert(3, "old_spec", old_spec)

        folds = fold_signs(
            new_spec,
            old_spec,
            sub,
            date_df,
        )

        folds.insert(0, "model", model)
        folds.insert(1, "comparison", tag)

        date_df.insert(0, "model", model)
        date_df.insert(1, "comparison", tag)

        all_summary.append(summary)
        all_folds.append(folds)
        all_dates.append(date_df)

summary_df = pd.concat(
    all_summary,
    ignore_index=True,
)

fold_df = pd.concat(
    all_folds,
    ignore_index=True,
)

date_df = pd.concat(
    all_dates,
    ignore_index=True,
)


# ---------------------------------------------------------------------
# POOLED CROSS-MODEL DATE-LEVEL ROBUSTNESS
# ---------------------------------------------------------------------

pooled_rows = []

for new_spec, old_spec, tag in COMPARISONS:
    pieces = []

    for model in sorted(P["model"].unique()):
        sub = P[P["model"] == model]

        if not {
            new_spec,
            old_spec,
        }.issubset(
            set(sub["spec"].unique())
        ):
            continue

        d = paired_date_series(
            new_spec,
            old_spec,
            sub,
        )

        d["model"] = model
        pieces.append(d)

    if not pieces:
        continue

    dm = pd.concat(
        pieces,
        ignore_index=True,
    )

    pooled = (
        dm.groupby("forecast_date", as_index=False)
        .agg(
            brier_improvement=(
                "brier_improvement",
                "mean",
            ),
            logloss_improvement=(
                "logloss_improvement",
                "mean",
            ),
        )
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    for metric in [
        "brier_improvement",
        "logloss_improvement",
    ]:
        x = pooled[metric].to_numpy(dtype=float)

        row = {
            "scope": "pooled_models",
            "comparison": tag,
            "metric": metric,
            "n_dates": int(len(x)),
            "mean_improvement": float(
                np.mean(x)
            ),
            "median_improvement": float(
                np.median(x)
            ),
            "positive_date_share": float(
                np.mean(x > 0)
            ),
        }

        for lag in [9, 20, 40, 60]:
            h = hac_mean(x, lag)

            row[f"hac{lag}_estimate"] = h["estimate"]
            row[f"hac{lag}_se"] = h["se"]
            row[f"hac{lag}_p"] = h["p"]
            row[f"hac{lag}_ci_lo"] = h["ci_lo"]
            row[f"hac{lag}_ci_hi"] = h["ci_hi"]

        pooled_rows.append(row)

pooled_df = pd.DataFrame(pooled_rows)


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

summary_path = (
    OUT_DIR
    / "RS05_strongvol_inference_summary.csv"
)

fold_path = (
    OUT_DIR
    / "RS05_strongvol_fold_signs.csv"
)

date_path = (
    OUT_DIR
    / "RS05_strongvol_date_level_deltas.csv"
)

pooled_path = (
    OUT_DIR
    / "RS05_strongvol_pooled_inference.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)

fold_df.to_csv(
    fold_path,
    index=False,
)

date_df.to_csv(
    date_path,
    index=False,
)

pooled_df.to_csv(
    pooled_path,
    index=False,
)


# ---------------------------------------------------------------------
# REPORT
# ---------------------------------------------------------------------

print()
print("-" * 110)
print("PRIMARY TEST: S6L vs S4L")
print("-" * 110)

primary = summary_df[
    summary_df["comparison"]
    == "primary_S6L_minus_S4L"
].copy()

show_cols = [
    "model",
    "metric",
    "mean_improvement",
    "median_improvement",
    "positive_date_share",
    "hac9_p",
    "hac20_p",
    "hac40_p",
    "hac60_p",
    "bootstrap20_ci_lo",
    "bootstrap20_ci_hi",
    "bootstrap20_robust_positive",
]

print(
    primary[show_cols]
    .round(6)
    .to_string(index=False)
)

print()
print("-" * 110)
print("PRIMARY FOLD SIGN CONSISTENCY")
print("-" * 110)

pf = fold_df[
    fold_df["comparison"]
    == "primary_S6L_minus_S4L"
].copy()

for metric in [
    "brier_delta",
    "logloss_delta",
]:
    vals = pf[metric].dropna()

    print(
        f"{metric}: "
        f"{int((vals > 0).sum())}/{len(vals)} "
        f"positive folds "
        f"({float(np.mean(vals > 0)):.3f})"
    )

print()
print("-" * 110)
print("POOLED DATE-LEVEL PRIMARY TEST")
print("-" * 110)

print(
    pooled_df[
        pooled_df["comparison"]
        == "primary_S6L_minus_S4L"
    ]
    .round(6)
    .to_string(index=False)
)

print()
print("-" * 110)
print("OUTPUT FILES")
print("-" * 110)
print(summary_path)
print(fold_path)
print(date_path)
print(pooled_path)
print("=" * 110)
