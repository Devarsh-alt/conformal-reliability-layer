from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# CONFIG
# =============================================================================

DEFAULT_PREDICTIONS = Path(
    r"D:\reliability_layer\data\monitoring\regime_shifts\RS05"
    r"\RS05_oos_predictions_H10_main.csv.gz"
)

OUT_DIR = Path(
    r"D:\reliability_layer\data\monitoring\regime_shifts\RS05"
)

HEADLINE_START = "2020-01-01"

# Main confirmatory comparisons:
# H2: Full monitor vs persistence
# H3: Full monitor vs market + own volatility
# H4: Full monitor vs strongest non-novel baseline P + M + V
COMPARISONS = {
    "H2_full_vs_persistence": "S1",
    "H3_full_vs_market_ownvol": "S3",
    "H4_full_vs_PMV": "S4",
}

CHALLENGER_SPEC = "S6"

# H=10 labels overlap for 10 trading days. Use HAC lag H-1 by default.
DEFAULT_HAC_LAG = 9

# Moving block bootstrap. Keep this moderate for a desktop run.
DEFAULT_BLOCK = 10
DEFAULT_BOOT = 3000
DEFAULT_SEED = 42


# =============================================================================
# HELPERS
# =============================================================================

def logloss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1.0 - 1e-8)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def brier(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def newey_west_mean_se(
    x: np.ndarray,
    max_lag: int,
) -> float:
    """
    HAC standard error for the sample mean using Bartlett weights.
    x is already the DATE-AGGREGATED paired loss difference series.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return math.nan

    xbar = float(np.mean(x))
    centered = x - xbar

    gamma0 = float(np.mean(centered * centered))
    var_mean = gamma0 / n

    L = min(int(max_lag), n - 1)

    for lag in range(1, L + 1):
        cov = float(
            np.mean(centered[lag:] * centered[:-lag])
        )
        weight = 1.0 - lag / (L + 1.0)
        var_mean += (2.0 * weight * cov) / n

    return math.sqrt(max(var_mean, 0.0))


def dm_test(
    date_diff: pd.Series,
    hac_lag: int,
) -> tuple[float, float, float]:
    """
    DM-style paired predictive-loss test on DATE-AGGREGATED loss differences.

    Positive diff = baseline loss - challenger loss > 0,
    meaning the challenger (S6) is better.

    Returns:
      mean_improvement, HAC_SE, two-sided p-value
    """
    x = pd.to_numeric(date_diff, errors="coerce").dropna().to_numpy()

    if len(x) < 10:
        return math.nan, math.nan, math.nan

    mean_diff = float(np.mean(x))
    se = newey_west_mean_se(x, hac_lag)

    if not np.isfinite(se) or se <= 0:
        return mean_diff, se, math.nan

    z = mean_diff / se
    p = float(2.0 * norm.sf(abs(z)))
    return mean_diff, se, p


def moving_block_bootstrap(
    x: np.ndarray,
    block_len: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """
    Moving-block bootstrap over DATE-level observations.

    Blocks are sampled from the observed date sequence and concatenated
    until the original number of dates is recovered. This preserves local
    temporal dependence better than row-wise resampling.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return np.array([], dtype=float)

    L = max(1, min(int(block_len), n))
    starts = np.arange(0, n - L + 1)
    rng = np.random.default_rng(seed)

    out = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        pieces = []
        total = 0

        while total < n:
            s = int(rng.choice(starts))
            piece = x[s : s + L]
            pieces.append(piece)
            total += len(piece)

        sample = np.concatenate(pieces)[:n]
        out[b] = float(np.mean(sample))

    return out


def holm_adjust(pvals: pd.Series) -> pd.Series:
    """
    Holm step-down adjustment for a family of p-values.
    NaNs remain NaN.
    """
    out = pd.Series(np.nan, index=pvals.index, dtype=float)
    valid = pvals.dropna()

    if valid.empty:
        return out

    ordered = valid.sort_values()
    m = len(ordered)

    adjusted = []
    running = 0.0

    for rank, (idx, p) in enumerate(ordered.items(), start=1):
        value = min(1.0, (m - rank + 1) * float(p))
        running = max(running, value)
        adjusted.append((idx, running))

    for idx, value in adjusted:
        out.loc[idx] = value

    return out


def get_models_arg(
    requested: str,
    available: list[str],
) -> list[str]:
    if requested.upper() == "ALL":
        return available
    desired = [x.strip() for x in requested.split(",") if x.strip()]
    missing = [x for x in desired if x not in available]
    if missing:
        raise ValueError(
            f"Requested models not found in predictions: {missing}. "
            f"Available={available}"
        )
    return desired


# =============================================================================
# DATA
# =============================================================================

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    p = pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )

    required = {
        "forecast_date",
        "ticker",
        "model",
        "target",
        "H",
        "spec",
        "algo",
        "y",
        "p_cal",
    }

    missing = sorted(required - set(p.columns))
    if missing:
        raise KeyError(
            f"Prediction file missing required columns: {missing}"
        )

    p["forecast_date"] = pd.to_datetime(
        p["forecast_date"],
        errors="coerce",
    )
    p["y"] = pd.to_numeric(p["y"], errors="coerce")
    p["p_cal"] = pd.to_numeric(p["p_cal"], errors="coerce")

    p = p.dropna(
        subset=[
            "forecast_date",
            "y",
            "p_cal",
        ]
    ).copy()

    return p


def make_pair_date_series(
    subset: pd.DataFrame,
    baseline_spec: str,
    challenger_spec: str,
    metric: str,
) -> pd.DataFrame:
    """
    Build paired challenger-vs-baseline loss differences.

    S1 is the deterministic persistence RULE, while S3/S4/S6 are
    learned LOGIT specs in the main run. Therefore the algorithm is selected
    per specification rather than by one global dataframe filter.

    Model is part of the pairing key. Different VaR models can have different
    target labels on the same ticker/date.
    """
    algo_for_spec = {
        "S0": "base",
        "S1": "rule",
        "S2": "logit",
        "S3": "logit",
        "S4": "logit",
        "S5": "logit",
        "S6": "logit",
        "S7": "logit",
    }

    if baseline_spec not in algo_for_spec:
        raise ValueError(
            f"No algorithm mapping for baseline spec {baseline_spec}"
        )
    if challenger_spec not in algo_for_spec:
        raise ValueError(
            f"No algorithm mapping for challenger spec {challenger_spec}"
        )

    base_algo = algo_for_spec[baseline_spec]
    challenger_algo = algo_for_spec[challenger_spec]

    base = subset[
        (subset["spec"] == baseline_spec)
        & (subset["algo"] == base_algo)
    ][
        [
            "forecast_date",
            "ticker",
            "model",
            "y",
            "p_cal",
        ]
    ].copy()

    challenger = subset[
        (subset["spec"] == challenger_spec)
        & (subset["algo"] == challenger_algo)
    ][
        [
            "forecast_date",
            "ticker",
            "model",
            "y",
            "p_cal",
        ]
    ].copy()

    base = base.rename(
        columns={
            "p_cal": "p_base",
            "y": "y_base",
        }
    )

    challenger = challenger.rename(
        columns={
            "p_cal": "p_challenger",
            "y": "y_challenger",
        }
    )

    key = [
        "forecast_date",
        "ticker",
        "model",
    ]

    for label, frame in (
        ("baseline", base),
        ("challenger", challenger),
    ):
        dup = int(frame.duplicated(key).sum())
        if dup:
            raise ValueError(
                f"{label} contains {dup:,} duplicate rows on {key}."
            )

    merged = base.merge(
        challenger,
        on=key,
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame(
            columns=[
                "forecast_date",
                "improvement",
            ]
        )

    y_base = merged["y_base"].to_numpy(dtype=float)
    y_challenger = merged["y_challenger"].to_numpy(dtype=float)

    if not np.array_equal(
        y_base,
        y_challenger,
    ):
        bad = merged.loc[
            merged["y_base"] != merged["y_challenger"],
            [
                "forecast_date",
                "ticker",
                "model",
                "y_base",
                "y_challenger",
            ],
        ].head(10)

        raise ValueError(
            "Matched baseline/challenger rows have inconsistent y values "
            "after matching on model + ticker + date.\n"
            f"Examples:\n{bad.to_string(index=False)}"
        )

    y = y_base
    p_base = merged["p_base"].to_numpy(dtype=float)
    p_challenger = merged["p_challenger"].to_numpy(dtype=float)

    if metric == "brier":
        loss_base = brier(y, p_base)
        loss_challenger = brier(y, p_challenger)
    elif metric == "logloss":
        loss_base = logloss(y, p_base)
        loss_challenger = logloss(y, p_challenger)
    else:
        raise ValueError(metric)

    # Positive improvement means S6 has lower loss.
    merged["improvement"] = loss_base - loss_challenger

    # Inference unit = DATE. Ticker/model rows are averaged within date.
    return (
        merged.groupby(
            "forecast_date",
            as_index=False,
        )["improvement"]
        .mean()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_comparison(
    subset: pd.DataFrame,
    model_label: str,
    comparison_name: str,
    baseline_spec: str,
    metric: str,
    block_len: int,
    n_boot: int,
    seed: int,
    hac_lag: int,
) -> dict:
    date_series = make_pair_date_series(
        subset,
        baseline_spec=baseline_spec,
        challenger_spec=CHALLENGER_SPEC,
        metric=metric,
    )

    x = date_series["improvement"].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 10:
        return {
            "model": model_label,
            "comparison": comparison_name,
            "metric": metric,
            "baseline_spec": baseline_spec,
            "challenger_spec": CHALLENGER_SPEC,
            "n_dates": len(x),
            "mean_improvement": math.nan,
            "hac_se": math.nan,
            "dm_z": math.nan,
            "dm_p": math.nan,
            "bootstrap_ci_low": math.nan,
            "bootstrap_ci_high": math.nan,
            "mde_95_hac": math.nan,
            "bootstrap_p_two_sided": math.nan,
        }

    mean_improvement, hac_se, p = dm_test(
        pd.Series(x),
        hac_lag=hac_lag,
    )

    boot = moving_block_bootstrap(
        x=x,
        block_len=block_len,
        n_boot=n_boot,
        seed=seed,
    )

    ci_low, ci_high = np.quantile(
        boot,
        [0.025, 0.975],
    )

    # Under H0, the observed mean's distance from zero relative to the
    # bootstrap distribution gives a simple empirical two-sided tail area.
    # This is descriptive; HAC p-values are the main confirmatory inference.
    boot_centered = boot - np.mean(boot)
    observed_abs = abs(mean_improvement)
    tail = np.mean(
        np.abs(boot_centered) >= observed_abs
    )
    bootstrap_p = float(
        min(
            1.0,
            2.0 * min(tail, 1.0 - tail),
        )
    )

    mde = (
        1.96 * hac_se
        if np.isfinite(hac_se)
        else math.nan
    )

    dm_z = (
        mean_improvement / hac_se
        if np.isfinite(hac_se) and hac_se > 0
        else math.nan
    )

    return {
        "model": model_label,
        "comparison": comparison_name,
        "metric": metric,
        "baseline_spec": baseline_spec,
        "challenger_spec": CHALLENGER_SPEC,
        "n_dates": len(x),
        "mean_improvement": mean_improvement,
        "hac_se": hac_se,
        "dm_z": dm_z,
        "dm_p": p,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "mde_95_hac": mde,
        "bootstrap_p_two_sided": bootstrap_p,
    }


def run_analysis(
    predictions_path: Path,
    models_arg: str,
    target: str,
    algo: str,
    block_len: int,
    n_boot: int,
    seed: int,
    hac_lag: int,
    exclude_start: str | None,
    exclude_end: str | None,
) -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    p = load_predictions(
        predictions_path
    )

    print("=" * 110)
    print("RS05b STATISTICAL INFERENCE")
    print("=" * 110)
    print(f"Input:        {predictions_path}")
    print(f"Target:       {target}")
    print(
        "Algorithms:   spec-aware "
        "(S1=rule; S3/S4/S6=" + algo + ")"
    )
    print(f"Headline:     {HEADLINE_START} onward")
    print(f"HAC lag:      {hac_lag}")
    print(f"Block length: {block_len}")
    print(f"Bootstrap:    {n_boot}")
    print()

    available_models = sorted(
        p["model"].dropna().astype(str).unique().tolist()
    )
    models = get_models_arg(
        models_arg,
        available_models,
    )

    required_specs = {
        "S1": "rule",
        "S3": algo,
        "S4": algo,
        "S6": algo,
    }

    for spec_name, required_algo in required_specs.items():
        present = p[
            (p["target"] == target)
            & (p["spec"] == spec_name)
            & (p["algo"] == required_algo)
            & (p["forecast_date"] >= pd.Timestamp(HEADLINE_START))
            & (p["model"].isin(models))
        ]
        if present.empty:
            raise RuntimeError(
                f"No predictions found for required spec/algorithm "
                f"{spec_name}/{required_algo}. "
                "Check the RS05a prediction file and --algo."
            )

    # Do not globally filter on algorithm:
    # S1 is "rule", while S3/S4/S6 use the requested learned algorithm.
    p = p[
        (p["target"] == target)
        & (p["forecast_date"] >= pd.Timestamp(HEADLINE_START))
        & (p["model"].isin(models))
    ].copy()

    if p.empty:
        raise RuntimeError(
            "No rows remain after target/algo/date/model filtering."
        )

    if exclude_start and exclude_end:
        es = pd.Timestamp(exclude_start)
        ee = pd.Timestamp(exclude_end)
        before = len(p)

        p = p[
            ~p["forecast_date"].between(
                es,
                ee,
                inclusive="both",
            )
        ].copy()

        print(
            f"Excluded dates: {es.date()} to {ee.date()} "
            f"({before - len(p):,} rows removed)"
        )

    results = []

    # -------------------------------------------------------------------------
    # Pooled across all selected VaR models.
    # Date is the inference unit. Rows within each date are averaged first.
    # -------------------------------------------------------------------------
    pooled = p.copy()

    print()
    print("POOLED ACROSS SELECTED MODELS")
    print("-" * 110)

    for comparison_name, baseline_spec in COMPARISONS.items():
        for metric in ("brier", "logloss"):
            r = analyze_comparison(
                subset=pooled,
                model_label="POOLED",
                comparison_name=comparison_name,
                baseline_spec=baseline_spec,
                metric=metric,
                block_len=block_len,
                n_boot=n_boot,
                seed=seed,
                hac_lag=hac_lag,
            )
            results.append(r)

            print(
                f"{comparison_name:30s} {metric:8s} "
                f"N={r['n_dates']:4d} "
                f"improvement={r['mean_improvement']:+.6g} "
                f"HAC-SE={r['hac_se']:.3g} "
                f"p={r['dm_p']:.4g} "
                f"CI=[{r['bootstrap_ci_low']:.6g}, "
                f"{r['bootstrap_ci_high']:.6g}] "
                f"MDE95={r['mde_95_hac']:.3g}"
            )

    # -------------------------------------------------------------------------
    # Per-model descriptive inference.
    # -------------------------------------------------------------------------
    for model_name in models:
        mdf = p[p["model"] == model_name].copy()

        print()
        print(f"MODEL-SPECIFIC: {model_name}")
        print("-" * 110)

        for comparison_name, baseline_spec in COMPARISONS.items():
            for metric in ("brier", "logloss"):
                r = analyze_comparison(
                    subset=mdf,
                    model_label=model_name,
                    comparison_name=comparison_name,
                    baseline_spec=baseline_spec,
                    metric=metric,
                    block_len=block_len,
                    n_boot=n_boot,
                    seed=seed,
                    hac_lag=hac_lag,
                )
                results.append(r)

                print(
                    f"{comparison_name:30s} {metric:8s} "
                    f"N={r['n_dates']:4d} "
                    f"improvement={r['mean_improvement']:+.6g} "
                    f"p={r['dm_p']:.4g}"
                )

    R = pd.DataFrame(results)

    # Confirmatory Holm family: pooled H2-H4 × Brier/logloss.
    pooled_mask = R["model"].eq("POOLED")
    R["holm_p"] = np.nan
    R.loc[pooled_mask, "holm_p"] = holm_adjust(
        R.loc[
            pooled_mask,
            "dm_p",
        ].reset_index(drop=True)
    ).to_numpy()

    # The indexing above loses original labels; redo by explicit assignment.
    pooled_idx = R.index[pooled_mask].tolist()
    pooled_p = R.loc[pooled_mask, "dm_p"].to_numpy(dtype=float)
    pooled_adj = holm_adjust(
        pd.Series(
            pooled_p,
            index=pooled_idx,
        )
    )

    for idx, val in pooled_adj.items():
        R.loc[idx, "holm_p"] = val

    R["significant_05_holm"] = (
        R["holm_p"] < 0.05
    )

    # Save primary results.
    result_path = (
        OUT_DIR
        / "RS05b_pooled_and_model_inference.csv"
    )
    R.to_csv(
        result_path,
        index=False,
    )

    # Compact pooled table.
    pooled_out = R[
        R["model"] == "POOLED"
    ][
        [
            "comparison",
            "metric",
            "n_dates",
            "mean_improvement",
            "hac_se",
            "dm_z",
            "dm_p",
            "holm_p",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "mde_95_hac",
            "significant_05_holm",
        ]
    ].copy()

    pooled_path = (
        OUT_DIR
        / "RS05b_confirmatory_pooled.csv"
    )
    pooled_out.to_csv(
        pooled_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("CONFIRMATORY POOLED RESULTS")
    print("=" * 110)
    print(
        pooled_out.to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print()
    print("Interpretation:")
    print(
        "  improvement > 0  => S6 has lower predictive loss than the baseline."
    )
    print(
        "  HAC p-value       => primary paired predictive-loss inference."
    )
    print(
        "  Holm p-value      => multiplicity-adjusted confirmatory p-value."
    )
    print(
        "  bootstrap CI      => moving-block date bootstrap 95% interval."
    )
    print(
        "  MDE95             => approximate minimum detectable mean improvement "
        "at 95% using the HAC standard error."
    )

    print()
    print(f"Saved: {result_path}")
    print(f"Saved: {pooled_path}")

    # Optional leave-window result.
    if exclude_start and exclude_end:
        excl_name = (
            f"RS05b_excluded_"
            f"{pd.Timestamp(exclude_start):%Y%m%d}_"
            f"{pd.Timestamp(exclude_end):%Y%m%d}.csv"
        )
        excl_path = OUT_DIR / excl_name

        R[
            R["model"] == "POOLED"
        ].to_csv(
            excl_path,
            index=False,
        )

        print(f"Saved exclusion run: {excl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "RS05b: date-block bootstrap + HAC/DM-style paired "
            "predictive-loss inference for RS05a OOS predictions."
        )
    )

    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
    )

    parser.add_argument(
        "--models",
        default="ALL",
        help="ALL or comma-separated models, e.g. HS,GARCH-t",
    )

    parser.add_argument(
        "--target",
        default="T1",
        choices=["T1", "T2"],
    )

    parser.add_argument(
        "--algo",
        default="logit",
        choices=["logit", "xgb"],
    )

    parser.add_argument(
        "--block-len",
        type=int,
        default=DEFAULT_BLOCK,
    )

    parser.add_argument(
        "--boot",
        type=int,
        default=DEFAULT_BOOT,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--hac-lag",
        type=int,
        default=DEFAULT_HAC_LAG,
    )

    parser.add_argument(
        "--exclude-start",
        default=None,
    )

    parser.add_argument(
        "--exclude-end",
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (
        (args.exclude_start is None)
        != (args.exclude_end is None)
    ):
        raise ValueError(
            "--exclude-start and --exclude-end must be supplied together."
        )

    run_analysis(
        predictions_path=Path(args.predictions),
        models_arg=args.models,
        target=args.target,
        algo=args.algo,
        block_len=args.block_len,
        n_boot=args.boot,
        seed=args.seed,
        hac_lag=args.hac_lag,
        exclude_start=args.exclude_start,
        exclude_end=args.exclude_end,
    )


if __name__ == "__main__":
    main()
