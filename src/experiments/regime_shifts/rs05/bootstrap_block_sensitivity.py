from __future__ import annotations

"""
RS05 — BOOTSTRAP BLOCK-LENGTH SENSITIVITY
==========================================

Purpose
-------
Check whether the RED-vs-GREEN empirical reliability-band separation is
robust to reasonable choices of temporal bootstrap block length.

Block lengths tested:
    5, 10, 20 trading dates

Everything else is FIXED:
    - same combined OOS RS05 prediction file
    - same 2020+ headline sample
    - same T1 / T2 targets
    - same S4 / S6 specifications
    - same logistic predictions
    - same frozen thr05 / thr10 / thr20 thresholds
    - same date-level moving-block bootstrap logic
    - 2000 bootstrap repetitions

The bootstrap samples DATE blocks, not individual ticker rows, because:
    1. all tickers share the same dates
    2. H=10 labels overlap through time
    3. ticker-level iid bootstrap would understate temporal dependence

Primary output:
    RED - GREEN realized-rate difference
    95% bootstrap percentile CI

Interpretation:
    Main robustness question:
        Does the 95% CI for RED-GREEN remain above zero across
        block lengths 5, 10, and 20?

The bootstrap probability P(RED-GREEN > 0) is reported descriptively.
It is NOT treated as a conventional hypothesis-test p-value.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd


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

# Same headline start used in the existing audit.
HEADLINE_START = pd.Timestamp("2020-01-01")

MODELS = [
    "EWMA",
    "GARCH-t",
    "HS",
    "QRF",
]

SPECS = [
    "S4",
    "S6",
]

TARGETS = [
    "T1",
    "T2",
]

ALGO = "logit"

# Sensitivity values.
BLOCK_LENGTHS = [
    5,
    10,
    20,
]

BOOT_REPS = 2000
RANDOM_SEED = 42

EPS = 1e-12


# =============================================================================
# OUTPUTS
# =============================================================================

DETAIL_OUT = (
    RS05_DIR
    / "RS05_bootstrap_block_sensitivity_H10.csv"
)

SUMMARY_OUT = (
    RS05_DIR
    / "RS05_bootstrap_block_sensitivity_summary_H10.csv"
)


# =============================================================================
# HELPERS
# =============================================================================

def stable_group_seed(
    target: str,
    model: str,
    spec: str,
    block_length: int,
) -> int:
    """
    Deterministic seed.

    Do NOT use Python's hash() because hash randomization can change
    between processes.
    """

    key = (
        f"{target}|{model}|{spec}|"
        f"BLOCK{block_length}"
    )

    value = 0

    for ch in key:
        value = (
            value * 131
            + ord(ch)
        ) % 1_000_000_007

    return RANDOM_SEED + value


def moving_block_sample(
    n_dates: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Moving-block bootstrap over DATE indices.

    Returns exactly n_dates sampled date indices.
    """

    if n_dates <= 1:
        return np.arange(n_dates)

    L = int(
        min(
            max(1, block_length),
            n_dates,
        )
    )

    n_blocks = int(
        math.ceil(
            n_dates / L
        )
    )

    starts = rng.integers(
        low=0,
        high=n_dates - L + 1,
        size=n_blocks,
    )

    pieces = []

    for start in starts:
        pieces.append(
            np.arange(
                start,
                start + L,
            )
        )

    sampled = np.concatenate(
        pieces
    )

    return sampled[:n_dates]


def assign_trust_band(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Assign GREEN / AMBER / RED using frozen RS05 thresholds.

    IMPORTANT:
    Thresholds are already present in the OOS prediction file.
    They are NOT estimated inside this script.
    """

    return pd.Series(
        np.select(
            [
                df["p_cal"] >= df["thr10"],
                df["p_cal"] >= df["thr20"],
            ],
            [
                "RED",
                "AMBER",
            ],
            default="GREEN",
        ),
        index=df.index,
        name="trust_band",
    )


def prepare_date_level_arrays(
    g: pd.DataFrame,
):
    """
    Aggregate the group to date level.

    For every date we keep:
        GREEN row count
        GREEN event count
        RED row count
        RED event count

    This is mathematically equivalent to resampling all rows belonging
    to the selected dates, while being much faster than repeatedly
    concatenating row indices inside 2000 bootstrap repetitions.
    """

    x = g[
        [
            "forecast_date",
            "y",
            "trust_band",
        ]
    ].copy()

    # Aggregate counts and event totals by date and band.
    agg = (
        x.groupby(
            [
                "forecast_date",
                "trust_band",
            ],
            observed=True,
        )
        .agg(
            n=("y", "size"),
            events=("y", "sum"),
        )
        .reset_index()
    )

    dates = np.sort(
        agg["forecast_date"].unique()
    )

    # Empty date frame ensures dates containing only AMBER observations
    # are still represented.
    date_frame = pd.DataFrame(
        {
            "forecast_date": dates
        }
    )

    for band in [
        "GREEN",
        "RED",
    ]:

        sub = agg[
            agg["trust_band"] == band
        ][
            [
                "forecast_date",
                "n",
                "events",
            ]
        ].copy()

        sub = sub.rename(
            columns={
                "n":
                    f"{band.lower()}_n",
                "events":
                    f"{band.lower()}_events",
            }
        )

        date_frame = date_frame.merge(
            sub,
            on="forecast_date",
            how="left",
        )

    date_frame = date_frame.fillna(0)

    return (
        dates,
        date_frame[
            "red_n"
        ].to_numpy(
            dtype=float
        ),
        date_frame[
            "red_events"
        ].to_numpy(
            dtype=float
        ),
        date_frame[
            "green_n"
        ].to_numpy(
            dtype=float
        ),
        date_frame[
            "green_events"
        ].to_numpy(
            dtype=float
        ),
    )


def bootstrap_red_green_difference(
    g: pd.DataFrame,
    block_length: int,
    reps: int,
    seed: int,
) -> dict:
    """
    Bootstrap RED - GREEN realized-rate separation.

    Trust-band assignment is fixed before bootstrap because the RS05
    thresholds are frozen.

    Returns:
        point estimate
        percentile 95% CI
        bootstrap probability of positive separation
        number of valid bootstrap repetitions
    """

    g = g.copy()

    g["trust_band"] = assign_trust_band(g)

    red = g[
        g["trust_band"] == "RED"
    ]

    green = g[
        g["trust_band"] == "GREEN"
    ]

    if red.empty:
        raise RuntimeError(
            "No RED observations in group."
        )

    if green.empty:
        raise RuntimeError(
            "No GREEN observations in group."
        )

    # -------------------------------------------------------------------------
    # Point estimate from the actual OOS sample.
    # -------------------------------------------------------------------------

    red_rate = float(
        red["y"].mean()
    )

    green_rate = float(
        green["y"].mean()
    )

    point_difference = (
        red_rate
        - green_rate
    )

    # -------------------------------------------------------------------------
    # Date-level aggregates.
    # -------------------------------------------------------------------------

    (
        dates,
        red_n,
        red_events,
        green_n,
        green_events,
    ) = prepare_date_level_arrays(
        g
    )

    n_dates = len(dates)

    rng = np.random.default_rng(
        seed
    )

    bootstrap_values = []

    for _ in range(reps):

        selected = moving_block_sample(
            n_dates=n_dates,
            block_length=block_length,
            rng=rng,
        )

        selected_red_n = (
            red_n[selected].sum()
        )

        selected_red_events = (
            red_events[selected].sum()
        )

        selected_green_n = (
            green_n[selected].sum()
        )

        selected_green_events = (
            green_events[selected].sum()
        )

        if (
            selected_red_n <= EPS
            or selected_green_n <= EPS
        ):
            continue

        boot_red_rate = (
            selected_red_events
            / selected_red_n
        )

        boot_green_rate = (
            selected_green_events
            / selected_green_n
        )

        difference = (
            boot_red_rate
            - boot_green_rate
        )

        if np.isfinite(difference):
            bootstrap_values.append(
                difference
            )

    if len(bootstrap_values) < max(
        20,
        reps // 4,
    ):
        raise RuntimeError(
            "Too few valid bootstrap repetitions."
        )

    bootstrap_values = np.asarray(
        bootstrap_values,
        dtype=float,
    )

    ci_low = float(
        np.quantile(
            bootstrap_values,
            0.025,
        )
    )

    ci_high = float(
        np.quantile(
            bootstrap_values,
            0.975,
        )
    )

    probability_positive = float(
        np.mean(
            bootstrap_values > 0.0
        )
    )

    return {
        "point_red_minus_green":
            point_difference,

        "red_rate":
            red_rate,

        "green_rate":
            green_rate,

        "ci_low":
            ci_low,

        "ci_high":
            ci_high,

        "bootstrap_probability_positive":
            probability_positive,

        "n_bootstrap_valid":
            int(
                len(
                    bootstrap_values
                )
            ),

        "n_dates":
            int(n_dates),

        "n_red_rows":
            int(len(red)),

        "n_green_rows":
            int(len(green)),
    }


# =============================================================================
# INTEGRITY CHECKS
# =============================================================================

def validate_input(
    raw: pd.DataFrame,
) -> pd.DataFrame:

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
    # Normalize.
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

    for col in [
        "thr05",
        "thr10",
        "thr20",
    ]:
        raw[col] = pd.to_numeric(
            raw[col],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Exact intended audit filter.
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
        (raw["algo"] == ALGO)
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

    # -------------------------------------------------------------------------
    # Validate targets.
    # -------------------------------------------------------------------------

    unique_y = set(
        df["y"]
        .astype(int)
        .unique()
    )

    if not unique_y.issubset({0, 1}):
        raise RuntimeError(
            f"Non-binary target values found: {unique_y}"
        )

    df["y"] = (
        df["y"]
        .astype(int)
    )

    # -------------------------------------------------------------------------
    # Validate probabilities.
    # -------------------------------------------------------------------------

    p = df[
        "p_cal"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(p).all():
        raise RuntimeError(
            "Non-finite p_cal values found."
        )

    if (
        (p < 0)
        | (p > 1)
    ).any():
        raise RuntimeError(
            "p_cal contains values outside [0,1]."
        )

    # -------------------------------------------------------------------------
    # Validate frozen thresholds.
    # -------------------------------------------------------------------------

    thresholds = df[
        [
            "thr05",
            "thr10",
            "thr20",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        thresholds
    ).all():
        raise RuntimeError(
            "Non-finite frozen thresholds found."
        )

    # Expected ordering:
    #
    # thr05 >= thr10 >= thr20
    #
    # Because higher forecast probability corresponds to higher reliability
    # concern.

    bad_order = (
        (df["thr05"] < df["thr10"])
        |
        (df["thr10"] < df["thr20"])
    )

    if bad_order.any():
        raise RuntimeError(
            "Frozen threshold ordering violation detected."
        )

    return df


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def build_stability_summary(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    summary_rows = []

    for (
        target,
        model,
        spec,
    ), g in detail.groupby(
        [
            "target",
            "model",
            "spec",
        ],
        sort=True,
    ):

        g = g.sort_values(
            "block_length"
        )

        # All three CIs above zero?
        all_ci_positive = bool(
            (
                g["ci_low"] > 0
            ).all()
        )

        # All point estimates positive?
        all_point_positive = bool(
            (
                g["point_red_minus_green"] > 0
            ).all()
        )

        # Direction stable across all lengths.
        directions = np.sign(
            g[
                "point_red_minus_green"
            ].to_numpy()
        )

        direction_stable = bool(
            np.all(
                directions == directions[0]
            )
        )

        summary_rows.append(
            {
                "target":
                    target,

                "model":
                    model,

                "spec":
                    spec,

                "all_three_CIs_above_zero":
                    all_ci_positive,

                "all_three_point_estimates_positive":
                    all_point_positive,

                "point_estimate_direction_stable":
                    direction_stable,

                "min_point_estimate":
                    float(
                        g[
                            "point_red_minus_green"
                        ].min()
                    ),

                "max_point_estimate":
                    float(
                        g[
                            "point_red_minus_green"
                        ].max()
                    ),

                "min_ci_low":
                    float(
                        g[
                            "ci_low"
                        ].min()
                    ),

                "max_ci_low":
                    float(
                        g[
                            "ci_low"
                        ].max()
                    ),

                "min_ci_high":
                    float(
                        g[
                            "ci_high"
                        ].min()
                    ),

                "max_ci_high":
                    float(
                        g[
                            "ci_high"
                        ].max()
                    ),
            }
        )

    return pd.DataFrame(
        summary_rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 110)
    print(
        "RS05 — BOOTSTRAP BLOCK-LENGTH SENSITIVITY — H10"
    )
    print("=" * 110)

    print(
        f"Input:\n{INPUT_PATH}"
    )

    print(
        f"Block lengths: {BLOCK_LENGTHS}"
    )

    print(
        f"Bootstrap repetitions per group/block: "
        f"{BOOT_REPS}"
    )

    print(
        f"Headline start: "
        f"{HEADLINE_START.date()}"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_PATH}"
        )

    # -------------------------------------------------------------------------
    # Load.
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Validate + filter.
    # -------------------------------------------------------------------------

    df = validate_input(
        raw
    )

    print(
        f"Filtered rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['forecast_date'].min().date()} "
        f"to "
        f"{df['forecast_date'].max().date()}"
    )

    # -------------------------------------------------------------------------
    # Check group structure.
    # -------------------------------------------------------------------------

    groups = (
        df.groupby(
            [
                "target",
                "model",
                "spec",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="n_rows"
        )
    )

    expected_groups = (
        len(TARGETS)
        * len(MODELS)
        * len(SPECS)
    )

    print(
        f"Calibration groups found: "
        f"{len(groups)}"
    )

    if len(groups) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} groups, "
            f"found {len(groups)}."
        )

    # -------------------------------------------------------------------------
    # Run sensitivity.
    # -------------------------------------------------------------------------

    results = []

    total_jobs = (
        expected_groups
        * len(BLOCK_LENGTHS)
    )

    completed = 0

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
        observed=True,
    ):

        print()
        print(
            "-" * 100
        )

        print(
            f"Group: "
            f"{target} | {model} | {spec}"
        )

        print(
            f"Rows: {len(g):,} | "
            f"Dates: {g['forecast_date'].nunique():,} | "
            f"Tickers: {g['ticker'].nunique():,}"
        )

        for block_length in BLOCK_LENGTHS:

            completed += 1

            print(
                f"  [{completed}/{total_jobs}] "
                f"Block length = {block_length} ..."
            )

            seed = stable_group_seed(
                target=target,
                model=model,
                spec=spec,
                block_length=block_length,
            )

            result = (
                bootstrap_red_green_difference(
                    g=g,
                    block_length=block_length,
                    reps=BOOT_REPS,
                    seed=seed,
                )
            )

            row = {
                "target":
                    target,

                "model":
                    model,

                "spec":
                    spec,

                "block_length":
                    block_length,

                "bootstrap_reps":
                    BOOT_REPS,

                "n_bootstrap_valid":
                    result[
                        "n_bootstrap_valid"
                    ],

                "n_dates":
                    result[
                        "n_dates"
                    ],

                "n_red_rows":
                    result[
                        "n_red_rows"
                    ],

                "n_green_rows":
                    result[
                        "n_green_rows"
                    ],

                "red_rate":
                    result[
                        "red_rate"
                    ],

                "green_rate":
                    result[
                        "green_rate"
                    ],

                "point_red_minus_green":
                    result[
                        "point_red_minus_green"
                    ],

                "ci_low":
                    result[
                        "ci_low"
                    ],

                "ci_high":
                    result[
                        "ci_high"
                    ],

                "bootstrap_probability_positive":
                    result[
                        "bootstrap_probability_positive"
                    ],

                # Convenient robustness flag.
                "ci_excludes_zero":
                    bool(
                        result["ci_low"] > 0
                    ),
            }

            results.append(
                row
            )

            print(
                f"      RED     = "
                f"{result['red_rate']:.6f}"
            )

            print(
                f"      GREEN   = "
                f"{result['green_rate']:.6f}"
            )

            print(
                f"      RED-GREEN = "
                f"{result['point_red_minus_green']:.6f}"
            )

            print(
                f"      95% CI   = "
                f"["
                f"{result['ci_low']:.6f}, "
                f"{result['ci_high']:.6f}"
                f"]"
            )

            print(
                f"      CI > 0?  = "
                f"{result['ci_low'] > 0}"
            )

    # -------------------------------------------------------------------------
    # Save detail table.
    # -------------------------------------------------------------------------

    detail = pd.DataFrame(
        results
    )

    detail = detail.sort_values(
        [
            "target",
            "model",
            "spec",
            "block_length",
        ]
    )

    detail.to_csv(
        DETAIL_OUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Build stability summary.
    # -------------------------------------------------------------------------

    summary = build_stability_summary(
        detail
    )

    summary = summary.sort_values(
        [
            "target",
            "model",
            "spec",
        ]
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "BLOCK-LENGTH SENSITIVITY SUMMARY"
    )
    print("=" * 110)

    display_cols = [
        "target",
        "model",
        "spec",
        "all_three_CIs_above_zero",
        "min_point_estimate",
        "max_point_estimate",
        "min_ci_low",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 110
    )
    print(
        "OVERALL COUNTS"
    )
    print(
        "=" * 110
    )

    print(
        "Groups with all 3 bootstrap CIs > 0: "
        f"{int(summary['all_three_CIs_above_zero'].sum())}"
        f"/{len(summary)}"
    )

    print(
        "Groups with all 3 point estimates > 0: "
        f"{int(summary['all_three_point_estimates_positive'].sum())}"
        f"/{len(summary)}"
    )

    print()
    print(
        f"Detailed output:\n{DETAIL_OUT}"
    )

    print(
        f"Summary output:\n{SUMMARY_OUT}"
    )

    print()
    print(
        "DONE."
    )


if __name__ == "__main__":
    main()