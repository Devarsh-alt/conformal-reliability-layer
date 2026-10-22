
from __future__ import annotations

"""
RS05-CONT — Continuous VaR Deterioration Monitor
=================================================

Research role
-------------
Estimate, strictly out of sample:

    E[ log(Future 10-day normalized pinball loss /
           causal historical 20-day normal loss) | information at t ]

After exponentiating the prediction:

    predicted deterioration multiplier
        = exp(predicted_log_ratio)

Interpretation:
    1.0x = normal expected loss
    1.5x = expected loss 50% above normal
    2.0x = expected loss twice normal
    3.0x = expected loss three times normal

Primary comparison
------------------
S4  = P + M + V
S4L = P + M + V + L
S6  = P + M + V + S + D
S6L = P + M + V + L + S + D

L = validated long-memory volatility features from strong_vol_baseline.py:
    longvol_own_rv60
    longvol_own_rv120
    longvol_own_rv252
    longvol_own_ewma_vol
    longvol_median_rv60
    longvol_median_rv120
    longvol_median_rv252
    longvol_median_ewma_vol

Causal safeguards
-----------------
- quarterly date-based walk-forward test blocks
- expanding training
- H-day purge: training forecast dates must end before the H-day
  forward label window can overlap the first test forecast date
- all tickers on a date stay on the same side of a split
- model-specific fits: HS, EWMA, GARCH-t, QRF separately
- preprocessing is fitted on training rows only
- long-memory volatility is reconstructed causally from returns known at t
- no target/outcome columns enter the features
- no random train/test split

This is deliberately lightweight:
- primary estimator = Ridge regression
- no hyperparameter search
- no XGBoost in the primary run
- metrics are calculated row-level and date-level
"""

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

BASE = Path(r"D:\reliability_layer")
RS_DIR = BASE / "data" / "monitoring" / "regime_shifts"

RS04B_PATH = RS_DIR / "RS04b_causal_reliability_panel.csv"
RS01_PATH = RS_DIR / "RS01_long_history_var_panel.csv"

TARGET_PATH = (
    BASE
    / "data"
    / "results"
    / "continuous_deterioration"
    / "continuous_deterioration_target_H10.csv.gz"
)

OUT_DIR = (
    BASE
    / "data"
    / "results"
    / "continuous_deterioration"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_COL = "forecast_date"
TICKER_COL = "ticker"
MODEL_COL = "model_name"

EXPECTED_MODELS = [
    "EWMA",
    "GARCH-t",
    "HS",
    "QRF",
]

H = 10
HEADLINE_START = pd.Timestamp("2020-01-01")
MIN_FIT_DATES = 300

RIDGE_ALPHA = 10.0

GROUPS = {
    "P": [
        "past_5d_breach_rate",
        "past_10d_breach_rate",
        "past_20d_breach_rate",
        "past_5d_mean_norm_loss",
        "past_10d_mean_norm_loss",
        "past_20d_mean_norm_loss",
        "past_5d_median_norm_loss",
        "past_10d_median_norm_loss",
        "past_20d_median_norm_loss",
        "past_5d_max_norm_loss",
        "past_10d_max_norm_loss",
        "past_20d_max_norm_loss",
        "past_breach_diff_5d_20d",
        "past_normloss_ratio_5d_20d",
        "historical_5d_mean_norm_loss",
        "historical_10d_mean_norm_loss",
        "historical_20d_mean_norm_loss",
        "causal_5d_loss_threshold",
        "causal_10d_loss_threshold",
        "causal_20d_loss_threshold",
        "hist_ratio_5d",
        "hist_ratio_10d",
        "hist_ratio_20d",
    ],
    "M": [
        "vix",
        "median_rv20",
        "vix_change_10d",
        "rv20_change_10d",
        "vix_pct_change_10d",
        "rv20_pct_change_10d",
        "vix_zscore_252d",
        "rv20_zscore_252d",
        "stress_dimensions",
        "regime",
    ],
    "V": [
        "own_rv10",
        "own_rv20",
        "own_rv10_z252",
        "own_rv20_z252",
        "own_abs_return_1d",
        "own_abs_return_mean_5d",
        "own_abs_return_max_5d",
        "own_rv10_rv20_ratio",
    ],
    "L": [
        "longvol_own_rv60",
        "longvol_own_rv120",
        "longvol_own_rv252",
        "longvol_own_ewma_vol",
        "longvol_median_rv60",
        "longvol_median_rv120",
        "longvol_median_rv252",
        "longvol_median_ewma_vol",
    ],
    "S": [
        "current_abs_var",
        "current_var_change_pct_1d",
        "current_var_change_pct_5d",
        "current_var_change_pct_20d",
        "current_abs_var_zscore_252d",
    ],
    "D": [
        "cross_model_disagreement_cv",
        "cross_model_disagreement_range_pct",
        "model_abs_var_to_cross_median",
        "model_abs_var_rank_pct",
    ],
}

SPECS = {
    "S4": ["P", "M", "V"],
    "S4L": ["P", "M", "V", "L"],
    "S6": ["P", "M", "V", "S", "D"],
    "S6L": ["P", "M", "V", "L", "S", "D"],
}


# =============================================================================
# HELPERS
# =============================================================================

def emit(msg: str = ""):
    print(msg, flush=True)


def build_feature_list(spec: str) -> list[str]:
    if spec not in SPECS:
        raise ValueError(f"Unknown spec: {spec}. Expected {list(SPECS)}")

    cols = []
    for group in SPECS[spec]:
        cols.extend(GROUPS[group])

    # Preserve declared order and remove duplicates.
    return list(dict.fromkeys(cols))


def load_base_panel() -> pd.DataFrame:
    if not RS04B_PATH.exists():
        raise FileNotFoundError(
            f"RS04b panel not found:\n{RS04B_PATH}"
        )

    if not TARGET_PATH.exists():
        raise FileNotFoundError(
            f"Continuous target file not found:\n{TARGET_PATH}\n"
            "Run continuous_deterioration_target_audit.py first."
        )

    df = pd.read_csv(RS04B_PATH, low_memory=False)
    target = pd.read_csv(TARGET_PATH, low_memory=False)

    # Normalize keys.
    for frame in (df, target):
        frame[TICKER_COL] = (
            frame[TICKER_COL].astype(str).str.strip().str.upper()
        )
        frame[DATE_COL] = pd.to_datetime(
            frame[DATE_COL], errors="coerce"
        )
        frame[MODEL_COL] = frame[MODEL_COL].astype(str).str.strip()

    # Target columns used by this stage.
    target_cols = [
        TICKER_COL,
        DATE_COL,
        MODEL_COL,
        "future_10d_mean_norm_loss",
        "future_10d_breach_rate",
        "relative_to_hist20",
    ]

    # historical_20d_mean_norm_loss is a feature as well as a field saved
    # in the audit target. Do not merge it blindly when RS04b already has
    # the same column, because pandas would create _x/_y duplicates.

    missing_target = [
        c for c in target_cols if c not in target.columns
    ]
    if missing_target:
        raise KeyError(
            f"Continuous target is missing required columns: {missing_target}"
        )

    target = target[target_cols].copy()

    target = (
        target
        .drop_duplicates(
            [TICKER_COL, DATE_COL, MODEL_COL]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------------
    # Merge future-target columns collision-free.
    #
    # RS04b can already contain some target columns. Earlier versions of this
    # script merged the same columns a second time, causing pandas to create
    # *_x / *_y names and later KeyErrors. We instead make the audit target
    # authoritative for the continuous target fields by removing those
    # duplicate columns before merging them.
    # ---------------------------------------------------------------------

    target_value_cols = [
        "future_10d_mean_norm_loss",
        "future_10d_breach_rate",
        "relative_to_hist20",
    ]

    overlap = [
        c for c in target_value_cols
        if c in df.columns
    ]

    if overlap:
        df = df.drop(columns=overlap)

    target_merge = target[
        [
            TICKER_COL,
            DATE_COL,
            MODEL_COL,
            *target_value_cols,
        ]
    ].copy()

    out = df.merge(
        target_merge,
        on=[TICKER_COL, DATE_COL, MODEL_COL],
        how="inner",
        validate="one_to_one",
    )

    if out.empty:
        raise AssertionError(
            "Feature/target merge returned zero rows."
        )

    # RS04b's causal historical 20-day mean is a FEATURE, not a future
    # outcome. We preserve/recover it separately without creating x/y
    # duplicates.
    if "historical_20d_mean_norm_loss" not in out.columns:
        hist20 = (
            target[
                [
                    TICKER_COL,
                    DATE_COL,
                    MODEL_COL,
                    "historical_20d_mean_norm_loss",
                ]
            ]
            .drop_duplicates(
                [TICKER_COL, DATE_COL, MODEL_COL]
            )
        )

        out = out.merge(
            hist20,
            on=[TICKER_COL, DATE_COL, MODEL_COL],
            how="left",
            validate="one_to_one",
        )

    # The validated RS05 feature manifest includes these three persistence
    # ratios. Reconstruct them whenever an RS04b version does not contain
    # them, using only causal historical quantities.
    for w in (5, 10, 20):
        hist_col = f"historical_{w}d_mean_norm_loss"
        thr_col = f"causal_{w}d_loss_threshold"
        ratio_col = f"hist_ratio_{w}d"

        if (
            ratio_col not in out.columns
            and hist_col in out.columns
            and thr_col in out.columns
        ):
            denom = pd.to_numeric(
                out[thr_col],
                errors="coerce",
            ).replace(0, np.nan)

            out[ratio_col] = (
                pd.to_numeric(
                    out[hist_col],
                    errors="coerce",
                )
                / denom
            )

    required_p = [
        "historical_5d_mean_norm_loss",
        "historical_10d_mean_norm_loss",
        "historical_20d_mean_norm_loss",
        "hist_ratio_5d",
        "hist_ratio_10d",
        "hist_ratio_20d",
    ]

    missing_reconstructed = [
        c for c in required_p
        if c not in out.columns
    ]

    if missing_reconstructed:
        raise KeyError(
            "RS04b/target panel lacks required persistence features "
            f"after collision-safe loading: {missing_reconstructed}"
        )

    # Final hard assertion: no pandas merge suffixes may exist for the target
    # fields used by the continuous stage.
    suffixed = [
        c for c in out.columns
        if c in {
            "future_10d_mean_norm_loss_x",
            "future_10d_mean_norm_loss_y",
            "future_10d_breach_rate_x",
            "future_10d_breach_rate_y",
            "relative_to_hist20_x",
            "relative_to_hist20_y",
        }
    ]

    if suffixed:
        raise AssertionError(
            "Unexpected target merge suffixes remain: "
            f"{suffixed}"
        )

    return out


def add_causal_long_memory_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the validated long-memory volatility features used by
    strong_vol_baseline.py.

    actual_return at RS01 row t corresponds to a return already known at t.
    Rolling statistics therefore use data through t and are causal for
    a forecast made at t.
    """

    required = [
        TICKER_COL,
        DATE_COL,
        "actual_return",
    ]

    if not RS01_PATH.exists():
        raise FileNotFoundError(
            f"RS01 panel not found:\n{RS01_PATH}"
        )

    raw = pd.read_csv(
        RS01_PATH,
        usecols=required,
        low_memory=False,
    )

    raw[TICKER_COL] = (
        raw[TICKER_COL]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    raw[DATE_COL] = pd.to_datetime(
        raw[DATE_COL],
        errors="coerce",
    )
    raw["actual_return"] = pd.to_numeric(
        raw["actual_return"],
        errors="coerce",
    )

    known = (
        raw
        .dropna(subset=[TICKER_COL, DATE_COL, "actual_return"])
        .drop_duplicates([TICKER_COL, DATE_COL])
        .sort_values([TICKER_COL, DATE_COL])
        .reset_index(drop=True)
    )

    pieces = []

    for ticker, g in known.groupby(TICKER_COL, sort=True):
        g = g.copy()
        r = g["actual_return"].astype(float)

        for w in (60, 120, 252):
            g[f"longvol_own_rv{w}"] = (
                r.rolling(w, min_periods=w).std()
            )

        # Same causal EWMA construction used in the validated
        # strong-volatility baseline.
        lam = 0.94
        sq = r.to_numpy(dtype=float) ** 2

        ew = np.full(len(g), np.nan, dtype=float)

        if len(g):
            init_n = min(20, len(g))
            init = (
                np.nanvar(sq[:init_n], ddof=1)
                if init_n > 1
                else sq[0]
            )
            init = float(init) if np.isfinite(init) else 0.0
            v = max(init, 1e-12)

            for i, x in enumerate(sq):
                if np.isfinite(x):
                    v = (
                        lam * v
                        + (1.0 - lam) * float(x)
                    )
                ew[i] = np.sqrt(max(v, 1e-12))

        g["longvol_own_ewma_vol"] = ew
        pieces.append(g)

    stock = pd.concat(pieces, ignore_index=True)

    own_cols = [
        TICKER_COL,
        DATE_COL,
        "longvol_own_rv60",
        "longvol_own_rv120",
        "longvol_own_rv252",
        "longvol_own_ewma_vol",
    ]

    own = stock[own_cols].copy()

    df = df.merge(
        own,
        on=[TICKER_COL, DATE_COL],
        how="left",
        validate="many_to_one",
    )

    # Market-level cross-sectional median.
    market = (
        stock
        .groupby(DATE_COL, as_index=False)
        .agg(
            longvol_median_rv60=(
                "longvol_own_rv60",
                "median",
            ),
            longvol_median_rv120=(
                "longvol_own_rv120",
                "median",
            ),
            longvol_median_rv252=(
                "longvol_own_rv252",
                "median",
            ),
            longvol_median_ewma_vol=(
                "longvol_own_ewma_vol",
                "median",
            ),
            longvol_n_stocks=(
                TICKER_COL,
                "nunique",
            ),
        )
    )

    df = df.merge(
        market,
        on=DATE_COL,
        how="left",
        validate="many_to_one",
    )

    return df


def make_pipeline(feature_cols: list[str]) -> Pipeline:
    numeric_cols = [c for c in feature_cols if c != "regime"]
    categorical_cols = [
        c for c in feature_cols if c == "regime"
    ]

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = Ridge(alpha=RIDGE_ALPHA)

    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )


def quarterly_periods(
    dates: pd.Series,
    start: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    unique = (
        pd.Series(
            pd.to_datetime(dates).dropna().unique()
        )
        .sort_values()
    )

    unique = unique[
        unique >= np.datetime64(start)
    ]

    if unique.empty:
        return []

    # Series.to_period() operates on the Series index in some pandas
    # versions. Use PeriodIndex explicitly so the conversion is applied
    # to the date values rather than the RangeIndex.
    period_series = pd.PeriodIndex(
        pd.to_datetime(unique),
        freq="Q",
    )
    periods = sorted(period_series.unique())

    out = []

    for p in periods:
        p_start = p.start_time
        p_end = p.end_time.normalize()

        test_dates = unique[
            (unique >= np.datetime64(max(p_start, start)))
            & (unique <= np.datetime64(p_end))
        ]

        if len(test_dates):
            out.append(
                (
                    pd.Timestamp(test_dates.min()),
                    pd.Timestamp(test_dates.max()),
                )
            )

    return out


def purge_cutoff(
    all_dates: np.ndarray,
    test_start: pd.Timestamp,
    h: int,
) -> pd.Timestamp:
    """
    Exclude the final h-1 forecast dates immediately before the test block.

    A training row at forecast date t uses outcomes through t+h.
    Therefore, if its label window reaches the first test forecast date,
    its target window overlaps the test labels. We remove those dates.
    """

    dates = pd.Series(
        pd.to_datetime(all_dates).dropna().unique()
    ).sort_values()

    previous = dates[
        dates < test_start
    ]

    if len(previous) <= h:
        return test_start - pd.Timedelta(days=1)

    return pd.Timestamp(
        previous.iloc[-h]
    )


def evaluate_predictions(
    pred: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (model, spec), g in pred.groupby(
        ["model_name", "spec"],
        sort=True,
    ):
        g = g.copy()

        y = g["target_log_ratio"].to_numpy(dtype=float)
        p = g["pred_log_ratio"].to_numpy(dtype=float)

        finite = (
            np.isfinite(y)
            & np.isfinite(p)
        )

        y = y[finite]
        p = p[finite]

        if len(y) < 2:
            continue

        rho = spearmanr(y, p).statistic

        rows.append(
            {
                "model_name": model,
                "spec": spec,
                "level": "row",
                "n": int(len(y)),
                "n_dates": int(
                    g.loc[finite, DATE_COL].nunique()
                ),
                "mae_log": float(
                    mean_absolute_error(y, p)
                ),
                "rmse_log": float(
                    mean_squared_error(y, p) ** 0.5
                ),
                "r2_log": float(
                    r2_score(y, p)
                ),
                "spearman_log": float(rho),
                "mae_multiplier": float(
                    np.mean(
                        np.abs(
                            np.exp(p)
                            - np.exp(y)
                        )
                    )
                ),
                "baseline_mae_log": float(
                    np.mean(np.abs(y))
                ),
            }
        )

        # Date-level aggregation is the dependence-aware evaluation unit
        # used throughout the reliability study.
        d = (
            g.loc[finite]
            .groupby(DATE_COL, as_index=False)
            .agg(
                target_log_ratio=(
                    "target_log_ratio",
                    "mean",
                ),
                pred_log_ratio=(
                    "pred_log_ratio",
                    "mean",
                ),
            )
        )

        yd = d["target_log_ratio"].to_numpy(dtype=float)
        pd_ = d["pred_log_ratio"].to_numpy(dtype=float)

        rho_d = spearmanr(
            yd,
            pd_,
        ).statistic

        rows.append(
            {
                "model_name": model,
                "spec": spec,
                "level": "date",
                "n": int(len(d)),
                "n_dates": int(len(d)),
                "mae_log": float(
                    mean_absolute_error(
                        yd,
                        pd_,
                    )
                ),
                "rmse_log": float(
                    mean_squared_error(
                        yd,
                        pd_,
                    ) ** 0.5
                ),
                "r2_log": float(
                    r2_score(
                        yd,
                        pd_,
                    )
                ),
                "spearman_log": float(rho_d),
                "mae_multiplier": float(
                    np.mean(
                        np.abs(
                            np.exp(pd_)
                            - np.exp(yd)
                        )
                    )
                ),
                "baseline_mae_log": float(
                    np.mean(np.abs(yd))
                ),
            }
        )

    return pd.DataFrame(rows)


def run(args: argparse.Namespace):
    t0 = time.time()

    emit("=" * 110)
    emit("RS05-CONT — CONTINUOUS WALK-FORWARD VA R DETERIORATION MONITOR")
    emit("=" * 110)
    emit(
        f"Target: log(future 10d normalized loss / "
        f"causal historical 20d loss)"
    )
    emit(f"Estimator: Ridge(alpha={RIDGE_ALPHA})")
    emit(f"Headline evaluation starts: {HEADLINE_START.date()}")
    emit()

    df = load_base_panel()

    emit(f"Base merged rows: {len(df):,}")
    emit(
        f"Base forecast range: "
        f"{df[DATE_COL].min().date()} "
        f"to "
        f"{df[DATE_COL].max().date()}"
    )
    emit(
        f"Models: {sorted(df[MODEL_COL].unique())}"
    )

    # Add the same long-memory volatility group used by the validated
    # strong-volatility robustness experiment.
    emit()
    emit("ADDING CAUSAL LONG-MEMORY VOLATILITY FEATURES...")
    df = add_causal_long_memory_features(df)

    missing_l = [
        c
        for c in GROUPS["L"]
        if c not in df.columns
    ]

    if missing_l:
        raise AssertionError(
            f"Missing long-memory features: {missing_l}"
        )

    # Build continuous target.
    ratio = pd.to_numeric(
        df["relative_to_hist20"],
        errors="coerce",
    )

    df["target_log_ratio"] = np.where(
        ratio > 0,
        np.log(ratio),
        np.nan,
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove impossible / missing target rows.
    before = len(df)
    df = df.dropna(
        subset=["target_log_ratio"]
    ).copy()

    emit(
        f"Rows after finite target filter: "
        f"{len(df):,} "
        f"(dropped {before - len(df):,})"
    )

    emit()
    emit("=" * 110)
    emit("FEATURE GROUPS")
    emit("=" * 110)

    for k in ["P", "M", "V", "L", "S", "D"]:
        emit(
            f"{k} ({len(GROUPS[k])}): "
            f"{GROUPS[k]}"
        )

    # One-time schema audit before any expensive fitting.
    all_requested_features = sorted(
        set(
            c
            for spec_name in args.specs
            for c in build_feature_list(spec_name)
        )
    )
    missing_features = [
        c for c in all_requested_features
        if c not in df.columns
    ]

    required_targets = [
        "target_log_ratio",
        "relative_to_hist20",
        "future_10d_mean_norm_loss",
        "future_10d_breach_rate",
    ]
    missing_targets = [
        c for c in required_targets
        if c not in df.columns
    ]

    if missing_features or missing_targets:
        raise AssertionError(
            "PRE-FIT SCHEMA AUDIT FAILED. "
            f"Missing features={missing_features}; "
            f"missing targets={missing_targets}"
        )

    emit(
        "PRE-FIT SCHEMA AUDIT: PASS "
        f"({len(all_requested_features)} features, "
        f"{len(required_targets)} target fields)"
    )

    specs = args.specs
    models = args.models

    invalid_specs = sorted(
        set(specs) - set(SPECS)
    )
    if invalid_specs:
        raise ValueError(
            f"Invalid specs: {invalid_specs}"
        )

    invalid_models = sorted(
        set(models) - set(EXPECTED_MODELS)
    )
    if invalid_models:
        raise ValueError(
            f"Invalid models: {invalid_models}"
        )

    all_dates = np.array(
        sorted(
            pd.to_datetime(
                df[DATE_COL]
                .dropna()
                .unique()
            )
        )
    )

    # Start the walk-forward folds in 2019Q1 so 2019 acts as burn-in,
    # matching the binary RS05 design. Avoid pandas version-specific
    # QuarterBegin(start=...) arguments.
    periods = quarterly_periods(
        pd.Series(all_dates),
        pd.Timestamp("2019-01-01"),
    )

    if args.test_start:
        requested_start = pd.Timestamp(
            args.test_start
        )
        periods = [
            p for p in periods
            if p[1] >= requested_start
        ]

    if args.test_end:
        requested_end = pd.Timestamp(
            args.test_end
        )
        periods = [
            p for p in periods
            if p[0] <= requested_end
        ]

    emit()
    emit(
        f"Quarterly test blocks: {len(periods)}"
    )

    predictions = []

    for model_name in models:
        model_df = df[
            df[MODEL_COL] == model_name
        ].copy()

        if model_df.empty:
            raise AssertionError(
                f"No rows for model {model_name}"
            )

        emit()
        emit("=" * 110)
        emit(
            f"MODEL: {model_name}"
        )
        emit("=" * 110)

        for spec in specs:
            feature_cols = build_feature_list(spec)

            missing = [
                c for c in feature_cols
                if c not in model_df.columns
            ]

            if missing:
                raise KeyError(
                    f"{model_name} {spec} missing "
                    f"features: {missing}"
                )

            emit()
            emit(
                f"{model_name} {spec}: "
                f"{len(feature_cols)} features"
            )

            spec_rows = []

            for fold_idx, (
                test_start,
                test_end,
            ) in enumerate(periods, start=1):

                test = model_df[
                    (
                        model_df[DATE_COL]
                        >= test_start
                    )
                    & (
                        model_df[DATE_COL]
                        <= test_end
                    )
                ].copy()

                if test.empty:
                    continue

                cutoff = purge_cutoff(
                    all_dates,
                    test_start,
                    H,
                )

                train = model_df[
                    model_df[DATE_COL] < cutoff
                ].copy()

                fit_dates = train[
                    DATE_COL
                ].nunique()

                test_dates = test[
                    DATE_COL
                ].nunique()

                if fit_dates < MIN_FIT_DATES:
                    emit(
                        f"  Fold {fold_idx:02d} "
                        f"{test_start.date()} skipped: "
                        f"fit_dates={fit_dates} "
                        f"< {MIN_FIT_DATES}"
                    )
                    continue

                X_train = train[
                    feature_cols
                ].copy()
                y_train = train[
                    "target_log_ratio"
                ].astype(float)

                X_test = test[
                    feature_cols
                ].copy()

                # Pipeline ensures imputation, one-hot encoding and scaling
                # are estimated only from the training block.
                pipe = make_pipeline(
                    feature_cols
                )

                fit_t = time.time()

                pipe.fit(
                    X_train,
                    y_train,
                )

                p = pipe.predict(
                    X_test
                )

                fold_cols = [
                    TICKER_COL,
                    DATE_COL,
                    MODEL_COL,
                    "target_log_ratio",
                    "relative_to_hist20",
                    "future_10d_mean_norm_loss",
                    "future_10d_breach_rate",
                ]

                missing_fold_cols = [
                    c for c in fold_cols
                    if c not in test.columns
                ]

                if missing_fold_cols:
                    raise AssertionError(
                        f"{model_name} {spec} fold is missing required "
                        f"target columns: {missing_fold_cols}"
                    )

                fold_pred = test[fold_cols].copy()

                # Preserve the specification identifier. Without this field,
                # multiple specifications are concatenated correctly but cannot
                # be distinguished during final sorting/evaluation.
                fold_pred["spec"] = spec

                fold_pred["pred_log_ratio"] = p

                # Expected future deterioration multiplier.
                fold_pred[
                    "predicted_deterioration_multiplier"
                ] = np.exp(p)

                fold_pred[
                    "realized_deterioration_multiplier"
                ] = fold_pred[
                    "relative_to_hist20"
                ]

                fold_pred[
                    "fold"
                ] = fold_idx
                fold_pred[
                    "test_start"
                ] = test_start
                fold_pred[
                    "test_end"
                ] = test_end

                spec_rows.append(
                    fold_pred
                )

                fit_seconds = time.time() - fit_t

                emit(
                    f"  {model_name} {spec} "
                    f"{test_start.strftime('%YQ') if False else str(test_start.year) + 'Q' + str(test_start.quarter)} "
                    f"done | "
                    f"fit_dates={fit_dates:,} "
                    f"test_dates={test_dates:,} "
                    f"test_rows={len(test):,} "
                    f"fit={fit_seconds:.2f}s"
                )

            if spec_rows:
                predictions.extend(
                    spec_rows
                )

    if not predictions:
        raise RuntimeError(
            "No predictions were produced."
        )

    pred = pd.concat(
        predictions,
        ignore_index=True,
    )

    # Final prediction schema audit before any evaluation/sorting.
    required_pred_cols = [
        TICKER_COL,
        DATE_COL,
        MODEL_COL,
        "spec",
        "target_log_ratio",
        "relative_to_hist20",
        "future_10d_mean_norm_loss",
        "future_10d_breach_rate",
        "pred_log_ratio",
        "predicted_deterioration_multiplier",
    ]

    missing_pred_cols = [
        c for c in required_pred_cols
        if c not in pred.columns
    ]

    if missing_pred_cols:
        raise AssertionError(
            "FINAL PREDICTION SCHEMA AUDIT FAILED: "
            f"{missing_pred_cols}"
        )

    expected_specs = set(specs)
    actual_specs = set(pred["spec"].dropna().astype(str))
    if actual_specs != expected_specs:
        raise AssertionError(
            "Specification coverage mismatch: "
            f"expected={sorted(expected_specs)}, "
            f"actual={sorted(actual_specs)}"
        )

    expected_models = set(models)
    actual_models = set(
        pred[MODEL_COL].dropna().astype(str)
    )
    if actual_models != expected_models:
        raise AssertionError(
            "Model coverage mismatch: "
            f"expected={sorted(expected_models)}, "
            f"actual={sorted(actual_models)}"
        )

    pred = pred.sort_values(
        [
            MODEL_COL,
            "spec",
            DATE_COL,
            TICKER_COL,
        ]
    ).reset_index(drop=True)

    dup_keys = pred.duplicated(
        [MODEL_COL, "spec", DATE_COL, TICKER_COL]
    )

    if bool(dup_keys.any()):
        raise AssertionError(
            "Duplicate OOS prediction keys detected: "
            f"{int(dup_keys.sum())}"
        )

    # Save raw OOS predictions.
    pred_path = (
        OUT_DIR
        / "continuous_deterioration_oos_predictions_H10.csv.gz"
    )

    pred.to_csv(
        pred_path,
        index=False,
        compression="gzip",
    )

    # Headline evaluation starts in 2020.
    headline = pred[
        pred[DATE_COL] >= HEADLINE_START
    ].copy()

    metrics = evaluate_predictions(
        headline
    )

    metrics_path = (
        OUT_DIR
        / "continuous_deterioration_oos_metrics_H10.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Spec comparison table: S6L minus S4L on date-level metrics.
    # -------------------------------------------------------------------------

    date_metrics = metrics[
        metrics["level"] == "date"
    ].copy()

    comparison_rows = []

    for model_name in models:
        m = date_metrics[
            date_metrics[MODEL_COL] == model_name
        ].set_index("spec")

        if "S4L" not in m.index or "S6L" not in m.index:
            continue

        comparison_rows.append(
            {
                "model_name": model_name,
                "S6L_minus_S4L_MAE_log": (
                    m.loc["S6L", "mae_log"]
                    - m.loc["S4L", "mae_log"]
                ),
                "S6L_minus_S4L_RMSE_log": (
                    m.loc["S6L", "rmse_log"]
                    - m.loc["S4L", "rmse_log"]
                ),
                "S6L_minus_S4L_R2": (
                    m.loc["S6L", "r2_log"]
                    - m.loc["S4L", "r2_log"]
                ),
                "S6L_minus_S4L_Spearman": (
                    m.loc["S6L", "spearman_log"]
                    - m.loc["S4L", "spearman_log"]
                ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        OUT_DIR
        / "continuous_deterioration_S6L_vs_S4L_H10.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # COVID diagnostic: actual and predicted severity by model/spec.
    # -------------------------------------------------------------------------

    covid = headline[
        (
            headline[DATE_COL]
            >= pd.Timestamp("2020-02-24")
        )
        & (
            headline[DATE_COL]
            <= pd.Timestamp("2020-07-17")
        )
    ].copy()

    covid_summary = (
        covid
        .groupby(
            [MODEL_COL, "spec"],
            as_index=False,
        )
        .agg(
            n_rows=(
                TICKER_COL,
                "size",
            ),
            mean_actual_multiplier=(
                "realized_deterioration_multiplier",
                "mean",
            ),
            median_actual_multiplier=(
                "realized_deterioration_multiplier",
                "median",
            ),
            mean_predicted_multiplier=(
                "predicted_deterioration_multiplier",
                "mean",
            ),
            median_predicted_multiplier=(
                "predicted_deterioration_multiplier",
                "median",
            ),
            p90_actual_multiplier=(
                "realized_deterioration_multiplier",
                lambda x: x.quantile(0.90),
            ),
        )
    )

    covid_path = (
        OUT_DIR
        / "continuous_deterioration_COVID_summary_H10.csv"
    )

    covid_summary.to_csv(
        covid_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Daily model-monitoring series.
    # -------------------------------------------------------------------------

    date_series = (
        headline
        .groupby(
            [DATE_COL, MODEL_COL, "spec"],
            as_index=False,
        )
        .agg(
            mean_predicted_multiplier=(
                "predicted_deterioration_multiplier",
                "mean",
            ),
            median_predicted_multiplier=(
                "predicted_deterioration_multiplier",
                "median",
            ),
            mean_realized_multiplier=(
                "realized_deterioration_multiplier",
                "mean",
            ),
            median_realized_multiplier=(
                "realized_deterioration_multiplier",
                "median",
            ),
            mean_future_breach_rate=(
                "future_10d_breach_rate",
                "mean",
            ),
            n_tickers=(
                TICKER_COL,
                "nunique",
            ),
        )
    )

    date_series_path = (
        OUT_DIR
        / "continuous_deterioration_daily_series_H10.csv.gz"
    )

    date_series.to_csv(
        date_series_path,
        index=False,
        compression="gzip",
    )

    # -------------------------------------------------------------------------
    # Headline output.
    # -------------------------------------------------------------------------

    emit()
    emit("=" * 110)
    emit("HEADLINE CONTINUOUS-REGRESSION METRICS — 2020+")
    emit("=" * 110)

    if metrics.empty:
        emit("No metrics available.")
    else:
        emit(
            metrics[
                [
                    "model_name",
                    "spec",
                    "level",
                    "n",
                    "n_dates",
                    "mae_log",
                    "rmse_log",
                    "r2_log",
                    "spearman_log",
                    "mae_multiplier",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

    emit()
    emit("=" * 110)
    emit("S6L vs S4L — DATE-LEVEL DIFFERENCES")
    emit("=" * 110)

    if comparison.empty:
        emit("No S6L/S4L comparison available.")
    else:
        emit(
            comparison
            .round(6)
            .to_string(index=False)
        )

    emit()
    emit("=" * 110)
    emit("COVID — ACTUAL VS PREDICTED DETERIORATION")
    emit("=" * 110)

    if covid_summary.empty:
        emit("No COVID rows.")
    else:
        emit(
            covid_summary
            .round(4)
            .to_string(index=False)
        )

    emit()
    emit("OUTPUT FILES")
    emit(pred_path)
    emit(metrics_path)
    emit(comparison_path)
    emit(covid_path)
    emit(date_series_path)

    emit()
    emit(
        f"Runtime: {time.time() - t0:.1f}s"
    )
    emit("=" * 110)
    emit("RS05-CONT COMPLETE")
    emit("=" * 110)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=EXPECTED_MODELS,
        choices=EXPECTED_MODELS,
    )

    parser.add_argument(
        "--specs",
        nargs="+",
        default=["S4", "S4L", "S6", "S6L"],
        choices=list(SPECS),
    )

    parser.add_argument(
        "--test-start",
        default=None,
        help="Optional earliest test period, e.g. 2020-01-01",
    )

    parser.add_argument(
        "--test-end",
        default=None,
        help="Optional latest test period, e.g. 2026-06-12",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
