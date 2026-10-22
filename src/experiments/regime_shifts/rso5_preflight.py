r"""
RS05 PREFLIGHT — SECTION 0
==========================

This script performs the prerequisites required before the actual RS05 ML models:

1. Label diagnostics
2. Raw-data split-adjustment scan
3. RS04b own-stock volatility feature construction
4. RS04b truncation-invariance tests

It DOES NOT modify RS04.

Run from:
D:\reliability_layer\src\experiments\regime_shifts

Command:
python rso5_preflight.py
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(r"D:\reliability_layer")

DATA_DIR = ROOT / "data"
MONITOR_DIR = DATA_DIR / "monitoring"
RS_DIR = MONITOR_DIR / "regime_shifts"

RS04_PATH = RS_DIR / "RS04_causal_reliability_panel.csv"
QUARANTINE_PATH = RS_DIR / "RS04_quarantined_outcomes.csv"
RAW_ZIP = DATA_DIR / "raw.zip"

RS04B_PATH = RS_DIR / "RS04b_causal_reliability_panel.csv"
LABEL_DIAG_PATH = RS_DIR / "RS05_preflight_label_diagnostics.csv"
SPLIT_SCAN_PATH = RS_DIR / "RS05_preflight_split_adjustment_scan.csv"
TRUNCATION_AUDIT_PATH = RS_DIR / "RS05_preflight_truncation_audit.csv"
OWNVOL_SUMMARY_PATH = RS_DIR / "RS04b_own_volatility_summary.csv"
MANIFEST_PATH = RS_DIR / "RS04b_feature_manifest.json"
LOG_PATH = RS_DIR / "RS05_preflight.log"

EXPECTED_TICKERS = [
    "AAPL", "AMZN", "BA", "BAC", "CAT", "CVX", "DIS", "GME",
    "GOOGL", "GS", "JNJ", "JPM", "KO", "MCD", "MSFT", "NFLX",
    "NVDA", "PFE", "PG", "TSLA", "UNH", "V", "WMT", "XOM",
]

HORIZONS = [5, 10, 20]
MIN_HISTORY = 252
SPLIT_THRESHOLD = 0.40

TRUNCATION_DATES = [
    pd.Timestamp("2018-02-05"),
    pd.Timestamp("2020-03-16"),
    pd.Timestamp("2022-06-01"),
]

OWNVOL_FEATURES = [
    "own_rv10",
    "own_rv20",
    "own_rv10_z252",
    "own_rv20_z252",
    "own_abs_return_1d",
    "own_abs_return_mean_5d",
    "own_abs_return_max_5d",
    "own_rv10_rv20_ratio",
]


# =============================================================================
# LOGGING
# =============================================================================

LOG_LINES: list[str] = []


def log(message: str = "") -> None:
    print(message, flush=True)
    LOG_LINES.append(str(message))


def save_log() -> None:
    LOG_PATH.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")


def section(title: str) -> None:
    log("\n" + "=" * 100)
    log(title)
    log("=" * 100)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def clean_ticker(value) -> str:
    return str(value).strip().upper()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten normal or MultiIndex CSV headers."""
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        new_cols = []
        for tup in out.columns:
            parts = []
            for item in tup:
                s = str(item).strip()
                if s and s.lower() not in {"nan", "none"}:
                    parts.append(s)
            new_cols.append("_".join(parts))
        out.columns = new_cols

    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_column(
    columns,
    candidates: list[str],
    contains: list[str] | None = None,
) -> str | None:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}

    for candidate in candidates:
        if candidate in cols:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if contains:
        for c in cols:
            lc = str(c).lower()
            if all(term.lower() in lc for term in contains):
                return c

    return None


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def get_exact_horizon_column(
    df: pd.DataFrame,
    prefix: str,
    h: int,
) -> str | None:
    """
    RS04 uses:
        future_5d_breach_rate
        future_10d_breach_rate
        future_20d_breach_rate

    This function also accepts common alternate naming conventions.
    """
    exact_candidates = [
        f"{prefix}_{h}d",
        f"{prefix}{h}d",
        f"{prefix}_{h}",
    ]

    for c in exact_candidates:
        if c in df.columns:
            return c

    # Alternate placements of the horizon:
    alternate = [
        f"{prefix.split('_')[0]}_{h}d_" + "_".join(prefix.split("_")[1:]),
        f"{prefix.split('_')[0]}{h}d_" + "_".join(prefix.split("_")[1:]),
    ]

    for c in alternate:
        if c in df.columns:
            return c

    # Generic robust search.
    hpat = re.compile(rf"(^|_){h}d($|_)", re.I)
    candidates = []

    for c in df.columns:
        lc = str(c).lower()
        if hpat.search(lc) and prefix.lower().replace("_", "") in lc.replace("_", ""):
            candidates.append(c)

    if candidates:
        return candidates[0]

    return None


def find_future_breach_rate_column(
    df: pd.DataFrame,
    h: int,
) -> str | None:
    """Robustly identify future_Hd_breach_rate in the actual RS04 schema."""
    candidates = [
        f"future_{h}d_breach_rate",
        f"future_breach_rate_{h}d",
        f"future_breach_rate{h}d",
        f"future_breaches_rate_{h}d",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: exact semantic search.
    for c in df.columns:
        lc = str(c).lower()
        if (
            "future" in lc
            and "breach" in lc
            and "rate" in lc
            and f"{h}d" in lc
        ):
            return c

    return None


def find_deterioration_column(df: pd.DataFrame, h: int) -> str | None:
    candidates = [
        f"deterioration_{h}d",
        f"deterioration{h}d",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        lc = str(c).lower()
        if "deterioration" in lc and f"{h}d" in lc:
            return c

    return None


def find_coverage_column(df: pd.DataFrame, h: int) -> str | None:
    candidates = [
        f"coverage_failure_{h}d",
        f"coveragefailure_{h}d",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        lc = str(c).lower()
        if "coverage" in lc and "failure" in lc and f"{h}d" in lc:
            return c

    return None


# =============================================================================
# RS04 LOAD
# =============================================================================

def load_rs04() -> pd.DataFrame:
    section("LOADING RS04")

    if not RS04_PATH.exists():
        raise FileNotFoundError(f"Missing RS04:\n{RS04_PATH}")

    df = pd.read_csv(RS04_PATH)

    required = ["ticker", "forecast_date", "model_name", "current_abs_var"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"RS04 is missing required columns: {missing}")

    df["ticker"] = df["ticker"].map(clean_ticker)
    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    if df["forecast_date"].isna().any():
        raise ValueError("RS04 contains invalid forecast_date values.")

    df["model_name"] = df["model_name"].astype(str).str.strip()

    df = (
        df.sort_values(["ticker", "forecast_date", "model_name"])
        .reset_index(drop=True)
    )

    duplicates = int(
        df.duplicated(
            ["ticker", "forecast_date", "model_name"]
        ).sum()
    )

    log(f"RS04 path: {RS04_PATH}")
    log(f"RS04 shape: {df.shape}")
    log(f"Duplicate ticker/date/model rows: {duplicates}")
    log(f"Tickers: {df['ticker'].nunique()}")
    log(f"Models: {sorted(df['model_name'].unique().tolist())}")
    log(
        f"Forecast dates: {df['forecast_date'].min().date()} "
        f"to {df['forecast_date'].max().date()}"
    )

    if duplicates:
        raise ValueError("RS04 has duplicate ticker/date/model keys.")

    return df


# =============================================================================
# QUARANTINE / NORMALIZED LOSS
# =============================================================================

def load_and_reconstruct_normalized_loss(
    rs04: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("RECONSTRUCTING NORMALIZED-LOSS DIAGNOSTIC")

    if not QUARANTINE_PATH.exists():
        raise FileNotFoundError(
            f"Missing RS04 quarantine file:\n{QUARANTINE_PATH}"
        )

    q = pd.read_csv(QUARANTINE_PATH)

    required_q = [
        "ticker",
        "forecast_date",
        "realized_date",
        "outcome_next_day_breach",
        "outcome_next_day_pinball_loss",
        "model_name",
    ]

    missing = [c for c in required_q if c not in q.columns]
    if missing:
        raise ValueError(
            f"Quarantine is missing required columns: {missing}"
        )

    q["ticker"] = q["ticker"].map(clean_ticker)
    q["model_name"] = q["model_name"].astype(str).str.strip()
    q["forecast_date"] = pd.to_datetime(
        q["forecast_date"],
        errors="coerce",
    )
    q["realized_date"] = pd.to_datetime(
        q["realized_date"],
        errors="coerce",
    )

    q["outcome_next_day_breach"] = numeric(
        q["outcome_next_day_breach"]
    )
    q["outcome_next_day_pinball_loss"] = numeric(
        q["outcome_next_day_pinball_loss"]
    )

    q = q[
        [
            "ticker",
            "forecast_date",
            "realized_date",
            "outcome_next_day_breach",
            "outcome_next_day_pinball_loss",
            "model_name",
        ]
    ].copy()

    rs_key = rs04[
        ["ticker", "forecast_date", "model_name", "current_abs_var"]
    ].copy()

    q_key = q[
        ["ticker", "forecast_date", "model_name"]
    ].copy()

    rs_keys = set(
        map(
            tuple,
            rs_key[["ticker", "forecast_date", "model_name"]].to_numpy(),
        )
    )

    q_keys = set(
        map(
            tuple,
            q_key[["ticker", "forecast_date", "model_name"]].to_numpy(),
        )
    )

    log(f"Quarantine path: {QUARANTINE_PATH}")
    log(f"Quarantine shape: {q.shape}")
    log(f"Quarantine columns: {q.columns.tolist()}")
    log(
        "Quarantined pinball-loss column: "
        "outcome_next_day_pinball_loss"
    )

    if rs_keys != q_keys:
        raise ValueError(
            "RS04 and quarantine model keys are not identical."
        )

    merged = q.merge(
        rs_key,
        on=["ticker", "forecast_date", "model_name"],
        how="left",
        validate="one_to_one",
    )

    merged["current_abs_var"] = numeric(
        merged["current_abs_var"]
    )

    positive_var = merged["current_abs_var"] > 0

    merged["normalized_loss"] = np.nan
    merged.loc[positive_var, "normalized_loss"] = (
        merged.loc[
            positive_var,
            "outcome_next_day_pinball_loss",
        ]
        / merged.loc[positive_var, "current_abs_var"]
    )

    log(f"Reconstructed normalized-loss rows: {len(merged):,}")
    log(
        "Non-positive |VaR| rows: "
        f"{int((merged['current_abs_var'] <= 0).sum()):,}"
    )
    log(
        "Non-finite normalized-loss rows: "
        f"{int((~np.isfinite(merged['normalized_loss'])).sum()):,}"
    )

    valid_norm = merged["normalized_loss"].replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    log(f"Normalized loss min:    {valid_norm.min():.12g}")
    log(f"Normalized loss median: {valid_norm.median():.12g}")
    log(f"Normalized loss max:    {valid_norm.max():.12g}")
    log(
        "|VaR| min among positive values: "
        f"{merged.loc[positive_var, 'current_abs_var'].min():.12g}"
    )

    log(f"RS04 model keys:       {len(rs_keys):,}")
    log(f"Quarantine model keys: {len(q_keys):,}")
    log(f"Key coverage exact:     {rs_keys == q_keys}")

    return merged, rs_key


# =============================================================================
# LABEL DIAGNOSTICS
# =============================================================================

def count_positive_episodes(
    series: pd.Series,
) -> int:
    s = numeric(series)
    valid = s.notna()

    if not valid.any():
        return 0

    b = (s.loc[valid] == 1).astype(int)
    return int(
        (
            (b == 1)
            & (b.shift(1, fill_value=0) == 0)
        ).sum()
    )


def compute_future_breach_rate_from_quarantine(
    q: pd.DataFrame,
    h: int,
) -> pd.DataFrame:
    """
    Reconstruct future H-day breach rate directly from the quarantined
    next-day outcomes. For each forecast date t, the H outcomes are the
    next H realized outcomes for that ticker/model.

    This is only used as a diagnostic and is not added to RS04b features.
    """
    out_parts = []

    for (ticker, model), g in q.groupby(
        ["ticker", "model_name"],
        sort=False,
    ):
        g = (
            g.sort_values("forecast_date")
            .reset_index(drop=True)
            .copy()
        )

        breach = numeric(g["outcome_next_day_breach"]).astype(float)

        # Current row's outcome is the next-day outcome relative to the
        # current VaR forecast. Therefore H consecutive rows correspond
        # to H future trading-day outcomes.
        future_sum = (
            breach.rolling(h, min_periods=h)
            .sum()
            .shift(-(h - 1))
        )

        future_rate = future_sum / h

        temp = g[
            ["ticker", "forecast_date", "model_name"]
        ].copy()

        temp[f"diagnostic_future_{h}d_breach_rate"] = future_rate

        out_parts.append(temp)

    return pd.concat(out_parts, ignore_index=True)


def label_diagnostics(
    rs04: pd.DataFrame,
    quarantine_with_norm: pd.DataFrame,
) -> pd.DataFrame:
    section("1. LABEL DIAGNOSTICS")

    records = []

    # -------------------------------------------------------------------------
    # Exact target discovery
    # -------------------------------------------------------------------------
    target_map = {}

    for h in HORIZONS:
        dcol = find_deterioration_column(rs04, h)
        ccol = find_coverage_column(rs04, h)
        rcol = find_future_breach_rate_column(rs04, h)

        log(f"H={h}: deterioration column = {dcol}")
        log(f"H={h}: coverage failure column = {ccol}")
        log(f"H={h}: future breach-rate column = {rcol}")

        if dcol is None or ccol is None:
            raise ValueError(
                f"Could not identify H={h} target columns."
            )

        target_map[h] = {
            "deterioration": dcol,
            "coverage": ccol,
            "breach_rate": rcol,
        }

    # -------------------------------------------------------------------------
    # Normalized-loss diagnostics
    # -------------------------------------------------------------------------
    norm = quarantine_with_norm["normalized_loss"]

    log("\nNORMALIZED-LOSS DIAGNOSTICS")
    log(f"Mean normalized loss:   {norm.mean():.10g}")
    log(f"Median normalized loss: {norm.median():.10g}")
    log(f"95th percentile:        {norm.quantile(0.95):.10g}")
    log(f"99th percentile:        {norm.quantile(0.99):.10g}")
    log(f"Maximum:                {norm.max():.10g}")

    # -------------------------------------------------------------------------
    # Overall and yearly target statistics
    # -------------------------------------------------------------------------
    rs04_tmp = rs04.copy()
    rs04_tmp["_year"] = rs04_tmp["forecast_date"].dt.year

    # Save overall diagnostics per model/horizon.
    for model_name, gm in rs04_tmp.groupby("model_name", sort=True):
        gm = gm.sort_values(["ticker", "forecast_date"])

        for h in HORIZONS:
            dcol = target_map[h]["deterioration"]
            ccol = target_map[h]["coverage"]
            rcol = target_map[h]["breach_rate"]

            d = numeric(gm[dcol])
            c = numeric(gm[ccol])

            d_valid = d.notna()
            c_valid = c.notna()

            d_episodes = sum(
                count_positive_episodes(g2[dcol])
                for _, g2 in gm.groupby("ticker")
            )

            c_episodes = sum(
                count_positive_episodes(g2[ccol])
                for _, g2 in gm.groupby("ticker")
            )

            records.extend(
                [
                    {
                        "diagnostic": "overall_target",
                        "model_name": model_name,
                        "year": np.nan,
                        "horizon": h,
                        "metric": "deterioration_prevalence",
                        "value": float(d.mean()),
                        "n": int(d_valid.sum()),
                        "positives": int((d == 1).sum()),
                        "positive_episodes": int(d_episodes),
                    },
                    {
                        "diagnostic": "overall_target",
                        "model_name": model_name,
                        "year": np.nan,
                        "horizon": h,
                        "metric": "coverage_failure_prevalence",
                        "value": float(c.mean()),
                        "n": int(c_valid.sum()),
                        "positives": int((c == 1).sum()),
                        "positive_episodes": int(c_episodes),
                    },
                ]
            )

            # T1/T2 overlap.
            valid = d.notna() & c.notna()
            if valid.any():
                overlap = int(
                    ((d == 1) & (c == 1) & valid).sum()
                )
                t1_pos = int(((d == 1) & valid).sum())
                t2_pos = int(((c == 1) & valid).sum())

                records.extend(
                    [
                        {
                            "diagnostic": "target_overlap",
                            "model_name": model_name,
                            "year": np.nan,
                            "horizon": h,
                            "metric": "t1_t2_overlap_among_t1",
                            "value": (
                                overlap / t1_pos
                                if t1_pos
                                else np.nan
                            ),
                            "n": int(valid.sum()),
                            "positives": overlap,
                            "positive_episodes": np.nan,
                        },
                        {
                            "diagnostic": "target_overlap",
                            "model_name": model_name,
                            "year": np.nan,
                            "horizon": h,
                            "metric": "t1_t2_overlap_among_t2",
                            "value": (
                                overlap / t2_pos
                                if t2_pos
                                else np.nan
                            ),
                            "n": int(valid.sum()),
                            "positives": overlap,
                            "positive_episodes": np.nan,
                        },
                    ]
                )

            # -----------------------------------------------------------------
            # Zero-breach share among T1 positives.
            #
            # Prefer actual RS04 future breach rate if present.
            # Otherwise reconstruct it from quarantine outcomes.
            # -----------------------------------------------------------------
            if rcol is not None:
                rate = numeric(gm[rcol])
                rate_source = "RS04_future_breach_rate"
            else:
                reconstructed = compute_future_breach_rate_from_quarantine(
                    quarantine_with_norm,
                    h,
                )

                gm_keys = gm[
                    ["ticker", "forecast_date", "model_name"]
                ]

                gm_rate = gm_keys.merge(
                    reconstructed,
                    on=["ticker", "forecast_date", "model_name"],
                    how="left",
                    validate="one_to_one",
                )

                rate = numeric(
                    gm_rate[
                        f"diagnostic_future_{h}d_breach_rate"
                    ]
                )
                rate_source = "quarantine_reconstruction"

            t1 = numeric(gm[dcol])

            mask = t1.eq(1) & rate.notna()

            if mask.any():
                # A rate of 0 is exactly zero breaches.
                zero_share = float(
                    (rate.loc[mask] == 0).mean()
                )
            else:
                zero_share = np.nan

            records.append(
                {
                    "diagnostic": "t1_zero_breach",
                    "model_name": model_name,
                    "year": np.nan,
                    "horizon": h,
                    "metric": "share_t1_positive_windows_with_zero_breaches",
                    "value": zero_share,
                    "n": int(mask.sum()),
                    "positives": int((rate.loc[mask] == 0).sum()),
                    "positive_episodes": np.nan,
                    "source": rate_source,
                }
            )

            log(
                f"{model_name} H={h}: "
                f"T1-positive windows with zero breaches = "
                f"{zero_share:.4%} "
                f"[source={rate_source}]"
            )

            # -----------------------------------------------------------------
            # T1/T2 yearly prevalence.
            # -----------------------------------------------------------------
            for year, gy in gm.groupby("_year", sort=True):
                dy = numeric(gy[dcol])
                cy = numeric(gy[ccol])

                records.extend(
                    [
                        {
                            "diagnostic": "yearly_prevalence",
                            "model_name": model_name,
                            "year": int(year),
                            "horizon": h,
                            "metric": "deterioration_prevalence",
                            "value": float(dy.mean()),
                            "n": int(dy.notna().sum()),
                            "positives": int((dy == 1).sum()),
                            "positive_episodes": count_positive_episodes(dy),
                        },
                        {
                            "diagnostic": "yearly_prevalence",
                            "model_name": model_name,
                            "year": int(year),
                            "horizon": h,
                            "metric": "coverage_failure_prevalence",
                            "value": float(cy.mean()),
                            "n": int(cy.notna().sum()),
                            "positives": int((cy == 1).sum()),
                            "positive_episodes": count_positive_episodes(cy),
                        },
                    ]
                )

    # -------------------------------------------------------------------------
    # Cross-model same-date label correlations.
    # -------------------------------------------------------------------------
    log("\nCROSS-MODEL LABEL CORRELATIONS")

    corr_records = []

    for h in HORIZONS:
        for target_type in ["deterioration", "coverage"]:
            col = target_map[h][target_type]

            pivot = rs04[
                ["ticker", "forecast_date", "model_name", col]
            ].copy()

            pivot[col] = numeric(pivot[col])

            wide = pivot.pivot_table(
                index=["ticker", "forecast_date"],
                columns="model_name",
                values=col,
                aggfunc="first",
            )

            corr = wide.corr()

            for m1 in corr.index:
                for m2 in corr.columns:
                    if m1 >= m2:
                        continue

                    value = float(corr.loc[m1, m2])

                    corr_records.append(
                        {
                            "diagnostic": "cross_model_label_correlation",
                            "model_name": f"{m1} vs {m2}",
                            "year": np.nan,
                            "horizon": h,
                            "metric": f"{target_type}_correlation",
                            "value": value,
                            "n": int(wide[[m1, m2]].dropna().shape[0]),
                            "positives": np.nan,
                            "positive_episodes": np.nan,
                        }
                    )

    records.extend(corr_records)

    # -------------------------------------------------------------------------
    # Effective sample size.
    # -------------------------------------------------------------------------
    unique_dates = (
        rs04["forecast_date"]
        .drop_duplicates()
        .sort_values()
    )

    for h in HORIZONS:
        effective_windows = len(unique_dates) // h

        records.append(
            {
                "diagnostic": "effective_sample",
                "model_name": "ALL",
                "year": np.nan,
                "horizon": h,
                "metric": "unique_forecast_dates",
                "value": float(len(unique_dates)),
                "n": int(len(unique_dates)),
                "positives": np.nan,
                "positive_episodes": np.nan,
            }
        )

        records.append(
            {
                "diagnostic": "effective_sample",
                "model_name": "ALL",
                "year": np.nan,
                "horizon": h,
                "metric": "approx_nonoverlapping_windows",
                "value": float(effective_windows),
                "n": int(effective_windows),
                "positives": np.nan,
                "positive_episodes": np.nan,
            }
        )

    out = pd.DataFrame(records)

    out.to_csv(
        LABEL_DIAG_PATH,
        index=False,
    )

    log(f"\nSaved label diagnostics:\n{LABEL_DIAG_PATH}")

    log("\nOVERALL TARGET SUMMARY")
    overall = out[
        out["diagnostic"] == "overall_target"
    ].copy()

    if not overall.empty:
        cols = [
            "model_name",
            "horizon",
            "metric",
            "value",
            "n",
            "positives",
            "positive_episodes",
        ]
        log(
            overall[cols].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    return out


# =============================================================================
# RAW ZIP / SPLIT SCAN
# =============================================================================

def infer_ticker(member: str, df: pd.DataFrame | None = None) -> str | None:
    stem = Path(member).stem.strip().upper()

    if stem in EXPECTED_TICKERS:
        return stem

    for ticker in EXPECTED_TICKERS:
        if re.search(
            rf"(^|[_\-.]){re.escape(ticker)}([_\-.]|$)",
            stem,
        ):
            return ticker

    if df is not None:
        ticker_col = find_column(
            df.columns,
            ["ticker", "Ticker", "symbol", "Symbol"],
        )

        if ticker_col is not None:
            values = (
                df[ticker_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
            )

            if len(values) == 1 and values[0] in EXPECTED_TICKERS:
                return values[0]

    return None


def prepare_raw_price_series(
    raw: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    raw = flatten_columns(raw)

    date_col = find_column(
        raw.columns,
        ["Date", "date", "datetime", "timestamp"],
    )

    close_col = find_column(
        raw.columns,
        ["Close", "close"],
        contains=["close"],
    )

    adj_close_col = find_column(
        raw.columns,
        ["Adj Close", "Adj_Close", "adj_close", "adj close"],
    )

    if date_col is None or close_col is None:
        raise ValueError(
            f"{ticker}: unable to identify date/close columns. "
            f"Columns={list(raw.columns)}"
        )

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(
                raw[date_col],
                errors="coerce",
            ),
            "close": numeric(raw[close_col]),
        }
    )

    if adj_close_col is not None:
        out["adj_close"] = numeric(
            raw[adj_close_col]
        )
    else:
        out["adj_close"] = np.nan

    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )

    return out


def split_adjustment_scan() -> pd.DataFrame:
    section("2. RAW-DATA SPLIT-ADJUSTMENT SCAN")

    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Raw zip not found:\n{RAW_ZIP}")

    results = []

    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        members = [
            m
            for m in zf.namelist()
            if m.lower().endswith(".csv")
            and not m.endswith("/")
        ]

        log(f"CSV members in raw.zip: {len(members)}")

        seen_tickers: set[str] = set()

        for member in sorted(members):
            try:
                raw = pd.read_csv(
                    io.BytesIO(zf.read(member))
                )

                ticker = infer_ticker(
                    member,
                    raw,
                )

                if ticker not in EXPECTED_TICKERS:
                    continue

                if ticker in seen_tickers:
                    continue

                series = prepare_raw_price_series(
                    raw,
                    ticker,
                )

                seen_tickers.add(ticker)

                series["raw_log_return"] = np.log(
                    series["close"]
                    / series["close"].shift(1)
                )

                if series["adj_close"].notna().any():
                    series["adj_log_return"] = np.log(
                        series["adj_close"]
                        / series["adj_close"].shift(1)
                    )
                else:
                    series["adj_log_return"] = np.nan

                large = series[
                    series["raw_log_return"].abs()
                    > SPLIT_THRESHOLD
                ].copy()

                log(
                    f"{ticker}: {len(large)} raw |log return| > "
                    f"{SPLIT_THRESHOLD:.2f}"
                )

                for _, row in large.iterrows():
                    raw_ret = float(row["raw_log_return"])

                    adj_ret = row["adj_log_return"]
                    adj_ret = (
                        float(adj_ret)
                        if pd.notna(adj_ret)
                        else np.nan
                    )

                    likely_split = False

                    if np.isfinite(adj_ret):
                        # Very large raw jump that largely disappears in
                        # adjusted prices is a likely split artifact.
                        likely_split = (
                            abs(raw_ret) > SPLIT_THRESHOLD
                            and abs(adj_ret) < 0.10
                        )

                    if likely_split:
                        interpretation = (
                            "Likely split artifact: "
                            "large raw jump largely disappears after adjustment."
                        )
                    elif np.isfinite(adj_ret):
                        interpretation = (
                            "Large move remains material after adjustment; "
                            "manual review."
                        )
                    else:
                        interpretation = (
                            "No separate adjusted-close series; "
                            "manual review."
                        )

                    results.append(
                        {
                            "member": member,
                            "ticker": ticker,
                            "date": row["date"],
                            "raw_log_return": raw_ret,
                            "adj_log_return": adj_ret,
                            "abs_raw_log_return": abs(raw_ret),
                            "abs_adj_log_return": (
                                abs(adj_ret)
                                if np.isfinite(adj_ret)
                                else np.nan
                            ),
                            "likely_split_artifact": bool(likely_split),
                            "interpretation": interpretation,
                        }
                    )

            except Exception as exc:
                log(
                    f"WARNING: failed to inspect {member}: {exc}"
                )

    if not results:
        # Still save an empty-but-well-formed table.
        out = pd.DataFrame(
            columns=[
                "member",
                "ticker",
                "date",
                "raw_log_return",
                "adj_log_return",
                "abs_raw_log_return",
                "abs_adj_log_return",
                "likely_split_artifact",
                "interpretation",
            ]
        )
    else:
        out = pd.DataFrame(results)

    out.to_csv(
        SPLIT_SCAN_PATH,
        index=False,
    )

    found = sorted(
        set(out["ticker"].dropna().astype(str))
        & set(EXPECTED_TICKERS)
    )

    log(f"\nExpected tickers found/scanned: {len(found)} / 24")

    if len(found) < len(EXPECTED_TICKERS):
        missing = sorted(
            set(EXPECTED_TICKERS) - set(found)
        )
        log(f"WARNING — ticker files not identified: {missing}")

    flagged = int(
        out["likely_split_artifact"].sum()
    ) if not out.empty else 0

    log(f"Likely split artifacts: {flagged}")
    log(
        f"Saved split scan:\n{SPLIT_SCAN_PATH}"
    )

    if not out.empty:
        log("\nLARGE-MOVE EVENTS")
        log(
            out.to_string(
                index=False,
            )
        )

    return out


# =============================================================================
# RAW RETURNS
# =============================================================================

def load_raw_returns() -> pd.DataFrame:
    section("BUILDING STOCK-LEVEL RAW RETURNS")

    if not RAW_ZIP.exists():
        raise FileNotFoundError(f"Raw zip not found:\n{RAW_ZIP}")

    frames = []

    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        members = [
            m
            for m in zf.namelist()
            if m.lower().endswith(".csv")
            and not m.endswith("/")
        ]

        seen_tickers: set[str] = set()

        for member in sorted(members):
            raw = pd.read_csv(
                io.BytesIO(zf.read(member))
            )

            ticker = infer_ticker(
                member,
                raw,
            )

            if ticker not in EXPECTED_TICKERS:
                continue

            if ticker in seen_tickers:
                continue

            series = prepare_raw_price_series(
                raw,
                ticker,
            )

            series["log_return"] = np.log(
                series["close"]
                / series["close"].shift(1)
            )

            frames.append(
                series[
                    [
                        "ticker",
                        "date",
                        "close",
                        "log_return",
                    ]
                ]
            )

            seen_tickers.add(ticker)

    if len(seen_tickers) != len(EXPECTED_TICKERS):
        missing = sorted(
            set(EXPECTED_TICKERS) - seen_tickers
        )
        raise ValueError(
            f"Could not load all 24 stock series. Missing: {missing}"
        )

    raw_returns = pd.concat(
        frames,
        ignore_index=True,
    )

    raw_returns = (
        raw_returns
        .sort_values(["ticker", "date"])
        .drop_duplicates(
            ["ticker", "date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    log(f"Raw return rows: {len(raw_returns):,}")
    log(f"Tickers: {raw_returns['ticker'].nunique()}")
    log(
        f"Date range: {raw_returns['date'].min().date()} "
        f"to {raw_returns['date'].max().date()}"
    )

    return raw_returns


# =============================================================================
# CAUSAL OWN-VOLATILITY FEATURES
# =============================================================================

def build_own_vol_features(
    raw_returns: pd.DataFrame,
) -> pd.DataFrame:
    section("BUILDING RS04b OWN-VOLATILITY FEATURES")

    parts = []

    for ticker, g in raw_returns.groupby(
        "ticker",
        sort=True,
    ):
        g = (
            g.sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        r = numeric(g["log_return"])
        abs_r = r.abs()

        rv10 = r.rolling(
            10,
            min_periods=10,
        ).std(ddof=1)

        rv20 = r.rolling(
            20,
            min_periods=20,
        ).std(ddof=1)

        # Historical reference distribution excludes the current date.
        rv10_mean_prev = (
            rv10.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .mean()
            .shift(1)
        )

        rv10_std_prev = (
            rv10.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .std(ddof=1)
            .shift(1)
        )

        rv20_mean_prev = (
            rv20.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .mean()
            .shift(1)
        )

        rv20_std_prev = (
            rv20.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .std(ddof=1)
            .shift(1)
        )

        temp = pd.DataFrame(
            {
                "ticker": ticker,
                "forecast_date": g["date"],
                "own_rv10": rv10,
                "own_rv20": rv20,
                "own_rv10_z252": (
                    (rv10 - rv10_mean_prev)
                    / rv10_std_prev
                ),
                "own_rv20_z252": (
                    (rv20 - rv20_mean_prev)
                    / rv20_std_prev
                ),
                "own_abs_return_1d": abs_r,
                "own_abs_return_mean_5d": (
                    abs_r.rolling(
                        5,
                        min_periods=5,
                    ).mean()
                ),
                "own_abs_return_max_5d": (
                    abs_r.rolling(
                        5,
                        min_periods=5,
                    ).max()
                ),
                "own_rv10_rv20_ratio": (
                    rv10
                    / rv20.replace(0, np.nan)
                ),
            }
        )

        parts.append(temp)

    features = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["ticker", "forecast_date"])
        .drop_duplicates(
            ["ticker", "forecast_date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    log(
        f"Own-volatility rows: {len(features):,}"
    )

    summary = []

    for col in OWNVOL_FEATURES:
        s = numeric(features[col])

        summary.append(
            {
                "feature": col,
                "rows": len(s),
                "valid": int(s.notna().sum()),
                "missing_rate": float(s.isna().mean()),
                "min": float(s.min()) if s.notna().any() else np.nan,
                "median": (
                    float(s.median())
                    if s.notna().any()
                    else np.nan
                ),
                "max": float(s.max()) if s.notna().any() else np.nan,
            }
        )

    pd.DataFrame(summary).to_csv(
        OWNVOL_SUMMARY_PATH,
        index=False,
    )

    log(
        f"Saved own-volatility summary:\n"
        f"{OWNVOL_SUMMARY_PATH}"
    )

    return features


# =============================================================================
# RS04b MERGE
# =============================================================================

def load_existing_feature_manifest() -> list[str]:
    if not MANIFEST_PATH.exists():
        return []

    try:
        obj = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return []

    features = obj.get("features", [])
    if not isinstance(features, list):
        return []

    return [
        str(x)
        for x in features
    ]


def build_rs04b(
    rs04: pd.DataFrame,
    own_features: pd.DataFrame,
) -> pd.DataFrame:
    section("MERGING RS04 + OWN-VOLATILITY FEATURES")

    key = ["ticker", "forecast_date"]

    dup = int(
        own_features.duplicated(key).sum()
    )

    if dup:
        raise ValueError(
            f"Own-vol table has {dup} duplicate keys."
        )

    before = len(rs04)

    rs04b = rs04.merge(
        own_features,
        on=key,
        how="left",
        validate="many_to_one",
    )

    if len(rs04b) != before:
        raise ValueError(
            "RS04b row count changed during merge."
        )

    for col in OWNVOL_FEATURES:
        if col not in rs04b.columns:
            raise ValueError(
                f"Missing RS04b feature after merge: {col}"
            )

    existing = load_existing_feature_manifest()

    # The original RS04 manifest should exist, but if it does not,
    # infer feature columns conservatively.
    if not existing:
        excluded = {
            "ticker",
            "forecast_date",
            "realized_date",
            "model_name",
        }

        target_prefixes = (
            "future_",
            "deterioration_",
            "coverage_failure_",
            "coverage_failure_k_",
            "label_start_date_",
            "label_end_date_",
        )

        existing = [
            c
            for c in rs04.columns
            if (
                c not in excluded
                and not c.startswith(target_prefixes)
            )
        ]

    existing = [
        c
        for c in existing
        if c in rs04b.columns
    ]

    feature_manifest = list(
        dict.fromkeys(
            existing + OWNVOL_FEATURES
        )
    )

    manifest = {
        "base_panel": str(RS04_PATH),
        "output_panel": str(RS04B_PATH),
        "feature_count": len(feature_manifest),
        "features": feature_manifest,
        "new_features": OWNVOL_FEATURES,
        "causality": {
            "rolling_volatility": (
                "Computed from returns through forecast_date."
            ),
            "zscore": (
                "Current RV is standardized against the "
                "previous 252 RV observations; current RV "
                "is excluded from its historical reference."
            ),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    rs04b.to_csv(
        RS04B_PATH,
        index=False,
    )

    log(f"RS04 rows:  {before:,}")
    log(f"RS04b rows: {len(rs04b):,}")
    log(f"RS04b feature count: {len(feature_manifest)}")
    log(
        f"Saved RS04b panel:\n{RS04B_PATH}"
    )
    log(
        f"Saved feature manifest:\n{MANIFEST_PATH}"
    )

    return rs04b


# =============================================================================
# TRUNCATION TEST
# =============================================================================

def own_vol_features_from_raw(
    raw_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Silent version used by truncation tests."""
    parts = []

    for ticker, g in raw_returns.groupby(
        "ticker",
        sort=True,
    ):
        g = (
            g.sort_values("date")
            .reset_index(drop=True)
        )

        r = numeric(g["log_return"])
        abs_r = r.abs()

        rv10 = r.rolling(
            10,
            min_periods=10,
        ).std(ddof=1)

        rv20 = r.rolling(
            20,
            min_periods=20,
        ).std(ddof=1)

        rv10_mean_prev = (
            rv10.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .mean()
            .shift(1)
        )

        rv10_std_prev = (
            rv10.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .std(ddof=1)
            .shift(1)
        )

        rv20_mean_prev = (
            rv20.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .mean()
            .shift(1)
        )

        rv20_std_prev = (
            rv20.rolling(
                MIN_HISTORY,
                min_periods=MIN_HISTORY,
            )
            .std(ddof=1)
            .shift(1)
        )

        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "forecast_date": g["date"].to_numpy(),
                    "own_rv10": rv10.to_numpy(),
                    "own_rv20": rv20.to_numpy(),
                    "own_rv10_z252": (
                        (
                            rv10
                            - rv10_mean_prev
                        )
                        / rv10_std_prev
                    ).to_numpy(),
                    "own_rv20_z252": (
                        (
                            rv20
                            - rv20_mean_prev
                        )
                        / rv20_std_prev
                    ).to_numpy(),
                    "own_abs_return_1d": abs_r.to_numpy(),
                    "own_abs_return_mean_5d": (
                        abs_r.rolling(
                            5,
                            min_periods=5,
                        )
                        .mean()
                        .to_numpy()
                    ),
                    "own_abs_return_max_5d": (
                        abs_r.rolling(
                            5,
                            min_periods=5,
                        )
                        .max()
                        .to_numpy()
                    ),
                    "own_rv10_rv20_ratio": (
                        rv10
                        / rv20.replace(
                            0,
                            np.nan,
                        )
                    ).to_numpy(),
                }
            )
        )

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )


def truncation_test(
    full_features: pd.DataFrame,
    raw_returns: pd.DataFrame,
) -> pd.DataFrame:
    section("3. RS04b TRUNCATION-INVARIANCE TEST")

    full = (
        full_features[
            [
                "ticker",
                "forecast_date",
            ]
            + OWNVOL_FEATURES
        ]
        .drop_duplicates(
            ["ticker", "forecast_date"]
        )
        .set_index(
            ["ticker", "forecast_date"]
        )
        .sort_index()
    )

    audit = []

    for T in TRUNCATION_DATES:
        cut_raw = raw_returns[
            raw_returns["date"] <= T
        ].copy()

        truncated = own_vol_features_from_raw(
            cut_raw
        )

        truncated = (
            truncated[
                [
                    "ticker",
                    "forecast_date",
                ]
                + OWNVOL_FEATURES
            ]
            .drop_duplicates(
                ["ticker", "forecast_date"]
            )
            .set_index(
                ["ticker", "forecast_date"]
            )
            .sort_index()
        )

        keys = full.index[
            full.index.get_level_values(
                "forecast_date"
            ) <= T
        ]

        a = full.loc[keys]
        b = truncated.reindex(keys)

        if a.shape != b.shape:
            raise AssertionError(
                f"Truncation shape mismatch at {T.date()}: "
                f"{a.shape} vs {b.shape}"
            )

        for col in OWNVOL_FEATURES:
            av = numeric(a[col]).to_numpy(
                dtype=float
            )
            bv = numeric(b[col]).to_numpy(
                dtype=float
            )

            both_nan = (
                np.isnan(av)
                & np.isnan(bv)
            )

            equal = (
                np.isfinite(av)
                & np.isfinite(bv)
                & np.isclose(
                    av,
                    bv,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )

            passed = bool(
                np.all(
                    both_nan | equal
                )
            )

            max_diff = np.nan
            finite = (
                np.isfinite(av)
                & np.isfinite(bv)
            )

            if finite.any():
                max_diff = float(
                    np.max(
                        np.abs(
                            av[finite] - bv[finite]
                        )
                    )
                )

            audit.append(
                {
                    "truncation_date": T.date(),
                    "feature": col,
                    "rows_compared": len(av),
                    "passed": passed,
                    "max_abs_diff": max_diff,
                }
            )

            if not passed:
                raise AssertionError(
                    f"Truncation test failed: "
                    f"{T.date()} / {col}"
                )

        log(
            f"{T.date()} — "
            f"{len(OWNVOL_FEATURES)} / "
            f"{len(OWNVOL_FEATURES)} features PASSED"
        )

    out = pd.DataFrame(audit)

    out.to_csv(
        TRUNCATION_AUDIT_PATH,
        index=False,
    )

    log(
        f"\nSaved truncation audit:\n"
        f"{TRUNCATION_AUDIT_PATH}"
    )

    return out


# =============================================================================
# FINAL SANITY
# =============================================================================

def final_sanity(
    rs04: pd.DataFrame,
    rs04b: pd.DataFrame,
    truncation: pd.DataFrame,
) -> None:
    section("FINAL PREFLIGHT SANITY CHECK")

    checks: list[tuple[str, bool]] = []

    checks.append(
        (
            "RS04 row count preserved",
            len(rs04b) == len(rs04),
        )
    )

    checks.append(
        (
            "Zero duplicate model keys",
            int(
                rs04b.duplicated(
                    [
                        "ticker",
                        "forecast_date",
                        "model_name",
                    ]
                ).sum()
            ) == 0,
        )
    )

    for col in OWNVOL_FEATURES:
        checks.append(
            (
                f"Feature present: {col}",
                col in rs04b.columns,
            )
        )

    checks.append(
        (
            "All truncation tests passed",
            not truncation.empty
            and bool(truncation["passed"].all()),
        )
    )

    for name, passed in checks:
        log(
            f"{name:<65} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    failures = [
        name
        for name, passed in checks
        if not passed
    ]

    if failures:
        raise AssertionError(
            "RS05 preflight failed:\n"
            + "\n".join(
                f"- {x}"
                for x in failures
            )
        )

    log("\nRS05 PREFLIGHT: ALL REQUIRED CHECKS PASSED")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    section("RS05 PREFLIGHT — SECTION 0")

    log(f"Project root: {ROOT}")
    log(f"RS04:         {RS04_PATH}")
    log(f"Quarantine:   {QUARANTINE_PATH}")
    log(f"Raw zip:      {RAW_ZIP}")

    try:
        rs04 = load_rs04()

        quarantine, _ = (
            load_and_reconstruct_normalized_loss(
                rs04
            )
        )

        label_diagnostics(
            rs04,
            quarantine,
        )

        split_adjustment_scan()

        raw_returns = load_raw_returns()

        own_features = build_own_vol_features(
            raw_returns
        )

        rs04b = build_rs04b(
            rs04,
            own_features,
        )

        truncation = truncation_test(
            own_features,
            raw_returns,
        )

        final_sanity(
            rs04,
            rs04b,
            truncation,
        )

        section("OUTPUTS")

        log(f"RS04b panel:       {RS04B_PATH}")
        log(f"Label diagnostics: {LABEL_DIAG_PATH}")
        log(f"Split scan:        {SPLIT_SCAN_PATH}")
        log(f"Truncation audit:  {TRUNCATION_AUDIT_PATH}")
        log(f"Own-vol summary:   {OWNVOL_SUMMARY_PATH}")
        log(f"Feature manifest:  {MANIFEST_PATH}")
        log(f"Log:               {LOG_PATH}")

        log("\nRS05 PREFLIGHT COMPLETE")

    except Exception as exc:
        log("\n" + "!" * 100)
        log("RS05 PREFLIGHT FAILED")
        log("!" * 100)
        log(repr(exc))
        raise

    finally:
        save_log()


if __name__ == "__main__":
    main()
