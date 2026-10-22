
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# =============================================================================
# PATHS
# =============================================================================

BASE = Path(r"D:\reliability_layer")
RS_DIR = BASE / "data" / "monitoring" / "regime_shifts"
RS05_DIR = RS_DIR / "RS05"

DEFAULT_PREDICTIONS = RS05_DIR / "RS05_oos_predictions_H10_main.csv.gz"
RS04B_PATH = RS_DIR / "RS04b_causal_reliability_panel.csv"
OUT_DIR = RS05_DIR

# =============================================================================
# DESIGN
# =============================================================================

HEADLINE_START = pd.Timestamp("2020-01-01")

HORIZON = 10

# Outcome-defined failure event:
# fraction of 24 tickers with coverage_failure_10d == 1.
FAILURE_FRACTION = 0.50

# Sensitivity event thresholds.
FAILURE_SENS = [0.25, 0.75]

# Merge nearby pieces of the same event.
MAX_GAP_TRADING_DATES = 5

# Look back before each event onset.
PRE_ONSET_DATES = 20

# Main cross-sectional alert rule.
MAIN_CROSS_SECTION = 0.25
CROSS_SECTION_SENS = [0.10, 0.50]

# Main matched alert budget.
BUDGET = 0.10

# RS05a calibration block and fit requirements.
CAL_LEN = 252
MIN_FIT_DATES = 300


# =============================================================================
# LOADING
# =============================================================================

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found:\n{path}")

    df = pd.read_csv(path, compression="gzip", low_memory=False)

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
        "thr10",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Prediction file missing required columns: {missing}")

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["p_cal"] = pd.to_numeric(df["p_cal"], errors="coerce")
    df["thr10"] = pd.to_numeric(df["thr10"], errors="coerce")

    df = df.dropna(
        subset=["forecast_date", "ticker", "model", "target", "H", "spec", "algo", "y"]
    ).copy()

    return df


def load_vix(panel_path: Path) -> pd.DataFrame:
    if not panel_path.exists():
        raise FileNotFoundError(f"RS04b panel not found:\n{panel_path}")

    df = pd.read_csv(panel_path, low_memory=False)

    required = {"forecast_date", "vix", "label_end_date_10d"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"RS04b is missing VIX/calibration columns: {missing}"
        )

    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df["label_end_date_10d"] = pd.to_datetime(
        df["label_end_date_10d"], errors="coerce"
    )
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")

    d = (
        df[["forecast_date", "vix", "label_end_date_10d"]]
        .dropna(subset=["forecast_date", "vix"])
        .groupby("forecast_date", as_index=False)
        .agg(
            vix=("vix", "mean"),
            label_end_date_10d=("label_end_date_10d", "min"),
        )
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    return d


# =============================================================================
# RS05 FOLD RECONSTRUCTION
# =============================================================================

def reconstruct_quarter_folds(
    prediction_dates: pd.Series,
    vix_dates: pd.Series,
    vix_label_end: pd.Series,
) -> dict[str, dict]:
    """
    Reconstruct the RS05a quarterly test/calibration dates.

    Test dates come from the OOS prediction file.
    Calibration dates mirror RS05a:
        latest 252 dates whose label_end_date_10d < test_first.
    Fit dates are not needed for the VIX threshold.
    """
    idx = pd.DatetimeIndex(
        pd.to_datetime(prediction_dates, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    quarter_index = pd.PeriodIndex(idx, freq="Q")

    vix_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(vix_dates, errors="coerce"),
            "label_end": pd.to_datetime(vix_label_end, errors="coerce"),
        }
    ).dropna(subset=["date", "label_end"])

    vix_frame = vix_frame.drop_duplicates("date").sort_values("date")

    folds: dict[str, dict] = {}

    for q in sorted(quarter_index.unique()):
        test_dates = idx[quarter_index == q]

        if len(test_dates) < 10:
            continue

        test_first = test_dates.min()

        eligible = vix_frame.loc[
            vix_frame["label_end"] < test_first, "date"
        ].drop_duplicates().sort_values()

        if len(eligible) < CAL_LEN + MIN_FIT_DATES:
            continue

        cal_dates = pd.DatetimeIndex(eligible.iloc[-CAL_LEN:])

        folds[str(q)] = {
            "test_dates": pd.DatetimeIndex(test_dates),
            "cal_dates": cal_dates,
            "test_first": test_first,
        }

    return folds


def attach_fold_to_predictions(
    df: pd.DataFrame,
    folds: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for q, f in folds.items():
        test_set = set(f["test_dates"])
        part = df[df["forecast_date"].isin(test_set)].copy()
        part["fold"] = q
        rows.append(part)

    if not rows:
        raise RuntimeError("Could not attach any quarterly folds.")

    out = pd.concat(rows, ignore_index=True)
    return out


# =============================================================================
# VIX BASELINE
# =============================================================================

def build_vix_thresholds(
    vix_df: pd.DataFrame,
    folds: dict[str, dict],
    budget: float,
) -> pd.DataFrame:
    """
    For each RS05 test fold, choose a raw-VIX threshold from the same
    252-date causal calibration block such that approximately 'budget'
    of calibration dates exceed it.
    """
    rows = []

    vix_map = vix_df.set_index("forecast_date")["vix"]

    for q, f in folds.items():
        cal = vix_map.reindex(f["cal_dates"]).dropna()

        if len(cal) < CAL_LEN:
            raise RuntimeError(
                f"{q}: only {len(cal)} usable VIX calibration dates; expected {CAL_LEN}."
            )

        threshold = float(cal.quantile(1.0 - budget))

        for d in f["test_dates"]:
            if d in vix_map.index:
                rows.append(
                    {
                        "fold": q,
                        "forecast_date": d,
                        "vix": float(vix_map.loc[d]),
                        "vix_threshold": threshold,
                        "vix_alert": float(vix_map.loc[d]) >= threshold,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No VIX test rows were produced.")

    return out.sort_values("forecast_date").reset_index(drop=True)


# =============================================================================
# EVENT EXTRACTION
# =============================================================================

def build_failure_daily(
    model_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build daily realized failure fraction from one model's T2 labels.
    """
    g = (
        model_outcomes
        .groupby("forecast_date")
        .agg(
            n_tickers=("ticker", "nunique"),
            failure_fraction=("y", "mean"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    g["complete_24_tickers"] = g["n_tickers"].eq(24)
    return g


def extract_events(
    daily: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Contiguous periods with failure_fraction >= threshold.
    Small gaps of <= MAX_GAP_TRADING_DATES are merged.
    """
    x = daily[
        daily["complete_24_tickers"]
        & (daily["forecast_date"] >= HEADLINE_START)
    ].copy()

    x["is_failure"] = x["failure_fraction"] >= threshold

    events = []
    i = 0
    event_id = 0

    while i < len(x):
        if not bool(x.iloc[i]["is_failure"]):
            i += 1
            continue

        event_id += 1
        start_i = i
        end_i = i

        while end_i + 1 < len(x):
            next_i = end_i + 1

            if bool(x.iloc[next_i]["is_failure"]):
                end_i = next_i
                continue

            gap = next_i - end_i - 1

            if gap <= MAX_GAP_TRADING_DATES:
                found = None
                j = next_i + 1

                while j < len(x) and (j - next_i - 1) <= MAX_GAP_TRADING_DATES:
                    if bool(x.iloc[j]["is_failure"]):
                        found = j
                        break
                    j += 1

                if found is not None:
                    end_i = found
                    continue

            break

        block = x.iloc[start_i:end_i + 1].copy()

        events.append(
            {
                "event_id": event_id,
                "start_date": block["forecast_date"].min(),
                "end_date": block["forecast_date"].max(),
                "n_dates": int(len(block)),
                "max_failure_fraction": float(block["failure_fraction"].max()),
                "mean_failure_fraction": float(block["failure_fraction"].mean()),
            }
        )

        i = end_i + 1

    return pd.DataFrame(events)


# =============================================================================
# SIGNAL CONSTRUCTION
# =============================================================================

def model_signal_date_level(
    model_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Model-based date-level alert fractions.

    The prediction file contains fold-frozen thr10, so the threshold is
    already causal and calibration-derived.
    """
    x = model_predictions.copy()

    x["alert"] = x["p_cal"].to_numpy(float) >= x["thr10"].to_numpy(float)

    daily = (
        x.groupby("forecast_date")
        .agg(
            n_tickers=("ticker", "nunique"),
            alert_fraction=("alert", "mean"),
            mean_p=("p_cal", "mean"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    daily["complete_24_tickers"] = daily["n_tickers"].eq(24)

    return daily


def vix_signal_date_level(
    vix_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    x = vix_thresholds.copy()

    daily = (
        x.groupby("forecast_date")
        .agg(
            n_tickers=("forecast_date", "size"),
            vix=("vix", "mean"),
            vix_threshold=("vix_threshold", "mean"),
            vix_alert=("vix_alert", "mean"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    # VIX is a market-wide signal: on an alert date all 24 tickers would be
    # considered alerted.
    daily["alert_fraction"] = np.where(
        daily["vix_alert"] >= 1.0,
        1.0,
        0.0,
    )
    daily["mean_p"] = np.nan
    daily["complete_24_tickers"] = True

    return daily


# =============================================================================
# ALERT / EVENT ANALYSIS
# =============================================================================

def trading_lead(
    all_dates: pd.Series,
    alert_date: pd.Timestamp,
    event_date: pd.Timestamp,
) -> int:
    dates = (
        pd.to_datetime(all_dates, errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    window = dates[
        (dates >= alert_date)
        & (dates <= event_date)
    ]
    return max(len(window) - 1, 0)


def analyze_signal(
    signal_name: str,
    signal_daily: pd.DataFrame,
    events: pd.DataFrame,
    cross_section: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each event:
      - inspect the 20 prior trading dates + onset date
      - alert when date-level alert_fraction >= cross_section
      - record first alert and lead time

    Also calculates descriptive false-alarm runs.
    """
    signal_daily = signal_daily.copy()

    all_dates = signal_daily["forecast_date"]

    event_rows = []

    event_or_pre = pd.Series(False, index=signal_daily.index)

    for _, event in events.iterrows():
        start = pd.Timestamp(event["start_date"])
        end = pd.Timestamp(event["end_date"])

        prior = (
            all_dates[all_dates < start]
            .drop_duplicates()
            .sort_values()
            .tail(PRE_ONSET_DATES)
        )

        window_dates = pd.concat(
            [prior, pd.Series([start])],
            ignore_index=True,
        ).drop_duplicates().sort_values()

        window = signal_daily[
            signal_daily["forecast_date"].isin(window_dates)
        ].copy()

        # Mark event + pre-event dates for false-alarm accounting.
        mask = (
            signal_daily["forecast_date"].between(start, end, inclusive="both")
            | signal_daily["forecast_date"].isin(prior)
        )
        event_or_pre.loc[mask.index[mask]] = True

        eligible = window[
            window["alert_fraction"] >= cross_section
        ]

        if eligible.empty:
            first_alert = pd.NaT
            lead = math.nan
            hit = 0
        else:
            first_alert = pd.Timestamp(eligible["forecast_date"].min())
            lead = trading_lead(all_dates, first_alert, start)
            hit = int(first_alert < start)

        event_rows.append(
            {
                "signal": signal_name,
                "event_id": int(event["event_id"]),
                "failure_start": start,
                "failure_end": end,
                "failure_n_dates": int(event["n_dates"]),
                "failure_peak_fraction": float(event["max_failure_fraction"]),
                "first_alert_date": first_alert,
                "hit": hit,
                "lead_days": lead,
            }
        )

    # False alarms: alert dates outside event/pre-event windows.
    signal_daily["event_or_pre_event"] = event_or_pre.to_numpy()
    signal_daily["alert"] = signal_daily["alert_fraction"] >= cross_section

    fa = signal_daily[
        signal_daily["alert"]
        & (~signal_daily["event_or_pre_event"])
    ].copy()

    false_runs = []
    if not fa.empty:
        fa = fa.sort_values("forecast_date").reset_index(drop=True)
        start_idx = 0
        run_id = 0

        for i in range(1, len(fa)):
            date_gap = (
                fa.loc[i, "forecast_date"]
                - fa.loc[i - 1, "forecast_date"]
            ).days

            if date_gap > 7:
                block = fa.iloc[start_idx:i]
                run_id += 1
                false_runs.append(
                    {
                        "signal": signal_name,
                        "run_id": run_id,
                        "start_date": block["forecast_date"].min(),
                        "end_date": block["forecast_date"].max(),
                        "n_dates": int(len(block)),
                        "max_alert_fraction": float(block["alert_fraction"].max()),
                        "max_mean_p": (
                            float(block["mean_p"].max())
                            if block["mean_p"].notna().any()
                            else math.nan
                        ),
                    }
                )
                start_idx = i

        block = fa.iloc[start_idx:]
        run_id += 1
        false_runs.append(
            {
                "signal": signal_name,
                "run_id": run_id,
                "start_date": block["forecast_date"].min(),
                "end_date": block["forecast_date"].max(),
                "n_dates": int(len(block)),
                "max_alert_fraction": float(block["alert_fraction"].max()),
                "max_mean_p": (
                    float(block["mean_p"].max())
                    if block["mean_p"].notna().any()
                    else math.nan
                ),
            }
        )

    return pd.DataFrame(event_rows), pd.DataFrame(false_runs)


def summarize_events(
    event_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for signal_name, g in event_results.groupby("signal"):
        n = len(g)
        hits = int(g["hit"].sum())
        leads = pd.to_numeric(
            g.loc[g["hit"] == 1, "lead_days"],
            errors="coerce",
        ).dropna()

        pval = (
            float(
                binomtest(
                    hits,
                    n,
                    p=0.5,
                    alternative="greater",
                ).pvalue
            )
            if n
            else math.nan
        )

        rows.append(
            {
                "signal": signal_name,
                "n_events": n,
                "n_hits": hits,
                "hit_rate": hits / n if n else math.nan,
                "sign_test_p_gt_0.5": pval,
                "mean_lead_days": float(leads.mean()) if len(leads) else math.nan,
                "median_lead_days": float(leads.median()) if len(leads) else math.nan,
                "min_lead_days": float(leads.min()) if len(leads) else math.nan,
                "max_lead_days": float(leads.max()) if len(leads) else math.nan,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-alert RS05c event study: S3 vs S4 vs S6 vs a causal VIX-only rule."
        )
    )
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
    )
    parser.add_argument(
        "--rs04b",
        default=str(RS04B_PATH),
    )
    parser.add_argument(
        "--cross-section",
        type=float,
        default=MAIN_CROSS_SECTION,
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=BUDGET,
    )
    parser.add_argument(
        "--models",
        default="ALL",
        help="ALL or comma-separated model names such as HS,GARCH-t",
    )

    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    vix = load_vix(Path(args.rs04b))

    # We only need T2/H=10/logit and the three requested specs.
    p = predictions[
        (predictions["target"] == "T2")
        & (predictions["H"] == HORIZON)
        & (predictions["algo"] == "logit")
        & (predictions["forecast_date"] >= HEADLINE_START)
        & (predictions["spec"].isin(["S3", "S4", "S6"]))
    ].copy()

    if p.empty:
        raise RuntimeError(
            "No T2/H10/logit S3/S4/S6 predictions found."
        )

    available_models = sorted(p["model"].astype(str).unique())

    if args.models.upper() == "ALL":
        models = available_models
    else:
        models = [
            x.strip()
            for x in args.models.split(",")
            if x.strip()
        ]
        missing = sorted(set(models) - set(available_models))
        if missing:
            raise ValueError(
                f"Requested models unavailable: {missing}. "
                f"Available={available_models}"
            )

    # Make sure each spec has the same outcome rows.
    outcome = (
        p[p["spec"] == "S6"]
        [["forecast_date", "ticker", "model", "y"]]
        .drop_duplicates(["forecast_date", "ticker", "model"])
    )

    # Reconstruct the RS05 quarterly folds.
    folds = reconstruct_quarter_folds(
        prediction_dates=p["forecast_date"],
        vix_dates=vix["forecast_date"],
        vix_label_end=vix["label_end_date_10d"],
    )

    p = attach_fold_to_predictions(p, folds)

    # VIX-only signal uses the same causal 10% budget on each fold.
    vix_thresholds = build_vix_thresholds(
        vix_df=vix,
        folds=folds,
        budget=args.budget,
    )

    signal_summaries = []
    all_event_results = []
    all_false_runs = []
    all_events = []

    print("=" * 112)
    print("MATCHED-ALERT RS05c EVENT STUDY")
    print("=" * 112)
    print(f"Predictions:             {args.predictions}")
    print(f"RS04b:                   {args.rs04b}")
    print(f"Target:                  T2 / coverage_failure_10d")
    print(f"Algorithms:              logistic")
    print(f"Headline start:          {HEADLINE_START.date()}")
    print(f"Failure event threshold: {FAILURE_FRACTION:.0%}")
    print(f"Alert budget:            {args.budget:.0%}")
    print(f"Cross-sectional rule:    >= {args.cross_section:.0%}")
    print(f"Pre-onset window:        {PRE_ONSET_DATES} trading dates")
    print(f"Models:                  {models}")
    print()

    for model_name in models:
        print("=" * 112)
        print(f"MODEL: {model_name}")
        print("=" * 112)

        # Outcomes are independent of spec; use S6 rows to define realized events.
        m_out = outcome[outcome["model"] == model_name].copy()

        daily_out = build_failure_daily(m_out)

        events = extract_events(
            daily_out,
            threshold=FAILURE_FRACTION,
        )

        all_events.append(
            events.assign(model=model_name)
        )

        print(
            f"Main failure events: {len(events)} | "
            f"complete 24-ticker dates: "
            f"{int(daily_out['complete_24_tickers'].sum()):,}"
        )

        if not events.empty:
            print(events.to_string(index=False))

        # -------------------------
        # S3 / S4 / S6
        # -------------------------
        for spec in ["S3", "S4", "S6"]:
            mp = p[
                (p["model"] == model_name)
                & (p["spec"] == spec)
            ].copy()

            daily_signal = model_signal_date_level(mp)

            event_results, false_runs = analyze_signal(
                signal_name=spec,
                signal_daily=daily_signal,
                events=events,
                cross_section=args.cross_section,
            )

            if not event_results.empty:
                event_results["model"] = model_name
                event_results["spec"] = spec
                all_event_results.append(event_results)

            if not false_runs.empty:
                false_runs["model"] = model_name
                false_runs["spec"] = spec
                all_false_runs.append(false_runs)

            # Realized alert burden on ticker-days and dates.
            row_alert = (
                mp["p_cal"].to_numpy(float)
                >= mp["thr10"].to_numpy(float)
            )
            ticker_day_rate = float(np.mean(row_alert))

            daily_alert = daily_signal[
                daily_signal["complete_24_tickers"]
            ].copy()

            date_alert_rate = float(
                np.mean(
                    daily_alert["alert_fraction"].to_numpy(float)
                    >= args.cross_section
                )
            )

            signal_summaries.append(
                {
                    "model": model_name,
                    "signal": spec,
                    "budget": args.budget,
                    "cross_section": args.cross_section,
                    "ticker_day_alert_rate": ticker_day_rate,
                    "date_alert_rate": date_alert_rate,
                    "mean_cross_section_alert_fraction": float(
                        daily_alert["alert_fraction"].mean()
                    ),
                }
            )

        # -------------------------
        # VIX-only
        # -------------------------
        vd = vix_thresholds.copy()

        # The outcome event is model-specific, but the VIX signal is shared.
        event_results, false_runs = analyze_signal(
            signal_name="VIX",
            signal_daily=vix_signal_date_level(vd),
            events=events,
            cross_section=args.cross_section,
        )

        if not event_results.empty:
            event_results["model"] = model_name
            event_results["spec"] = "VIX"
            all_event_results.append(event_results)

        if not false_runs.empty:
            false_runs["model"] = model_name
            false_runs["spec"] = "VIX"
            all_false_runs.append(false_runs)

        vd_complete = vd.copy()
        ticker_day_rate = float(vd_complete["vix_alert"].mean())
        date_alert_rate = ticker_day_rate

        signal_summaries.append(
            {
                "model": model_name,
                "signal": "VIX",
                "budget": args.budget,
                "cross_section": args.cross_section,
                "ticker_day_alert_rate": ticker_day_rate,
                "date_alert_rate": date_alert_rate,
                "mean_cross_section_alert_fraction": date_alert_rate,
            }
        )

    event_results_df = (
        pd.concat(all_event_results, ignore_index=True)
        if all_event_results else pd.DataFrame()
    )
    false_runs_df = (
        pd.concat(all_false_runs, ignore_index=True)
        if all_false_runs else pd.DataFrame()
    )
    events_df = (
        pd.concat(all_events, ignore_index=True)
        if all_events else pd.DataFrame()
    )
    burden_df = pd.DataFrame(signal_summaries)

    summary_df = (
        summarize_events(event_results_df)
        if not event_results_df.empty
        else pd.DataFrame()
    )

    # Add model to summary.
    if not event_results_df.empty:
        summary_df = (
            event_results_df.groupby("signal", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "model": g["model"].iloc[0]
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
            .merge(
                summarize_events(event_results_df),
                on="signal",
                how="right",
            )
        )
        # The same signal labels occur separately by model, so rebuild cleanly.
        summary_rows = []
        for (model_name, signal_name), g in event_results_df.groupby(
            ["model", "signal"]
        ):
            n = len(g)
            hits = int(g["hit"].sum())
            leads = pd.to_numeric(
                g.loc[g["hit"] == 1, "lead_days"],
                errors="coerce",
            ).dropna()

            summary_rows.append(
                {
                    "model": model_name,
                    "signal": signal_name,
                    "n_events": n,
                    "n_hits": hits,
                    "hit_rate": hits / n if n else math.nan,
                    "sign_test_p_gt_0.5": (
                        float(
                            binomtest(
                                hits,
                                n,
                                p=0.5,
                                alternative="greater",
                            ).pvalue
                        )
                        if n else math.nan
                    ),
                    "mean_lead_days": float(leads.mean()) if len(leads) else math.nan,
                    "median_lead_days": float(leads.median()) if len(leads) else math.nan,
                    "min_lead_days": float(leads.min()) if len(leads) else math.nan,
                    "max_lead_days": float(leads.max()) if len(leads) else math.nan,
                }
            )

        summary_df = pd.DataFrame(summary_rows)

    print()
    print("=" * 112)
    print("MATCHED-ALERT EVENT RESULTS")
    print("=" * 112)
    if summary_df.empty:
        print("No event results.")
    else:
        print(
            summary_df.sort_values(["model", "signal"]).to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            )
        )

    print()
    print("=" * 112)
    print("REALIZED ALERT BURDEN")
    print("=" * 112)
    print(
        burden_df.sort_values(["model", "signal"]).to_string(
            index=False,
            float_format=lambda x: f"{x:.6g}",
        )
    )

    print()
    print("=" * 112)
    print("IMPORTANT INTERPRETATION")
    print("=" * 112)
    print(
        "S3/S4/S6 use their own fold-frozen RS05a thr10 thresholds, so they "
        "receive the same nominal 10% calibration alert budget."
    )
    print(
        "VIX uses a causal fold-specific 90th-percentile threshold over the "
        "same 252-date calibration block, creating the same nominal 10% budget."
    )
    print(
        "The VIX signal is market-wide: when it alerts, all 24 tickers count "
        "as alerted, so its cross-sectional fraction is 100% on alert dates."
    )
    print(
        "Event definitions are outcome-based and identical across signals "
        "for a given VaR model."
    )
    print(
        "Event-level statistics are descriptive because the number of major "
        "failure events is small."
    )

    # Save outputs.
    events_path = OUT_DIR / "RS05c_matched_alert_failure_events_H10.csv"
    results_path = OUT_DIR / "RS05c_matched_alert_event_results_H10.csv"
    false_path = OUT_DIR / "RS05c_matched_alert_false_alarm_runs_H10.csv"
    burden_path = OUT_DIR / "RS05c_matched_alert_burden_H10.csv"
    summary_path = OUT_DIR / "RS05c_matched_alert_summary_H10.csv"
    vix_path = OUT_DIR / "RS05c_matched_alert_vix_thresholds_H10.csv"

    events_df.to_csv(events_path, index=False)
    event_results_df.to_csv(results_path, index=False)
    false_runs_df.to_csv(false_path, index=False)
    burden_df.to_csv(burden_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    vix_thresholds.to_csv(vix_path, index=False)

    print()
    print("Saved:")
    print(events_path)
    print(results_path)
    print(false_path)
    print(burden_path)
    print(summary_path)
    print(vix_path)


if __name__ == "__main__":
    main()
